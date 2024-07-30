import abc
import math
import os
import socket
from collections import deque
from typing import List
from warnings import warn

import deepspeed
import launchpad as lp
import torch
import vllm
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers.trainer import get_scheduler

from ellm.actor import Actor
from ellm.model import LLM
from ellm.preference import PreferenceCollector
from ellm.types import PreferenceData
from ellm.utils.data import (PreferenceDataset, PromptDataset,
                             blending_datasets, get_tokenizer)
from ellm.utils.distributed import (init_process_group,
                                    node_ip_address_from_perspective)
from ellm.utils.launcher import DistributedLauncher
from ellm.utils.setup import get_strategy


class LearnerBase(abc.ABC, DistributedLauncher):
    """Learner updates the LLM policy from preference data."""

    def __init__(
        self,
        world_size,
        rank,
        local_rank,
        master_addr,
        master_port,
        is_master,
        args,
        actors,
    ) -> None:
        super().__init__(
            world_size, rank, local_rank, master_addr, master_port, is_master
        )
        self.args = args
        self.actors = actors

    def _init(self, args, actors: List[Actor]) -> None:
        strategy = get_strategy(args)
        strategy.setup_distributed()

        model = LLM(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            ds_config=strategy.get_ds_train_config(is_wrapped=True),
        )

        tokenizer = get_tokenizer(
            args.pretrain,
            model.model,
            "right",
            use_fast=not args.disable_fast_tokenizer,
        )

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": args.gradient_checkpointing_use_reentrant
                }
            )

        optimizer = strategy.create_optimizer(
            model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2
        )

        # prepare datasets
        prompts_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=False,
        )
        prompts_data = prompts_data.select(
            range(min(args.max_samples, len(prompts_data)))
        )
        prompts_dataset = PromptDataset(
            prompts_data, tokenizer, strategy, input_template=args.input_template
        )
        prompts_dataloader = strategy.setup_dataloader(
            prompts_dataset, args.micro_rollout_batch_size, True, True
        )

        # configure scheduler
        num_update_steps_per_episodes = (
            int(
                len(prompts_dataloader)
                * (args.micro_rollout_batch_size / args.micro_train_batch_size)
            )
            * args.max_epochs
            // strategy.accumulated_gradient
        )
        max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)
        scheduler = get_scheduler(
            "cosine_with_min_lr",
            optimizer,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
        )

        # prepare models/optimizers...
        (self.model, self.optimizer, self.scheduler) = strategy.prepare(
            (model, optimizer, scheduler),
            is_rlhf=True,
        )

        # load checkpoint
        if args.load_checkpoint:
            strategy.print("Load checkpoint: ", args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

        # logger
        self._wandb = None
        if strategy.args.use_wandb and strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=args.wandb_org,
                project=args.wandb_project,
                group=args.wandb_group,
                name=args.wandb_run_name,
                config=args.__dict__,
                reinit=True,
            )

        self.preference_collector = PreferenceCollector(
            actors, tokenizer, args.prompt_max_length, strategy, self._wandb
        )
        self.pi_buffer = deque(maxlen=args.micro_pi_buffer_maxlen)
        self.r_buffer = deque(maxlen=args.micro_r_buffer_maxlen)

        self.strategy = strategy
        self.tokenizer = tokenizer
        self.prompts_dataloader = prompts_dataloader
        self.update_step = 0

        # Log summary of the learner
        strategy.print(self.model)
        strategy.print(self.optimizer)
        strategy.print(self.scheduler)
        strategy.pprint(vars(args))

        # prepare parameter syncing to actors (reference to openrlhf)
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if strategy.is_rank_0():
            master_addr = node_ip_address_from_perspective()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            world_size = len(actors) + 1
            backend = "nccl"
            if vllm.__version__ > "0.4.2":
                backend = "gloo"
                warn(f"Using gloo backend for vLLM version {vllm.__version__}")
            futs = [
                actor.futures.init_process_group(
                    master_addr,
                    master_port,
                    i + 1,
                    world_size,
                    "ellm",
                    backend=backend,
                )
                for i, actor in enumerate(actors)
            ]
            self._model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name="ellm",
            )
            _ = [fut.result() for fut in futs]

        torch.distributed.barrier()

    def run(self):
        self._init(self.args, self.actors)
        update_interval = self.args.rollout_batch_size // (
            self.strategy.world_size * self.args.micro_rollout_batch_size
        )
        self.strategy.print(f"Update interval = {update_interval}")
        steps = 1

        for episode in range(self.args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(episode)
                self.strategy.print(f"Set DistributedSampler at epoch {episode}")
            progress_bar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for processed_prompts, raw_prompts in self.prompts_dataloader:
                preference_data = self.preference_collector(processed_prompts)
                for i, pref in enumerate(preference_data):
                    # Replace with raw prompts instead of templated ones
                    new_pref = PreferenceData(
                        prompt=raw_prompts[i],
                        chosen_response=pref.chosen_response,
                        rejected_response=pref.rejected_response,
                    )
                    self.pi_buffer.append(new_pref)
                    self.r_buffer.append(new_pref)

                if steps % update_interval == 0:
                    torch.cuda.empty_cache()
                    self.preference_learning(steps // update_interval)
                    torch.cuda.empty_cache()
                    torch.distributed.barrier()
                    self._broadcast_to_vllm()

                progress_bar.update()
                steps += 1

        lp.stop()

    def preference_learning(self, update_step):
        torch.distributed.barrier()
        dataset = PreferenceDataset(
            self.pi_buffer,
            self.tokenizer,
            self.args.prompt_max_length,
            self.args.generate_max_length,
            self.strategy,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.micro_train_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
        for epoch in range(self.args.max_epochs):
            dataloader = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.args.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            acc_mean = 0
            loss_mean = 0
            self.model.train()
            for data in dataloader:
                infos = self.learning_step(data)

                # metrics
                loss = infos["loss"]
                chosen_reward = infos["chosen_reward"]
                rejected_reward = infos["rejected_reward"]
                acc_mean = (
                    acc_mean * 0.9
                    + 0.1 * (chosen_reward > rejected_reward).float().mean().item()
                )
                loss_mean = loss_mean * 0.9 + 0.1 * loss.item()
                logs_dict = {
                    "chosen_reward": chosen_reward.mean().item(),
                    "rejected_reward": rejected_reward.mean().item(),
                    "acc_mean": acc_mean,
                    "loss_mean": loss_mean,
                }

                self.update_step += 1
                self.save_logs_and_checkpoints(
                    self.args,
                    self.update_step,
                    logs_dict,
                )

    @abc.abstractmethod
    def learning_step(self, data):
        """Preference learning step."""

    def save_logs_and_checkpoints(self, args, global_step, logs_dict={}):
        # logs
        if global_step % args.logging_steps == 0:
            logs_dict = self.strategy.all_reduce(logs_dict)

            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {
                    "train/%s" % k: v
                    for k, v in {**logs_dict, "global_step": global_step}.items()
                }
                self._wandb.log(logs)

        # eval
        # if global_step % args.eval_steps == 0:
        #     self.evaluate(self.eval_dataloader, global_step)
        # # save ckpt
        # # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        # if global_step % args.save_steps == 0:
        #     tag = f"global_step{global_step}"
        #     self.strategy.save_ckpt(
        #         self.model.model,
        #         args.ckpt_path,
        #         tag,
        #         args.max_ckpt_num,
        #         args.max_ckpt_mem,
        #     )

    def _broadcast_to_vllm(self):
        model = self.model.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # Fire all vllm engines for broadcast
            if self.strategy.is_rank_0():
                shape = (
                    param.shape
                    if self.strategy.args.zero_stage != 3
                    else param.ds_shape
                )
                futs = [
                    actor.futures.update_weight(
                        name,
                        dtype=param.dtype,
                        shape=shape,
                        empty_cache=count == num_params,
                    )
                    for actor in self.actors
                ]

            # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
            with deepspeed.zero.GatheredParameters(
                [param], enabled=self.strategy.args.zero_stage == 3
            ):
                if self.strategy.is_rank_0():
                    torch.distributed.broadcast(
                        param.data, 0, group=self._model_update_group
                    )
                    _ = [fut.result() for fut in futs]
