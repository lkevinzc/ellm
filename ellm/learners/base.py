import abc
import dataclasses
import math
import os
import socket
import time
from argparse import Namespace
from collections import deque
from datetime import datetime
from typing import Any, Dict, List
from warnings import warn

import deepspeed
import launchpad as lp
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tree
import vllm
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers.trainer import get_scheduler

from ellm.actor import Actor
from ellm.model import LLM
from ellm.preference import PreferenceCollector
from ellm.types import DAPAlgo, PreferenceData
from ellm.utils.data import (PreferenceDataset, PromptDataset,
                             blending_datasets, get_tokenizer)
from ellm.utils.deepspeed import get_strategy
from ellm.utils.distributed import (init_process_group,
                                    node_ip_address_from_perspective,
                                    torch_type_codec)
from ellm.utils.ipc import PlasmaShmServer
from ellm.utils.launcher import DistributedLauncher


class LearnerBase(abc.ABC, DistributedLauncher):
    """Learner updates the LLM policy from preference data."""

    def __init__(
        self,
        world_size: int,
        rank: int,
        local_rank: int,
        master_addr: str,
        master_port: str,
        is_master: bool,
        args: Namespace,
        actors: List[Actor],
        ipc_server: PlasmaShmServer,
    ) -> None:
        super().__init__(
            world_size, rank, local_rank, master_addr, master_port, is_master
        )
        self.args = args
        self.actors = actors
        self.ipc_server = ipc_server

    def _init(self, args: Namespace, actors: List[Actor]) -> None:
        args, strategy = get_strategy(args)
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
        self.algo = args.dap_algo

        if self.algo != DAPAlgo.SimPO:
            strategy.print("Running reference-based algorithm... (DPO, IPO, etc.)")
            assert args.ref_pretrain, "Reference model must be non-empty"
            ref_model = LLM(
                args.ref_pretrain,
                use_flash_attention_2=args.flash_attn,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=args.ref_offload),
            )
        else:
            strategy.print("Running reference-free algorithm... (SimPO)")

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
        prompts_data, eval_prompts_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=True,
            eval_sample=args.max_eval,
        )
        prompts_data = prompts_data.select(
            range(min(args.max_samples, len(prompts_data)))
        )
        prompts_dataset = PromptDataset(
            prompts_data,
            tokenizer,
            strategy,
            input_template=args.input_template,
            get_reference=True,
        )
        prompts_dataloader = strategy.setup_dataloader(
            prompts_dataset,
            args.micro_rollout_batch_size,
            pin_memory=True,
            shuffle=False,
        )
        strategy.print("Prompt dataset example:")
        strategy.print("Processed:", prompts_dataset[0][0])
        strategy.print("Raw:", prompts_dataset[0][1])
        strategy.print("Prompt dataloader len:", len(prompts_dataloader))

        if strategy.is_rank_0:
            eval_prompts_dataset = PromptDataset(
                eval_prompts_data,
                tokenizer,
                strategy,
                input_template=args.input_template,
                get_reference=True,
            )
            self.eval_prompts_dataloader = DataLoader(
                eval_prompts_dataset,
                batch_size=args.micro_rollout_batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )
            # self.sample_eval_indices = np.random.choice(len(eval_prompts_dataset), 10)

        # configure scheduler
        num_policy_sgd_steps_per_episodes = int(
            len(prompts_dataset) // args.train_batch_size
        )
        max_steps = math.ceil(
            args.num_episodes
            * num_policy_sgd_steps_per_episodes
            * args.max_step_adjustment
        )
        scheduler = get_scheduler(
            "cosine_with_min_lr",
            optimizer,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
        )
        strategy.print(
            f"num_policy_sgd_steps_per_episodes={num_policy_sgd_steps_per_episodes}; max_steps={max_steps}"
        )

        # prepare models/optimizers...
        if self.algo != DAPAlgo.SimPO:
            ((self.model, self.optimizer, self.scheduler), self.ref_model) = (
                strategy.prepare(
                    (model, optimizer, scheduler),
                    ref_model,
                    is_rlhf=True,
                )
            )
        else:
            (self.model, self.optimizer, self.scheduler) = strategy.prepare(
                (model, optimizer, scheduler),
                is_rlhf=True,
            )
            self.ref_model = None

        # load checkpoint
        if args.load_checkpoint:
            strategy.print("Load checkpoint: ", args.save_path)

        exp_name = args.wandb_run_name + "_" + datetime.now().strftime("%m%dT%H:%M")
        self.save_path = os.path.join(args.save_path, exp_name)
        os.makedirs(self.save_path, exist_ok=True)

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
                name=exp_name,
                config=args.__dict__,
                reinit=True,
            )

        self.preference_collector = PreferenceCollector(
            actors,
            tokenizer,
            self.ipc_server,
            args.prompt_max_length,
            strategy,
            self._wandb,
        )
        self.pi_buffer = deque(maxlen=args.micro_pi_buffer_maxlen)

        self.strategy = strategy
        self.tokenizer = tokenizer
        self.prompts_dataloader = prompts_dataloader
        self.update_interval = args.rollout_batch_size // (
            strategy.world_size * args.micro_rollout_batch_size
        )

        self.global_step = 0
        self.pi_beta_version = 0
        self.policy_sgd_step = 0
        self.query_step = 0
        self.prompt_consumed = 0

        # Log summary of the learner
        strategy.print(self.model)
        strategy.print(self.optimizer)
        strategy.print(self.scheduler)
        strategy.pprint(vars(args))
        strategy.print(f"Update interval = {self.update_interval}")

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

        dist.barrier()

    def run(self):
        self._init(self.args, self.actors)

        steps = 1
        self.start_time = time.time()

        self.actor_info = {}

        if not self.strategy.args.debug:
            self.save_logs_and_checkpoints(
                self.args,
                self.policy_sgd_step,
                {},
            )

        for episode in range(self.args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(episode)
                self.strategy.print(f"Set DistributedSampler at epoch {episode}")
            progress_bar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for processed_prompts, raw_prompts, refs in self.prompts_dataloader:
                preference_data, self.actor_info = self.preference_collector(
                    processed_prompts, refs
                )
                self.prompt_consumed += len(processed_prompts)
                self.query_step += np.sum(
                    [not p.is_model_data for p in preference_data]
                )
                self.process_preference_data(preference_data, raw_prompts)

                if steps % self.update_interval == 0:
                    train_info = self.preference_learning(steps // self.update_interval)

                    self.save_logs_and_checkpoints(
                        self.args,
                        steps,
                        train_info,
                    )

                    if (
                        steps // self.update_interval
                    ) % self.args.sync_params_every == 0:
                        self.sync_params_to_actors()

                    if (
                        steps // self.update_interval
                    ) % self.args.buffer_clear_every == 0:
                        self.pi_buffer.clear()

                progress_bar.update()
                steps += 1

        # if self.args.dump_reward_buffer:  # For debug purpose.
        #     if not self.strategy.is_rank_0():
        #         dist.gather_object(self.r_buffer)
        #     else:
        #         gather_r_buffer = [None] * self.strategy.world_size
        #         dist.gather_object(self.r_buffer, gather_r_buffer)
        #         pd.to_pickle(
        #             gather_r_buffer, os.path.join(self.save_path, "buffer.pkl")
        #         )

        if self.strategy.is_rank_0():
            self._wandb.finish()
            lp.stop()

    def process_preference_data(self, data_list: List[PreferenceData], raw_prompts):
        for i, pref in enumerate(data_list):
            # Replace with raw prompts instead of templated ones
            new_pref = dataclasses.replace(pref, prompt=raw_prompts[i])  # shallow copy
            self.pi_buffer.append(new_pref)

    def preference_learning(self, learning_round):
        torch.cuda.empty_cache()
        dist.barrier()
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
            step_bar = tqdm(
                range(dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )
            acc_mean = []
            loss_mean = []
            reward_margin = []
            self.model.train()
            for data in dataloader:
                infos = self.learning_step(data)

                # metrics
                loss = infos.pop("loss")
                chosen_reward = infos.pop("chosen_reward")
                rejected_reward = infos.pop("rejected_reward")
                acc_mean.append((chosen_reward > rejected_reward).float().mean().item())
                loss_mean.append(loss.cpu().item())
                reward_margin.append((chosen_reward - rejected_reward).mean().item())

                step_bar.update()
                self.global_step += 1
                if self.global_step % self.strategy.accumulated_gradient == 0:
                    self.policy_sgd_step += 1

        torch.cuda.empty_cache()
        dist.barrier()

        train_info = {
            "epoch": epoch + 1,
            "chosen_reward": chosen_reward.mean().item(),
            "rejected_reward": rejected_reward.mean().item(),
            "acc_mean": np.mean(acc_mean),
            "loss_mean": np.mean(loss_mean),
            "reward_margin": np.mean(reward_margin),
            "learning_round": learning_round,
            **tree.map_structure(lambda x: x.cpu().float().mean().item(), infos),
        }
        train_info = {
            "train/%s" % k: v
            for k, v in {
                **train_info,
            }.items()
        }
        return train_info

    @abc.abstractmethod
    def learning_step(self, data):
        """Preference learning step."""

    def get_misc_info(self) -> Dict[str, Any]:
        return {
            "pi_beta_version": self.pi_beta_version,
            "global_step": self.global_step,
            "policy_sgd_step": self.policy_sgd_step,
            "pi_buffer_len": len(self.pi_buffer),
            "elapse": time.time() - self.start_time,
            "update_interval": self.update_interval,
        }

    def save_logs_and_checkpoints(self, args, batch_steps, train_info):
        # eval
        eval_info = {}
        if batch_steps % args.eval_steps == 0:
            eval_info = self.evaluate(self.eval_prompts_dataloader, batch_steps)

        # logs
        if batch_steps % args.logging_steps == 0:
            misc_info = self.get_misc_info()
            try:
                last_lr = self.scheduler.get_last_lr()[0]
                misc_info["lr"] = last_lr
            except:
                pass
            misc_info = {
                "misc/%s" % k: v
                for k, v in {
                    **misc_info,
                }.items()
            }
            logs_dict = {**train_info, **eval_info, **self.actor_info, **misc_info}
            logs_dict = self.strategy.all_reduce(logs_dict)
            logs_dict.update(
                self.strategy.all_reduce(
                    {
                        "misc/query_step": self.query_step,
                        "misc/prompt_consumed": self.prompt_consumed,
                    },
                    op="sum",
                )
            )

            if self.strategy.is_rank_0():
                if self.pi_buffer:
                    self.strategy.pprint(np.random.choice(self.pi_buffer))
                self.strategy.pprint(logs_dict)
                if self._wandb is not None:
                    self._wandb.log(logs_dict)

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

    def evaluate(self, dataloader, steps):
        self.strategy.print(f"Start generating evaluation responses at step {steps}")
        st_time = time.time()
        # 1) Let Actors cache the current behavior policy.
        if self.strategy.is_rank_0():
            done = [actor.futures.notify_eval_start() for actor in self.actors]
            _ = [d.result() for d in done]

        # 2) Push the latest policy for fast vLLM generation.
        dist.barrier()
        self._broadcast_to_vllm()

        # 3) Generate and process results

        win_rate = 0
        win_rate_prob = 0
        if self.strategy.is_rank_0():
            prompts = []
            responses = []
            references = []
            futs = []
            win_probs = []
            wins = []
            for i, (processed_prompts, _, refs) in enumerate(dataloader):
                prompts.extend(processed_prompts)
                references.extend(refs)

                actor = self.actors[i % len(self.actors)]
                fut = actor.futures.generate_and_maybe_eval(processed_prompts, refs)
                futs.append(fut)
                if len(futs) == len(self.actors) or i == len(dataloader) - 1:
                    for fut in futs:
                        resp, win_prob = fut.result()
                        responses.extend(resp)
                        wins.extend(win_prob > 0.5)
                        win_probs.extend(win_prob)
                    futs.clear()

            eval_res_path = os.path.join(self.save_path, "eval_results")
            os.makedirs(eval_res_path, exist_ok=True)
            pd.DataFrame(
                {"prompt": prompts, "reference": references, "response": responses}
            ).to_json(
                os.path.join(eval_res_path, f"{steps}.json"),
                orient="records",
                lines=True,
            )
            win_rate = np.mean(wins).item()
            win_rate_prob = np.mean(win_probs).item()

        win_rate = self.strategy.broadcast(win_rate)
        win_rate_prob = self.strategy.broadcast(win_rate_prob)

        # 4) Recover Actors' original behavior policy.
        if self.strategy.is_rank_0():
            done = [actor.futures.notify_eval_done() for actor in self.actors]
            _ = [d.result() for d in done]

        return {
            "eval/rm_win_rate": win_rate,
            "eval/rm_win_rate_prob": win_rate_prob,
            "eval/elapse": time.time() - st_time,
        }

    def sync_params_to_actors(self):
        self._broadcast_to_vllm()
        self.pi_beta_version += 1

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
                        dtype=torch_type_codec(param.dtype),
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
                    dist.broadcast(param.data, 0, group=self._model_update_group)
                    _ = [fut.result() for fut in futs]
