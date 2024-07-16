"""The training loop for Online Iterative Direct Alignment from Preferences."""

import itertools
import math
import pdb

import ray
import torch
from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.utils import DeepspeedStrategy, blending_datasets
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers.trainer import get_scheduler

from ellm.model import Model
from ellm.ray_utils import BaseDAP
from ellm.utils import get_tokenizer


@ray.remote
class Learner(BaseDAP):

    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        model = Model(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
        )

        # configure tokenizer
        self.tokenizer = get_tokenizer(
            pretrain,
            model.model,
            "left",
            strategy,
            use_fast=not strategy.args.disable_fast_tokenizer,
        )

        strategy.print(model)
        self.prepare_datasets()

        args = strategy.args

        if args.enable_ema:
            ema_model = Model(
                pretrain,
                use_flash_attention_2=strategy.args.flash_attn,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
            )
        else:
            ema_model = None

        # configure optimizer
        model_optim = strategy.create_optimizer(
            model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2
        )

        # configure scheduler
        num_update_steps_per_episodes = (
            int(
                len(self.prompts_dataloader)
                * (args.micro_rollout_batch_size / args.micro_train_batch_size)
            )
            * args.max_epochs
            // strategy.accumulated_gradient
        )

        max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)
        self.max_steps = max_steps

        model_scheduler = get_scheduler(
            "cosine_with_min_lr",
            model_optim,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": args.gradient_checkpointing_use_reentrant
                }
            )

        # prepare models/optimizers...
        self.model, self.model_optim, self.model_scheduler = strategy.prepare(
            (model, model_optim, model_scheduler),
            is_rlhf=True,
        )

        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None

    def prepare_datasets(self):
        strategy = self.strategy
        args = self.strategy.args

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
            prompts_data, self.tokenizer, strategy, input_template=args.input_template
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            prompts_dataset, args.micro_rollout_batch_size, True, True
        )

        if args.pretrain_data:
            pretrain_data = blending_datasets(
                args.pretrain_data,
                args.pretrain_data_probs,
                strategy,
                args.seed,
                return_eval=False,
            )
            pretrain_max_len = (
                args.max_len
                if args.max_len
                else args.prompt_max_len + args.generate_max_len
            )
            pretrain_dataset = SFTDataset(
                pretrain_data.select(
                    range(
                        min(len(pretrain_data), args.max_epochs * len(prompts_dataset))
                    )
                ),
                self.tokenizer,
                pretrain_max_len,
                strategy,
                pretrain_mode=True,
            )
            self.pretrain_dataloader = itertools.cycle(
                iter(
                    strategy.setup_dataloader(
                        pretrain_dataset,
                        args.micro_train_batch_size,
                        True,
                        True,
                        pretrain_dataset.collate_fn,
                    )
                )
            )
        else:
            self.pretrain_dataloader = None

    def max_steps(self):
        """Return the maximum number of steps."""
        return self.max_steps

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            self.tokenizer,
            args.save_path,
        )

    def rollout(self):
        # simple round-robin
        rank = torch.distributed.get_rank()
        print(rank)

    def policy_optimization(self):
        pass

    def run(self):
        step = 1
        train_interval = 100
        for episode in range(self.cfg.num_episodes):
            for prompts in self.prompts_dataloader:
                self.rollout(prompts)

                pdb.set_trace()

                if step % train_interval == 0:
                    self.policy_optimization()


class LearnerGroup:

    def __init__(self, num_gpus_per_node: int, num_gpus_per_actor: int = 1) -> None:
        self._num_gpus_per_node = num_gpus_per_node
        self._init_learners(num_gpus_per_actor)

    def _init_learners(self, num_gpus_per_actor: int):
        pg = None
        world_size = self._num_gpus_per_node

        if self._num_gpus_per_node > 1:
            bundles = [{"GPU": self._num_gpus_per_node, "CPU": self._num_gpus_per_node}]
            pg = placement_group(bundles, strategy="STRICT_SPREAD")
            ray.get(pg.ready())
        if pg:
            learner_master = Learner.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0
                ),
            ).remote(world_size, 0, 0, None, None)
        else:
            learner_master = Learner.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
            ).remote(world_size, 0, 0, None, None)
        learners = [learner_master]

        # Create worker ray-actors
        if world_size > 1:
            master_addr, master_port = ray.get(
                learner_master.get_master_addr_port.remote()
            )
            for rank in range(1, world_size):
                local_rank = rank % self._num_gpus_per_node
                if pg:
                    learner_worker_actor = Learner.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=rank
                            // self._num_gpus_per_node,
                        ),
                    ).remote(world_size, rank, local_rank, master_addr, master_port)
                else:
                    learner_worker_actor = Learner.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                    ).remote(world_size, rank, local_rank, master_addr, master_port)
                learners.append(learner_worker_actor)

        self._learners_handle = learners

    def async_init_from_pretrain(self, strategy, pretrain_model):
        return [
            learner.init_model_from_pretrained.remote(strategy, pretrain_model)
            for learner in self._learners_handle
        ]
