import argparse
import pdb
from datetime import datetime
from typing import List

import launchpad as lp
from launchpad.nodes.python import local_multi_processing
from openrlhf.datasets import RewardDataset
from openrlhf.utils import blending_datasets
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from ellm.learners.dap import DAPLearner
from ellm.types import PreferenceData
from ellm.utils.data import PreferenceDataset, get_tokenizer
from ellm.utils.launcher import get_free_port
from ellm.utils.setup import get_strategy


class Strategy:
    def __init__(self, args) -> None:
        self.args = args

    def print(self, *args):
        print(*args)

    def is_rank_0(self):
        return True


class DatasetActor:
    def __init__(self, args, strategy, rank=0, world_size=1) -> None:
        train_data, eval_data = blending_datasets(
            args.dataset,
            args.dataset_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            stopping_strategy="all_exhausted",
        )
        train_data = train_data.select(range(min(args.max_samples, len(train_data))))

        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrain, trust_remote_code=True, use_fast=True
        )
        tokenizer.padding_side = "right"

        train_dataset = RewardDataset(
            train_data,
            tokenizer,
            args.max_len,
            strategy,
            input_template=args.input_template,
            is_dpo=True,
        )

        sub_ds_size = len(train_dataset) // world_size
        rejected_key = getattr(args, "rejected_key")
        chosen_key = getattr(args, "chosen_key")

        self.samples = []
        for data in train_data:
            if data[chosen_key][0]["role"] == "user":
                prompt = data[chosen_key][0]["content"]
                chosen = data[chosen_key][1]["content"]
            else:
                prompt = data[chosen_key][1]["content"]
                chosen = data[chosen_key][0]["content"]
            if data[rejected_key][0]["role"] == "user":
                rejected = data[rejected_key][1]["content"]
            else:
                rejected = data[rejected_key][0]["content"]

            sample = PreferenceData(prompt, chosen, rejected)
            self.samples.append(sample)

        self.train_dataloader = DataLoader(
            self.samples[rank * sub_ds_size : (rank + 1) * sub_ds_size],
            args.micro_train_batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=lambda item_list: item_list,
        )
        self.data_iter = iter(self.train_dataloader)

    def step(self, prompt: List[str]):
        del prompt  # dummy
        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_dataloader)
            data = next(self.data_iter)
        return data


def main(args):
    dummy_strategy = Strategy(args)

    # local test
    # actor = DatasetActor(args, dummy_strategy)
    # print(actor.step(None))

    # mp test
    program = lp.Program("dataset_actor_learner")
    actors = []
    local_resources = {}
    for i in range(4):
        label = f"actor_{i}"
        actors.append(
            program.add_node(
                lp.CourierNode(DatasetActor, args, dummy_strategy), label=label
            )
        )
    _gpu_offset = 0

    master_addr = "0.0.0.0"
    master_port = get_free_port()
    args.local_rank = 0
    label = "learner_0"
    master_learner = lp.PyClassNode(
        DAPLearner,
        4,
        0,
        0,
        master_addr,
        master_port,
        True,
        args,
        actors,
    )
    program.add_node(master_learner, label=label)
    local_resources[label] = local_multi_processing.PythonProcess(
        env=dict(CUDA_VISIBLE_DEVICES=str(_gpu_offset))
    )
    for i in range(1, 4):
        args.local_rank = 0
        label = f"learner_{i}"
        worker_learner = lp.PyClassNode(
            DAPLearner,
            4,
            i,
            i,
            master_addr,
            master_port,
            False,
            args,
            actors,
        )
        program.add_node(worker_learner, label=label)
        local_resources[label] = local_multi_processing.PythonProcess(
            env=dict(CUDA_VISIBLE_DEVICES=str(i + _gpu_offset))
        )

    lp.launch(
        program,
        launch_type="local_mp",
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrain", type=str, default="OpenLLMAI/Llama-3-8b-sft-mixture"
    )
    parser.add_argument("--ref_pretrain", type=str, default=None)

    # Prompts dataset
    parser.add_argument(
        "--prompt_data", type=str, default="OpenLLMAI/prompt-collection-v0.1"
    )
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--max_samples", type=int, default=10000)

    # Offline preference dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="OpenLLMAI/preference_dataset_mixture2_and_safe_pku",
    )
    parser.add_argument(
        "--dataset_probs", type=str, default="1.0", help="sampling probs for datasets"
    )

    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_dpo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB

    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--rollout_batch_size", type=int, default=512)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--micro_pi_buffer_maxlen", type=int, default=999999999)
    parser.add_argument("--micro_r_buffer_maxlen", type=int, default=999999999)
    parser.add_argument("--prompt_max_length", type=int, default=1024)
    parser.add_argument("--generate_max_length", type=int, default=1024)

    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument(
        "--ipo", action="store_true", default=False
    )  # IPO https://arxiv.org/pdf/2310.12036v2.pdf
    parser.add_argument(
        "--label_smoothing", type=float, default=0.0
    )  # cDPO https://arxiv.org/pdf/2305.18290.pdf
    parser.add_argument(
        "--gamma_beta_ratio", type=float, default=0.5
    )  # SimPO https://arxiv.org/pdf/2405.14734
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local_rank for deepspeed"
    )
    parser.add_argument("--zero_stage", type=int, default=3)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--ref_offload", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")

    # custom dataset key name
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--chosen_key", type=str, default="chosen")
    parser.add_argument("--rejected_key", type=str, default="rejected")
    parser.add_argument("--input_key", type=str, default="input")
    parser.add_argument("--input_template", type=str, default="")
    parser.add_argument("--apply_chat_template", action="store_true", default=True)

    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="sea_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()

    if args.ref_pretrain is None or args.ref_pretrain == "":
        args.ref_pretrain = args.pretrain

    main(args)

"""
python tests/lp_learner.py --apply_chat_template --chosen_key chosen --rejected_key rejected --micro_train_batch_size 2 --input_key context_messages
"""
