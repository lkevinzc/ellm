import argparse
import datetime
import pdb
from datetime import datetime
from typing import List

import ray

from ellm.learner import LearnerGroup
from ellm.utils import get_strategy


def train(args):
    # configure strategy
    strategy = get_strategy(args)

    learner = LearnerGroup(num_gpus_per_node=4)

    refs = learner.async_init_from_pretrain(strategy, "google/gemma-2b")
    ray.get(refs)

    pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_num_nodes", type=int, default=1, help="number of nodes for reference"
    )
    parser.add_argument(
        "--ref_num_gpus_per_node",
        type=int,
        default=8,
        help="number of gpus per node for reference",
    )
    parser.add_argument(
        "--reward_num_nodes",
        type=int,
        default=1,
        help="number of nodes for reward model",
    )
    parser.add_argument(
        "--reward_num_gpus_per_node",
        type=int,
        default=8,
        help="number of gpus per node for reward model",
    )
    parser.add_argument(
        "--colocate_actor_ref",
        action="store_true",
        default=False,
        help="whether to colocate reference and actor model, if true, they will share same gpus.",
    )

    parser.add_argument(
        "--actor_num_nodes", type=int, default=1, help="number of nodes for actor"
    )
    parser.add_argument(
        "--actor_num_gpus_per_node",
        type=int,
        default=8,
        help="number of gpus per node for actor",
    )
    parser.add_argument(
        "--critic_num_nodes", type=int, default=1, help="number of nodes for critic"
    )
    parser.add_argument(
        "--critic_num_gpus_per_node",
        type=int,
        default=8,
        help="number of gpus per node for critic",
    )
    parser.add_argument(
        "--colocate_critic_reward",
        action="store_true",
        default=False,
        help="whether to colocate critic and reward model, if true, they will share same gpus.",
    )

    # optional vLLM for text generation
    parser.add_argument(
        "--vllm_num_engines", type=int, default=None, help="number of vLLM Engines"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )
    parser.add_argument(
        "--vllm_sync_backend",
        type=str,
        default="nccl",
        help="DeepSpeed -> vLLM weight sync backend",
    )

    parser.add_argument("--prompt_data", type=str, default=None)
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_data", type=str, default=None)
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--reward_pretrain", type=str, default=None)
    parser.add_argument("--critic_pretrain", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=512)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024)
    parser.add_argument("--generate_max_len", type=int, default=1024)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--ptx_coef", type=float, default=0.05)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--value_clip", type=float, default=0.2)
    parser.add_argument("--lambd", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--normalize_reward", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local_rank for deepspeed"
    )
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.02)
    ## Make EMA as an optional feature
    parser.add_argument(
        "--enable_ema", action="store_true", help="Enable EMA checkpoint for the model."
    )
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
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
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--freezing_actor_steps", type=int, default=-1)
    parser.add_argument("--n_samples_per_prompt", type=int, default=1)

    parser.add_argument("--save_value_network", action="store_true", default=False)

    # reward model
    parser.add_argument("--head_prefix", type=str, default="value_head")
    parser.add_argument("--ref_reward_offload", action="store_true", default=False)

    # custom dataset key name
    parser.add_argument("--input_key", type=str, default="input")
    parser.add_argument("--input_template", type=str, default="User: {}\nAssistant: ")
    parser.add_argument("--apply_chat_template", action="store_true", default=False)

    # evaluation
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)

    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # performance tuning
    parser.add_argument("--perf", action="store_true", default=False)

    args = parser.parse_args()

    # if args.critic_pretrain is None:
    #     args.critic_pretrain = args.reward_pretrain.split(",")[0]

    # if (
    #     args.vllm_num_engines >= 1
    #     and args.n_samples_per_prompt > 1
    #     and not args.enable_prefix_caching
    # ):
    #     print(
    #         "[Warning] Please --enable_prefix_caching to accelerate when --n_samples_per_prompt > 1."
    #     )

    train(args)


"""
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 /home/aiops/liuzc/ellm/experiment/main.py \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 2 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 2 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --ref_reward_offload \
   --pretrain OpenLLMAI/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenLLMAI/Llama-3-8b-rm-mixture \
   --save_path /openrlhf/examples/checkpoint/llama3-8b-rlhf \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenLLMAI/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing
"""
