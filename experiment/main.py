import argparse

import launchpad as lp
import vllm
from launchpad.nodes.python import local_multi_processing

from ellm.actor import Actor
from ellm.learners.dap import DAPLearner
from ellm.utils.launcher import get_free_port


def main(args):
    # actor setup
    vllm_args = {
        "model": args.pretrain,
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.5,
        "dtype": "bfloat16",  # TODO(liuzc) check whether to use bfloat
        "seed": 0,
        "enable_prefix_caching": True,
    }
    sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.generate_max_length,
        seed=0,
        n=args.num_samples,
    )
    program = lp.Program("online_dap")
    actors = []
    local_resources = {}
    for i in range(4):
        label = f"actor_{i}"
        actors.append(
            program.add_node(
                lp.CourierNode(Actor, vllm_args, sampling_params), label=label
            )
        )
        local_resources[label] = local_multi_processing.PythonProcess(
            env=dict(CUDA_VISIBLE_DEVICES=str(i))
        )
    _gpu_offset = 4

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
    parser.add_argument("--pretrain", type=str, default="google/gemma-2b")
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

    # Generation params
    parser.add_argument("--generate_max_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_samples", type=int, default=2)

    parser.add_argument("--save_path", type=str, default="./output")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_dpo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB

    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--rollout_batch_size", type=int, default=512)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--micro_pi_buffer_maxlen", type=int, default=999999999)
    parser.add_argument("--micro_r_buffer_maxlen", type=int, default=999999999)
    parser.add_argument("--prompt_max_length", type=int, default=1024)

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
    parser.add_argument("--apply_chat_template", action="store_true", default=False)

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="ellm_simpo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="online_SimPO",
    )

    args = parser.parse_args()

    if args.ref_pretrain is None or args.ref_pretrain == "":
        args.ref_pretrain = args.pretrain

    main(args)
