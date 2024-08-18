import argparse

import launchpad as lp

from ellm.interface import get_program
from ellm.learners import DAPLearner, DAPwRMLearner


def main(args):
    if args.learn_rm:
        learner_cls = DAPwRMLearner
    else:
        learner_cls = DAPLearner
    program, local_resources = get_program(args, learner_cls)
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
    parser.add_argument("--max_eval", type=int, default=1000)

    # Offline preference dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="OpenLLMAI/preference_dataset_mixture2_and_safe_pku",
    )
    parser.add_argument(
        "--dataset_probs", type=str, default="1.0", help="sampling probs for datasets"
    )

    # Online DAP
    parser.add_argument("--buffer_clear_every", type=int, default=1)
    parser.add_argument("--sync_params_every", type=int, default=1)
    parser.add_argument("--dump_reward_buffer", action="store_true")

    # Exploration
    parser.add_argument("--learn_rm", action="store_true")
    parser.add_argument(
        "--exp_method",
        type=str,
        choices=["no", "enn_dts"],
        default=["no"],
        help="Types of exploration.",
    )
    parser.add_argument("--exp_pretrain", type=str, default="")
    ## enn_dts
    parser.add_argument("--num_ensemble", type=int, default=10)
    parser.add_argument("--enn_lr", type=float, default=1e-3)
    parser.add_argument("--enn_lambda", type=float, default=0.1)
    parser.add_argument("--enn_hidden_dim", type=int, default=128)
    parser.add_argument("--enn_sgd_steps", type=int, default=1)

    # Generation params
    parser.add_argument("--generate_max_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=float, default=-1)
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
    parser.add_argument("--r_buffer_maxlen", type=int, default=3200)
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
    parser.add_argument("--learning_rate", type=float, default=5e-7)
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
    parser.add_argument("--output_key", type=str, default="output")
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

    # Validation.
    if args.ref_pretrain is None or args.ref_pretrain == "":
        args.ref_pretrain = args.pretrain
    if args.learn_rm:
        args.exp_method != "no"
        args.exp_pretrain == ""

    main(args)
