python experiment/main.py \
    --pretrain cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --prompt_data trl-internal-testing/tldr-preference-sft-trl-style \
    --input_key prompt \
    --micro_train_batch_size 1 \
    --flash_attn \
    --gradient_checkpointing \
    --use_wandb True
