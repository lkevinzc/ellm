cd /home/aiops/liuzc
source ./.zshrc
conda activate ellm
cd ellm

python experiment/main.py \
    --pretrain cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --prompt_data trl-internal-testing/tldr-preference-sft-trl-style \
    --input_key prompt \
    --micro_train_batch_size 1 \
    --micro_rollout_batch_size 128 \
    --eval_steps 100 \
    --flash_attn \
    --gradient_checkpointing \
    --use_wandb True


python experiment/main.py \
    --pretrain cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --prompt_data trl-internal-testing/tldr-preference-sft-trl-style \
    --input_key prompt \
    --micro_train_batch_size 1 \
    --micro_rollout_batch_size 625 \
    --rollout_batch_size 2500 \
    --eval_steps 100 \
    --flash_attn \
    --gradient_checkpointing \
    --use_wandb True \
    --wandb_run_name iterative4_SimPO

# Offline SimPO
python experiment/main.py --pretrain cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr --prompt_data trl-internal-testing/tldr-preference-sft-trl-style --input_key prompt --output_key summary --max_train 50000 --micro_train_batch_size 1 --micro_rollout_batch_size 128 --eval_steps 80 --buffer_clear_every 999999999 --sync_params_every 9999999 --micro_pi_buffer_maxlen 128 --generate_max_length 53 --flash_attn --gradient_checkpointing --use_wandb True --wandb_run_name offline_SimPO

# Online SimPO
python experiment/main.py --pretrain cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr --prompt_data trl-internal-testing/tldr-preference-sft-trl-style --input_key prompt --output_key summary --max_train 50000 --micro_train_batch_size 1 --micro_rollout_batch_size 128 --eval_steps 80 --buffer_clear_every 999999999 --sync_params_every 1 --micro_pi_buffer_maxlen 128 --generate_max_length 53 --flash_attn --gradient_checkpointing --use_wandb True --wandb_run_name online_SimPO
