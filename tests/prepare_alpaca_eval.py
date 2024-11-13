import pdb

import datasets
from transformers import AutoTokenizer

dataset = datasets.load_dataset(
    "tatsu-lab/alpaca_eval",
    "alpaca_eval_gpt4_baseline",
)["eval"]
pdb.set_trace()


data = datasets.load_from_disk(
    "/home/aiops/liuzc/SimPO/on_policy_data_gen/datasets/llama3.1_8b_ultrafeedback_final"
)


tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2-2b-it", trust_remote_code=True, use_fast=True
)

prompts = data["prompt"]
chosens = data["chosen"]

conversations = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    for prompt in prompts
]

chosen_lens = [len(tokenizer(x)["input_ids"]) for x in chosens]

lens = [len(c) for c in conversations]
import numpy as np

np.sum([l <= 2048 for l in lens])
np.sum([l <= 2048 - 1224 for l in chosen_lens])

np.sum([(l1 + l2) <= 2048 for l1, l2 in zip(lens, chosen_lens)])
