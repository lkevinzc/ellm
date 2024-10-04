from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
from datasets import load_dataset
from transformers import HfArgumentParser

from ellm.oracles import get_cls

data_paths = [
    # DPO
    # ## 1B STF
    # "./output/skyworkrm_dpo_offline_0921T01:12/eval_results/0.json",
    # "./output/skyworkrm_dpo_offline_0921T00:15/eval_results/0.json",
    # "./output/skyworkrm_dpo_offline_0917T07:16/eval_results/0.json",
    # ## 1B Offline
    # "./output/skyworkrm_dpo_offline_0921T01:12/eval_results/391.json",
    # "./output/skyworkrm_dpo_offline_0921T00:15/eval_results/391.json",
    # "./output/skyworkrm_dpo_offline_0917T07:16/eval_results/391.json",
    # ## 1B Online passive
    # "./output/skyworkrm_dpo_online_0920T16:13/eval_results/391.json",
    # "./output/skyworkrm_dpo_online_0920T15:15/eval_results/391.json",
    # "./output/skyworkrm_dpo_online_0917T08:29/eval_results/380.json",
    ## 1B Online sea
    "./output/skyworkrm_dpo_sea_0925T15:53/eval_results/567.json",
    "./output/skyworkrm_dpo_sea_0925T11:49/eval_results/567.json",
    "./output/skyworkrm_dpo_sea_0925T07:47/eval_results/567.json",
    ## 2.8B STF
    # "./output/skyworkrm_dpo_offline_0921T06:59/eval_results/0.json",
    # "./output/skyworkrm_dpo_offline_0921T03:35/eval_results/0.json",
    # "./output/skyworkrm_dpo_offline_0918T02:32/eval_results/0.json",
    # ## 2.8B Offline
    # "./output/skyworkrm_dpo_offline_0921T06:59/eval_results/391.json",
    # "./output/skyworkrm_dpo_offline_0921T03:35/eval_results/391.json",
    # "./output/skyworkrm_dpo_offline_0918T02:32/eval_results/391.json",
    # ## 2.8B Online passive
    # "./output/skyworkrm_dpo_online_0921T06:59/eval_results/391.json",
    # "./output/skyworkrm_dpo_online_0921T03:38/eval_results/391.json",
    # "./output/skyworkrm_dpo_online_0918T01:53/eval_results/391.json",
    ## 2.8B Online sea
    "./output/skyworkrm_dpo_sea_lmd0.2_0925T20:10/eval_results/567.json",
    "./output/skyworkrm_dpo_sea_lmd0.2_0926T05:34/eval_results/567.json",
    "./output/skyworkrm_dpo_sea_lmd0.2_0926T10:32/eval_results/567.json",
    # ## 6.9B STF
    # "./output/skyworkrm_dpo_offline_0921T21:17/eval_results/0.json",
    # "./output/skyworkrm_dpo_offline_0921T17:56/eval_results/0.json",
    # "./output/skyworkrm_dpo_offline_0918T03:59/eval_results/0.json",
    # ## 6.9B Offline
    # "./output/skyworkrm_dpo_offline_0921T21:17/eval_results/391.json",
    # "./output/skyworkrm_dpo_offline_0921T17:56/eval_results/391.json",
    # "./output/skyworkrm_dpo_offline_0918T03:59/eval_results/391.json",
    # ## 6.9B Online passive
    # "./output/skyworkrm_dpo_online_0921T21:16/eval_results/391.json",
    # "./output/skyworkrm_dpo_online_0921T17:50/eval_results/391.json",
    # "./output/skyworkrm_dpo_online_0918T04:02/eval_results/391.json",
    ## 6.9B Online sea
    "./output/skyworkrm_dpo_sea_0926T03:44/eval_results/567.json",
    "./output/skyworkrm_dpo_sea_0925T20:05/eval_results/567.json",
    "./output/skyworkrm_dpo_sea_0925T12:16/eval_results/567.json",
]


@dataclass
class ScriptArguments:
    data_paths: List[str] = field(
        default=None,
        metadata={"help": "The directory containing evaluation responses."},
    )
    judge_model: str = field(
        default="meta-llama/Meta-Llama-3-70B-Instruct",
        metadata={
            "help": "The model name or path to the model to use as a judge. E.g., 'gpt-3.5-turbo-0125', 'meta-llama/Meta-Llama-3-70B-Instruct'."
        },
    )
    sft_ref: bool = field(
        default=False, metadata={"help": "Whether to use SFT target as reference."}
    )
    parallel: Optional[int] = field(
        default=4, metadata={"help": "The number of parallel calls."}
    )


# Parse the arguments
parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]


data_paths = args.data_paths or data_paths

print(data_paths)


# Judge the outputs
judge_cls = get_cls(args.judge_model)
judge = judge_cls(
    reward_model_path=args.judge_model,
    max_workers=args.parallel,
    remote_rm_url="http://remote-rm",  # Only for remote RM.
)
sft_loaded = False
print("Judge:", judge)
for path in data_paths:
    # Load evaluation data
    data = pd.read_json(
        path,
        orient="records",
        lines=True,
    )
    prompts = data["prompt"]
    model_completions = data["response"]
    if args.sft_ref and not sft_loaded:
        print("SFT target as reference")
        # Load the dataset
        raw_dataset = load_dataset("lkevinzc/tldr-with-sft-reference", split="test")
        raw_dataset = raw_dataset.select(range(len(prompts)))

        # Extract the prompts and reference completions
        prompts_data = raw_dataset["prompt"]
        assert all([prompts[i] == prompts_data[i] for i in range(len(prompts))])
        reference_completions = raw_dataset["summary"]
        sft_loaded = True
    elif not args.sft_ref:
        print("taking model generation as reference")
        reference_completions = data["reference"]

    win_probs = judge.compare(
        prompts, model_completions, reference_completions, return_probs=True
    )
    model_win_rate = win_probs.mean()
    print(f"{path},{model_win_rate*100:.2f}%")
