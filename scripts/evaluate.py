from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from datasets import load_dataset
from transformers import HfArgumentParser

from ellm.oracles.gpt import GPTJudgeOracle


@dataclass
class ScriptArguments:
    data_path: str = field(
        metadata={"help": "The directory containing evaluation responses."}
    )
    judge_model: str = field(
        default="meta-llama/Meta-Llama-3-70B-Instruct",
        metadata={
            "help": "The model name or path to the model to use as a judge. E.g., 'gpt-3.5-turbo-0125', 'meta-llama/Meta-Llama-3-70B-Instruct'."
        },
    )
    sft_ref: bool = field(
        default=True, metadata={"help": "Whether to use SFT target as reference."}
    )
    parallel: Optional[int] = field(
        default=4, metadata={"help": "The number of parallel calls."}
    )


# Parse the arguments
parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

# Load evaluation data
data = pd.read_json(
    args.data_path,
    orient="records",
    lines=True,
)
prompts = data["prompt"]
model_completions = data["response"]
reference_completions = data["reference"]

if args.sft_ref:
    # Load the dataset
    raw_dataset = load_dataset("lkevinzc/tldr-with-sft-reference", split="test")
    raw_dataset = raw_dataset.select(range(len(prompts)))

    # Extract the prompts and reference completions
    prompts_data = raw_dataset["prompt"]
    assert all([prompts[i] == prompts_data[i] for i in range(len(prompts))])

    reference_completions = raw_dataset["summary"]

# Judge the outputs
judge = GPTJudgeOracle(args.judge_model, max_workers=args.parallel)

win_probs = judge.compare(
    prompts, model_completions, reference_completions, return_probs=True
)
model_win_rate = win_probs.mean()
print(f"Model win rate: {model_win_rate*100:.2f}%")
