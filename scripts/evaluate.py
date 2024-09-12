from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from transformers import HfArgumentParser

from ellm.oracles.gpt import GPTJudgeOracle


@dataclass
class ScriptArguments:
    data_path: str = field(
        metadata={"help": "The directory containing evaluation responses."}
    )
    ref_data_path: str = field(
        default="", metadata={"help": "The directory containing reference responses."}
    )
    judge_model: str = field(
        default="meta-llama/Meta-Llama-3-70B-Instruct",
        metadata={
            "help": "The model name or path to the model to use as a judge. E.g., 'gpt-3.5-turbo-0125', 'meta-llama/Meta-Llama-3-70B-Instruct'."
        },
    )
    num_examples: Optional[int] = field(
        default=None, metadata={"help": "The number of examples to evaluate."}
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

if args.ref_data_path:
    # For head-to-head comparison.
    reference_completions = pd.read_json(
        args.ref_data_path,
        orient="records",
        lines=True,
    )["response"]

# Judge the outputs
judge = GPTJudgeOracle(args.judge_model, max_workers=16)

win_probs = judge.compare(
    prompts, model_completions, reference_completions, return_probs=True
)
model_win_rate = win_probs.mean()
print(f"Model win rate: {model_win_rate*100:.2f}%")
