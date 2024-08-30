import concurrent.futures
import logging
import random
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from openai import OpenAI
from transformers import HfArgumentParser
from trl import BasePairwiseJudge, HfPairwiseJudge, is_openai_available
from trl.trainer.judges import DEFAULT_PAIRWISE_SYSTEM_PROMPT


class OpenAIPairwiseJudge(BasePairwiseJudge):
    """
    Judge based on the OpenAI API.

    This judge is relevant for assessing the quality chat models, where the completion is a response to a given prompt.

    Args:
        model (`str`, *optional*): The model to use for the judge. Defaults to `"gpt-4-turbo-preview"`.
        system_prompt (`str`, *optional*): The system prompt to be used for the judge. If not provided, a default prompt is used.
            Note that the system prompt should contain the following placeholders: `{prompt}`, `{response0}`, and `{response1}`.
            Also, the inference is called with `max_tokens=1`, consequently the system prompt should ask for a single token response.
        max_requests (`int`, *optional*): The maximum number of requests to make to the OpenAI API. Defaults to 1000. If set to `None`, there is no limit.

    """

    def __init__(
        self,
        model="gpt-4-turbo-preview",
        system_prompt: Optional[str] = None,
        max_requests: Union[int, None] = 1_000,
    ):
        if not is_openai_available():
            raise ValueError(
                "OpenAI client is not installed. Please install it with 'pip install openai'."
            )
        self.client = OpenAI(
            api_key="5mMAS9UeX4F42DX5034CiceJ5iVs6v4X43UlsI6g8ccHGGz6",
            base_url="https://chatgpt.alpha.insea.io/openai/v1",
        )
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_PAIRWISE_SYSTEM_PROMPT
        self.max_requests = max_requests
        self.num_requests = 0
        self._warned = False

    def judge(
        self,
        prompts: List[str],
        completions: List[List[str]],
        shuffle_order: bool = True,
    ) -> List[int]:
        # Check if the limit of requests is reached, if so, use random choice instead
        if self.max_requests is not None and self.num_requests >= self.max_requests:
            if not self._warned:  # Print the warning only once
                logging.warning(
                    f"Reached the maximum number of requests ({self.max_requests}). From now on, using random choice instead. "
                    " To increase the limit, set `max_requests` to a higher value, or to `None` for no limit."
                )
                self._warned = True
            return [random.choice([0, 1]) for _ in prompts]

        # Shuffle the order of the completions to avoid positional bias
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [
                pair[::-1] if flip else pair
                for flip, pair in zip(flip_mask, completions)
            ]

        # Define a function to get the rank for a single prompt, will be called concurrently
        def get_rank(prompt, candidates):
            content = self.system_prompt.format(
                prompt=prompt, response0=candidates[0], response1=candidates[1]
            )
            messages = [{"role": "user", "content": content}]
            completion = self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=1
            )
            response = completion.choices[0].message.content
            print(response)
            if response in ["0", "1"]:
                return int(response)
            else:
                logging.warning(
                    f"Invalid response from the model: {response}, using random choice instead."
                )
                return random.choice([0, 1])

        # Call the completions concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            ranks = list(executor.map(get_rank, prompts, completions))

        # Flip back the ranks to the original order if needed
        if shuffle_order:
            ranks = [
                ranks[i] if not flip else 1 - ranks[i]
                for i, flip in enumerate(flip_mask)
            ]

        # Update the number of requests
        self.num_requests += len(prompts)

        # Return the ranks
        return ranks


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
if "gpt" in args.judge_model:
    judge = OpenAIPairwiseJudge(args.judge_model)
else:
    judge = HfPairwiseJudge(args.judge_model)

completions = [[c0, c1] for c0, c1 in zip(reference_completions, model_completions)]
best_idxs = judge.judge(prompts, completions)
model_win_rate = best_idxs.count(1) / len(best_idxs)
print(f"Model win rate: {model_win_rate*100:.2f}%")
