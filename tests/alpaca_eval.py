# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER" # this is recommended for gemma-2 models; otherwise it is not needed
import argparse
import json
import os
import time

from datasets import load_dataset
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser(description="Generate AlpacaEval.")
parser.add_argument(
    "--model", type=str, default="google/gemma-2-9b-it", help="Path to the LLM model"
)
parser.add_argument(
    "--temperature", type=float, default=0.8, help="Temperature for sampling"
)
parser.add_argument(
    "--top_p", type=float, default=0.95, help="Top-p probability for sampling"
)
parser.add_argument(
    "--max_tokens", type=int, default=4096, help="Maximum number of tokens to generate"
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--output_dir", type=str, default="alpaca_eval_outputs", help="output_dir"
)
args = parser.parse_args()
args.seed = int(time.time_ns() // 2 * 20)

print(args)

llm = LLM(model=args.model)
tokenizer = llm.get_tokenizer()

eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

prompts = eval_set["instruction"]

conversations = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for prompt in prompts
]

sampling_params = SamplingParams(
    temperature=args.temperature,
    top_p=args.top_p,
    max_tokens=args.max_tokens,
    seed=args.seed,
)

outputs = llm.generate(conversations, sampling_params)

# Save the outputs as a JSON file.
output_data = []
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    output_data.append(
        {
            "instruction": prompts[i],
            "format_instruction": prompt,
            "output": generated_text,
            "generator": args.model,
        }
    )

model_fn = args.model.replace("/", "_")
output_file = f"output_{model_fn}_{args.seed}.json"
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(os.path.join(args.output_dir, output_file), "w") as f:
    json.dump(output_data, f, indent=4)

print(f"Outputs saved to {os.path.join(args.output_dir, output_file)}")
