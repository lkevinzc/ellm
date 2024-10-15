"""Validate weights consistency."""

import argparse
import pdb

import torch
import tree
import vllm
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrain", type=str, default="google/gemma-2-2b-it", help="Path to the LLM model"
)
args = parser.parse_args()

llm = vllm.LLM(
    **{
        "model": args.pretrain,
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.5,
    }
)
vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model

vllm_model_param_keys = set([k for k, v in vllm_model.named_parameters()])

transformer_model = AutoModelForCausalLM.from_pretrained(
    args.pretrain,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    quantization_config=None,
    torch_dtype=torch.bfloat16,
)

transformer_model_param_keys = set([k for k, v in transformer_model.named_parameters()])


print("loading weights...")
vllm_model.load_weights(transformer_model.named_parameters())

cache = tree.map_structure(lambda x: x.cpu(), vllm_model.state_dict())
vllm_model.load_state_dict(cache)

pdb.set_trace()

all_match = True
if not vllm_model_param_keys.issubset(transformer_model_param_keys):
    print("Parameters in vllm but not in transformer:")
    diff = vllm_model_param_keys.difference(transformer_model_param_keys)
    print(len(diff), sorted(list(diff)))
    all_match = False

print()

if not transformer_model_param_keys.issubset(vllm_model_param_keys):
    print("Parameters in transformer but not in vllm:")
    diff = transformer_model_param_keys.difference(vllm_model_param_keys)
    print(len(diff), sorted(list(diff)))
    all_match = False

if all_match:
    print("All parameters match!")

if input("Print vllm params? (y/n)") == "y":
    print(vllm_model_param_keys)
if input("Print transformer params? (y/n)") == "y":
    print(transformer_model_param_keys)

"""
python tests/test_weight_synchronization.py --pretrain cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr
python tests/test_weight_synchronization.py --pretrain google/gemma-2-2b-it
python tests/test_weight_synchronization.py --pretrain meta-llama/Llama-3.2-1B-Instruct
"""
