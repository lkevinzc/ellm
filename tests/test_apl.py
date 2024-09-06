import pdb

import numpy as np
import torch
import vllm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ellm.utils.data import zero_pad_sequences

model_name = "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr"
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "Hello how are you"
output = tokenizer(
    text,
    max_length=512,
    padding=False,
    truncation=True,
    return_tensors="pt",
)

sampling_params = vllm.SamplingParams(
    temperature=0.7, top_p=0.9, max_tokens=512, seed=0, n=8
)
llm = vllm.LLM(
    **{
        "model": model_name,
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.25,
        "dtype": "bfloat16",
    }
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    quantization_config=None,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)

outputs = llm.generate(
    ["give me one word", "Singapore is", "hurry up", "write the paper"], sampling_params
)

# 1) Predictive entropy estimation
entropy_estimations = []
for output in outputs:
    entropy = 0
    for resp_output in output.outputs:
        entropy += resp_output.cumulative_logprob
    entropy /= len(output.outputs)
    entropy_estimations.append(entropy)
ent_filtered_indices = np.argsort(entropy_estimations)[-2:]  # online, on-policy
outputs = [outputs[i] for i in ent_filtered_indices]


@torch.no_grad
def compute_logp(model, prompt_response_ids, prompt_response_masks, prompt_len: int):
    model_output = model(prompt_response_ids, attention_mask=prompt_response_masks)
    all_logits = model_output["logits"]

    loss_masks = prompt_response_masks.clone().bool()
    prompt_id_lens = [prompt_len] * len(loss_masks)
    # mask prompts
    for mask, source_len in zip(loss_masks, prompt_id_lens):
        mask[:source_len] = False
    loss_masks = loss_masks[:, 1:]

    labels = prompt_response_ids[:, 1:].clone()
    # dummy token; we'll ignore the losses on these tokens later
    labels[loss_masks == False] = 0

    per_token_logps = torch.gather(
        all_logits[:, :-1, :].log_softmax(-1),
        dim=2,
        index=labels.unsqueeze(2),
    ).squeeze(2)

    logprobs = (per_token_logps * loss_masks).sum(-1)
    return logprobs


prompts = []
candidates = {}
for i, output in enumerate(outputs):
    # for each prompt
    prompt_response_ids = [
        torch.tensor(output.prompt_token_ids + o.token_ids) for o in output.outputs
    ]
    prompt_response_masks = [torch.ones_like(ids) for ids in prompt_response_ids]

    prompt_response_ids = zero_pad_sequences(
        prompt_response_ids, side="right", value=tokenizer.pad_token_id
    )
    prompt_response_masks = zero_pad_sequences(prompt_response_masks, side="right")

    prompt_response_ids = prompt_response_ids.cuda()
    prompt_response_masks = prompt_response_masks.cuda()

    logprobs = compute_logp(
        model, prompt_response_ids, prompt_response_masks, len(output.prompt_token_ids)
    )

    logprobs_ref = torch.randn_like(logprobs)

    implicit_rewards = logprobs - logprobs_ref

    M = len(prompt_response_ids)
    reward_margins = torch.abs(
        implicit_rewards.view(M, 1) - implicit_rewards.view(1, M)
    )
    pair_indices = torch.where(reward_margins == reward_margins.max())[0].cpu().tolist()

    prompts.append(output.prompt)
    candidates[i] = [output.outputs[j].text for j in pair_indices]

pdb.set_trace()
