import argparse
import os
from typing import List

import torch
from mosec import Runtime, Server, Worker
from mosec.mixin import TypedMsgPackMixin
from msgspec import Struct
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Request(Struct, kw_only=True):
    prompt: str
    candidates: List[str]


class Response(Struct, kw_only=True):
    first_win_prob: float


MODEL_CONFIGS = {
    "Skywork/Skywork-Reward-Llama-3.1-8B": {
        "attn_implementation": "flash_attention_2",
        "num_labels": 1,
    }
}


class RewardModel(TypedMsgPackMixin, Worker):
    def __init__(self):
        super().__init__()
        model_name = os.environ.get("RM_MODEL_NAME")
        configs = MODEL_CONFIGS.get(model_name, {})
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **configs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.example = Request(
            prompt="What is the range of the numeric output of a sigmoid node in a neural network?",
            candidates=[
                "The output of a sigmoid node is bounded between -1 and 1.",
                "The output of a sigmoid node is bounded between 0 and 1.",
            ],
        )  # To warmup: do one forward pass to allocate GPU memory
        if self.max_batch_size > 1:
            self.example = [self.example] * self.max_batch_size

    def forward(self, requests: List[Request]) -> List[Response]:
        if self.max_batch_size == 1:
            requests = [requests]
        batch_msg1 = []
        batch_msg2 = []
        for r in requests:
            msg1 = [
                {"role": "user", "content": r.prompt},
                {"role": "assistant", "content": r.candidates[0]},
            ]
            msg2 = [
                {"role": "user", "content": r.prompt},
                {"role": "assistant", "content": r.candidates[1]},
            ]
            batch_msg1.append(msg1)
            batch_msg2.append(msg2)
        pair = self.tokenizer.apply_chat_template(
            batch_msg1 + batch_msg2, tokenize=False
        )
        pair = self.tokenizer(pair, return_tensors="pt", padding=True).to(
            self.model.device
        )
        with torch.no_grad():
            logits = self.model(**pair).logits.cpu().squeeze()
        batch_scores_1 = logits[: self.max_batch_size]
        batch_scores_2 = logits[self.max_batch_size :]
        # Apply BT model.
        batch_first_win_prob = (batch_scores_1 - batch_scores_2).sigmoid().tolist()

        responses = [Response(first_win_prob=p) for p in batch_first_win_prob]
        if self.max_batch_size == 1:
            responses = responses[0]
        return responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remote_rm_model", type=str, default="Skywork/Skywork-Reward-Llama-3.1-8B"
    )
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--cuda_devices", type=str, default="all")
    args = parser.parse_args()

    if args.cuda_devices == "all":
        NUM_DEVICE = torch.cuda.device_count()
        devices = list(range(NUM_DEVICE))
    else:
        devices = args.cuda_devices.split(",")
        NUM_DEVICE = len(devices)

    def _prepare_env(cid: int) -> dict:
        return {"CUDA_VISIBLE_DEVICES": str(cid), "RM_MODEL_NAME": args.remote_rm_model}

    server = Server()
    runtime = Runtime(
        worker=RewardModel,
        num=NUM_DEVICE,
        max_batch_size=args.max_batch_size,
        env=[_prepare_env(x) for x in devices],
        timeout=10,
    )
    server.register_runtime(
        {
            "/compare": [runtime],
        }
    )
    server.run()
