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


class RewardModel(TypedMsgPackMixin, Worker):
    def __init__(self):
        super().__init__()
        model_name = os.environ.get("RM_MODEL_NAME")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.example = Request(
            prompt="What is the range of the numeric output of a sigmoid node in a neural network?",
            candidates=[
                "The output of a sigmoid node is bounded between -1 and 1.",
                "The output of a sigmoid node is bounded between 0 and 1.",
            ],
        )  # To warmup: do one forward pass to allocate GPU memory

    def forward(self, request: Request) -> Response:
        msg1 = [
            {"role": "user", "content": request.prompt},
            {"role": "assistant", "content": request.candidates[0]},
        ]
        msg2 = [
            {"role": "user", "content": request.prompt},
            {"role": "assistant", "content": request.candidates[1]},
        ]
        pair = self.tokenizer.apply_chat_template([msg1, msg2], tokenize=False)
        pair = self.tokenizer(pair, return_tensors="pt", padding=True).to(
            self.model.device
        )
        with torch.no_grad():
            logits = self.model(**pair).logits.cpu().squeeze()
        # Apply BT model.
        first_win_prob = (logits[0] - logits[1]).sigmoid().item()
        return Response(first_win_prob=first_win_prob)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remote_rm_model", type=str, default="LxzGordon/URM-LLaMa-3.1-8B"
    )
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
        env=[_prepare_env(x) for x in devices],
        timeout=10,
    )
    server.register_runtime(
        {
            "/compare": [runtime],
        }
    )
    server.run()
