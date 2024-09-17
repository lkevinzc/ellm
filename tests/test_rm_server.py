import time

import httpx
import msgspec
import pandas as pd
from fire import Fire

from ellm.oracles.remote.client import RemoteRMOracle


def main(max_workers: int = 4, server_addr: str = "0.0.0.0:8000"):
    # A quick validation.
    req = {
        "prompt": "What is the range of the numeric output of a sigmoid node in a neural network?",
        "candidates": [
            "The output of a tanh node is bounded between -1 and 1.",
            "The output of a sigmoid node is bounded between 0 and 1.",
        ],
    }
    resp = httpx.post(
        f"http://{server_addr}/compare", content=msgspec.msgpack.encode(req)
    )
    print(resp.status_code, msgspec.msgpack.decode(resp.content))

    # Speed test.
    n = 50
    remote_oracle = RemoteRMOracle(
        remote_rm_url=f"http://{server_addr}", max_workers=max_workers
    )
    data = pd.read_json(
        "output/neworacle_dpo_offline_0911T19:18/eval_results/380.json",
        orient="records",
        lines=True,
    )
    prompts = data["prompt"][:n]
    candidate_1 = data["response"][:n]
    candidate_2 = data["reference"][:n]

    st = time.time()
    result = remote_oracle.compare(prompts, candidate_1, candidate_2, return_probs=True)
    print(time.time() - st)


if __name__ == "__main__":
    Fire(main)
