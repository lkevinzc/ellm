import time

import pandas as pd

from ellm.oracles.remote.client import RemoteRMOracle

n = 50
remote_oracle = RemoteRMOracle(remote_rm_url="http://0.0.0.0:8000", max_workers=4)
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
