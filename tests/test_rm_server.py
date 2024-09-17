import time

import pandas as pd

from ellm.oracles.remote.client import RemoteRMOracle

remote_oracle = RemoteRMOracle(remote_rm_url="http://0.0.0.0:8000", max_workers=1)
data = pd.read_json(
    "/home/aiops/liuzc/ellm/output/apl_ent_pref_certainty_4ep_0910T07:53/eval_results/380.json",
    orient="records",
    lines=True,
)
prompts = data["prompt"]
candidate_1 = data["response"]
candidate_2 = data["reference"]

st = time.time()
result = remote_oracle.compare(prompts, candidate_1, candidate_2, return_probs=True)
print(time.time() - st)
