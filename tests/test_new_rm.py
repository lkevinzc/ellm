import pandas as pd

from ellm.oracles.pair import PairRMOracle
from ellm.oracles.scalar import ScalarRMOracle

pair_rm = PairRMOracle()
pythia_rm = ScalarRMOracle(
    "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr",
    "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr",
)

data = pd.read_json(
    "/home/aiops/liuzc/ellm/output/apl_ent_pref_certainty_4ep_0910T07:53/eval_results/380.json",
    orient="records",
    lines=True,
)
prompts = data["prompt"]
candidate_1 = data["response"]
candidate_2 = data["reference"]


pythia_rm_results = pythia_rm.compare(prompts, candidate_1, candidate_2, batch_size=16)
pair_rm_results = pair_rm.compare(prompts, candidate_1, candidate_2, batch_size=16)

print(
    "Two models' results match ratio: ", (pair_rm_results == pythia_rm_results).mean()
)
