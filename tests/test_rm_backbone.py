import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pdb
from typing import List

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from ellm.rm.backbone import DebertaV2PairRM
from ellm.types import PreferenceData

buffer = pd.read_pickle(
    "/home/aiops/liuzc/ellm/output/online_SimPO_dump_0811T15:29/buffer.pkl"
)
data = []
for b in buffer:
    data += list(b)

backbone = DebertaV2PairRM.from_pretrained(
    "llm-blender/PairRM-hf", device_map="cuda:0"
).eval()
tokenizer = AutoTokenizer.from_pretrained("llm-blender/PairRM-hf")


source_prefix = "<|source|>"
cand_prefix = "<|candidate|>"
max_length = 2048
source_max_length = 1224


def collate_fn(item_list: List[PreferenceData]):
    prompt_ids = []
    chosen_ids = []
    rejected_ids = []
    for pref in item_list:
        source_ids = tokenizer.encode(
            source_prefix + pref.prompt, max_length=source_max_length, truncation=True
        )
        candidate_max_length = max_length - len(source_ids)
        chosen_candidate_ids = tokenizer.encode(
            cand_prefix + pref.chosen_response,
            max_length=candidate_max_length,
            truncation=True,
        )
        rejected_candidate_ids = tokenizer.encode(
            cand_prefix + pref.rejected_response,
            max_length=candidate_max_length,
            truncation=True,
        )

        prompt_ids.append(source_ids)
        chosen_ids.append(chosen_candidate_ids)
        rejected_ids.append(rejected_candidate_ids)

    encodings = tokenizer.pad(
        {
            "input_ids": [a + b for a, b in zip(prompt_ids, chosen_ids)]
            + [a + b for a, b in zip(prompt_ids, rejected_ids)]
        },
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
    )
    return encodings


test_out = collate_fn(data[:4])
test_enc = {k: v.to(backbone.device) for k, v in test_out.items()}
pdb.set_trace()

dl = DataLoader(data, batch_size=4, drop_last=False, collate_fn=collate_fn)
outputs = []
for enc in tqdm(dl):
    enc = {k: v.to(backbone.device) for k, v in enc.items()}
    outputs.append(backbone.get_feature(**enc).detach().cpu())


pd.to_pickle(outputs, "processed_features_for_rm.pt")
