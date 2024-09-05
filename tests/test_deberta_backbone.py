import pdb
from dataclasses import dataclass
from typing import List

import fire
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from ellm.rm.backbone import DebertaV2Vanilla


@dataclass
class PreferenceData:
    prompt: str
    chosen_response: str
    rejected_response: str


def main(model_name: str):

    buffer = pd.read_pickle(
        "/home/aiops/liuzc/ellm/output/online_SimPO_dump_0811T15:29/buffer.pkl"
    )
    data = []
    for b in buffer:
        data += list(b)

    backbone = DebertaV2Vanilla.from_pretrained(model_name, device_map="cuda:0").eval()

    max_length = 2048
    source_max_length = 1224

    def collate_fn(item_list: List[PreferenceData]):
        chosen_ids = []
        rejected_ids = []
        for pref in item_list:
            chosen_ids.append(
                backbone.tokenize_pair(
                    pref.prompt, pref.chosen_response, source_max_length, max_length
                )
            )
            rejected_ids.append(
                backbone.tokenize_pair(
                    pref.prompt, pref.rejected_response, source_max_length, max_length
                )
            )

        encodings = backbone.tokenizer.pad(
            {"input_ids": chosen_ids + rejected_ids},
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
        )
        return encodings

    test_out = collate_fn(data[:4])
    keep_column_mask = test_out["attention_mask"].ne(0).any(dim=0)
    input_ids = test_out["input_ids"][:, keep_column_mask]
    test_enc = {k: v.to(backbone.device) for k, v in test_out.items()}
    pdb.set_trace()
    test_feat = backbone.get_feature(**test_enc).detach().cpu()
    print("test feat", test_feat.shape)

    dl = DataLoader(data, batch_size=4, drop_last=False, collate_fn=collate_fn)
    pdb.set_trace()

    outputs = []
    for enc in tqdm(dl):
        enc = {k: v.to(backbone.device) for k, v in enc.items()}
        outputs.append(backbone.get_feature(**enc).detach().cpu())

    model_name = model_name.replace("/", "_")
    pd.to_pickle(outputs, f"processed_features_for_rm{model_name}.pt")


if __name__ == "__main__":
    fire.Fire(main)
# microsoft/deberta-v3-large,
