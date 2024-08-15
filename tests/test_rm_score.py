import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from typing import List

from transformers import AutoTokenizer

from ellm.rm.backbone import DebertaV2PairRM

pairrm = DebertaV2PairRM.from_pretrained(
    "llm-blender/PairRM-hf", device_map="cuda:0"
).eval()
tokenizer = AutoTokenizer.from_pretrained("llm-blender/PairRM-hf")
source_prefix = "<|source|>"
cand1_prefix = "<|candidate1|>"
cand2_prefix = "<|candidate2|>"
inputs = ["hello!", "I love you!"]
candidates_A = ["hi!", "I hate you!"]
candidates_B = ["f**k off!", "I love you, too!"]


def tokenize_pair(
    sources: List[str],
    candidate1s: List[str],
    candidate2s: List[str],
    source_max_length=1224,
    candidate_max_length=412,
):
    ids = []
    assert len(sources) == len(candidate1s) == len(candidate2s)
    max_length = source_max_length + 2 * candidate_max_length
    for i in range(len(sources)):
        source_ids = tokenizer.encode(
            source_prefix + sources[i], max_length=source_max_length, truncation=True
        )
        candidate_max_length = (max_length - len(source_ids)) // 2
        candidate1_ids = tokenizer.encode(
            cand1_prefix + candidate1s[i],
            max_length=candidate_max_length,
            truncation=True,
        )
        candidate2_ids = tokenizer.encode(
            cand2_prefix + candidate2s[i],
            max_length=candidate_max_length,
            truncation=True,
        )
        ids.append(source_ids + candidate1_ids + candidate2_ids)
    encodings = tokenizer.pad(
        {"input_ids": ids},
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
    )
    return encodings


encodings = tokenize_pair(inputs, candidates_A, candidates_B)
encodings = {k: v.to(pairrm.device) for k, v in encodings.items()}
outputs = pairrm(**encodings)
logits = outputs.logits.tolist()
comparison_results = outputs.logits > 0
print(logits)
# [1.9003021717071533, -1.2547134160995483]
print(comparison_results)
# tensor([ True, False], device='cuda:0'), which means whether candidate A is better than candidate B for each input

# results = pairrm.compare(
#     inputs, candidates_A, candidates_B
# )
# print(results)
