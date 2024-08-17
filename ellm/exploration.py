from typing import Dict, List

import torch
import tree
from transformers import AutoTokenizer

from ellm.rm.backbone import DebertaV2PairRM
from ellm.rm.model import RewardModel


class Explorer:
    def __init__(self, reward_model: RewardModel) -> None:
        self.backbone = DebertaV2PairRM.from_pretrained(
            "llm-blender/PairRM-hf", device_map="cuda:0"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained("llm-blender/PairRM-hf")
        self.source_prefix = "<|source|>"
        self.cand_prefix = "<|candidate|>"
        self.max_length = 2048
        self.source_max_length = 1224
        self.backbone_bs = 8

        self.reward_model = reward_model

    def select(
        self, prompts: List[str], candidates: Dict[int, List[str]]
    ) -> Dict[int, List[str]]:
        """Select dueling responses from candidates.

        Args:
            prompts: A list of prompt texts, M
            candidates: Lists of responses per prompt, M -> N

        Returns:
            Dict[int, List[str]]: Pair of responses per prompt, M -> 2
        """
        input_ids = []
        M = len(prompts)
        N = len(candidates[0])
        for i in range(M):
            source_ids = self.tokenizer.encode(
                self.source_prefix + prompts[i],
                max_length=self.source_max_length,
                truncation=True,
            )
            candidate_max_length = self.max_length - len(source_ids)
            for j in range(N):
                candidate_ids = self.tokenizer.encode(
                    self.cand_prefix + candidates[i][j],
                    max_length=candidate_max_length,
                    truncation=True,
                )
                input_ids.append(source_ids + candidate_ids)
        encodings = self.tokenizer.pad(
            {"input_ids": input_ids},
            return_tensors="pt",
        )

        features = []
        for ndx in range(0, M * N, self.backbone_bs):
            batch_enc = tree.map_structure(
                lambda x: x[ndx : min(ndx + self.backbone_bs, M * N)].to(
                    self.backbone.device
                ),
                encodings,
            )
            features.append(self.backbone.get_feature(**batch_enc))
        features = torch.cat(features, dim=0)  # (M*N, d)
        features = features.reshape(M, N, -1)

        selected_candidate_indices = self.reward_model.get_duel_actions(
            features
        ).cpu()  # (M, 2)
        dueling_candidates = {}
        for i, sel_idx in enumerate(selected_candidate_indices):
            dueling_candidates[i] = [candidates[i][j] for j in sel_idx]
        return dueling_candidates
