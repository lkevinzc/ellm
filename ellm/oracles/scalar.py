from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl.trainer.utils import get_reward

from ellm.oracles.base import OracleBase
from ellm.utils.data import RankDataset


class ScalarRMOracle(OracleBase):
    def __init__(self, reward_model_path: str, tokenizer_path: str, **_) -> None:
        super().__init__()
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path, num_labels=1, trust_remote_code=False, device_map="cuda"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side="left",
            trust_remote_code=False,
        )

    @torch.no_grad
    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> torch.List[torch.Any]:
        assert len(candidates_A) == len(
            candidates_B
        ), "Number of candidates_A and candidates_B must be the same"
        assert len(inputs) == len(
            candidates_A
        ), "Number of inputs and candidates must be the same"
        dataset = RankDataset(
            inputs,
            candidates_A,
            candidates_B,
            self.tokenizer,
            prompt_max_length=1224,
            completion_max_length=412,
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn
        )
        all_scores = []
        for batch in tqdm(
            iter(dataloader), desc="Ranking candidates", disable=disable_tqdm
        ):
            device = torch.cuda.current_device()
            prompt_ids, completion_ids = batch
            num_data = len(prompt_ids)
            prompt_ids = prompt_ids.squeeze(1).to(device).repeat(2, 1)
            completion_ids = completion_ids.squeeze(1).to(device)
            context_length = prompt_ids.shape[1]

            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            _, scores, _ = get_reward(
                self.reward_model,
                prompt_completion_ids,
                self.tokenizer.pad_token_id,
                context_length,
            )
            scores = scores.cpu().squeeze().numpy()
            scores_1 = scores[:num_data]
            scores_2 = scores[num_data:]
            all_scores.append(np.stack([scores_1, scores_2], axis=-1))
        all_scores = np.concatenate(all_scores)
        logits = all_scores[:, 0] - all_scores[:, 1]
        probs = torch.from_numpy(logits).sigmoid().numpy()
        if return_probs:
            return probs
        else:
            return probs > 0.5
