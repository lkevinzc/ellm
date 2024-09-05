import abc
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import tree

from ellm.rm.backbone import RMBackbone
from ellm.rm.model import RewardModel
from ellm.types import Metric


@dataclass
class ExplorationResults:
    dueling_candidates: Dict[int, List[str]]
    candidate_features: torch.Tensor
    init_clash: List[bool]
    info: Metric


class ExplorerBase(abc.ABC):
    @abc.abstractmethod
    def best_of_n(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
    ) -> List[str]:
        """Best-of-N generation given the reward model.

        Args:
            prompts (List[str]): A list of prompt texts, M
            candidates (Dict[int, List[str]]): Lists of responses per prompt, M -> N

        Returns:
            List[str]: A list of the best response per prompt.
        """

    @abc.abstractmethod
    def select(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
    ) -> ExplorationResults:
        """Select dueling responses from candidates.

        Args:
            prompts (List[str]): A list of prompt texts, M
            candidates (Dict[int, List[str]]): Lists of responses per prompt, M -> N

        Returns:
            ExplorationResults: Pair of responses per prompt (and features), M -> 2
        """


class Explorer(ExplorerBase):
    def __init__(
        self, reward_model: RewardModel, rm_backbone: RMBackbone, args: Namespace
    ) -> None:
        self.backbone = rm_backbone
        self.reward_model = reward_model

        self.max_length = 2048
        self.source_max_length = 1224
        self.backbone_bs = 8

        self.random_sampling = args.exp_rnd_sample

    def best_of_n(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
    ) -> List[str]:
        """Best-of-N generation given the reward model.

        Args:
            prompts (List[str]): A list of prompt texts, M
            candidates (Dict[int, List[str]]): Lists of responses per prompt, M -> N

        Returns:
            List[str]: A list of the best response per prompt.
        """
        features = self._get_features(prompts, candidates)  # (M, N, d)
        best_response_indices = (
            self.reward_model.get_best_action(features).cpu().squeeze()
        )  # (M,)
        best_responses = [
            candidates[i][sel_idx] for i, sel_idx in enumerate(best_response_indices)
        ]
        return best_responses

    def select(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
    ) -> ExplorationResults:
        """Select dueling responses from candidates.

        Args:
            prompts (List[str]): A list of prompt texts, M
            candidates (Dict[int, List[str]]): Lists of responses per prompt, M -> N

        Returns:
            ExplorationResults: Pair of responses per prompt (and features), M -> 2
        """
        features = self._get_features(prompts, candidates)  # (M, N, d)
        rewards, first_indices, second_indices = self.reward_model.get_duel_actions(
            features
        )  # rewards: (E or 2, M, N, 1); indices: both (M, 1)

        init_clash = (second_indices == first_indices).cpu().squeeze()
        rewards_with_agreed_best = rewards[:, init_clash]
        clashed_best_indices = second_indices[init_clash]
        agreed_best_resp_std = np.mean(
            [
                torch.std(rewards_with_agreed_best[:, i, clashed_best_indices[i]]).cpu()
                for i in range(len(clashed_best_indices))
            ]
        )
        rewards_without_agreed_best = rewards[:, ~init_clash]
        not_clashed_best_indices = second_indices[~init_clash]
        not_agreed_best_resp_std = np.mean(
            [
                torch.std(
                    rewards_without_agreed_best[:, i, not_clashed_best_indices[i]]
                ).cpu()
                for i in range(len(not_clashed_best_indices))
            ]
        )
        # In the case where both responses are the same, do random sampling
        if self.random_sampling:
            N = features.shape[1]
            rnd_second_indices = torch.ones_like(second_indices) * -1
            for _ in range(3):
                # Clash prob 1 / N^3
                rand_indices = torch.randint_like(second_indices, N)
                valid_idx = (rand_indices != first_indices) * (rnd_second_indices == -1)
                rnd_second_indices[valid_idx] = rand_indices[valid_idx]
                if -1 not in rnd_second_indices:
                    break

            second_indices = torch.where(
                second_indices == first_indices, rnd_second_indices, second_indices
            )

        selected_candidate_indices = torch.cat(
            [first_indices, second_indices], dim=-1
        ).cpu()
        dueling_candidates = {}
        for i, sel_idx in enumerate(selected_candidate_indices):
            dueling_candidates[i] = [candidates[i][j] for j in sel_idx]
        return ExplorationResults(
            dueling_candidates=dueling_candidates,
            candidate_features=(
                torch.stack(
                    [
                        features[i][selected_candidate_indices[i]]
                        for i in range(len(prompts))
                    ]
                ).cpu()
            ),
            init_clash=init_clash.tolist(),
            info={
                "explorer/agreed_best_resp_std": np.nan_to_num(agreed_best_resp_std),
                "explorer/not_agreed_best_resp_std": np.nan_to_num(
                    not_agreed_best_resp_std
                ),
            },
        )

    def _get_features(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
    ):
        input_ids = []
        M = len(prompts)
        N = len(candidates[0])
        for i in range(M):
            for j in range(N):
                pair_ids = self.backbone.tokenize_pair(
                    prompt=prompts[i],
                    candidate=candidates[i][j],
                    source_max_length=self.source_max_length,
                    max_length=self.max_length,
                )
                input_ids.append(pair_ids)
        encodings = self.backbone.tokenizer.pad(
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
        features = features.view(M, N, -1)
        return features
