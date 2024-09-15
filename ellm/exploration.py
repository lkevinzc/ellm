import abc
from argparse import Namespace
from collections import deque
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import tree

from ellm.rm.backbone import RMBackbone
from ellm.rm.model import RewardModel
from ellm.rm.uncertainty import kl_ensemble
from ellm.types import Metric


@dataclass
class ExplorationResults:
    dueling_candidates: Dict[int, List[str]]
    candidate_features: torch.Tensor
    init_clash: List[bool]
    is_model_data: List[bool]
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
        (
            _,
            dueling_candidates,
            features,
            selected_candidate_indices,
            init_clash,
            info,
        ) = self._inner_select(prompts, candidates)
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
            is_model_data=[False] * len(prompts),
            info=info,
        )

    def _inner_select(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
    ):
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

        info = {
            "explorer/agreed_best_resp_std": np.nan_to_num(agreed_best_resp_std),
            "explorer/not_agreed_best_resp_std": np.nan_to_num(
                not_agreed_best_resp_std
            ),
        }
        return (
            rewards,
            dueling_candidates,
            features,
            selected_candidate_indices,
            init_clash,
            info,
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


class ModelBasedExplorer(Explorer):
    """It not only explores based on Thompson sampling, but also synthesizes
    model rollout when it trusts itself to boot sample efficiency."""

    def __init__(
        self, reward_model: RewardModel, rm_backbone: RMBackbone, args: Namespace
    ) -> None:
        super().__init__(reward_model, rm_backbone, args)
        self.count = 1
        self.trust_region_scale = args.trust_region_scale
        self.burn_in_period = args.burn_in_period
        self.history_prompts = deque()
        self.history_dueling_candidates = deque()
        self.thresholds = deque(maxlen=1)

    def select(
        self, prompts: List[str], candidates: Dict[int, List[str]]
    ) -> ExplorationResults:
        (
            rewards,  # rewards: (E, M, N, 1)
            dueling_candidates,
            features,
            selected_candidate_indices,
            init_clash,
            info,
        ) = self._inner_select(prompts, candidates)
        is_model_data = [False] * len(prompts)
        model_chosen_rewards = []
        model_rejected_rewards = []
        if self.count > self.burn_in_period:
            # Estimate trust region boundary from history.
            b = min(128, len(self.history_prompts))
            tr_ind = np.random.choice(len(self.history_prompts), size=b, replace=False)
            tr_features = self._get_features(
                [self.history_prompts[i] for i in tr_ind],
                [self.history_dueling_candidates[i] for i in tr_ind],
            )  # (b, 2, d)
            tr_rewards = self.reward_model.get_rewards(tr_features)  # (E, b, 2, 1)
            tr_kl = kl_ensemble(tr_rewards)  # (b, 2, 2)
            assert not torch.isnan(tr_kl).any()
            threshold = torch.quantile(
                _tril_flatten(tr_kl), q=0.01, interpolation="nearest"
            ).item()
            self.thresholds.append(threshold)

            # Construct the trust region.
            kl = kl_ensemble(rewards)  # (M, N, N')
            assert not torch.isnan(kl).any()
            trusted = kl < self.trust_region_scale * np.mean(self.thresholds)
            valid = trusted * torch.triu(torch.ones_like(kl), diagonal=1)
            mean_rewards = rewards.mean(0)  # (M, N, 1)
            for i in range(len(prompts)):
                is_model_data[i] = (valid[i].sum() > 0).item()
                if is_model_data[i]:
                    tr_pairs = torch.where(valid[i])
                    sel_idx = np.random.choice(len(tr_pairs[0]))
                    # logging.info(f"{tr_pairs}, {sel_idx}")
                    tr_rewards = mean_rewards[
                        i, [tr_pairs[0][sel_idx], tr_pairs[1][sel_idx]]
                    ].reshape(-1)
                    tr_rank = tr_rewards.argsort()
                    assert len(tr_rank) == 2
                    tr_chosen = tr_pairs[tr_rank[1]][sel_idx]
                    tr_rejected = tr_pairs[tr_rank[0]][sel_idx]
                    # logging.info(f"{tr_rewards},{tr_rank},{tr_chosen}, {tr_rejected}")
                    dueling_candidates[i] = [
                        candidates[i][tr_chosen],
                        candidates[i][tr_rejected],
                    ]
                    selected_candidate_indices[i] = torch.tensor(
                        [tr_chosen, tr_rejected]
                    )
                    model_chosen_rewards.append(mean_rewards[i, tr_chosen].item())
                    model_rejected_rewards.append(mean_rewards[i, tr_rejected].item())

        # Update history.
        for i in range(len(prompts)):
            # Only record true environment experiences.
            if not is_model_data[i]:
                self.history_prompts.append(prompts[i])
                self.history_dueling_candidates.append(dueling_candidates[i])
        self.count += 1

        info.update(
            {
                "explorer/tr_threshold_mean": np.mean(self.thresholds),
                "explorer/model_chosen_rewards": np.mean(model_chosen_rewards),
                "explorer/model_rejected_rewards": np.mean(model_rejected_rewards),
                "explorer/history_length": len(self.history_prompts),
                "explorer/model_data_ratio": np.mean(is_model_data),
            }
        )
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
            is_model_data=is_model_data,
            info=info,
        )


def _tril_flatten(batch_sm: torch.Tensor):
    """Take out off-diagonal lower triangular elements of batched square matrices."""
    N = batch_sm.size(-1)
    indices = torch.tril_indices(N, N)
    off_diag = indices[0, :] != indices[1, :]
    indices = indices[:, off_diag]
    indices = N * indices[0] + indices[1]
    return batch_sm.flatten(-2)[..., indices]
