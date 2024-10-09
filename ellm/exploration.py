import abc
import random
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict, List

import einops
import numpy as np
import torch
import tree

from ellm.rm import uncertainty
from ellm.rm.backbone import RMBackbone
from ellm.rm.model import RewardModel
from ellm.types import Metric


@dataclass
class ExplorationResults:
    dueling_candidates: Dict[int, List[str]]
    candidate_features: torch.Tensor
    init_clash: List[bool]
    is_model_data: List[bool]
    all_rewards: torch.Tensor
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

    @abc.abstractmethod
    def compare(self, candidate_features: torch.Tensor) -> torch.Tensor:
        """Compare candidates using the reward model.

        Args:
            candidate_features (torch.Tensor): (M, 2, d)

        Returns:
            torch.Tensor: (M,), 1 means the first wins
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
            rewards,
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
                )
            ),
            init_clash=init_clash.tolist(),
            is_model_data=[False] * len(prompts),
            all_rewards=rewards,
            info=info,
        )

    def compare(self, candidate_features: torch.Tensor) -> torch.Tensor:
        rewards = self.reward_model.get_rewards(candidate_features).mean(0)  # (M, 2, 1)
        return (rewards[:, 0] > rewards[:, 1]).squeeze().float().cpu().numpy()

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
        self.burn_in_period = args.burn_in_period
        self.max_model_data_ratio = args.max_model_data_ratio
        self.model_data_selector = getattr(self, f"_{args.model_data_strategy}_select")
        self.pure_model_based = args.pure_model_based

    def _random_select(
        self,
        candidates,
        rewards,
        dueling_candidates,
        selected_candidate_indices,
        is_model_data,
    ):
        reward_margin = rewards - einops.rearrange(rewards, "e m n 1 -> e m 1 n")
        E, M, _, _ = reward_margin.shape
        random_belief_reward_margin = reward_margin[
            torch.randint(E, (M,)), torch.arange(M)
        ]  # M, N, N'
        # mean_rewards = rewards.mean(0)
        max_model_data = int(len(is_model_data) * self.max_model_data_ratio)
        is_model_data[:max_model_data] = 1
        random.shuffle(is_model_data)
        for i, imd in enumerate(is_model_data):
            if imd:
                # candidate_1, candidate_2 = np.random.choice(
                #     len(candidates[i]), 2, replace=False
                # )
                # if single_rewards[i, candidate_1] > single_rewards[i, candidate_2]:
                #     rnd_chosen, rnd_rejected = candidate_1, candidate_2
                # else:
                #     rnd_chosen, rnd_rejected = candidate_2, candidate_1
                # rnd_chosen = mean_rewards[i].squeeze().argmax()
                # rnd_rejected = mean_rewards[i].squeeze().argmin()
                margin_i = random_belief_reward_margin[i]
                margin_i_abs = torch.abs(margin_i)
                tr_pairs = torch.where(margin_i_abs == margin_i_abs.max())
                sel_idx = np.random.choice(len(tr_pairs[0]))  # break tie
                candidate_1, candidate_2 = tr_pairs[0][sel_idx], tr_pairs[1][sel_idx]
                if margin_i[candidate_1, candidate_2] > 0:
                    rnd_chosen, rnd_rejected = candidate_1, candidate_2
                else:
                    rnd_chosen, rnd_rejected = candidate_2, candidate_1
                dueling_candidates[i] = [
                    candidates[i][rnd_chosen],
                    candidates[i][rnd_rejected],
                ]
                selected_candidate_indices[i] = torch.tensor([rnd_chosen, rnd_rejected])
        return dueling_candidates, selected_candidate_indices, is_model_data

    def _epistemic_uct_select(
        self,
        candidates,
        rewards,
        dueling_candidates,
        selected_candidate_indices,
        is_model_data,
    ):
        mean_rewards = rewards.mean(0)
        reward_margin_abs = torch.abs(
            mean_rewards - einops.rearrange(mean_rewards, "m n 1 -> m 1 n")
        )
        max_model_data = int(len(is_model_data) * self.max_model_data_ratio)
        uct = uncertainty.logits_variance(rewards)  # (M, N, N')
        prompt_uct = uct.mean(-1).mean(-1)
        model_data_indices = prompt_uct.argsort()[:max_model_data].tolist()
        for i in model_data_indices:
            is_model_data[i] = 1
            # candidate_1, candidate_2 = np.random.choice(
            #     len(candidates[i]), 2, replace=False
            # )
            # if mean_rewards[i, candidate_1] > mean_rewards[i, candidate_2]:
            #     rnd_chosen, rnd_rejected = candidate_1, candidate_2
            # else:
            #     rnd_chosen, rnd_rejected = candidate_2, candidate_1
            valid = uct[i] <= torch.quantile(uct[i], 0.5)
            valid_margin = reward_margin_abs[i] * valid
            tr_pairs = torch.where(valid_margin == valid_margin.max())
            sel_idx = np.random.choice(len(tr_pairs[0]))  # break tie
            candidate_1, candidate_2 = tr_pairs[0][sel_idx], tr_pairs[1][sel_idx]
            if mean_rewards[i, candidate_1] > mean_rewards[i, candidate_2]:
                rnd_chosen, rnd_rejected = candidate_1, candidate_2
            else:
                rnd_chosen, rnd_rejected = candidate_2, candidate_1
            dueling_candidates[i] = [
                candidates[i][rnd_chosen],
                candidates[i][rnd_rejected],
            ]
            selected_candidate_indices[i] = torch.tensor([rnd_chosen, rnd_rejected])
        return dueling_candidates, selected_candidate_indices, is_model_data

    def _total_uct_select(
        self,
        candidates,
        rewards,
        dueling_candidates,
        selected_candidate_indices,
        is_model_data,
    ):
        mean_rewards = rewards.mean(0)
        max_model_data = int(len(is_model_data) * self.max_model_data_ratio)
        uct = uncertainty.logits_variance(rewards)  # (M, N, N')
        al_uct = uncertainty.bernoulli_variance(rewards)  # (M, N, N')

        prompt_uct = uct.mean(-1).mean(-1)
        model_data_indices = prompt_uct.argsort()[:max_model_data].tolist()
        for i in model_data_indices:
            is_model_data[i] = 1
            tr_pairs = torch.where(al_uct[i] == al_uct[i].min())
            sel_idx = np.random.choice(len(tr_pairs[0]))  # break tie
            candidate_1, candidate_2 = tr_pairs[0][sel_idx], tr_pairs[1][sel_idx]
            if mean_rewards[i, candidate_1] > mean_rewards[i, candidate_2]:
                rnd_chosen, rnd_rejected = candidate_1, candidate_2
            else:
                rnd_chosen, rnd_rejected = candidate_2, candidate_1
            dueling_candidates[i] = [
                candidates[i][rnd_chosen],
                candidates[i][rnd_rejected],
            ]
            selected_candidate_indices[i] = torch.tensor([rnd_chosen, rnd_rejected])
        return dueling_candidates, selected_candidate_indices, is_model_data

    def select(
        self, prompts: List[str], candidates: Dict[int, List[str]]
    ) -> ExplorationResults:
        # Select the query points using exploration strategies.
        # Be optimistic and reduce uncertainty.
        (
            rewards,  # rewards: (E, M, N, 1)
            dueling_candidates,
            features,
            selected_candidate_indices,
            init_clash,
            info,
        ) = self._inner_select(prompts, candidates)
        # Replace queries that the agent is already confident about the results.
        # Utilize uncertainty to build trust region.
        is_model_data = np.zeros(len(prompts))
        model_chosen_rewards = []
        model_rejected_rewards = []
        model_pred_prob = []
        sel_pair_ep_uct = []
        sel_prompt_ep_uct = []
        uct_mean = 0
        if self.count > self.burn_in_period:
            dueling_candidates, selected_candidate_indices, is_model_data = (
                self.model_data_selector(
                    candidates,
                    rewards,
                    dueling_candidates,
                    selected_candidate_indices,
                    is_model_data,
                )
            )
            mean_rewards = rewards.mean(0)  # (M, N, 1)
            uct = uncertainty.logits_variance(rewards)
            uct_mean = uct.mean().item()

        for i in range(len(prompts)):
            if is_model_data[i]:
                tr_chosen = selected_candidate_indices[i, 0]
                tr_rejected = selected_candidate_indices[i, 1]

                model_chosen_rewards.append(mean_rewards[i, tr_chosen].item())
                model_rejected_rewards.append(mean_rewards[i, tr_rejected].item())
                model_pred_prob.append(
                    (mean_rewards[i, tr_chosen] - mean_rewards[i, tr_rejected])
                    .sigmoid()
                    .item()
                )
                sel_pair_ep_uct.append(uct[i][tr_chosen, tr_rejected].item())
                sel_prompt_ep_uct.append(uct[i].mean().item())
            else:
                if self.pure_model_based:
                    # Disable learning.
                    dueling_candidates[i] = ["dummy", "dummy"]
            # # Construct the trust region.
            # uct = self.uncertainty_fn(rewards)  # (M, N, N')
            # assert not torch.isnan(uct).any()
            # prompt_uct = uct.mean(-1).mean(-1)

            # max_model_data = int(len(prompts) * self.max_model_data_ratio)
            # prompt_valid = valid.sum(-1).sum(-1)
            # # Take prompt which has the most pairs in the trust region.
            # maybe_model_data_i = prompt_valid.argsort()[-max_model_data:].tolist()
            # prompt_uct = (uct * valid).sum(-1).sum(-1)
            # maybe_model_data_i = prompt_uct.argsort()[-max_model_data:].tolist()

            # if self.random_model_data:
            #     maybe_model_data_i = list(range(len(uct)))
            #     random.shuffle(maybe_model_data_i)
            #     maybe_model_data_i = maybe_model_data_i[:max_model_data]

            # single_rewards = rewards[0]
            # for i in maybe_model_data_i:
            #     if valid[i].sum() > 0:
            #         is_model_data[i] = True
            #         inv_uct = (uct[i].max() - uct[i]) * valid[i]
            #         tr_pairs = torch.where(inv_uct == inv_uct.max())
            #         # tr_pairs = torch.where(valid[i])
            #         sel_idx = np.random.choice(len(tr_pairs[0]))  # break tie

            #         logging.info(f"{tr_pairs}, {sel_idx}")
            #         tr_rewards = mean_rewards[
            #             i, [tr_pairs[0][sel_idx], tr_pairs[1][sel_idx]]
            #         ].reshape(-1)
            #         tr_rank = tr_rewards.argsort()
            #         assert len(tr_rank) == 2
            #         tr_chosen = tr_pairs[tr_rank[1]][sel_idx]
            #         tr_rejected = tr_pairs[tr_rank[0]][sel_idx]
            #         logging.info(f"{tr_rewards},{tr_rank},{tr_chosen}, {tr_rejected}")
            #         dueling_candidates[i] = [
            #             candidates[i][tr_chosen],
            #             candidates[i][tr_rejected],
            #         ]
            #         selected_candidate_indices[i] = torch.tensor(
            #             [tr_chosen, tr_rejected]
            #         )

        self.count += 1

        info.update(
            {
                "explorer/model_chosen_rewards": np.mean(model_chosen_rewards),
                "explorer/model_rejected_rewards": np.mean(model_rejected_rewards),
                "explorer/model_pred_prob_min": (
                    np.min(model_pred_prob) if model_pred_prob else np.nan
                ),
                "explorer/model_pred_prob_max": (
                    np.max(model_pred_prob) if model_pred_prob else np.nan
                ),
                "explorer/model_pred_prob_mean": np.mean(model_pred_prob),
                "explorer/sel_pair_ep_uct_mean": np.mean(sel_pair_ep_uct),
                "explorer/sel_pair_ep_uct_std": np.std(sel_pair_ep_uct),
                "explorer/sel_prompt_ep_uct_mean": np.std(sel_prompt_ep_uct),
                "explorer/sel_prompt_ep_uct_std": np.std(sel_prompt_ep_uct),
                "explorer/all_ep_uct_mean": uct_mean,
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
                )
            ),
            init_clash=init_clash.tolist(),
            is_model_data=is_model_data.astype("bool").tolist(),
            all_rewards=rewards,
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
