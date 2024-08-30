import abc
import random
from argparse import Namespace
from typing import Any, Dict, Tuple

import einops
import torch
import torch.nn.functional as F
from torch import nn, optim

from ellm.rm.networks import EnsembleModel, MLPModel
from ellm.rm.optim import LAdam
from ellm.utils.buffer import UniformBuffer


class RewardModel(abc.ABC, nn.Module):

    train_bs = 32
    infer_bs = 32

    @abc.abstractclassmethod
    def get_metrics(cls):
        """Get learning metrics."""

    @abc.abstractmethod
    def get_duel_actions(
        self, features: torch.Tensor
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Get dueling actions based on rewards of given features.

        Args:
            features (torch.Tensor): (M, N, d)

        Returns:
            Tuple[torch.LongTensor]: [(M, 1), (M, 1)]
        """

    @abc.abstractmethod
    def get_best_action(self, features: torch.Tensor) -> torch.LongTensor:
        """Get Best-of-N action based on rewards of given features.

        Args:
            features (torch.Tensor): (M, N, d)

        Returns:
            torch.LongTensor: (M, 1)
        """

    @abc.abstractmethod
    def learn(self, buffer: UniformBuffer) -> Dict[str, Any]:
        """Learn the reward model based on preference data."""


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self,
        chosen_reward: torch.Tensor,
        reject_reward: torch.Tensor,
        mask: torch.Tensor,
        margin: torch.Tensor = None,
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return (loss * mask).mean()


class LmcFGTS(RewardModel):
    "Feel-Good Thompson Sampling for contextual dueling bandits based on Langevin Monte Carlo."

    @classmethod
    def get_metrics(cls):
        return {
            "train/rm/loss_rew": 0,
            "train/rm/loss_reg": 0,
        }

    def __init__(self, args: Namespace) -> None:
        super().__init__()
        encoding_dim = 2048  # Fixed due to PairRM's backbone
        # Posterior models
        self.model_1 = MLPModel(
            encoding_dim=encoding_dim,
            hidden_dim=args.rm_hidden_dim,
        )
        self.model_2 = MLPModel(
            encoding_dim=encoding_dim,
            hidden_dim=args.rm_hidden_dim,
        )
        self.model_1.init()
        self.model_2.init()

        # LMC optimizers
        self.optimizer_1 = LAdam(
            self.model_1.parameters(),
            lr=args.rm_lr,
            temperature=args.lmc_temp,
            a=args.lmc_a,
            asgld=args.lmc_asgld,
        )
        self.optimizer_2 = LAdam(
            self.model_2.parameters(),
            lr=args.rm_lr,
            temperature=args.lmc_temp,
            a=args.lmc_a,
            asgld=args.lmc_asgld,
        )

        self.reg_lambda = args.reg_lambda
        self.sgd_steps = args.rm_sgd_steps
        self.loss_fn = PairWiseLoss()

    @torch.no_grad
    def _get_rewards(self, features: torch.Tensor) -> torch.Tensor:
        M, N, _ = features.shape
        features = einops.rearrange(features, "m n d -> (m n) d")
        rewards_1 = []
        rewards_2 = []
        for ndx in range(0, len(features), self.infer_bs):
            batch_feat = features[ndx : min(ndx + self.infer_bs, len(features))]
            rewards_1.append(self.model_1(batch_feat))
            rewards_2.append(self.model_2(batch_feat))
        rewards_1 = torch.cat(rewards_1).view(M, N, 1)
        rewards_2 = torch.cat(rewards_2).view(M, N, 1)
        return torch.stack([rewards_1, rewards_2])  # (2, M, N, 1)

    @torch.no_grad
    def get_best_action(self, features: torch.Tensor) -> torch.LongTensor:
        rewards = self._get_rewards(features)  # (2, M, N, 1)
        avg_rewards = rewards.mean(0)  # (M, N, 1)
        best_actions = avg_rewards.argmax(dim=1)  # (M, 1)
        return best_actions

    @torch.no_grad
    def get_duel_actions(
        self, features: torch.Tensor
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        rewards = self._get_rewards(features)
        best_actions = rewards.argmax(dim=2)
        first_actions, second_actions = best_actions
        return first_actions, second_actions

    def learn(self, buffer: UniformBuffer) -> Dict[str, Any]:
        total_num_queries = buffer.total_num_queries

        for _ in range(self.sgd_steps):
            batch = buffer.sample(self.train_bs)
            pair_feats = batch.pair_features.view(2 * self.train_bs, -1)
            scores_1 = self.model_1(pair_feats).view(self.train_bs, 2, 1)
            scores_2 = self.model_2(pair_feats).view(self.train_bs, 2, 1)
            loss_rew_1 = self.loss_fn(
                scores_1[..., 0, :], scores_1[..., 1, :], batch.loss_masks
            )
            loss_rew_2 = self.loss_fn(
                scores_2[..., 0, :], scores_2[..., 1, :], batch.loss_masks
            )
            loss_reg_1 = (
                self.reg_lambda
                * self.train_bs
                / total_num_queries
                * self.model_1.regularization()
            )
            loss_reg_2 = (
                self.reg_lambda
                * self.train_bs
                / total_num_queries
                * self.model_2.regularization()
            )

            self.optimizer_1.zero_grad()
            (loss_rew_1 + loss_reg_1).backward()
            self.optimizer_1.step()

            self.optimizer_2.zero_grad()
            (loss_rew_2 + loss_reg_2).backward()
            self.optimizer_2.step()

        return {
            "train/rm/loss_rew": ((loss_rew_1 + loss_rew_2) / 2).detach(),
            "train/rm/loss_reg": ((loss_reg_1 + loss_reg_2) / 2).detach(),
        }


class EnnDTS(RewardModel):
    """Double Thompson Sampling based on ensemble."""

    @classmethod
    def get_metrics(cls):
        return {
            "train/rm/loss_rew": 0,
            "train/rm/loss_reg": 0,
            "train/rm/lambda": 0,
        }

    def __init__(self, args: Namespace) -> None:
        super().__init__()
        assert args.enn_max_try <= args.num_ensemble // 2

        self.model = EnsembleModel(
            encoding_dim=2048,  # Fixed due to PairRM's backbone
            num_ensemble=args.num_ensemble,
            hidden_dim=args.rm_hidden_dim,
        )
        self.model.init()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.rm_lr)
        self.reg_lambda = args.enn_lambda
        self.max_resample = args.enn_max_try
        self.allow_second_best = args.exp_allow_second_best
        self.sgd_steps = args.rm_sgd_steps
        self.loss_fn = PairWiseLoss()

    @torch.no_grad
    def _get_rewards(self, features: torch.Tensor) -> torch.Tensor:
        M, N, _ = features.shape
        E = self.model.num_ensemble
        features = einops.rearrange(features, "m n d -> (m n) d")
        rewards = []
        for ndx in range(0, len(features), self.infer_bs):
            batch_feat = features[ndx : min(ndx + self.infer_bs, len(features))]
            batch_feat = batch_feat[None, :, :].repeat([E, 1, 1])
            rewards.append(self.model(batch_feat))
        rewards = torch.cat(rewards, dim=1)  # (E, M*N, 1)
        rewards = rewards.view(E, M, N, 1)
        return rewards

    @torch.no_grad
    def get_best_action(self, features: torch.Tensor) -> torch.LongTensor:
        rewards = self._get_rewards(features)  # (E, M, N, 1)
        avg_rewards = rewards.mean(0)  # (M, N, 1)
        best_actions = avg_rewards.argmax(dim=1)  # (M, 1)
        return best_actions

    @torch.no_grad
    def get_duel_actions(
        self, features: torch.Tensor
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        rewards = self._get_rewards(features)
        E = rewards.shape[0]
        best_actions = rewards.argmax(dim=2)  # (E, M, 1)
        # sample without replacement
        s1 = list(range(E // 2))
        random.shuffle(s1)
        s2 = list(range(E // 2, E))
        random.shuffle(s2)
        first_actions = best_actions[s1[0]]
        second_actions = torch.ones_like(first_actions) * -1
        for actions in best_actions[s2[: self.max_resample]]:
            valid_idx = (actions != first_actions) * (second_actions == -1)
            second_actions[valid_idx] = actions[valid_idx]
            if -1 not in second_actions:
                break
        if self.allow_second_best:
            second_best_actions = rewards.argsort(dim=2)[..., -2, :]
            for actions in second_best_actions[s2[: self.max_resample]]:
                valid_idx = (actions != first_actions) * (second_actions == -1)
                second_actions[valid_idx] = actions[valid_idx]
                if -1 not in second_actions:
                    break
        second_actions = torch.where(
            second_actions == -1, first_actions, second_actions
        )
        return first_actions, second_actions

    def learn(self, buffer: UniformBuffer) -> Dict[str, Any]:
        total_num_queries = buffer.total_num_queries
        for _ in range(self.sgd_steps):
            batch = buffer.sample(self.train_bs)
            pair_feats = batch.pair_features.view(2 * self.train_bs, -1)
            batch_inp = pair_feats[None, :, :].repeat([self.model.num_ensemble, 1, 1])
            scores = self.model(batch_inp)
            scores = scores.view(self.model.num_ensemble, self.train_bs, 2, 1)
            chosen_scores, rejected_scores = scores[..., 0, :], scores[..., 1, :]
            loss_rew = self.loss_fn(
                chosen_scores, rejected_scores, batch.loss_masks[None]
            )
            loss_reg = (
                self.reg_lambda
                * self.train_bs
                / total_num_queries
                * self.model.regularization()
            )
            self.optimizer.zero_grad()
            (loss_rew + loss_reg).backward()
            self.optimizer.step()

        return {
            "train/rm/loss_rew": loss_rew.detach(),
            "train/rm/loss_reg": loss_reg.detach(),
            "train/rm/lambda": self.reg_lambda * self.train_bs / total_num_queries,
        }


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # "broadcast" instead of copy
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
    except Exception:
        # NOTE: This exception is added for the purpose of setting breakpoint to
        # debug weight loading issues.
        raise
