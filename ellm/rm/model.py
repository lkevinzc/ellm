import abc
import random
from argparse import Namespace
from typing import Any, Dict, Iterable

import einops
import torch
import torch.nn.functional as F
from torch import nn, optim

from ellm.rm.ensemble import EnsembleModel
from ellm.utils.buffer import UniformBuffer


class RewardModel(abc.ABC, nn.Module):

    @abc.abstractmethod
    def get_duel_actions(self, features: torch.Tensor) -> torch.LongTensor:
        """Get dueling actions based on rewards of given features.

        Args:
            features (torch.Tensor): (M, N, d)

        Returns:
            torch.LongTensor: (M, 2)
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
    def learn(self, dataset: Iterable) -> Dict[str, Any]:
        """Learn the reward model based on preference data."""


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self,
        chosen_reward: torch.Tensor,
        reject_reward: torch.Tensor,
        margin: torch.Tensor = None,
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


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
        self.model = EnsembleModel(
            encoding_dim=2048,  # Fixed due to PairRM
            num_ensemble=args.num_ensemble,
            hidden_dim=args.enn_hidden_dim,
        )
        self.model.init()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.enn_lr)
        self.reg_lambda = args.enn_lambda
        self.sgd_steps = args.enn_sgd_steps
        self.loss_fn = PairWiseLoss()
        self.train_bs = 32
        self.infer_bs = 32
        # self.max_trial = 3

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
    def get_duel_actions(self, features: torch.Tensor) -> torch.LongTensor:
        rewards = self._get_rewards(features)
        E, _, N, _ = rewards.shape
        best_actions = rewards.argmax(dim=2)  # (E, M, 1)
        # sample without replacement
        s = list(range(E))
        random.shuffle(s)
        first_actions = best_actions[s[0]]
        second_actions = torch.ones_like(first_actions) * -1
        for actions in best_actions[s[1:]]:
            valid_idx = (actions != first_actions) * (second_actions == -1)
            second_actions[valid_idx] = actions[valid_idx]
            if -1 not in second_actions:
                break
        rand_actions = torch.randint_like(second_actions, N)
        second_actions = torch.where(second_actions == -1, rand_actions, second_actions)
        return torch.cat([first_actions, second_actions], dim=-1)

    def learn(self, buffer: UniformBuffer, total_num_queries: int) -> Dict[str, Any]:
        for _ in range(self.sgd_steps):
            batch = buffer.sample(self.train_bs).view(2 * self.train_bs, -1)
            batch_inp = batch[None, :, :].repeat([self.model.num_ensemble, 1, 1])
            scores = self.model(batch_inp)
            scores = scores.view(self.model.num_ensemble, self.train_bs, 2, 1)
            chosen_scores, rejected_scores = scores[..., 0, :], scores[..., 1, :]
            loss_rew = self.loss_fn(chosen_scores, rejected_scores)
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
