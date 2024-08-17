import abc
import random

import einops
import torch

from ellm.rm.ensemble import EnsembleModel


class RewardModel(abc.ABC):

    @abc.abstractmethod
    def get_duel_actions(self, features: torch.Tensor) -> torch.LongTensor:
        """Get dueling actions based on rewards of given features.

        Args:
            features (torch.Tensor): (M, N, d)

        Returns:
            torch.LongTensor: (M, 2)
        """


class EnnDTS(RewardModel):
    """Double Thompson Sampling based on ensemble."""

    def __init__(self, model: EnsembleModel) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        self.infer_bs = 32
        self.max_trial = 3

    @torch.no_grad
    def get_duel_actions(self, features: torch.Tensor) -> torch.LongTensor:
        M, N, _ = features.shape
        E = self.model.num_ensemble
        features = einops.rearrange(features, "m n d -> (m n) d")
        rewards = []
        for ndx in range(0, len(features), self.infer_bs):
            batch_feat = features[ndx : min(ndx + self.infer_bs, len(features))]
            batch_feat = batch_feat[None, :, :].repeat([E, 1, 1])
            rewards.append(self.model(batch_feat))
        rewards = torch.cat(rewards, dim=1)  # (E, M*N, 1)
        rewards = rewards.reshape(E, M, N, 1)
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
