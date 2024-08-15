from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimPOLoss(nn.Module):
    def __init__(
        self,
        beta: float,
        gamma_beta_ratio: float,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.gamma_beta_ratio = gamma_beta_ratio
        self.loss_type = loss_type
        assert loss_type in (
            "sigmoid",
            "hinge",
        ), f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        logits = pi_logratios - self.gamma_beta_ratio
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)

        loss = losses.mean()
        chosen_rewards = self.beta * policy_chosen_logps.detach()
        rejected_rewards = self.beta * policy_rejected_logps.detach()

        return loss, chosen_rewards, rejected_rewards
