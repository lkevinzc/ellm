from dataclasses import dataclass
from typing import Any, Dict, NamedTuple

import torch

Metric = Dict[str, Any]


@dataclass
class PreferenceData:
    prompt: str
    chosen_response: str
    rejected_response: str
    chosen_feature: torch.Tensor
    rejected_feature: torch.Tensor
    max_rewarding_feature: torch.Tensor
    chosen_idx: int
    init_clash: bool
    same: bool
    info: Metric


class RewardData(NamedTuple):
    pair_features: torch.Tensor  # (B, 2, d)
    max_rewarding_features: torch.Tensor  # (B, E, d) for ensemble
    loss_masks: torch.Tensor  # (B,)
    chosen_idx: torch.Tensor  # (B,)
