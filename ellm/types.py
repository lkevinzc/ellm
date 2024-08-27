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
    init_clash: bool
    same: bool
    info: Metric


class RewardData(NamedTuple):
    pair_features: torch.Tensor  # (B, 2, d)
    loss_masks: torch.Tensor  # (B,)
