from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, NamedTuple

import torch

Metric = Dict[str, Any]


class DAPAlgo(Enum):
    DPO = 0
    IPO = 1
    SimPO = 2


@dataclass
class PreferenceData:
    prompt: str
    chosen_response: str
    rejected_response: str
    chosen_feature: torch.Tensor
    rejected_feature: torch.Tensor
    init_clash: bool
    same: bool
    is_model_data: bool
    info: Metric


class RewardData(NamedTuple):
    pair_features: torch.Tensor  # (B, 2, d)
    loss_masks: torch.Tensor  # (B,)
