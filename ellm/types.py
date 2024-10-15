from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, NamedTuple

import torch

Metric = Dict[str, Any]


class DAPAlgo(Enum):
    DPO = 0
    IPO = 1
    SLiC = 2
    SimPO = 3


@dataclass
class PreferenceData:
    prompt: str
    chosen_response: str
    rejected_response: str
    chosen_id: int = 0
    chosen_feature: torch.Tensor = None
    rejected_feature: torch.Tensor = None
    init_clash: bool = False
    same: bool = False
    is_model_data: bool = False
    info: Metric = None
    env_chosen_response: str = ""
    env_rejected_response: str = ""


class RewardData(NamedTuple):
    pair_features: torch.Tensor  # (B, 2, d)
    loss_masks: torch.Tensor  # (B,)
