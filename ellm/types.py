from dataclasses import dataclass
from typing import Any, Dict

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
