from dataclasses import dataclass

import torch


@dataclass
class PreferenceData:
    prompt: str
    chosen_response: str
    rejected_response: str
    chosen_feature: torch.Tensor
    rejected_feature: torch.Tensor
    init_clash: bool
    same: bool
