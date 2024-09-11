from typing import Any, List

import llm_blender

from ellm.oracles.base import OracleBase


class PairRMOracle(OracleBase):
    def __init__(self, **_) -> None:
        super().__init__()
        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM")

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_logits: bool = False,
        disable_tqdm: bool = False,
    ) -> List[Any]:
        return self.blender.compare(
            inputs,
            candidates_A,
            candidates_B,
            batch_size=batch_size,
            return_logits=return_logits,
            disable_tqdm=disable_tqdm,
        )
