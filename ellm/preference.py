import time
from typing import List, Union

import torch

from ellm.actor import Actor
from ellm.types import PreferenceData
from ellm.utils.deepspeed import DeepspeedStrategy


class PreferenceCollector:
    def __init__(
        self,
        actors: List[Actor],
        tokenizer,
        prompt_max_len: int,
        strategy: DeepspeedStrategy,
        logger=None,
    ) -> None:
        self.actors = actors
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.prompt_max_len = prompt_max_len
        self.logger = logger

    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def __call__(self, prompts: Union[str, List[str]]) -> List[PreferenceData]:
        device = torch.cuda.current_device()

        # generate response & get feedback
        st_time = time.time()
        rank = torch.distributed.get_rank()
        actor = self.actors[rank % len(self.actors)]
        fut = actor.futures.step(prompts)
        preference_data = fut.result()
        actor_time = time.time() - st_time

        info = {"actor/generate_time": actor_time}
        if self.logger is not None:
            self.logger.log(info)

        return preference_data
