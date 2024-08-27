import time
from typing import List, Tuple, Union

import numpy as np
import torch

from ellm.actor import Actor
from ellm.types import Metric, PreferenceData
from ellm.utils.deepspeed import DeepspeedStrategy
from ellm.utils.ipc import PlasmaShmClient, PlasmaShmServer


class PreferenceCollector:
    def __init__(
        self,
        actors: List[Actor],
        tokenizer,
        ipc_server: PlasmaShmServer,
        prompt_max_len: int,
        strategy: DeepspeedStrategy,
        logger=None,
    ) -> None:
        self.actors = actors
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.prompt_max_len = prompt_max_len
        self.logger = logger
        self.ipc_client = PlasmaShmClient(ipc_server)

    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    def __call__(
        self, prompts: Union[str, List[str]], refs: Union[str, List[str]]
    ) -> Tuple[List[PreferenceData], Metric]:
        # generate response & get feedback
        st_time = time.time()
        rank = torch.distributed.get_rank()
        actor = self.actors[rank % len(self.actors)]

        if self.strategy.args.online_evaluation:
            handle = actor.step(prompts, refs)
        else:
            handle = actor.step(prompts)

        preference_data: List[PreferenceData] = self.ipc_client.deserialize_ipc(handle)

        actor_time = time.time() - st_time

        metric = {
            "actor/generate_time": actor_time,
            "actor/chosen_avg_str_len": np.mean(
                [len(p.chosen_response) for p in preference_data]
            ),
            "actor/rejected_avg_str_len": np.mean(
                [len(p.rejected_response) for p in preference_data]
            ),
            "actor/init_clash_ratio": np.mean([p.init_clash for p in preference_data]),
            "actor/same_response_ratio": np.mean([p.same for p in preference_data]),
        }

        metric.update(preference_data.info)

        return preference_data, metric
