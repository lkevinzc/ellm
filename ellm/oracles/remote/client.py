import concurrent.futures
import logging
import threading
import time
from http import HTTPStatus
from typing import Any, List
from warnings import warn

import httpx
import msgspec
import numpy as np

from ellm.oracles.base import OracleBase

logging.getLogger("httpx").setLevel(logging.WARNING)


class RemoteRMOracle(OracleBase):
    def __init__(
        self,
        remote_rm_url: str,
        max_workers: int = 4,
        max_retry: int = 10,
        **_,
    ) -> None:
        super().__init__()
        self.client = httpx.Client(
            base_url=remote_rm_url,
        )
        self.invalid_count = 0
        self.max_workers = max_workers
        self.max_retry = max_retry
        self.mutex = threading.Lock()

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> List[Any]:
        del batch_size, disable_tqdm

        completions = list(zip(candidates_A, candidates_B))

        # Define a function to get the rank for a single prompt, will be called concurrently
        def get_rank(prompt, candidates):

            wait_time = 1
            for _ in range(self.max_retry):
                try:
                    resp = self.client.post(
                        "/compare",
                        content=msgspec.msgpack.encode(
                            {"prompt": prompt, "candidates": candidates}
                        ),
                    )
                    if resp.status_code == HTTPStatus.OK:
                        result = msgspec.msgpack.decode(resp.content)
                        first_win_prob = result["first_win_prob"]
                        break
                    else:
                        raise RuntimeError(f"{resp.status_code}, {resp.content}")
                except Exception as e:
                    warn(f"Remote RM server failure: {e}")
                    time.sleep(wait_time)
                    wait_time *= 1.3
            else:
                raise RuntimeError("Remote RM server error!")

            return first_win_prob

        # Call the remote server concurrently
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            first_win_probs = list(executor.map(get_rank, inputs, completions))

        first_win_probs = np.array(first_win_probs)
        if return_probs:
            return first_win_probs
        else:
            return first_win_probs > 0.5
