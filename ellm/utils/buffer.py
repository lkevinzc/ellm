import numpy as np
import torch


class UniformBuffer(object):
    def __init__(self, max_len: int, device: torch.device):
        self._max_len = max_len
        self._storage = None
        self._device = device
        self._n = 0
        self._idx = 0

    def extend(self, batch: torch.Tensor):
        if self._storage is None:
            self._storage = torch.empty(
                (self._max_len, *batch.shape[1:]),
                device=self._device,
                dtype=batch.dtype,
            )

        num_steps = len(batch)
        indices = torch.arange(self._idx, self._idx + num_steps) % self._max_len
        self._storage[indices] = batch
        self._idx = (self._idx + num_steps) % self._max_len
        self._n = min(self._n + num_steps, self._max_len)

    def sample(self, batch_size: int):
        if batch_size > self._n:
            return None
        start_indices = np.random.choice(self._n, batch_size, replace=False)
        base_idx = 0 if self._n < self._max_len else self._idx
        all_indices = (start_indices + base_idx) % self._max_len
        return self._storage[all_indices]

    @property
    def size(self):
        return self._n
