import os
from socket import socket

import launchpad as lp
import torch
import torch.distributed as dist
from launchpad.nodes.python import local_multi_processing

from ellm.utils.distributed import (init_process_group,
                                    node_ip_address_from_perspective)

device = "cuda" if torch.cuda.is_available() else "cpu"

# !!! IMPORTANT NOTE !!!(liuzc)
# torch.dtype cannot be passed through lp's rpc due to segmentation fault; use string instead.
torch_type_decode = {
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}
torch_type_encode = {
    torch.bfloat16: "bf16",
    torch.float32: "f32",
}


def init_default(rank, size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        "nccl" if device == "cuda" else "gloo", rank=rank, world_size=size
    )


class Consumer:
    def __init__(self) -> None:
        init_default(1, 2)
        self.data = None
        self._group = None

    def init_process_group(
        self,
        master_address,
        master_port,
        rank,
        world_size,
        group_name,
        backend,
    ) -> None:
        assert torch.distributed.is_initialized()
        self._group = None
        self._group = init_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )

    def update_data(self, name, dtype, shape):
        print("receiving...")
        weight = torch.empty(shape, dtype=torch_type_decode[dtype], device=device)
        torch.distributed.broadcast(weight, 0, group=self._group)
        self.data = weight
        print(weight)


class Producer:
    def __init__(self, consumers) -> None:
        init_default(0, 2)

        self.data = torch.rand((7, 7), device=device)
        self.consumers = consumers

        master_addr = node_ip_address_from_perspective()
        with socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]
        world_size = len(consumers) + 1

        futs = [
            c.futures.init_process_group(
                master_addr,
                master_port,
                i + 1,
                world_size,
                "ellm",
                backend="nccl" if device == "cuda" else "gloo",
            )
            for i, c in enumerate(consumers)
        ]
        self._group = None
        self._group = init_process_group(
            backend="nccl" if device == "cuda" else "gloo",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=0,
            group_name="ellm",
        )
        _ = [fut.result() for fut in futs]

    def run(self):
        print("producer sending data to consumers...")
        print(self.data)

        futs = [
            c.futures.update_data(
                "data",
                dtype=torch_type_encode[self.data.dtype],
                shape=self.data.shape,
            )
            for c in self.consumers
        ]

        torch.distributed.broadcast(self.data, 0, group=self._group)
        _ = [fut.result() for fut in futs]

        lp.stop()


if __name__ == "__main__":
    program = lp.Program("text_nccl")
    consumers = []
    local_resources = {}

    consumers.append(program.add_node(lp.CourierNode(Consumer), label="c"))
    local_resources["c"] = local_multi_processing.PythonProcess(
        env=dict(CUDA_VISIBLE_DEVICES=str(0))
    )

    producer = lp.PyClassNode(Producer, consumers)
    program.add_node(producer, label="p")
    local_resources["c"] = local_multi_processing.PythonProcess(
        env=dict(CUDA_VISIBLE_DEVICES=str(1))
    )

    lp.launch(
        program,
        launch_type="local_mp",
        local_resources=local_resources,
        terminal="current_terminal",
    )
