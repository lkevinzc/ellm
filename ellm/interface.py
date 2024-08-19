from argparse import Namespace
from typing import Type

import launchpad as lp
import vllm
from launchpad.nodes.python import local_multi_processing

from ellm.actor import Actor
from ellm.learners.base import LearnerBase
from ellm.utils.ipc import PlasmaShmServer
from ellm.utils.launcher import get_free_port


def get_program(args: Namespace, learner_cls: Type[LearnerBase]):
    """Define the default distributed program topology with configs."""

    # Actor.
    vllm_args = {
        "model": args.pretrain,
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.5,
        "dtype": "bfloat16",
    }
    sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.generate_max_length,
        n=args.num_samples,
    )
    program = lp.Program("online_dap")
    actors = []
    local_resources = {}
    for i in range(4):
        label = f"actor_{i}"
        actors.append(
            program.add_node(
                lp.CourierNode(Actor, vllm_args, sampling_params, args),
                label=label,
            )
        )
        local_resources[label] = local_multi_processing.PythonProcess(
            env=dict(CUDA_VISIBLE_DEVICES=str(i))
        )
    _gpu_offset = 4

    # Learner.
    master_addr = "0.0.0.0"
    master_port = get_free_port()
    args.local_rank = 0
    label = "learner_0"
    master_learner = lp.PyClassNode(
        learner_cls,
        4,
        0,
        0,
        master_addr,
        master_port,
        True,
        args,
        actors,
    )
    program.add_node(master_learner, label=label)
    local_resources[label] = local_multi_processing.PythonProcess(
        env=dict(CUDA_VISIBLE_DEVICES=str(_gpu_offset))
    )
    for i in range(1, 4):
        label = f"learner_{i}"
        worker_learner = lp.PyClassNode(
            learner_cls,
            4,
            i,
            i,
            master_addr,
            master_port,
            False,
            args,
            actors,
        )
        program.add_node(worker_learner, label=label)
        local_resources[label] = local_multi_processing.PythonProcess(
            env=dict(CUDA_VISIBLE_DEVICES=str(i + _gpu_offset))
        )
    program.add_node(lp.CourierNode(PlasmaShmServer, size_mb=1000), label="ipc")
    return program, local_resources
