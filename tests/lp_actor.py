import launchpad as lp
import vllm
from absl import app, flags, logging
from launchpad.nodes.python import local_multi_processing
from ml_collections import ConfigDict

from ellm.actor import Actor
from ellm.utils.ipc import PlasmaShmClient, PlasmaShmServer

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_actors", 2, "The number of concurrent actors.")
flags.DEFINE_enum("exp_method", "no", ["no", "enn_dts"], "exploration method")
flags.DEFINE_string("exp_pretrain", "", "pretrained exploration model")


class Controller:
    def __init__(self, actors):
        self._actors = actors
        self._dataloader = ["San Franciso is a", "OpenAI is"]
        self._ipc_client = PlasmaShmClient()

    def run(self):

        futures = [
            actor.futures.step([self._dataloader[i % 2]])
            for i, actor in enumerate(self._actors)
        ]

        results = [
            self._ipc_client.deserialize_ipc(future.result()) for future in futures
        ]
        logging.info("Results: %s", results)
        lp.stop()


def make_program(num_actors, vllm_args, sampling_params, args):
    program = lp.Program("query_actors")
    actors = []
    local_resources = {}
    for i in range(num_actors):
        label = f"actor_{i}"
        actors.append(
            program.add_node(
                lp.CourierNode(Actor, vllm_args, sampling_params, args), label=label
            )
        )
        local_resources[label] = local_multi_processing.PythonProcess(
            env=dict(CUDA_VISIBLE_DEVICES=str(i))
        )
    program.add_node(lp.CourierNode(PlasmaShmServer), label="ipc")
    node = lp.CourierNode(Controller, actors=actors)
    program.add_node(node, label="controller")
    return program, local_resources


def main(_):
    vllm_args = {
        "model": "google/gemma-2b",
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.5,
        "dtype": "bfloat16",
        "enable_prefix_caching": True,
    }
    sampling_params = vllm.SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        seed=0,
        n=2 if FLAGS.exp_method == "no" else 5,
    )
    program, local_resources = make_program(
        FLAGS.num_actors,
        vllm_args,
        sampling_params,
        ConfigDict(
            {
                **FLAGS.flag_values_dict(),
                "num_ensemble": 10,
                "enn_lr": 1e-3,
                "enn_lambda": 0.1,
                "enn_hidden_dim": 128,
                "enn_sgd_steps": 1,
            }
        ),
    )

    lp.launch(
        program,
        launch_type="local_mp",
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    app.run(main)
