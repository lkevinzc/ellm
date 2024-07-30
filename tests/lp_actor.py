import launchpad as lp
import vllm
from absl import app, flags, logging
from launchpad.nodes.python import local_multi_processing

from ellm.actor import Actor

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_actors", 2, "The number of concurrent actors.")


class Controller:
    def __init__(self, actors):
        self._actors = actors
        self._dataloader = ["San Franciso is a", "OpenAI is"]

    def run(self):

        futures = [
            actor.futures.step([self._dataloader[i % 2]])
            for i, actor in enumerate(self._actors)
        ]
        results = [future.result() for future in futures]
        logging.info("Results: %s", results)
        lp.stop()


def make_program(num_actors, vllm_args, sampling_params):
    program = lp.Program("query_actors")
    actors = []
    local_resources = {}
    for i in range(num_actors):
        label = f"actor_{i}"
        actors.append(
            program.add_node(
                lp.CourierNode(Actor, vllm_args, sampling_params), label=label
            )
        )
        local_resources[label] = local_multi_processing.PythonProcess(
            env=dict(CUDA_VISIBLE_DEVICES=str(i))
        )

    node = lp.CourierNode(Controller, actors=actors)
    program.add_node(node, label="controller")
    return program, local_resources


def main(_):
    vllm_args = {
        "model": "google/gemma-2b",
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.5,
        "dtype": "bfloat16",  # TODO(liuzc) check whether to use bfloat
        "seed": 0,
        "enable_prefix_caching": True,
    }
    sampling_params = vllm.SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=512, seed=0, n=2
    )
    program, local_resources = make_program(
        FLAGS.num_actors, vllm_args, sampling_params
    )

    lp.launch(
        program,
        launch_type="local_mp",
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    app.run(main)
