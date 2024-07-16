import pdb
from typing import List
from warnings import warn

import llm_blender
import ray
import vllm
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote
class Actor:
    def __init__(self, vllm_args, sampling_params, exploration=None) -> None:
        # ###################################
        # ####      vLLM Generation      ####
        # ###################################
        self.__vllm_version__ = vllm.__version__

        assert self.__vllm_version__ >= "0.4.1", "Upgrade to vLLM >= 0.4.1"
        assert sampling_params.n >= 2, "need to sample at least 2 responses per prompt"
        if sampling_params.n > 2 and exploration is None:
            warn(
                f"trying to sample {sampling_params.n} responses but no selection mechanism is provided"
            )

        self.vllm_use_gpu_executor = vllm_args["tensor_parallel_size"] == 1
        # See https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py
        if self.vllm_use_gpu_executor:
            from openrlhf.trainer.ray.vllm_worker_wrap import WorkerWrap

            vllm.worker.worker.Worker = WorkerWrap
        else:
            # RayGPUExecutor
            # See the patch https://github.com/vllm-project/vllm/commit/479d69fad0538f04cb22bf13e76ff91cfeb8a4e5
            vllm_args["worker_use_ray"] = True

            RayWorkerWrapperPath = vllm.executor.ray_utils

            class RayWorkerWrapper(RayWorkerWrapperPath.RayWorkerWrapper):
                def __init__(self, *args, **kwargs) -> None:
                    kwargs["worker_module_name"] = (
                        "openrlhf.trainer.ray.vllm_worker_wrap"
                    )
                    kwargs["worker_class_name"] = "WorkerWrap"
                    super().__init__(*args, **kwargs)

            RayWorkerWrapperPath.RayWorkerWrapper = RayWorkerWrapper
        self.llm = vllm.LLM(**vllm_args)
        self.sampling_params: vllm.SamplingParams = sampling_params

        # ###################################
        # ####    Oracle Reward Model    ####
        # ###################################
        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM")

        self.exploration = exploration

    def step(self, prompts: List[str]):
        """Step the actor.

        Given a prompt x, K responses {y_1, ..., y_K} are sample from the behavior LLM pi_beta,
        from which 2 responses are selected to query the oracle for preference signal.
        The final constructed pair (x, y_w, y_l) is inserted into the replay buffer for learners.

        Args:
            prompt: A list of prompt texts.
        """
        # step 1. generate
        outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)
        candidates = []
        for i in range(len(outputs)):
            # for each prompt
            candidates.append([])
            for k in range(self.sampling_params.n):
                # for each response
                candidates[i].append(outputs[i].outputs[k].text)

        # step 2. optional selection
        if self.sampling_params.n > 2:
            pass
            print("do response selection here (efficient exploration)")

        # step 3. query for oracle preference
        feedback = self.blender.compare(
            prompts, [c[0] for c in candidates], [c[1] for c in candidates]
        )
        return candidates, feedback

    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend
    ):
        if self.vllm_use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                "init_process_group",
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
            )

    def update_llm_weights(self, name, dtype, shape, empty_cache=False):
        def stop_remote_worker_execution_loop():
            # Fix error for using 2 communication group
            # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
            if self.__vllm_version__ > "0.4.2":
                self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()

        stop_remote_worker_execution_loop()
        if self.vllm_use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.update_weight(
                name, dtype, shape, empty_cache
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                "update_weight", name, dtype, shape, empty_cache
            )


def create_actors(
    num_nodes: int,
    num_gpus_per_node: int,
    llm_model_name: str,
    llm_sampling_params: vllm.SamplingParams,
    llm_gpu_memory_utilization: float = 0.95,
    seed: int = 0,
    enable_prefix_caching: bool = False,
) -> List[Actor]:
    actors = []
    for i in range(num_nodes):
        # When num_gpus_per_node=1, vLLM init model in LLMEngine directly and assign 1 GPU for it.
        num_gpus = int(num_gpus_per_node == 1)
        scheduling_strategy = None

        if num_gpus_per_node > 1:
            bundles = [{"GPU": 1, "CPU": 1}] * num_gpus_per_node
            pg = placement_group(bundles)
            ray.get(pg.ready())

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0,
            )

        vllm_args = {
            "model": llm_model_name,
            "trust_remote_code": True,
            "tensor_parallel_size": num_gpus_per_node,
            "gpu_memory_utilization": llm_gpu_memory_utilization,
            "dtype": "bfloat16",  # TODO(liuzc) check whether to use bfloat
            "seed": seed + i,
            "enable_prefix_caching": enable_prefix_caching,
        }
        actors.append(
            Actor.options(
                num_cpus=1,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                vllm_args=vllm_args,
                sampling_params=llm_sampling_params,
            )
        )

    return actors


if __name__ == "__main__":
    actors = create_actors(
        num_nodes=1,
        num_gpus_per_node=1,
        llm_model_name="google/gemma-2b",
        llm_sampling_params=vllm.SamplingParams(
            temperature=0.7, top_p=0.9, max_tokens=512, seed=0, n=2
        ),
        llm_gpu_memory_utilization=0.5,
        seed=0,
        enable_prefix_caching=True,
    )
    output = ray.get(
        [
            actor.step.remote(prompts=["San Franciso is a", "OpenAI is"])
            for actor in actors
        ]
    )
    pdb.set_trace()
    print(f"output: {output}")
