from typing import List
from warnings import warn

import llm_blender
import vllm

from ellm.types import PreferenceData


class Actor:
    """Actor handles the interaction between the exploration policy and the environment."""

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

        self.llm = vllm.LLM(**vllm_args)
        self.sampling_params: vllm.SamplingParams = sampling_params

        # ###################################
        # ####    Oracle Reward Model    ####
        # ###################################
        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM")

        self.exploration = exploration

    def step(self, prompts: List[str]) -> List[PreferenceData]:
        """Step the actor.

        Given a prompt x, K responses {y_1, ..., y_K} are sample from the behavior LLM pi_beta,
        from which 2 responses are selected to query the oracle for preference signal.
        The final constructed pair (x, y_w, y_l) is inserted into the replay buffer for learners.

        Args:
            prompt: A list of prompt texts.
        """
        # step 1. generate
        outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)
        candidates = {}
        for i in range(len(outputs)):
            # for each prompt
            candidates[i] = []
            for k in range(self.sampling_params.n):
                # for each response
                candidates[i].append(outputs[i].outputs[k].text)

        # step 2. optional selection
        if self.sampling_params.n > 2:
            pass
            print("do response selection here (efficient exploration)")

        # step 3. query for oracle preference
        feedback = self.blender.compare(
            prompts,
            [candidates[i][0] for i in range(len(prompts))],
            [candidates[i][1] for i in range(len(prompts))],
        )
        chosen = 1 - feedback
        rejected = 1 - chosen
        preference_data = [
            PreferenceData(
                prompt=prompts[i],
                chosen_response=candidates[i][chosen[i]],
                rejected_response=candidates[i][rejected[i]],
            )
            for i in range(len(prompts))
        ]

        return preference_data
