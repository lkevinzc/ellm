import time
from typing import List

import llm_blender
import torch
import vllm

from ellm.exploration import ExplorationResults, Explorer
from ellm.rm import model
from ellm.types import PreferenceData
from ellm.utils.distributed import WorkerWrap, torch_type_codec
from ellm.utils.ipc import PlasmaShmClient


class Actor:
    """Actor handles the interaction between the exploration policy and the environment."""

    def __init__(self, ipc_server, vllm_args, sampling_params, args) -> None:
        self.args = args
        self.eval_mode = False
        self.pi_beta_weights = None

        self.ipc_client = PlasmaShmClient(ipc_server)

        # ###################################
        # ####      vLLM Generation      ####
        # ###################################
        self.__vllm_version__ = vllm.__version__

        assert self.__vllm_version__ >= "0.4.1", "Upgrade to vLLM >= 0.4.1"
        assert sampling_params.n >= 2, "need to sample at least 2 responses per prompt"

        vllm.worker.worker.Worker = WorkerWrap
        vllm_args.update({"seed": int(time.time() * 1000) % 2**32})
        self.llm = vllm.LLM(**vllm_args)
        self.sampling_params: vllm.SamplingParams = sampling_params
        self.model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model

        # ###################################
        # ####    Oracle Reward Model    ####
        # ###################################
        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM")

        # ###################################
        # ####        Exploration        ####
        # ###################################
        self.learning_rm = False
        if args.exp_method == "no":
            assert (
                sampling_params.n == 2
            ), f"trying to sample {sampling_params.n} responses but no selection mechanism is provided"
        else:
            assert sampling_params.n > 2
            self.explorer = Explorer(getattr(model, args.exp_method)(args))
            if args.exp_pretrain:
                print(f"Loading pretrained ENN from {args.exp_pretrain}")
                self.explorer.reward_model.load_state_dict(
                    torch.load(args.exp_pretrain)
                )
            else:
                self.learning_rm = True  # Learn RM online.

        # ###################################
        # ####  Best-of-N for Evaluation ####
        # ###################################
        if args.best_of_n_eval:
            self.num_eval_gen = args.num_bon
        else:
            self.num_eval_gen = 1
        self.eval_sampling_params = vllm.SamplingParams(
            n=self.num_eval_gen,
            temperature=0.0 if self.num_eval_gen == 1 else args.bon_temperature,
            top_p=0.95,
            max_tokens=200,
        )  # TODO hard-code first for tl;dr

    def _generate(self, prompts: List[str], sampling_params: vllm.SamplingParams):
        outputs = self.llm.generate(prompts, sampling_params=sampling_params)
        candidates = {}
        for i in range(len(outputs)):
            # for each prompt
            candidates[i] = []
            for k in range(sampling_params.n):
                # for each response
                candidates[i].append(outputs[i].outputs[k].text.strip())
        return candidates

    def generate_and_maybe_eval(
        self,
        prompts: List[str],
        references: List[str] = None,
    ):
        """
        1) Generate responses for given prompts;
        2) Optionally evaluate the win rate over references based on the oracle reward model.
        """
        assert self.eval_mode
        candidates = self._generate(prompts, self.eval_sampling_params)

        if self.num_eval_gen > 1:
            # best of n sampling
            responses = self.explorer.best_of_n(prompts, candidates)
        else:
            responses = [candidates[i][0] for i in range(len(prompts))]

        if references:
            win_logits = self.blender.compare(
                prompts, responses, references, return_logits=True
            )
            win_probs = torch.from_numpy(win_logits).sigmoid().numpy()
            return responses, win_probs
        return responses, None

    def step(self, prompts: List[str]) -> List[PreferenceData]:
        """Step the actor.

        Given a prompt x, K responses {y_1, ..., y_K} are sample from the behavior LLM pi_beta,
        from which 2 responses are selected to query the oracle for preference signal.
        The final constructed pair (x, y_w, y_l) is inserted into the replay buffer for learners.

        Args:
            prompt: A list of prompt texts.
        """
        assert not self.eval_mode
        # step 1. generate
        candidates = self._generate(prompts, self.sampling_params)

        # step 2. optional selection
        if self.sampling_params.n > 2:
            print("Selecting dueling responses from candidates...")
            # TODO: we need raw prompts here, but currently they are processed from learner side (issue #10).
            results: ExplorationResults = self.explorer.select(
                prompts, candidates, return_features=self.learning_rm
            )
            candidates = results.dueling_candidates

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
                chosen_feature=(
                    results.candidate_features[i][chosen[i]]
                    if self.learning_rm
                    else None
                ),
                rejected_feature=(
                    results.candidate_features[i][rejected[i]]
                    if self.learning_rm
                    else None
                ),
            )
            for i in range(len(prompts))
        ]

        handle = self.ipc_client.serialize_ipc(preference_data)
        return handle

    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend
    ):
        self._model_update_group = (
            self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
            )
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self._stop_remote_worker_execution_loop()
        return self.llm.llm_engine.model_executor.driver_worker.update_weight(
            name, dtype, shape, empty_cache
        )

    def update_rm(self, name, dtype, shape):
        assert self.learning_rm
        dtype = torch_type_codec(dtype)
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        torch.distributed.broadcast(weight, 0, group=self._model_update_group)
        params_dict = dict(self.explorer.reward_model.named_parameters())
        model.default_weight_loader(params_dict[name], weight)
        print(f"update reward model weight {name}")
        del weight

    def notify_eval_start(self):
        """Temporarily cache the current behavior policy weights to CPU."""
        self.eval_mode = True
        print("Start offloading...")
        st = time.time()
        self.cache_model_state = {k: v.cpu() for k, v in self.model.named_parameters()}
        print(f"Finished offloading in {time.time() - st} seconds")

    def notify_eval_done(self):
        assert self.eval_mode
        print("Start loading from cpu...")
        st = time.time()
        self.model.load_state_dict(self.cache_model_state)
        print(f"Finished loading in {time.time() - st} seconds")
        self.eval_mode = False

    def _stop_remote_worker_execution_loop(self):
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if self.__vllm_version__ > "0.4.2":
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()
