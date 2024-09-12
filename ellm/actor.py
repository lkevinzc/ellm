import time
from typing import List
from warnings import warn

import numpy as np
import torch
import vllm

from ellm import oracles
from ellm.exploration import ExplorationResults, Explorer
from ellm.rm import backbone, model
from ellm.types import PreferenceData
from ellm.utils.distributed import WorkerWrap, torch_type_codec
from ellm.utils.ipc import PlasmaShmClient


class Actor:
    """Actor handles the interaction between the LLM policy and the environment."""

    def __init__(self, ipc_server, vllm_args, sampling_params, args) -> None:
        self.args = args
        self.eval_mode = False
        self.pi_beta_weights = None
        # Measuring the **online** performance
        self.enable_online_evaluation = args.online_evaluation

        self.ipc_client = PlasmaShmClient(ipc_server)

        # ###################################
        # ####      vLLM Generation      ####
        # ###################################
        self.__vllm_version__ = vllm.__version__

        assert self.__vllm_version__ >= "0.4.1", "Upgrade to vLLM >= 0.4.1"
        assert sampling_params.n >= 2, "need to sample at least 2 responses per prompt"

        vllm.worker.worker.Worker = WorkerWrap
        vllm_args.update({"seed": time.time_ns() % 2**32})
        self.llm = vllm.LLM(**vllm_args)
        self.sampling_params: vllm.SamplingParams = sampling_params
        self.model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model

        # ###################################
        # ####    Oracle Reward Model    ####
        # ###################################
        oracle_cls = oracles.get_cls(args.reward_oracle)
        print("Using reward oracle", args.reward_oracle, oracle_cls)
        self.oracle = oracle_cls(
            reward_model_path=args.reward_oracle,
            tokenizer_path=args.pretrain,
        )

        # ###################################
        # ####        Exploration        ####
        # ###################################
        self.learning_rm = False
        if args.exp_method == "no":
            if sampling_params.n == 2:
                warn(
                    f"trying to sample {sampling_params.n} responses but no selection mechanism is provided"
                )
        else:
            assert sampling_params.n > 2
            # We assume reward model-based explorer.
            rm_backbone_cls = backbone.get_cls(args.rm_backbone)
            print("Using RM backbone", args.rm_backbone, rm_backbone_cls)
            self.explorer = Explorer(
                reward_model=getattr(model, args.exp_method)(args).cuda(),
                rm_backbone=rm_backbone_cls.from_pretrained(
                    args.rm_backbone, device_map="cuda:0"
                ).eval(),
                args=args,
            )
            if args.exp_pretrain:
                print(f"Loading pretrained ENN from {args.exp_pretrain}")
                self.explorer.reward_model.load_state_dict(
                    torch.load(args.exp_pretrain)
                )
            else:
                self.learning_rm = True  # Learn RM online.
        self.model_rollout = args.model_rollout

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

    def generate(self, prompts: List[str], sampling_params: vllm.SamplingParams):
        outputs = self.llm.generate(
            prompts, sampling_params=sampling_params, use_tqdm=False
        )
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
        candidates = self.generate(prompts, self.eval_sampling_params)

        if self.num_eval_gen > 1:
            # best of n sampling
            responses = self.explorer.best_of_n(prompts, candidates)
        else:
            responses = [candidates[i][0] for i in range(len(prompts))]

        if references:
            print("Evaluating using oracle", self.oracle)
            win_probs = self.oracle.compare(
                prompts, responses, references, return_probs=True, disable_tqdm=True
            )
            return responses, win_probs
        return responses, None

    def online_eval(self, prompts, references, candidates):
        win_probs_1 = self.oracle.compare(
            prompts,
            [candidates[i][0] for i in range(len(prompts))],
            references,
            return_probs=True,
            disable_tqdm=True,
        )
        win_probs_2 = self.oracle.compare(
            prompts,
            [candidates[i][1] for i in range(len(prompts))],
            references,
            return_probs=True,
            disable_tqdm=True,
        )
        return (win_probs_1 + win_probs_2) / 2

    def step(
        self, prompts: List[str], references: List[str] = None
    ) -> List[PreferenceData]:
        """Step the actor.

        Given a prompt x, K responses {y_1, ..., y_K} are sample from the behavior LLM pi_beta,
        from which 2 responses are selected to query the oracle for preference signal.
        The final constructed pair (x, y_w, y_l) is inserted into the replay buffer for learners.

        Args:
            prompts: A list of prompt texts.
            references: A list of reference texts.
        """
        assert not self.eval_mode
        info = dict()
        is_model_data = [False] * len(prompts)

        # step 1. generate
        candidates = self.generate(prompts, self.sampling_params)

        # step 2a. optional selection
        results = None
        if self.sampling_params.n > 2:
            print("Selecting dueling responses from candidates...")
            # TODO: we need raw prompts here, but currently they are processed from learner side (issue #10).
            results: ExplorationResults = self.explorer.select(prompts, candidates)
            candidates = results.dueling_candidates

            if self.model_rollout:
                # Use random sampling from explorer first; refactor later
                maybe_model_data = results.init_clash

        # step 2b. optional online eval
        if self.enable_online_evaluation:
            assert references is not None
            win_probs = self.online_eval(prompts, references, candidates)
            info["eval/online_win_probs"] = win_probs.mean()

        # step 3. query for oracle preference
        bt_probs = self.oracle.compare(
            prompts,
            [candidates[i][0] for i in range(len(prompts))],
            [candidates[i][1] for i in range(len(prompts))],
            return_probs=True,
            disable_tqdm=True,
        )
        bt_probs = torch.from_numpy(bt_probs)
        info["actor/first_action_win_prob"] = bt_probs.mean().item()

        if self.args.bt_sample:
            binary_feedback = torch.bernoulli(bt_probs).bool().numpy()
        else:
            binary_feedback = bt_probs > 0.5

        chosen = 1 - binary_feedback
        if self.model_rollout:
            # Record metric and overwrite label.
            model_rollout_correct = chosen[maybe_model_data] == 0
            model_rollout_acc = np.sum(model_rollout_correct) / (
                np.sum(maybe_model_data) + 1e-8
            )
            info["eval/model_rollout_acc"] = model_rollout_acc

            if model_rollout_acc > 0.9:
                # privileged information
                is_model_data = maybe_model_data
                chosen[is_model_data] = 0

        rejected = 1 - chosen

        same_response = [
            candidates[i][chosen[i]] == candidates[i][rejected[i]]
            for i in range(len(prompts))
        ]

        if self.learning_rm:
            # Measure the internal RM accuracy
            support = [
                results.init_clash[i] and not same_response[i]
                for i in range(len(prompts))
            ]
            rm_acc = np.sum(
                [binary_feedback[i] and support[i] for i in range(len(prompts))]
            ) / (np.sum(support) + 1e-8)
            info["eval/rm_acc"] = rm_acc

        if results is not None:
            info.update(results.info)

        preference_data = [
            PreferenceData(
                prompt=prompts[i],
                chosen_id=chosen[i],
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
                init_clash=results.init_clash[i] if self.learning_rm else False,
                same=same_response[i],
                is_model_data=is_model_data[i],
                info=info,
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
