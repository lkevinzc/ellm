"""APL: https://arxiv.org/pdf/2402.08114.

Due to its design of using LLM as the reward model, we have to make the actor-
learner interface more complicated. We first generate responses and estimate the entropy in actor, then compute the implicit reward margin in learner, and finally
get oracle feedback in actor. 
"""

import time

import launchpad as lp
import numpy as np
import torch
import vllm
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from ellm import actor
from ellm.learners.dap import DAPLearner
from ellm.types import PreferenceData


class APLActor(actor.Actor):
    """Sample a large batch and filter with entropy and reward margin."""

    def __init__(self, ipc_server, vllm_args, sampling_params, args) -> None:
        super().__init__(ipc_server, vllm_args, sampling_params, args)
        self.ref_llm = vllm.LLM(**vllm_args)

    def step(
        self, prompts: torch.List[str], references: torch.List[str] = None
    ) -> torch.List[actor.PreferenceData]:
        assert not self.eval_mode
        info = dict()

        outputs = self.llm.generate(
            prompts, sampling_params=self.sampling_params, use_tqdm=False
        )
        # 1) Predictive entropy estimation
        entropy_estimations = []
        for output in outputs:
            entropy = 0
            for logps in output.prompt_logprobs:
                entropy -= np.sum([v.logprob for v in logps.values()])
            entropy = entropy / len(output.prompt_logprobs)
            entropy_estimations.append(entropy)
        ent_filtered_indices = np.argsort(entropy_estimations)[
            -self.args.micro_pi_buffer_maxlen :
        ]  # online, on-policy
        prompts = prompts[ent_filtered_indices]
        outputs = outputs[ent_filtered_indices]

        # 2) Implicit reward margin

        # Query reward oracle.
        logits = self.blender.compare(
            prompts,
            [candidates[i][0] for i in range(len(prompts))],
            [candidates[i][1] for i in range(len(prompts))],
            disable_tqdm=True,
        )
        bt_probs = torch.from_numpy(logits).sigmoid()
        info["actor/first_action_win_prob"] = bt_probs.mean().item()
        binary_feedback = logits > 0
        chosen = 1 - binary_feedback
        rejected = 1 - chosen

        same_response = [
            candidates[i][chosen[i]] == candidates[i][rejected[i]]
            for i in range(len(prompts))
        ]

        preference_data = [
            PreferenceData(
                prompt=prompts[i],
                chosen_id=chosen[i],
                chosen_response=candidates[i][chosen[i]],
                rejected_response=candidates[i][rejected[i]],
                chosen_feature=None,
                rejected_feature=None,
                init_clash=False,
                same=same_response[i],
                is_model_data=False,
                info=info,
            )
            for i in range(len(prompts))
        ]

        handle = self.ipc_client.serialize_ipc(preference_data)
        return handle


class APLLearner(DAPLearner):
    def run(self):
        self._init(self.args, self.actors)
        update_interval = self.args.rollout_batch_size // (
            self.strategy.world_size * self.args.micro_rollout_batch_size
        )
        self.strategy.print(f"Update interval = {update_interval}")
        steps = 1
        self.start_time = time.time()

        self.actor_info = {}

        if not self.strategy.args.debug:
            self.save_logs_and_checkpoints(
                self.args,
                self.policy_sgd_step,
                {},
            )

        for episode in range(self.args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(episode)
                self.strategy.print(f"Set DistributedSampler at epoch {episode}")
            progress_bar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for processed_prompts, raw_prompts, refs in self.prompts_dataloader:
                preference_data, self.actor_info = self.preference_collector(
                    processed_prompts, refs
                )
                self.prompt_consumed += len(preference_data)
                self.query_step += np.sum(
                    [not p.is_model_data for p in preference_data]
                )
                self.process_preference_data(preference_data, raw_prompts)

                if steps % update_interval == 0:
                    train_info = self.preference_learning(steps // update_interval)

                    self.save_logs_and_checkpoints(
                        self.args,
                        steps,
                        train_info,
                    )

                    if (steps // update_interval) % self.args.sync_params_every == 0:
                        self.sync_params_to_actors()

                    if (steps // update_interval) % self.args.buffer_clear_every == 0:
                        self.pi_buffer.clear()

                progress_bar.update()
                steps += 1

        if self.strategy.is_rank_0():
            self._wandb.finish()
            lp.stop()
