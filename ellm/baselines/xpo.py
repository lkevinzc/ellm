"""XPO: https://arxiv.org/pdf/2405.21046"""

import torch
import vllm

from ellm import actor
from ellm.learners.dap import DAPLearner
from ellm.types import DAPAlgo


class XPOActor(actor.Actor):
    """Sample one response from llm and another from ref_llm."""

    def __init__(self, ipc_server, vllm_args, sampling_params, args) -> None:
        super().__init__(ipc_server, vllm_args, sampling_params, args)
        self.ref_llm = vllm.LLM(**vllm_args)

    def generate(self, prompts: actor.List[str], sampling_params: vllm.SamplingParams):
        if self.eval_mode:
            return super().generate(prompts, sampling_params)

        assert sampling_params.n == 2
        sampling_params.n = 1
        candidates = {}

        for llm in [self.llm, self.ref_llm]:
            outputs = llm.generate(
                prompts, sampling_params=sampling_params, use_tqdm=False
            )
            for i in range(len(outputs)):
                # for each prompt
                if i not in candidates:
                    candidates[i] = []
                candidates[i].append(outputs[i].outputs[0].text.strip())

        return candidates


class XPOLearner(DAPLearner):
    """Additional optimism loss term: log(\pi(y_ref|x))."""

    def _init(self, args, actors) -> None:
        super()._init(args, actors)
        assert self.algo == DAPAlgo.DPO and self.ref_model is not None
        self.xpo_alpha = args.xpo_alpha

    def learning_step(self, data):
        device = torch.cuda.current_device()
        chosen_ids, c_mask, rejected_ids, r_mask, extra = data
        chosen_ids = chosen_ids.squeeze(1).to(device)
        c_mask = c_mask.squeeze(1).to(device)
        rejected_ids = rejected_ids.squeeze(1).to(device)
        r_mask = r_mask.squeeze(1).to(device)

        prompt_id_lens = extra["prompt_ids_lens"]
        loss_masks = 1 - torch.tensor(extra["same_masks"]).float().to(device)

        chosen_logps, rejected_logps, _ = self.concatenated_forward(
            self.model, chosen_ids, c_mask, rejected_ids, r_mask, prompt_id_lens
        )
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps, _ = (
                self.concatenated_forward(
                    self.ref_model,
                    chosen_ids,
                    c_mask,
                    rejected_ids,
                    r_mask,
                    prompt_id_lens,
                )
            )
        preference_loss, chosen_reward, rejected_reward = self.loss(
            chosen_logps,
            rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            loss_masks,
        )

        # `chosen` indicates the original sampling source:
        # 0 - rejected_ids are from the ref policy
        # 1 - chosen_ids are from the ref policy
        chosen = torch.tensor(extra["chosen_ids"]).to(device)
        ref_logps = torch.where(chosen == 0, rejected_logps, chosen_logps)
        optimism_loss = (ref_logps * loss_masks).mean()

        loss = preference_loss + self.xpo_alpha * optimism_loss
        self.strategy.backward(loss, self.model, self.optimizer)
        self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

        infos = {
            "loss": loss.detach(),
            "optimism_loss": optimism_loss.detach(),
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
        }
        return infos
