"""Rejection sampling."""

import torch

from ellm import actor
from ellm.exploration import ExplorationResults, Explorer
from ellm.rm import model


class RSExplorer(Explorer):
    def select(
        self, prompts: torch.List[str], candidates: torch.Dict[int, torch.List[str]]
    ) -> actor.ExplorationResults:
        features = self._get_features(prompts, candidates)  # (M, N, d)
        rewards = self.reward_model.get_rewards(features)  # (E, M, N, 1)
        avg_rewards = rewards.mean(0)  # (M, N, 1)
        best_response_indices = avg_rewards.argmax(dim=1).cpu().squeeze()  # (M,)
        worst_response_indices = avg_rewards.argmin(dim=1).cpu().squeeze()  # (M,)

        dueling_candidates = {}
        for i in range(len(prompts)):
            dueling_candidates[i] = [
                candidates[i][best_response_indices[i]],
                candidates[i][worst_response_indices[i]],
            ]

        selected_candidate_indices = torch.stack(
            [best_response_indices, worst_response_indices], dim=-1
        )

        init_clash = (best_response_indices == worst_response_indices).cpu().squeeze()

        return ExplorationResults(
            dueling_candidates=dueling_candidates,
            candidate_features=(
                torch.stack(
                    [
                        features[i][selected_candidate_indices[i]]
                        for i in range(len(prompts))
                    ]
                ).cpu()
            ),
            init_clash=init_clash.tolist(),
            is_model_data=[False] * len(prompts),
            info={},
        )


class RSActor(actor.Actor):
    """Sample N responses and choose the best/worse-of-N."""

    def __init__(self, ipc_server, vllm_args, sampling_params, args) -> None:
        super().__init__(ipc_server, vllm_args, sampling_params, args)
        # Replace the explorer.
        self.explorer = RSExplorer(
            reward_model=getattr(model, args.exp_method)(args).cuda(),
            rm_backbone=self.rm_backbone,
            args=args,
        )
