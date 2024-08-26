from typing import Any, Dict, List

import torch
import torch.distributed as dist

from ellm.learners.dap import DAPLearner
from ellm.rm import model
from ellm.types import PreferenceData
from ellm.utils.buffer import UniformBuffer
from ellm.utils.distributed import torch_type_codec


class DAPwRMLearner(DAPLearner):
    """Collocated DAP and reward model learning."""

    def _init(self, args, actors) -> None:
        super()._init(args, actors)
        self.rm = None
        self.learn_rm_only = args.learn_rm_only

        assert args.exp_method != "no" and args.exp_pretrain == ""
        rm_cls = getattr(model, args.exp_method)
        if self.strategy.is_rank_0():
            self.rm: model.RewardModel = rm_cls(args).to(torch.cuda.current_device())
            self.r_buffer = UniformBuffer(
                args.r_buffer_maxlen, device=torch.cuda.current_device()
            )
        self.train_rm_info = rm_cls.get_metrics()

    def process_preference_data(self, data_list: List[PreferenceData], raw_prompts):
        super().process_preference_data(data_list, raw_prompts)
        c_feats = torch.stack([data.chosen_feature for data in data_list]).unsqueeze(
            dim=1
        )
        r_feats = torch.stack([data.rejected_feature for data in data_list]).unsqueeze(
            dim=1
        )
        pair_feats = torch.cat([c_feats, r_feats], dim=1).to(
            torch.cuda.current_device()
        )  # (micro_b, 2, d)
        all_pair_feats = self.strategy.gather(pair_feats)
        if self.rm:
            self.r_buffer.extend(all_pair_feats)

    def preference_learning(self, learning_round):
        train_info = {}
        # NOTE Put reward learning after policy learning otherwise program gets stuck.
        if not self.learn_rm_only:
            train_info.update(super().preference_learning(learning_round))
        train_info.update(self._reward_learning())
        return train_info

    def get_misc_info(self) -> Dict[str, Any]:
        info = super().get_misc_info()
        r_buffer_len = 0
        if self.rm:
            r_buffer_len = self.r_buffer.size
        info.update({"r_buffer_len": self.strategy.all_reduce(r_buffer_len, "max")})
        return info

    def sync_params_to_actors(self):
        """Additionally sync reward model params."""
        # Sync RM.
        if self.rm:
            for name, param in self.rm.named_parameters():
                shape = param.shape
                futs = [
                    actor.futures.update_rm(
                        name,
                        dtype=torch_type_codec(param.dtype),
                        shape=shape,
                    )
                    for actor in self.actors
                ]
                dist.broadcast(param.data, 0, group=self._model_update_group)
                _ = [fut.result() for fut in futs]

        dist.barrier()

        if not self.learn_rm_only:
            # Sync policy.
            super().sync_params_to_actors()

    def _reward_learning(self):
        # Aggregate data from workers.
        total_num_queries = self.strategy.all_reduce(self.query_step, "sum")
        if self.rm:
            self.r_buffer.total_num_queries = total_num_queries
            self.train_rm_info = self.rm.learn(self.r_buffer)
        dist.barrier()
        self.strategy.broadcast(self.train_rm_info)
        return self.train_rm_info
