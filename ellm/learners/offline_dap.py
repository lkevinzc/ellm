import os
import random
import time
from argparse import Namespace
from typing import List

import pandas as pd
import torch
import tree
from tqdm import tqdm

from ellm import oracles
from ellm.learners import DAPLearner
from ellm.learners.dap import DAPLearner
from ellm.types import PreferenceData


class OfflineDAPLearner(DAPLearner):
    def __init__(
        self,
        args: Namespace,
    ) -> None:
        self.args = args
        assert os.path.exists(args.offline_buffer_path)
        self._init(args, actors=[])
        # TODO: 4 shards by default; remove this hard-coded parameter later.
        all_shards = pd.read_pickle(args.offline_buffer_path)
        self.all_buffer: List[PreferenceData] = list(
            all_shards[torch.distributed.get_rank()]
        )
        self.eval_generate_args = {
            "do_sample": False,
            "early_stopping": True,
            "max_new_tokens": 200,
            "top_p": 0.95,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Oracle reward model for evaluation.
        oracle_cls = oracles.get_cls(args.reward_oracle)
        print("Using reward oracle", args.reward_oracle, oracle_cls)
        self.oracle = oracle_cls(
            reward_model_path=args.reward_oracle,
            tokenizer_path=args.pretrain,
            max_workers=16,
        )

    def run(self):
        steps = 1
        self.start_time = time.time()
        self.actor_info = {}
        bs = self.args.micro_rollout_batch_size
        random.shuffle(self.all_buffer)

        if not self.strategy.args.debug:
            self.save_logs_and_checkpoints(
                self.args,
                self.policy_sgd_step,
                {},
            )

        for p_ep in range(self.args.num_prompt_epoch):
            progress_bar = tqdm(
                range(len(self.all_buffer)),
                desc=f"Prompt epoch [{p_ep + 1}/{self.args.num_prompt_epoch}]",
                disable=not self.strategy.is_rank_0(),
            )
            for ndx in range(0, len(self.all_buffer), bs):
                self.pi_buffer.extend(
                    self.all_buffer[ndx : min(ndx + bs, len(self.all_buffer))]
                )
                self.prompt_consumed += bs
                self.query_step += bs

                if steps % self.update_interval == 0:
                    train_info = self.preference_learning(steps // self.update_interval)

                    self.save_logs_and_checkpoints(
                        self.args,
                        steps,
                        train_info,
                    )

                progress_bar.update()
                steps += 1
            self.prompt_epoch = p_ep + 1

        if self.strategy.is_rank_0():
            self._wandb.finish()

    def evaluate(self, dataloader, steps):
        self.strategy.print(f"Start generating evaluation responses at step {steps}")
        st_time = time.time()
        self.model.eval()
        device = torch.cuda.current_device()

        win_rate = 0
        win_rate_prob = 0
        if self.strategy.is_rank_0():
            prompts = []
            responses = []
            references = []
            for processed_prompts, _, refs in tqdm(dataloader, desc="generating"):
                prompts.extend(processed_prompts)
                references.extend(refs)
                batch = self.tokenizer(
                    processed_prompts,
                    return_tensors="pt",
                    max_length=self.args.prompt_max_length,
                    padding=True,
                    truncation=True,
                )
                inputs = tree.map_structure(lambda x: x.to(device), batch)
                sequences = self.model.generate(
                    **inputs, **self.eval_generate_args
                ).cpu()
                completions = self.tokenizer.batch_decode(
                    sequences[:, batch["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                # print(processed_prompts[0], completions[0])
                responses.extend(completions)

            win_probs = self.oracle.compare(
                prompts, responses, references, return_probs=True, disable_tqdm=True
            )
            wins = win_probs > 0.5
            win_rate = wins.mean().item()
            win_rate_prob = win_probs.mean().item()

        win_rate = self.strategy.broadcast(win_rate)
        win_rate_prob = self.strategy.broadcast(win_rate_prob)

        self.model.train()
        return {
            "eval/rm_win_rate": win_rate,
            "eval/rm_win_rate_prob": win_rate_prob,
            "eval/elapse": time.time() - st_time,
        }
