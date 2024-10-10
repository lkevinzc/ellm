import logging
import os
import random
import time
from argparse import Namespace
from typing import List

import datasets
import launchpad as lp
import pandas as pd
import torch
import torch.distributed as dist
import tree
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GenerationConfig

from ellm import oracles
from ellm.learners import DAPLearner
from ellm.learners.dap import DAPLearner
from ellm.types import PreferenceData
from ellm.utils.data import get_datasets, shard_buffer


class OfflineDAPLearner(DAPLearner):
    def __init__(
        self,
        args: Namespace,
    ) -> None:
        self.args = args
        self._init(args, actors=[])

        self.eval_generate_config = GenerationConfig(
            max_new_tokens=args.eval_generate_max_length,
            temperature=args.eval_temperature,
            top_p=args.eval_top_p,
            do_sample=args.eval_temperature > 0,
            use_cache=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Oracle reward model for evaluation.
        oracle_cls = oracles.get_cls(args.reward_oracle)
        logging.info(f"Using reward oracle {args.reward_oracle} {oracle_cls}")
        self.oracle = oracle_cls(
            reward_model_path=args.reward_oracle,
            tokenizer_path=args.pretrain,
            remote_rm_url=args.remote_rm_url,  # Only for remote RM.
        )

    def prepare_data(self, strategy, tokenizer):
        args = self.args
        if args.preference_data:
            if os.path.exists(args.preference_data):
                data = datasets.load_from_disk(args.preference_data)
            else:
                data = datasets.load_dataset(args.preference_data)
            all_shards = []
            for item in tqdm(data, desc="loading preference data"):
                all_shards.append(
                    PreferenceData(
                        prompt=item[args.prompt_key],
                        chosen_response=item[args.chosen_key],
                        rejected_response=item[args.rejected_key],
                        chosen_id=0,
                        chosen_feature=None,
                        rejected_feature=None,
                        init_clash=False,
                        same=item[args.chosen_key] == item[args.rejected_key],
                        is_model_data=False,
                        info={},
                    )
                )
            all_shards = all_shards[: args.max_train]
            self.all_buffer: List[PreferenceData] = shard_buffer(
                all_shards,
                dist.get_rank(),
                dist.get_world_size(),
                args.seed,
                shuffle=True,
                drop_last=True,
            )
        else:
            # Load pre-dumped data.
            assert os.path.exists(args.offline_buffer_path)
            all_shards = pd.read_pickle(args.offline_buffer_path)
            self.all_buffer: List[PreferenceData] = list(
                all_shards[torch.distributed.get_rank()]
            )
        self.prompts_dataset = tree.flatten(
            all_shards
        )  # needed to calculate lr scheduler
        self.prompts_dataloader = None
        _, self.eval_prompts_dataset = get_datasets(tokenizer, strategy, eval_only=True)
        self.eval_prompts_dataloader = DataLoader(
            self.eval_prompts_dataset,
            batch_size=strategy.args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def run(self):
        self.steps = 0
        self.start_time = time.time()

        self.actor_info = {}
        bs = self.args.micro_rollout_batch_size

        if not self.strategy.args.debug:
            self.save_logs_and_checkpoints({}, eval=True)

        self.steps = 1
        print("sleeping")
        time.sleep(10)
        for p_ep in range(self.args.num_prompt_epoch):
            progress_bar = tqdm(
                range(len(self.all_buffer) // bs),
                desc=f"Prompt epoch [{p_ep + 1}/{self.args.num_prompt_epoch}]",
                disable=not self.strategy.is_rank_0(),
            )
            for ndx in range(0, len(self.all_buffer), bs):
                self.pi_buffer.extend(
                    self.all_buffer[ndx : min(ndx + bs, len(self.all_buffer))]
                )
                self.prompt_consumed += bs
                self.query_step += bs

                if self.steps % self.update_interval == 0:
                    train_info = self.preference_learning(
                        self.steps // self.update_interval
                    )

                    self.save_logs_and_checkpoints(train_info)

                progress_bar.update()
                self.steps += 1
            self.prompt_epoch = p_ep + 1
            # Reorder data for another epoch.
            random.Random(self.args.seed + p_ep).shuffle(self.all_buffer)

        self.save_logs_and_checkpoints(train_info, eval=True)

        if self.strategy.is_rank_0():
            self._wandb.finish() if self._wandb else None
            lp.stop()

    def evaluate(self, dataloader, steps):
        self.strategy.print(f"Start generating evaluation responses at step {steps}")
        st_time = time.time()
        self.model.eval()
        device = torch.cuda.current_device()

        win_rate = 0
        win_rate_prob = 0
        if self.strategy.is_rank_0():
            processed_prompts = []
            prompts = []
            responses = []
            references = []
            progress_bar = tqdm(range(dataloader.__len__()))
            for i, (batch_processed_prompts, batch_prompts, refs) in enumerate(
                dataloader
            ):
                processed_prompts.extend(batch_processed_prompts)
                prompts.extend(batch_prompts)
                references.extend(refs)
                batch = self.tokenizer(
                    batch_processed_prompts,
                    return_tensors="pt",
                    max_length=self.args.prompt_max_length,
                    padding=True,
                    truncation=True,
                )
                inputs = tree.map_structure(lambda x: x.to(device), batch)
                sequences = self.model.generate(
                    **inputs, generation_config=self.eval_generate_config
                ).cpu()
                completions = self.tokenizer.batch_decode(
                    sequences[:, batch["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                completions = [c.strip() for c in completions]
                print("prompt", batch_processed_prompts[0])
                print("response", completions[0])
                responses.extend(completions)
                progress_bar.update()

            eval_res_path = os.path.join(self.save_path, "eval_results")
            os.makedirs(eval_res_path, exist_ok=True)
            pd.DataFrame(
                {
                    self.eval_input_key: prompts,
                    self.eval_output_key: responses,
                    f"format_{self.eval_input_key}": processed_prompts,
                    "reference": references,
                    "generator": self.args.wandb_run_name,
                }
            ).to_json(
                os.path.join(eval_res_path, f"{steps}.json"),
                orient="records",
            )

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
