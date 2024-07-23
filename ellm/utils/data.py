import os
from pathlib import Path
from typing import Callable, List

import torch
import torch.nn.functional as F
from datasets import interleave_datasets, load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from ellm.types import PreferenceData
from ellm.utils.deepspeed import DeepspeedStrategy

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_tokenizer(pretrain, model, padding_side="left", use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrain, trust_remote_code=True, use_fast=use_fast
    )
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        dataset_subfold_list = dataset.split("@")
        strategy.print(f"dataset: {dataset}")
        # local dir with python script or common local file
        if os.path.isdir(os.path.join(os.getcwd(), dataset)) or dataset.endswith(
            (".json", ".jsonl", ".csv", ".parquet", ".txt")
        ):
            if dataset.endswith((".json", ".jsonl", ".csv", ".parquet", ".txt")):
                files = dataset
                data_type = os.path.splitext(files)[1][1:]
            else:
                path = Path(dataset)
                script = [str(file.resolve()) for file in Path(path).rglob("*.py")]
                extensions = ("*.json", "*.jsonl", "*.csv", "*.parquet", "*.txt")
                files = [
                    str(file) for ext in extensions for file in Path(path).rglob(ext)
                ]
                strategy.print(f"script: {script}")
                strategy.print(f"files: {files}")
                # For dir, follow python script or first file type
                data_type = (
                    script[0] if len(script) == 1 else os.path.splitext(files[0])[1][1:]
                )
            # reformat data type
            if data_type in ["json", "jsonl"]:
                data_type = "json"
            elif data_type == "txt":
                data_type = "text"
            elif data_type.endswith(".py"):
                # load local dir with python script
                files = None
            if data_type.endswith(".py"):
                strategy.print(f"load {dataset} with script {data_type}")
            else:
                strategy.print(f"load {files} from {dataset}")
            data = load_dataset(data_type, data_files=files)
        elif len(dataset_subfold_list) == 2:
            dataset = dataset_subfold_list[0]
            subfold = dataset_subfold_list[1]
            data = load_dataset(dataset, data_dir=subfold.strip())
        elif len(dataset_subfold_list) == 1:
            dataset = dataset_subfold_list[0]
            data = load_dataset(dataset)
        else:
            raise Exception(f"Dataset Name {dataset}: Format error")

        if "train" in data:
            train_data_list.append(
                data["train"].select(range(min(max_count, len(data["train"]))))
            )
        else:
            train_data_list.append(
                data.select(range(min(max_count, len(data))))
            )  # train will contains eval? TODO

        if return_eval:
            max_count01 = int(max_count * 0.1)
            if "test" in data:
                eval_data = data["test"].select(
                    range(min(max_count01, len(data["test"])))
                )
            elif "validation" in data:
                eval_data = data["validation"].select(
                    range(min(max_count01, len(data["validation"])))
                )
            elif "train" in data:
                eval_data = data["train"].select(
                    range(min(max_count01, int(len(data["train"]) * 0.01)))
                )
            else:
                eval_data = data.select(
                    range(min(int(max_count01), int(len(data) * 0.01)))
                )
            eval_data_list.append(eval_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset


def pad_to_length(tensor, length, pad_value, dim=-1):
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value
                * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )


def _zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)


def _preprocess_data(
    data: PreferenceData,
    apply_chat_template=None,
) -> str:
    prompt = {"content": data.prompt, "role": "user"}
    chosen = {"content": data.chosen_response, "role": "assistant"}
    rejected = {"content": data.rejected_response, "role": "assistant"}

    if apply_chat_template:
        chosen = apply_chat_template([prompt, chosen], tokenize=False)
        rejected = apply_chat_template([prompt, rejected], tokenize=False)

        prompt = apply_chat_template(
            [prompt], tokenize=False, add_generation_prompt=True
        )
        chosen = chosen[len(prompt) :]
        rejected = rejected[len(prompt) :]
    else:
        raise ValueError("must apply chat template")
        # chosen = data[chosen_key]
        # rejected = data[rejected_key]
        # if prompt_key:
        #     prompt = data[prompt_key]
        #     if input_template:
        #         prompt = input_template.format(prompt)

    return prompt, chosen, rejected


class PromptDataset(Dataset):
    """Dataset for processing prompts."""

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.input_template = input_template
        self.n_samples_per_prompt = getattr(
            self.strategy.args, "n_samples_per_prompt", 1
        )

        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []

        def preprocess_data(
            data, input_template=None, input_key="input", apply_chat_template=None
        ) -> str:
            if apply_chat_template:
                prompt = apply_chat_template(
                    data[input_key], tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = data[input_key]
                if input_template:
                    prompt = input_template.format(prompt)
            return prompt

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(
                data, input_template, input_key, apply_chat_template
            )
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length * self.n_samples_per_prompt

    def __getitem__(self, idx):
        return self.prompts[idx // self.n_samples_per_prompt]


class PreferenceDataset(Dataset):
    def __init__(
        self,
        buffer: List[PreferenceData],
        tokenizer: Callable,
        prompt_max_length: int,
        generate_max_length: int,
        strategy: DeepspeedStrategy,
    ) -> None:
        super().__init__()
        self.prompts = []
        self.chosen_responses = []
        self.rejected_responses = []
        self.prompt_ids_lens = []

        self.tokenizer = tokenizer
        self.strategy = strategy
        self.prompt_max_length = prompt_max_length
        self.generate_max_length = generate_max_length

        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", True)
        if apply_chat_template:
            strategy.print("Applying chat template...")
            apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(
                self.strategy.args, "tokenizer_chat_template", None
            )
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        self.strategy.print("Constructing preference dataset...")

        for data in tqdm(buffer, disable=not self.strategy.is_rank_0()):
            prompt, chosen, rejected = _preprocess_data(
                data,
                apply_chat_template,
            )
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.prompt_max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
            # filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.prompt_max_length - 2:
                continue
            else:
                self.prompt_ids_lens.append(prompt_ids_len)

            self.prompts.append(prompt)
            self.chosen_responses.append(chosen)
            self.rejected_responses.append(rejected)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt, chosen, rejected = (
            self.prompts[idx],
            self.chosen_responses[idx],
            self.rejected_responses[idx],
        )
        extra = self.prompt_ids_lens[idx]

        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.prompt_max_length + self.generate_max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        rejected = (prompt + rejected).rstrip("\n")
        if not rejected.endswith(self.tokenizer.eos_token):
            rejected += " " + self.tokenizer.eos_token
        rejected_token = self.tokenizer(
            rejected,
            max_length=self.prompt_max_length + self.generate_max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        rejected_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        rejected_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            rejected_token["input_ids"],
            rejected_token["attention_mask"],
            extra,
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        rejected_ids = []
        rejected_masks = []
        extras = []
        for chosen_id, chosen_mask, rejected_id, rejected_mask, extra in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            rejected_ids.append(rejected_id)
            rejected_masks.append(rejected_mask)
            extras.append(extra)

        padding_side = "right"
        chosen_ids = _zero_pad_sequences(
            chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id
        )
        chosen_masks = _zero_pad_sequences(chosen_masks, side=padding_side)
        rejected_ids = _zero_pad_sequences(
            rejected_ids, side=padding_side, value=self.tokenizer.pad_token_id
        )
        rejected_masks = _zero_pad_sequences(rejected_masks, side=padding_side)
        return chosen_ids, chosen_masks, rejected_ids, rejected_masks, extras
