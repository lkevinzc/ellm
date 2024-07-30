import pdb

import launchpad as lp
import vllm
from absl import app, flags, logging
from launchpad.nodes.python import local_multi_processing
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ellm.actor import Actor
from ellm.utils.data import PromptDataset, blending_datasets
from ellm.utils.deepspeed import DeepspeedStrategy, DummyStrategy

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "pretrain", "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr", "model name"
)
flags.DEFINE_string(
    "prompt_data", "HuggingFaceH4/helpful-anthropic-raw", "dataset name"
)
flags.DEFINE_string("input_key", "instruction", "key to get the input prompt")
flags.DEFINE_bool("apply_chat_template", True, "apply chat template")


def main(_):
    max_samples = 10000
    strategy = DummyStrategy(args=FLAGS)
    prompts_data = blending_datasets(
        FLAGS.prompt_data,
        "1",
        strategy,
        max_count=max_samples,
        return_eval=False,
    )
    prompts_data = prompts_data.select(range(min(max_samples, len(prompts_data))))
    tokenizer = AutoTokenizer.from_pretrained(
        FLAGS.pretrain, trust_remote_code=True, use_fast=True
    )
    ds = PromptDataset(prompts_data, tokenizer, strategy)

    dl = DataLoader(
        ds,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    dl = iter(dl)
    item = next(dl)
    pdb.set_trace()


if __name__ == "__main__":
    app.run(main)
