import pdb

from absl import app, flags
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ellm.utils.data import PromptDataset, blending_datasets
from ellm.utils.deepspeed import DummyStrategy

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
    max_train = 10000
    strategy = DummyStrategy(args=FLAGS)
    prompts_data = blending_datasets(
        FLAGS.prompt_data,
        "1",
        strategy,
        max_count=max_train,
        return_eval=False,
    )
    prompts_data = prompts_data.select(range(min(max_train, len(prompts_data))))
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
