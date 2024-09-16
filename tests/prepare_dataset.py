import vllm
from datasets import load_dataset
from fire import Fire


def main(
    dataset_name: str,
    model_name: str,
    push_to: str,
    write_token: str,
    column_name: str,
    commit_msg: str,
    n_gpu: int,
):
    print(write_token)

    dataset_version_list = dataset_name.split("@")
    dataset = dataset_version_list[0]
    if len(dataset_version_list) > 1:
        version = dataset_version_list[1]
    else:
        version = None
    data = load_dataset(dataset, revision=version)

    sampling_params = vllm.SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_tokens=200,
    )
    llm = vllm.LLM(model=model_name, tensor_parallel_size=n_gpu)
    for split in ["train", "validation", "test"]:
        outputs = llm.generate(data[split]["prompt"], sampling_params)
        model_completions = [output.outputs[0].text.strip() for output in outputs]
        data[split] = data[split].add_column(column_name, model_completions)

    data.push_to_hub(push_to, token=write_token, commit_message=commit_msg)


if __name__ == "__main__":
    Fire(main)

"""
python tests/prepare_dataset.py trl-internal-testing/tldr-preference-sft-trl-style@462bcbf9c2eb9d30b4a029c1ed561710a571398c cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr lkevinzc/tldr-with-sft-reference $HF_WRITE_TOKEN pythia-1b-reference "Add pythia-1b reference responses" 1

python tests/prepare_dataset.py lkevinzc/tldr-with-sft-reference cleanrl/EleutherAI_pythia-2.8b-deduped__sft__tldr lkevinzc/tldr-with-sft-reference $HF_WRITE_TOKEN pythia-2.8b-reference "Add pythia-2.8b reference responses" 1

python tests/prepare_dataset.py lkevinzc/tldr-with-sft-reference cleanrl/EleutherAI_pythia-6.9b-deduped__sft__tldr lkevinzc/tldr-with-sft-reference $HF_WRITE_TOKEN pythia-6.9b-reference "Add pythia-6.9b reference responses" 1
"""
