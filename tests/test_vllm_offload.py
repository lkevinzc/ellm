import vllm
from absl import app, flags

from ellm.actor import Actor

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "pretrain", "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr", "model name"
)


def main(_):
    vllm_args = {
        "model": FLAGS.pretrain,
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.5,
        "dtype": "bfloat16",  # TODO(liuzc) check whether to use bfloat
        "seed": 0,
        "enable_prefix_caching": True,
    }
    sampling_params = vllm.SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=512, seed=0, n=2
    )
    actor = Actor(vllm_args, sampling_params)

    actor.notify_eval_start()

    actor.notify_eval_done()


if __name__ == "__main__":
    app.run(main)
