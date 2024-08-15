import pdb

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

    preference_data = actor.step(
        [
            "Chengdu is a city ",
            "SUBREDDIT: r/AskReddit TITLE: How do you get someone out of your head? POST: Hi, I'm 22, and I have been with my girlfriend for 5 years now. We recently moved together. We've always loved each other intensely. Problem, I recently started to have feelings for an other person (a friend). This person has had a boyfriend for now 3 years, and has absolutely no ideas. Those feelings were so strong, it was hard to hide them. After 2 months of me being distant and really sad, my girlfriend forced me to say what was bothering me. I'm not a good liar, and now she knows. We decided to give us a week alone, I went to my parents. Now, I'm completely lost. I keep on thinking about this person, and I hate that. I would like for those feelings to go away, to leave me alone. But I can't. What do I do? It's been 3 months now, and I'm just desperate. TL;DR:",
        ]
    )
    pdb.set_trace()


if __name__ == "__main__":
    app.run(main)
