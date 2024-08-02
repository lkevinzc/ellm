import vllm
from absl import app, flags

from ellm.actor import Actor

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "pretrain", "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr", "model name"
)
prompts = [
    "SUBREDDIT: r/relationships TITLE: I [27 M] met an amazing girl [29 F], but my superficial prejudices could screw it up. POST: I went on an online date with an amazing girl. She's smart, funny, and we just had chemistry. I've been on many online dates, and the awkward silence during certain bits of the date is just par for the course. This date was different. We always had something to talk about, and more over we were on the same page. We'd even say the same thing at the same time - it was surreal. This girl is also beautiful. She was a bit heavier than her pictures, but this is something I've come to expect from online dating, and I didn't really mind. Date 2 was great too. More of that awesome chemistry, more of that great conversation. Then the clothes started coming off, and I don't know why, but she was just a lot heavier than I thought under the clothes. I don't know if she was much bigger than I thought, but her bodyfat ratio was high, and she didn't wear the fat well. I am very attracted to her when she's clothed, but I have to admit I became less attracted after she got naked. I hate that I even have to say it because our chemistry is so perfect, and she has a very pretty face even. So now I'm in conflict. I am thinking I want to hang out with her at least one more time to see if I can get over this, but I don't want to lead her on. I hate that my superficial prejudices could screw up an amazing connection like this. What should I do? TL;DR:"
]


def model_close(params1, params2):
    for key in params1.keys():
        p1 = params1[key]
        p2 = params2[key]
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def main(_):
    vllm_args = {
        "model": FLAGS.pretrain,
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.5,
        "dtype": "bfloat16",
    }
    sampling_params = vllm.SamplingParams(temperature=0.7, max_tokens=53, n=2)
    actor = Actor(vllm_args, sampling_params)

    param_dict_original = dict(
        actor.llm.llm_engine.model_executor.driver_worker.model_runner.model.named_parameters()
    )

    actor.notify_eval_start()
    resp, win = actor.generate_and_maybe_eval(
        prompts,
        ["I don't know."],
    )
    print(resp, win)

    actor.notify_eval_done()

    pref = actor.step(prompts)

    print(pref)

    param_dict_now = dict(
        actor.llm.llm_engine.model_executor.driver_worker.model_runner.model.named_parameters()
    )

    print(model_close(param_dict_original, param_dict_now))


if __name__ == "__main__":
    app.run(main)
