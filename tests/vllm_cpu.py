import vllm


def check_model_same(params1, params2):
    for key in params1.keys():
        p1 = params1[key]
        p2 = params2[key]
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def offload_vllm(llm):
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    state_dict = {k: v.cpu() for k, v in model.named_parameters()}
    return state_dict


def load_vllm(llm, state_dict):
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    model.load_state_dict(state_dict)


def get_params(llm):
    return dict(
        llm.llm_engine.model_executor.driver_worker.model_runner.model.named_parameters()
    )


llm = vllm.LLM(
    **{
        "model": "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr",
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.5,
        "dtype": "bfloat16",
    }
)


def get_sampling_params():
    return vllm.SamplingParams(temperature=0, top_p=1, max_tokens=53, n=1)


prompts = [
    "SUBREDDIT: r/relationships TITLE: I [27 M] met an amazing girl [29 F], but my superficial prejudices could screw it up. POST: I went on an online date with an amazing girl. She's smart, funny, and we just had chemistry. I've been on many online dates, and the awkward silence during certain bits of the date is just par for the course. This date was different. We always had something to talk about, and more over we were on the same page. We'd even say the same thing at the same time - it was surreal. This girl is also beautiful. She was a bit heavier than her pictures, but this is something I've come to expect from online dating, and I didn't really mind. Date 2 was great too. More of that awesome chemistry, more of that great conversation. Then the clothes started coming off, and I don't know why, but she was just a lot heavier than I thought under the clothes. I don't know if she was much bigger than I thought, but her bodyfat ratio was high, and she didn't wear the fat well. I am very attracted to her when she's clothed, but I have to admit I became less attracted after she got naked. I hate that I even have to say it because our chemistry is so perfect, and she has a very pretty face even. So now I'm in conflict. I am thinking I want to hang out with her at least one more time to see if I can get over this, but I don't want to lead her on. I hate that my superficial prejudices could screw up an amazing connection like this. What should I do? TL;DR:"
]
response = llm.generate(prompts, sampling_params=get_sampling_params())
print("initial", response)

old_params = get_params(llm)

cpu_model = offload_vllm(llm)
load_vllm(llm, cpu_model)

new_params = get_params(llm)
print(check_model_same(old_params, new_params))
response = llm.generate(prompts, sampling_params=get_sampling_params())
print("after", response)
