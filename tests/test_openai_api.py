import os
import pdb

from openai import OpenAI

from ellm.oracles.gpt import logprob_parser

client = OpenAI(
    api_key=os.environ.get("OAI_KEY"),
    base_url=os.environ.get("OAI_URL"),
)
completion = client.chat.completions.create(
    model="gpt-4o-mini-2024-07-18",
    messages=[{"role": "user", "content": "output 1 or 0 only! no any other text"}],
    max_tokens=1,
    logprobs=True,
    top_logprobs=5,
)


win_prob = logprob_parser(
    completion, numerator_token="0", denominator_tokens=["0", "1"]
)


pdb.set_trace()
