import json

import pandas as pd
import numpy as np

from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer

login()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

ds = load_dataset(
    "/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B"
)

ds = ds.filter(
    lambda x: len(
        tokenizer.apply_chat_template(
            x["chosen"][:-1], tokenize=True, add_generation_prompt=True
        )
    )
    < 1024
)

ds.push_to_hub("ktolnos/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B")
