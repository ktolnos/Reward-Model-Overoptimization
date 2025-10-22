import json

import pandas as pd
import numpy as np

from datasets import load_dataset

ds = load_dataset('/nas/ucb/eop/Reward-Model-Overoptimization/experimental/data/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B')

ds.push_to_hub('ktolnos/helpsteer3_goldSkywork-Reward-V2-Llama-3.1-8B')