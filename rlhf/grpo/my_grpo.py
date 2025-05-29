import os
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator, DeepSpeedPlugin
import torch
from tqdm import tqdm
from accelerate.utils import set_seed
import numpy as np
import pandas as pd
import shutil
tqdm.pandas()
from grpo_utils import (build_train_eval_datasets)
from rm_utils import load_reward_model
from config import get_config

from accelerate import PartialState
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import (
    ModelConfig,
    GRPOConfig,
    GRPOTrainer,
    AutoModelForCausalLMWithValueHead,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
)
from peft import get_peft_model


@dataclass
class ScriptArguments:
    max_length: Optional[int] = field(default=1024)
    dataset_path: Optional[str] = field(default='', metadata={'help': 'training dataset path'})
    dbg: Optional[bool] = field(default=False)
    reward_model_path: Optional[str] = field(default='google/gemma-2b-it', metadata={'help': 'path to the reward model'})

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    ################
    # Model & Tokenizer
    ################
    peft_config = get_peft_config(model_args)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    policy = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    ################
    # Dataset
    ################

    train_dataset, eval_dataset = build_train_eval_datasets(
        script_args.dataset_path, script_args, eval_proportion=0.1, size=100 if script_args.dbg else None)
    print(f"Size of the train set: {len(train_dataset)}, eval set: {len(eval_dataset)}")

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        args=training_args,
        # processing_class=tokenizer,
        model=policy,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        reward_funcs=reward_model,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()