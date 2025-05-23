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
from ppo_utils import (print_trainable_parameters, collator, eval_model, build_dataset_unified, transfer_template_rm,
                       plot_curve, build_train_eval_datasets)
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
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
)


@dataclass
class ScriptArguments:
    max_length: Optional[int] = field(default=1024)
    dataset_path: Optional[str] = field(default='', metadata={'help': 'training dataset path'})
    dbg: Optional[bool] = field(default=False)

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # deepspeed_plugin = DeepSpeedPlugin(
    #     zero_stage=3,
    # )
    # accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )
    print(policy.named_modules)

    policy.resize_token_embeddings(len(tokenizer))
    policy.config.pad_token_id = tokenizer.pad_token_id

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None

    # assert policy.config.vocab_size == len(tokenizer) == policy.get_input_embeddings().weight.shape[0] == \
    #        value_model.get_input_embeddings().weight.shape[0] == value_model.config.vocab_size, \
    #     f"Tokenizer vocab size: {tokenizer.vocab_size}, Policy model config vocab size: {policy.config.vocab_size}," \
    #     f"Policy model embedding matrix size: {policy.get_input_embeddings().weight.shape[0]}, " \
    #     f"Value model config vocab size: {value_model.config.vocab_size}, " \
    #     f"Value model embedding matrix size: {value_model.get_input_embeddings().weight.shape[0]}"

    ################
    # Dataset
    ################

    train_dataset, eval_dataset = build_train_eval_datasets(
        script_args.dataset_path, tokenizer, script_args, eval_proportion=0.1, size=100 if script_args.dbg else None)
    print(f"Size of the train set: {len(train_dataset)}, eval set: {len(eval_dataset)}")

    ################
    # Training
    ################
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()