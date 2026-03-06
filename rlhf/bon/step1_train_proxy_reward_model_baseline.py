from dataclasses import dataclass, field
from typing import List, Optional
from accelerate import Accelerator
import numpy as np
import sys
import os
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import RewardTrainer, RewardConfig
from load_datasets import load_train_eval_dataset
from utils import *

# Add the `./reward_models` path to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reward_models')))
from data_utils import setup_tokenizer, get_length_config


@dataclass
class ScriptArguments:
    attn_implementation: Optional[str] = field(default="flash_attention_2")
    # data
    dataset: Optional[str] = field(default='rlhf/bon/step1_obtain_gold_score/unified_sampled_gold_score')
    # lora
    use_lora: Optional[bool] = field(default=True)
    lora_target_modules: Optional[List[str]] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_r: Optional[int] = field(default=32)
    lora_alpha: Optional[int] = field(default=64)
    lora_dropout: Optional[float] = field(default=0.05)
    # model
    base_model: Optional[str] = field(default="google/gemma-2b-it")
    freeze_pretrained: Optional[bool] = field(default=False)
    # log
    log_dir: Optional[str] = field(default='./reward_models_train')
    wandb_name: Optional[str] = field(default="test")
    debug: Optional[bool] = field(default=False, metadata={'help': 'if debug=True, only train with 100 samples'})


parser = HfArgumentParser((ScriptArguments, RewardConfig))
script_args, training_args = parser.parse_args_into_dataclasses()
_max_conv_tokens = get_length_config("default")["max_conversation_tokens"]
model_name_split = script_args.base_model.split("/")[-1]
if script_args.use_lora:
    output_name = f"{script_args.log_dir}/{model_name_split}_{script_args.wandb_name}_len{_max_conv_tokens}_lora{script_args.lora_r}_{training_args.learning_rate}_data{script_args.dataset.split('/')[-1]}"
else:
    output_name = f"{script_args.log_dir}/{model_name_split}_{script_args.wandb_name}_len{_max_conv_tokens}_fulltrain_{training_args.learning_rate}_data{script_args.dataset.split('/')[-1]}"

# Set computed/hardcoded training args
training_args.output_dir = os.path.join(output_name, 'logs')
training_args.run_name = script_args.wandb_name
training_args.max_length = _max_conv_tokens
training_args.remove_unused_columns = False
training_args.ddp_find_unused_parameters = False

device = Accelerator().local_process_index

# Load the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(script_args.base_model, use_fast=False)
setup_tokenizer(tokenizer, model_name=script_args.base_model)

# Load datasets
train_dataset, eval_dataset = load_train_eval_dataset(script_args.dataset, tokenizer, size=100 if script_args.debug else None, length_config="default")
print('Training dataset size: {}, validation dataset size: {}'.format(len(train_dataset), len(eval_dataset)))


if len(script_args.attn_implementation):
    model_params = {
        "attn_implementation": script_args.attn_implementation,
    }
else:
    model_params = {}

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.base_model, num_labels=1, device_map=device,
    torch_dtype=torch.bfloat16,
    **model_params
)

if script_args.freeze_pretrained:
    mlp_layer = nn.Sequential(
        nn.Linear(model.config.hidden_size, 1024, dtype=torch.bfloat16),
        nn.ReLU(),
        nn.Linear(1024, 1, dtype=torch.bfloat16)
    )
    mlp_layer.to(device)
    freeze_trainable_parameters(model)
    model.score = mlp_layer

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
print_trainable_parameters(model)


# Define the trainer parameters
trainer_params = {
    "model": model,
    "args": training_args,
    "processing_class": tokenizer,
    "train_dataset": train_dataset,
    "eval_dataset": eval_dataset,
}


if script_args.use_lora:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=script_args.lora_target_modules,
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
    )
    trainer_params["peft_config"] = peft_config

trainer = RewardTrainer(**trainer_params)
print_trainable_parameters(trainer.model)


print('training start')
trainer.train()
