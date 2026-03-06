from dataclasses import dataclass, field
from typing import List, Optional, Union
from accelerate import Accelerator
import numpy as np
import os
import torch
import torch.nn as nn
from datasets import concatenate_datasets
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import RewardTrainer, RewardConfig
from load_datasets import load_train_eval_dataset, build_dataset
from data_utils import setup_tokenizer, get_length_config, DATASET_LENGTH_CONFIGS
from utils import (
    print_trainable_parameters,
    freeze_trainable_parameters,
)


@dataclass
class ScriptArguments:
    length_config: str = field(default="default", metadata={
        "help": f"Name of the length config from DATASET_LENGTH_CONFIGS. Available: {list(DATASET_LENGTH_CONFIGS.keys())}"
    })
    attn_implementation: Optional[str] = field(default="flash_attention_2")
    # data
    dataset: List[str] = field(
        default_factory=list,
        metadata={"help": "One or more dataset repo/path values."},
    )
    # lora
    use_lora: Optional[bool] = field(default=True)
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lora_r: Optional[int] = field(default=32)
    lora_alpha: Optional[int] = field(default=64)
    lora_dropout: Optional[float] = field(default=0.05)
    # model
    base_model: Optional[str] = field(default="google/gemma-2b-it")
    freeze_pretrained: Optional[bool] = field(default=False)
    # log
    log_dir: Optional[str] = field(default="./reward_models_train")
    wandb_name: Optional[str] = field(default="test")
    debug_dataset: Optional[bool] = field(
        default=False, metadata={"help": "if True, only train with 100 samples"}
    )


parser = HfArgumentParser((ScriptArguments, RewardConfig))
script_args, training_args = parser.parse_args_into_dataclasses()
torch.manual_seed(training_args.seed)
np.random.seed(training_args.seed)

# Resolve length config.
_length_cfg = get_length_config(script_args.length_config)

dataset_list = script_args.dataset
if isinstance(dataset_list, str):
    dataset_list = [dataset_list]
if not dataset_list:
    raise ValueError("--dataset must contain at least one dataset path/name.")

model_name_split = script_args.base_model.split("/")[-1]
dataset_name = dataset_list[0]
_max_conv_tokens = _length_cfg["max_conversation_tokens"]
if script_args.use_lora:
    output_name = f"{script_args.log_dir}/{training_args.seed}_{model_name_split}_len{_max_conv_tokens}_lora{script_args.lora_r}_{training_args.learning_rate}_data{dataset_name.split('/')[-1]}"
else:
    output_name = f"{script_args.log_dir}/{training_args.seed}_{model_name_split}_len{_max_conv_tokens}_fulltrain_{training_args.learning_rate}_data{dataset_name.split('/')[-1]}"

# Set computed/hardcoded training args
training_args.output_dir = os.path.join(output_name, "logs")
training_args.run_name = script_args.wandb_name
training_args.max_length = _max_conv_tokens
training_args.remove_unused_columns = False
training_args.ddp_find_unused_parameters = False

device = Accelerator().local_process_index

# Load the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(script_args.base_model, use_fast=False)
setup_tokenizer(tokenizer, model_name=script_args.base_model)

# Load datasets
train_dataset, eval_dataset = load_train_eval_dataset(
    dataset_list[0],
    tokenizer,
    size=100 if script_args.debug_dataset else None,
    seed=training_args.seed,
    length_config=script_args.length_config,
)
for i in range(1, len(dataset_list)):
    new_train_dataset = build_dataset(
        dataset_list[i],
        tokenizer,
        split="train",
        size=100 if script_args.debug_dataset else None,
        length_config=script_args.length_config,
    )
    train_dataset = concatenate_datasets([train_dataset, new_train_dataset])
train_dataset = train_dataset.shuffle(seed=training_args.seed)
print(
    "Training dataset size: {}, validation dataset size: {}".format(
        len(train_dataset), len(eval_dataset)
    )
)


if len(script_args.attn_implementation):
    model_params = {
        "attn_implementation": script_args.attn_implementation,
    }
else:
    model_params = {}

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.base_model,
    num_labels=1,
    device_map=device,
    torch_dtype=torch.bfloat16,
    **model_params,
)

if script_args.freeze_pretrained:
    mlp_layer = nn.Sequential(
        nn.Linear(model.config.hidden_size, 1024, dtype=torch.bfloat16),
        nn.ReLU(),
        nn.Linear(1024, 1, dtype=torch.bfloat16),
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


print("training start")
trainer.train()
