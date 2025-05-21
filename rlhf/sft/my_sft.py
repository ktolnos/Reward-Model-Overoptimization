import os
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from tqdm import tqdm
from accelerate.utils import set_seed
import numpy as np
import pandas as pd
import shutil
tqdm.pandas()
import datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    ScriptArguments,
    get_peft_config,
)

# Define a simple chat template if needed
SIMPLE_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ message['content'] }}\n{% endif %}\n{% endfor %}"


@dataclass
class ScriptArguments:
    max_length: Optional[int] = field(default=1024)
    dataset_path: Optional[str] = field(default='', metadata={'help': 'training dataset path'})
    dbg: Optional[bool] = field(default=False)

def build_train_eval_datasets(data_path_train, tokenizer, script_args, eval_proportion, size=None):
    ds = datasets.load_dataset(data_path_train, split="train")
    if size is not None:
        ds = ds.select(range(0, size))
    ds_dict = ds.train_test_split(test_size=eval_proportion, seed=42)
    ds_train = ds_dict['train']
    ds_eval = ds_dict['test']
    ds_train = post_process_common_dataset(ds_train, tokenizer, script_args)
    ds_eval = post_process_common_dataset(ds_eval, tokenizer, script_args)
    return ds_train, ds_eval


def build_dataset_common(data_path, tokenizer, script_args, split='', size=None):
    ds = datasets.load_dataset(data_path, split=split)

    if size is not None:
        ds = ds.select(range(0, size))

    ds = post_process_common_dataset(ds, tokenizer, script_args)
    return ds

def post_process_common_dataset(ds, tokenizer, script_args):
    def formatting_func(example):
        kwargs = {"return_tensors": "pt"}
        # kwargs = {"padding": 'max_length', "truncation": True, "max_length": script_args.max_length, "return_tensors": "pt"}
        messages = example['chosen']
        prompt_plus_response = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode_plus(prompt_plus_response, **kwargs)

        return {
            # 'query': prompt_plus_response,
            "input_ids": tokens["input_ids"][0],
            "attention_mask": tokens["attention_mask"][0],
        }

    ds = ds.map(formatting_func,
                remove_columns=ds.column_names,
                batched=False, num_proc=30)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= script_args.max_length, num_proc=30)
    ds.set_format(type="torch")
    return ds

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    ################
    # Model & Tokenizer
    ################
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set chat template if needed
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )
    
    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    
    ################
    # Dataset
    ################
    train_dataset, eval_dataset = build_train_eval_datasets(
        script_args.dataset_path, tokenizer, script_args,
        eval_proportion=0.1,
        size=100 if script_args.dbg else None
    )
    print(f"Size of the train set: {len(train_dataset)}, eval set: {len(eval_dataset)}")
    
    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    # Train the model
    trainer.train()
    
    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()