import os
from dataclasses import dataclass, field
from typing import Optional, Union, List
from accelerate import Accelerator, DeepSpeedPlugin
import torch
from tqdm import tqdm
from accelerate.utils import set_seed
import numpy as np
import pandas as pd
import shutil

from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, EncodedInput, TruncationStrategy
from transformers.utils import PaddingStrategy

tqdm.pandas()
from grpo_utils import (build_train_eval_datasets)

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser, PreTrainedTokenizerBase, TensorType, BatchEncoding,
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
        script_args.reward_model_path, trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    policy = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(script_args.reward_model_path,
                                                     trust_remote_code=model_args.trust_remote_code,
                                                     padding_side="left")
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    class TokenizerWrapper(PreTrainedTokenizerBase):
        def __init__(self, tokenizer: PreTrainedTokenizerBase):
            self.tokenizer = tokenizer
            self.pad_token_id = tokenizer.pad_token_id
            self.eos_token_id = tokenizer.eos_token_id
            self.bos_token_id = tokenizer.bos_token_id

        def __call__(self, *args, **kwargs):
            print("call")
            return self.tokenizer(*args, **kwargs)

        def apply_chat_template(self, *args, **kwargs):
            return self.tokenizer.apply_chat_template(*args, **kwargs)

        def encode(self, *args, **kwargs) -> List[int]:
            print("encode")
            return tokenizer.encode(*args, **kwargs)

        def encode_plus(self, *args, **kwargs) -> BatchEncoding:
            print("encode_plus")
            return tokenizer.encode_plus(*args, **kwargs)

        def batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
            print("batch_encode_plus")
            return tokenizer.batch_encode_plus(*args, **kwargs)

        def tokenize(self, *args, **kwargs) -> List[str]:
            print("tokenize")
            return tokenizer.tokenize(*args, **kwargs)
    reward_tokenizer = TokenizerWrapper(reward_tokenizer)


    ################
    # Dataset
    ################

    train_dataset, eval_dataset = build_train_eval_datasets(
        script_args.dataset_path, tokenizer, eval_proportion=0.1, size=100 if script_args.dbg else None)
    print(f"Size of the train set: {len(train_dataset)}, eval set: {len(eval_dataset)}")

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        args=training_args,
        processing_class=reward_tokenizer,
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