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
from trl.models import prepare_deepspeed

from qrm_gemma_tokenizer import TokenizerWrapper

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
    reward_tokenizer.max_length = script_args.max_length
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    if 'QRM' in model_args.reward_model_path:
        reward_tokenizer = TokenizerWrapper(reward_tokenizer)


    ################
    # Dataset
    ################

    train_dataset, eval_dataset = build_train_eval_datasets(
        script_args.dataset_path, tokenizer, eval_proportion=0.1, size=100 if script_args.dbg else None,
        max_length=training_args.max_prompt_length,
    )
    print(f"Size of the train set: {len(train_dataset)}, eval set: {len(eval_dataset)}")

    for prompt in train_dataset['prompt'][:5]:
        print(f"Sample prompt: \n{prompt}")

    avg_len = np.mean([len(tokenizer.encode(prompt)) for prompt in train_dataset['prompt']])
    max_len = max([len(tokenizer.encode(prompt)) for prompt in train_dataset['prompt']])
    print(f"Average length of prompts: {avg_len}, Max length of prompts: {max_len}")


    def model_reward_func(prompts, completions, **kwargs):
        print("Prompts:\n", prompts)
        print("Completions:\n", completions)
        texts = [p + c for p, c in zip(prompts, completions)]
        reward_inputs = reward_tokenizer(
            text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
        )
        with torch.inference_mode():
           reward = reward_model(**reward_inputs).logits[:, 0]  # Shape (B*G,)
        return reward


    ################
    # Training
    ################
    trainer = GRPOTrainer(
        args=training_args,
        # reward_processing_classes=reward_tokenizer,
        model=policy,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        reward_funcs=model_reward_func,
    )
    if trainer.is_deepspeed_enabled:
        prepare_deepspeed(reward_model, trainer.accelerator)
    else:
        trainer.accelerator.prepare_model(
            reward_model, evaluation_mode=True, device_placement=True
        )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()