import os
from dataclasses import dataclass
from typing import Union, Any, Mapping

import torch
from accelerate.test_utils.scripts.test_sync import step_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
import numpy as np
import pandas as pd
from trl import GRPOTrainer
import wandb

tqdm.pandas()
import matplotlib.pyplot as plt
from reward_utils import get_reward, is_reasoning

@dataclass
class RewardController:
    trainer: GRPOTrainer = None
    logging_steps: float = 1


def build_train_eval_datasets(data_path_train, tokenizer, eval_proportion, size=None, max_length=512,):
    ds = datasets.load_dataset(data_path_train, split="train")
    if size is not None:
        ds = ds.select(range(0, size))
    ds_dict = ds.train_test_split(test_size=eval_proportion, seed=42)
    ds_train = ds_dict['train']
    ds_eval = ds_dict['test']
    ds_train = post_process_common_dataset(ds_train, tokenizer, max_length)
    ds_eval = post_process_common_dataset(ds_eval, tokenizer, max_length)
    return ds_train, ds_eval

def post_process_common_dataset(ds, tokenizer, max_length):
    def formatting_func(example):
        messages = example['chosen'][:-1]
        chat = tokenizer.apply_chat_template(messages, tokenize=True,
                                             add_generation_prompt=True,
                                             enable_thinking=False,
                                             truncation=True,
                                             max_length=max_length,
                                             )
        prompt = tokenizer.decode(chat, skip_special_tokens=False) # This way we limit max_length
        return {
            "prompt": prompt,
        }

    columns_to_remove = ds.column_names
    if 'reference_reward' in columns_to_remove:
        columns_to_remove.remove('reference_reward')
    print(columns_to_remove, " will be removed")
    ds = ds.map(formatting_func,
                remove_columns=columns_to_remove,
                batched=False, num_proc=30)
    ds.set_format(type="torch")
    return ds


def build_reward_function(reward_models, reward_tokenizers, script_args, controller: RewardController):
    def model_reward_func(prompts, completions, **kwargs):
        texts = [p + c for p, c in zip(prompts, completions)]
        rewards = []
        for reward_model, reward_tokenizer in zip(reward_models, reward_tokenizers):
            rew = get_reward(reward_model, reward_tokenizer, prompts, completions, texts, reward_controller=controller)
            if script_args.reference_rewards:
                raise NotImplementedError("Reference rewards are not implemented yet.")
            if script_args.sigmoid_rewards:
                rew = torch.sigmoid(rew)
            rewards.append(rew)
            if controller.trainer.state.global_step % controller.logging_steps == 0 and wandb.run is not None:
                wandb.log({
                    f"reward/{reward_model.config._name_or_path}": rew.mean().item(),
                }, step=controller.trainer.state.global_step)

        rewards = torch.stack(rewards, dim=1)  # Shape (B*G, N)
        if script_args.ensemble_aggregation == 'mean':
            reward = rewards.mean(dim=1)
        elif script_args.ensemble_aggregation == 'min':
            reward = rewards.min(dim=1).values
        else:
            raise ValueError(f"Unknown ensemble aggregation method: {script_args.ensemble_aggregation}")
        return reward.tolist()

    return model_reward_func




