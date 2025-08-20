import os
from collections import defaultdict
from dataclasses import dataclass, field
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
from reward_utils import get_reward

@dataclass
class RewardController:
    trainer: GRPOTrainer = None
    logging_steps: float = 1
    save_path: str = None
    generations_df: pd.DataFrame = None
    k_top_responses: int = 0
    adversarial_responses_buffer: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.save_path and self.generations_df is None:
            if os.path.exists(self.save_path):
                print(f"Loading existing generations from {self.save_path}")
                self.generations_df = pd.read_csv(self.save_path)
            else:
                print(f"Creating new generations file at {self.save_path}")
                self.generations_df = pd.DataFrame(columns=['prompt', 'completion', 'reward'])

    def get_and_clear_adversarial_buffer(self):
        buffer_copy = list(self.adversarial_responses_buffer)
        self.adversarial_responses_buffer.clear()
        return buffer_copy


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
    if 'reference_reward_1' in columns_to_remove:
        columns_to_remove.remove('reference_reward_1')
    if 'reference_reward_2' in columns_to_remove:
        columns_to_remove.remove('reference_reward_2')
    if 'reference_response_1' in columns_to_remove:
        columns_to_remove.remove('reference_response_1')
    if 'reference_response_2' in columns_to_remove:
        columns_to_remove.remove('reference_response_2')
    print(columns_to_remove, " will be removed")
    ds = ds.map(formatting_func,
                remove_columns=columns_to_remove,
                batched=False, num_proc=30)
    ds.set_format(type="torch")
    return ds


rew_mean_sum = defaultdict(float)
rew_mean_count = defaultdict(int)
def build_reward_function(reward_models, reward_tokenizers, script_args, controller: RewardController):
    def model_reward_func(prompts, completions, **kwargs):
        global rew_mean_sum, rew_mean_count
        should_log = controller.trainer.state.global_step % controller.logging_steps == 0

        reference_rewards = None
        if script_args.reference_rewards:
            reference_rewards = kwargs.get('reference_reward', None)
            assert reference_rewards is not None, "Reference rewards must be provided in the dataset if reference_rewards is True"
            if isinstance(reference_rewards, list):
                reference_rewards = torch.stack(reference_rewards)

        texts = [p + c for p, c in zip(prompts, completions)]
        rewards = []
        rewards_dict = {}
        for reward_model, reward_tokenizer in zip(reward_models, reward_tokenizers):
            rew = get_reward(reward_model, reward_tokenizer, prompts, completions, texts, reward_controller=controller)
            rew_mean_sum[reward_model] += rew.mean().item()
            rew_mean_count[reward_model] += 1
            rewards_dict[reward_model] = rew.detach()
            if should_log and wandb.run is not None:
                model_for_name = reward_model
                if hasattr(reward_model, 'module'):
                    model_for_name = reward_model.module
                wandb.log({
                    f"reward/{model_for_name.config._name_or_path}": rew_mean_sum[reward_model] / rew_mean_count[reward_model],
                }, step=controller.trainer.state.global_step)

            if script_args.reference_rewards and script_args.adv_rm_lambda == 0:
                rew = rew - reference_rewards
            if script_args.sigmoid_rewards:
                rew = torch.sigmoid(rew)
            rewards.append(rew)

        rewards = torch.stack(rewards, dim=1)  # Shape (B*G, N)
        if script_args.adv_rm_lambda != 0:
            assert rewards.shape[1] == 2, "Adv-RM requires exactly 2 reward models"
            assert reference_rewards is not None, "Reference rewards must be provided for Adv-RM"
            rewards_above_ref = rewards[:, 0] - script_args.adv_rm_lambda * rewards[:, 1]
            rewards_below_ref = rewards[:, 0] - 25
            reward = torch.where(rewards[:, 0] > reference_rewards, rewards_above_ref, rewards_below_ref)
        elif script_args.ensemble_aggregation == 'mean':
            reward = rewards.mean(dim=1)
        elif script_args.ensemble_aggregation == 'min':
            reward = rewards.min(dim=1).values
        else:
            raise ValueError(f"Unknown ensemble aggregation method: {script_args.ensemble_aggregation}")

        if controller.k_top_responses > 0:
            # Group rewards by prompt, since prompts are repeated for each completion in a group.
            # This is robust to variable batch sizes.
            group_indices = defaultdict(list)
            for i, p in enumerate(prompts):
                group_indices[p].append(i)

            for p, indices in group_indices.items():
                group_rewards = reward[torch.tensor(indices, device=reward.device)]

                # Find top k in this group
                top_k_indices_in_group = torch.topk(
                    group_rewards, min(controller.k_top_responses, len(group_rewards))
                ).indices

                for k_idx in top_k_indices_in_group:
                    original_idx = indices[k_idx]
                    ref_rew = reference_rewards[original_idx].item() if reference_rewards is not None else None
                    controller.adversarial_responses_buffer.append(
                        (prompts[original_idx], completions[original_idx], reward[original_idx].item(), ref_rew)
                    )

        if controller.save_path is not None:
            new_data = {
                'prompt': prompts,
                'completion': completions,
                'reward': reward.tolist()
            }
            for k, v in kwargs.items():
                if k == 'completion_ids':
                    continue
                if isinstance(v, list):
                    new_data[k] = [it.cpu().numpy() if isinstance(it, torch.Tensor) else it for it in v]
                elif isinstance(v, torch.Tensor):
                    new_data[k] = v.tolist()
                else:
                    new_data[k] = [v] * len(prompts)

            for reward_model in reward_models:
                new_data[f'reward_{reward_model.config._name_or_path}'] = rewards_dict[reward_model].tolist()
            controller.generations_df = pd.concat([controller.generations_df, pd.DataFrame(new_data)], ignore_index=True)
            if should_log:
                controller.generations_df.to_csv(controller.save_path, index=False)

        if should_log:
            for reward_model in reward_models:
                rew_mean_sum[reward_model] = 0
                rew_mean_count[reward_model] = 0
        return reward.tolist()

    return model_reward_func