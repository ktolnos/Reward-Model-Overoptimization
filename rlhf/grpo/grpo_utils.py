import gc
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
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
from trl import GRPOTrainer
import wandb

tqdm.pandas()
import matplotlib.pyplot as plt
from reward_utils import get_reward
from rlhf.prompt_utils import build_prompt_from_chosen
import math


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
                self.generations_df = pd.DataFrame(
                    columns=["prompt", "completion", "reward"]
                )

    def get_and_clear_adversarial_buffer(self):
        buffer_copy = list(self.adversarial_responses_buffer)
        self.adversarial_responses_buffer.clear()
        return buffer_copy


def build_train_eval_datasets(
    data_path_train,
    tokenizer,
    eval_proportion,
    size=None,
    max_length=512,
):
    ds = datasets.load_dataset(data_path_train, split="train")
    if size is not None:
        ds = ds.select(range(0, size))
    ds_dict = ds.train_test_split(test_size=eval_proportion, seed=42)
    ds_train = ds_dict["train"]
    ds_eval = ds_dict["test"]
    ds_train = post_process_common_dataset(ds_train, tokenizer, max_length)
    ds_eval = post_process_common_dataset(ds_eval, tokenizer, max_length)
    return ds_train, ds_eval


def post_process_common_dataset(ds, tokenizer, max_length):
    def formatting_func(example):
        prompt = build_prompt_from_chosen(
            example["chosen"],
            tokenizer,
            max_length=max_length,
        )
        return {
            "prompt": prompt,
        }

    columns_to_remove = ds.column_names
    if "reference_reward" in columns_to_remove:
        columns_to_remove.remove("reference_reward")
    if "reference_reward_1" in columns_to_remove:
        columns_to_remove.remove("reference_reward_1")
    if "reference_reward_2" in columns_to_remove:
        columns_to_remove.remove("reference_reward_2")
    if "reference_response_1" in columns_to_remove:
        columns_to_remove.remove("reference_response_1")
    if "reference_response_2" in columns_to_remove:
        columns_to_remove.remove("reference_response_2")
    print(columns_to_remove, " will be removed")
    ds = ds.map(
        formatting_func, remove_columns=columns_to_remove, batched=False, num_proc=30
    )
    ds.set_format(type="torch")
    return ds


def _load_reward_model(model_path, tokenizer, trust_remote_code=True):
    """Loads a reward model from the given path."""
    # print(f"Loading reward model from {model_path}")
    if "RRM" in model_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


rew_mean_sum = defaultdict(float)
rew_mean_count = defaultdict(int)


def get_active_indices(current_step, total_steps, num_rms, args):
    if args.rm_switch_strategy == "ensemble":
        return list(range(num_rms))

    if args.rm_switch_strategy == "sequential":
        # Calculate active index based on steps and multiplier
        if total_steps == 0:
            return [0]
        # We assume total_steps is at least 1 or handle start 0
        rm_index = (
            (current_step * num_rms * args.rm_switches_multiplier) // total_steps
        ) % num_rms
        return [rm_index]

    if args.rm_switch_strategy == "mix":
        ensemble_size = args.mix_ensemble_size
        if ensemble_size >= num_rms:
            return list(range(num_rms))

        if args.mix_strategy == "disjoint":
            num_chunks = math.ceil(num_rms / ensemble_size)
            chunk_idx = (
                (current_step * num_chunks * args.rm_switches_multiplier) // total_steps
            ) % num_chunks

            start_idx = chunk_idx * ensemble_size
            end_idx = min(start_idx + ensemble_size, num_rms)
            return list(range(start_idx, end_idx))

        elif args.mix_strategy == "sliding":
            # Sliding window start index shifts
            # Number of positions to slide through is num_rms
            start_idx = (
                (current_step * num_rms * args.rm_switches_multiplier) // total_steps
            ) % num_rms

            indices = []
            for i in range(ensemble_size):
                indices.append((start_idx + i) % num_rms)
            return indices

    return list(range(num_rms))  # Default fallback


def build_reward_function(
    reward_models,
    reward_tokenizers,
    script_args,
    controller: RewardController,
    policy_tokenizer=None,
):
    def model_reward_func(prompts, completions, **kwargs):
        global rew_mean_sum, rew_mean_count
        should_log = (
            controller.trainer.state.global_step > 0
            and controller.trainer.state.global_step % controller.logging_steps == 0
        )

        # --- Common Setup ---
        reference_rewards = None
        if script_args.reference_rewards:
            reference_rewards = kwargs.get("reference_reward", None)
            assert (
                reference_rewards is not None
            ), "Reference rewards must be provided in the dataset if reference_rewards is True"
            if isinstance(reference_rewards, list):
                reference_rewards = torch.stack(reference_rewards)

        texts = [p + c for p, c in zip(prompts, completions)]

        num_rms = len(script_args.reward_model_paths)

        current_step = controller.trainer.state.global_step
        total_steps = controller.trainer.state.max_steps
        if total_steps is None or total_steps == 0:
            total_steps = 1

        active_indices = get_active_indices(
            current_step, total_steps, num_rms, script_args
        )

        # Unified Loading/Unloading Logic
        for i in range(num_rms):
            if i in active_indices:
                # Ensure Loaded
                if reward_models[i] is None:
                    print(f"Loading RM {i} on demand.")
                    reward_model_path = script_args.reward_model_paths[i]
                    reward_tokenizer = reward_tokenizers[i]
                    reward_model = _load_reward_model(
                        reward_model_path, reward_tokenizer, trust_remote_code=True
                    )

                    if controller.trainer.is_deepspeed_enabled:
                        raise ValueError(
                            "DeepSpeed dynamic loading of reward models is not supported."
                        )  # Warning: DeepSpeed dynamic loading might be unstable.

                    reward_models[i] = controller.trainer.accelerator.prepare_model(
                        reward_model, evaluation_mode=True
                    )
            else:
                # Ensure Unloaded
                if reward_models[i] is not None:
                    print(f"Unloading RM {i}")
                    controller.trainer.accelerator.free_memory(reward_models[i])
                    reward_models[i] = None
                    gc.collect()
                    torch.cuda.empty_cache()

        if should_log and wandb.run is not None:
            wandb.log(
                {"reward/active_rm_indices": active_indices},
                step=controller.trainer.state.global_step,
            )

        models_to_process = []
        for i in active_indices:
            models_to_process.append((reward_models[i], reward_tokenizers[i]))

        # --- Step 2: Calculate raw rewards and log ---
        all_rewards_raw = []
        rewards_dict = {}
        for reward_model, reward_tokenizer in models_to_process:
            model_name = reward_model.config._name_or_path
            rew = get_reward(
                reward_model,
                reward_tokenizer,
                prompts,
                completions,
                texts,
                reward_controller=controller,
            )
            rewards_dict[model_name] = rew.detach()

            rew_mean_sum[model_name] += rew.mean().item()
            rew_mean_count[model_name] += 1
            rew_mean_for_model = rew_mean_sum[model_name] / rew_mean_count[model_name]

            if script_args.rm_subtract_mean_reward_per_model:
                rew = rew - rew_mean_for_model
            all_rewards_raw.append(rew)

            if should_log and wandb.run is not None:
                wandb.log(
                    {f"reward/{model_name}": rew_mean_for_model}, step=wandb.run.step
                )

        # --- Step 3: Process and aggregate rewards ---
        processed_rewards = []
        for rew in all_rewards_raw:
            processed_rew = rew
            if script_args.reference_rewards and script_args.adv_rm_lambda == 0:
                processed_rew = processed_rew - reference_rewards
            if script_args.sigmoid_rewards:
                processed_rew = torch.sigmoid(processed_rew)
            processed_rewards.append(processed_rew)

        rewards_tensor = torch.stack(processed_rewards, dim=1)

        if len(active_indices) == 1:
            reward = rewards_tensor.squeeze(1)
        elif script_args.adv_rm_lambda != 0:
            assert (
                rewards_tensor.shape[1] == 2
            ), "Adv-RM requires exactly 2 reward models"
            assert (
                reference_rewards is not None
            ), "Reference rewards must be provided for Adv-RM"
            rewards_above_ref = (
                rewards_tensor[:, 0] - script_args.adv_rm_lambda * rewards_tensor[:, 1]
            )
            rewards_below_ref = rewards_tensor[:, 0] - 25
            reward = torch.where(
                rewards_tensor[:, 0] > reference_rewards,
                rewards_above_ref,
                rewards_below_ref,
            )
        elif script_args.ensemble_aggregation == "mean":
            reward = rewards_tensor.mean(dim=1)
        elif script_args.ensemble_aggregation == "min":
            reward = rewards_tensor.min(dim=1).values
        else:
            raise ValueError(
                f"Unknown ensemble aggregation method: {script_args.ensemble_aggregation}"
            )

        if script_args.penalize_no_eos:
            assert (
                policy_tokenizer is not None
            ), "policy_tokenizer must be provided if penalize_no_eos is True"
            completion_ids = kwargs.get("completion_ids", None)
            assert (
                completion_ids is not None
            ), "completion_ids must be provided if penalize_no_eos is True"

            has_eos_list = []
            for c_ids in completion_ids:
                if isinstance(c_ids, torch.Tensor):
                    c_ids = c_ids.tolist()
                if policy_tokenizer.eos_token_id in c_ids:
                    has_eos_list.append(True)
                else:
                    has_eos_list.append(False)

            has_eos = torch.tensor(has_eos_list, device=reward.device)

            # penalize_no_eos logic:
            # We want to penalize sequences that do NOT have EOS (i.e., truncated/unfinished).
            # We use the sequences that DO have EOS (finished) as the baseline.

            finished_indices = torch.nonzero(has_eos).squeeze(1)
            unfinished_indices = torch.nonzero(~has_eos).squeeze(1)

            if len(finished_indices) > 0 and len(unfinished_indices) > 0:
                min_finished_reward = reward[finished_indices].min()
                # Penalize unfinished to be lower than min_finished_reward
                margin = 1.0
                target_reward = min_finished_reward - margin

                # Only lower them if they are higher
                current_unfinished_rewards = reward[unfinished_indices]
                new_rewards = torch.min(current_unfinished_rewards, target_reward)
                reward[unfinished_indices] = new_rewards

        if controller.k_top_responses > 0:
            # Group rewards by prompt, since prompts are repeated for each completion in a group.
            group_indices = defaultdict(list)
            for i, p in enumerate(prompts):
                group_indices[p].append(i)

            for p, indices in group_indices.items():
                group_rewards = reward[torch.tensor(indices, device=reward.device)]
                top_k_indices_in_group = torch.topk(
                    group_rewards, min(controller.k_top_responses, len(group_rewards))
                ).indices

                for k_idx in top_k_indices_in_group:
                    original_idx = indices[k_idx]
                    ref_rew = (
                        reference_rewards[original_idx].item()
                        if reference_rewards is not None
                        else None
                    )
                    controller.adversarial_responses_buffer.append(
                        (
                            prompts[original_idx],
                            completions[original_idx],
                            reward[original_idx].item(),
                            ref_rew,
                        )
                    )

        if controller.save_path is not None and should_log:
            new_data = {
                "prompt": prompts,
                "completion": completions,
                "reward": reward.tolist(),
            }
            for k, v in kwargs.items():
                if k == "completion_ids":
                    continue
                if isinstance(v, list):
                    new_data[k] = [
                        it.cpu().numpy() if isinstance(it, torch.Tensor) else it
                        for it in v
                    ]
                elif isinstance(v, torch.Tensor):
                    new_data[k] = v.tolist()
                else:
                    new_data[k] = [v] * len(prompts)

            for reward_model in reward_models:
                if (
                    reward_model is not None
                    and reward_model.config._name_or_path in rewards_dict
                ):
                    new_data[f"reward_{reward_model.config._name_or_path}"] = (
                        rewards_dict[reward_model.config._name_or_path].tolist()
                    )
            controller.generations_df = pd.concat(
                [controller.generations_df, pd.DataFrame(new_data)], ignore_index=True
            )
            controller.generations_df.to_csv(controller.save_path, index=False)

        if should_log:
            for model_path in rew_mean_sum.keys():
                rew_mean_sum[model_path] = 0
                rew_mean_count[model_path] = 0
        return reward.tolist()

    return model_reward_func
