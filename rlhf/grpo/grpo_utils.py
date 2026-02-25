"""GRPO utilities.

W&B metrics emitted by this module:
- reward/<rm_name>:
  Mean normalized reward for each active RM, averaged over steps since the
  previous logging event.
- rewards/ensemble_mean:
  Mean across active RMs (per sample, then batch mean), interval-averaged.
- rewards/ensemble_min:
  Min across active RMs (per sample, then batch mean), interval-averaged.
- rewards/ensemble_max:
  Max across active RMs (per sample, then batch mean), interval-averaged.
- rewards/ensemble_std:
  Std across active RMs (per sample, then batch mean), interval-averaged.
- rewards/ensemble_mean_minus_std:
  (mean - std) across active RMs (per sample, then batch mean),
  interval-averaged.
- rewards/ensemble_range:
  (max - min) across active RMs (per sample, then batch mean),
  interval-averaged.
- rewards/ensemble_active_rms:
  Number of active RMs, interval-averaged.
- rewards/batch_mean:
  Batch-level scalar reward mean after aggregation and post-processing,
  interval-averaged. Different from ensemble_mean when aggregation is not
  "mean" (e.g. "min", "uwo", Adv-RM) or when penalize_no_eos modifies reward.
- rewards/batch_min:
  Batch-level scalar reward min after aggregation and post-processing,
  interval-averaged.
- rewards/batch_max:
  Batch-level scalar reward max after aggregation and post-processing,
  interval-averaged.
- rewards/batch_std:
  Batch-level scalar reward std after aggregation and post-processing,
  interval-averaged.
"""

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
from trl import GRPOTrainer
import wandb

tqdm.pandas()
import matplotlib.pyplot as plt
import json
from datetime import datetime
from reward_utils import (
    get_reward,
    get_reward_rm,
    build_reward_texts,
    is_reasoning,
    load_reward_model,
)
from data_utils import (
    format_and_validate_preference_sample,
    completion_has_stop_token,
    get_generation_stop_token_ids,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_MAX_CONVERSATION_TOKENS,
)
import math

REWARD_STATISTICS_CACHE_VERSION = 1
RM_STD_EPS = 1e-6


@dataclass
class RewardController:
    trainer: GRPOTrainer = None
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
    max_prompt_length=DEFAULT_MAX_PROMPT_TOKENS,
):
    ds = datasets.load_dataset(data_path_train, split="train")
    if size is not None:
        ds = ds.select(range(0, size))
    ds_dict = ds.train_test_split(test_size=eval_proportion, seed=42)
    ds_train = ds_dict["train"]
    ds_eval = ds_dict["test"]
    ds_train = post_process_common_dataset(ds_train, tokenizer, max_prompt_length)
    ds_eval = post_process_common_dataset(ds_eval, tokenizer, max_prompt_length)
    return ds_train, ds_eval


def post_process_common_dataset(ds, tokenizer, max_prompt_length=None):
    max_prompt_length = (
        DEFAULT_MAX_PROMPT_TOKENS if max_prompt_length is None else max_prompt_length
    )

    def formatting_func(example):
        # Keep structured messages for reward model formatting
        # chosen contains [User, Assistant] (usually). Strip the last message if it's the assistant's.
        prompt_msgs = example["chosen"]
        if prompt_msgs and prompt_msgs[-1]["role"] == "assistant":
            prompt_msgs = prompt_msgs[:-1]

        prompt, _, _ = format_and_validate_preference_sample(
            example["chosen"],
            tokenizer,
            rejected_messages=example.get("rejected"),
            max_prompt_length=max_prompt_length,
            max_conversation_length=DEFAULT_MAX_CONVERSATION_TOKENS,
            context="GRPO",
        )
        return {
            "prompt": prompt,
            "prompt_messages": prompt_msgs,
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
    """Load reward/reasoning model via the shared reward_utils loader."""
    reasoning = "RRM" in model_path
    model, _ = load_reward_model(
        model_path,
        reasoning=reasoning,
        tokenizer=tokenizer,
        trust_remote_code=trust_remote_code,
        device=None,
        use_device_map=False,
    )
    return model

# Important: use for logging only, the values are reset after logging to represent the current mean accurately. Do not use for calculations.
rew_mean_sum = defaultdict(float)
rew_mean_count = defaultdict(int)
ensemble_metric_sum = defaultdict(float)
ensemble_metric_count = defaultdict(int)
_last_logged_step = -1  # tracks which global_step we last logged+reset at
_reward_buffer = []  # raw rewards buffered across micro-batches within a global step
_prev_batch_step = -1  # tracks which global_step the buffer belongs to


def _accumulate_metric(metric_sum, metric_count, metric_name, metric_value):
    metric_sum[metric_name] += float(metric_value)
    metric_count[metric_name] += 1


def _log_mean_metric(metric_sum, metric_count, metric_name, step):
    if metric_count[metric_name] == 0:
        return
    wandb.log(
        {metric_name: metric_sum[metric_name] / metric_count[metric_name]},
        step=step,
    )


def _reset_metric_buffers(metric_sum, metric_count):
    for metric_name in list(metric_sum.keys()):
        metric_sum[metric_name] = 0.0
        metric_count[metric_name] = 0


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

        elif args.mix_strategy == "random_disjoint":
            # Random selection of ensemble_size models
            # Number of unique random ensembles to cycle through
            num_chunks = math.ceil(num_rms / ensemble_size)
            chunk_idx = (
                (current_step * num_chunks * args.rm_switches_multiplier) // total_steps
            ) % num_chunks

            # Use chunk_idx as seed for reproducible random selection
            rng = np.random.RandomState(chunk_idx)
            indices = rng.choice(num_rms, size=ensemble_size, replace=False).tolist()
            return sorted(indices)

    return list(range(num_rms))  # Default fallback


def precompute_reward_statistics(
    reward_model_paths,
    reward_tokenizers,
    dataset_path,
    output_dir,
    sample_size=1000,
    batch_size=64,
    trust_remote_code=True,
):
    """Pre-compute per-model reward statistics on a fixed dataset sample.

    Loads the raw dataset (before post-processing removes chosen/rejected),
    samples items, scores both chosen and rejected completions with each
    reward model, and returns per-model mean/std values. Results are cached to disk.
    """
    cache_dir = os.path.join(output_dir, "reward_statistics")
    os.makedirs(cache_dir, exist_ok=True)

    # First pass: check cache for all models, identify which need computation.
    precomputed_statistics = {}
    uncached = []  # list of (index, model_path, cache_file)

    for i, model_path in enumerate(reward_model_paths):
        safe_name = model_path.replace("/", "_").replace("\\", "_").replace(":", "_")
        if len(safe_name) > 500:
            safe_name = safe_name[-500:]
        cache_file = os.path.join(cache_dir, f"{safe_name}.json")

        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cached = json.load(f)

            cache_version = cached.get("version")
            if cache_version != REWARD_STATISTICS_CACHE_VERSION:
                print(
                    f"[PrecomputeStats] RM {i} ({model_path}): stale cache version "
                    f"{cache_version!r} != {REWARD_STATISTICS_CACHE_VERSION}; recomputing."
                )
                uncached.append((i, model_path, cache_file))
                continue

            cached_dataset = cached.get("dataset_path")
            if cached_dataset != dataset_path:
                print(
                    f"[PrecomputeStats] RM {i} ({model_path}): cache dataset mismatch "
                    f"({cached_dataset!r} != {dataset_path!r}); recomputing."
                )
                uncached.append((i, model_path, cache_file))
                continue

            if "mean_reward" not in cached or "std_reward" not in cached:
                print(
                    f"[PrecomputeStats] RM {i} ({model_path}): cache missing mean/std; recomputing."
                )
                uncached.append((i, model_path, cache_file))
                continue

            precomputed_statistics[model_path] = {
                "mean_reward": float(cached["mean_reward"]),
                "std_reward": float(cached["std_reward"]),
            }
            print(
                f"[PrecomputeStats] RM {i} ({model_path}): loaded cached "
                f"mean={cached['mean_reward']:.4f}, std={cached['std_reward']:.4f}"
            )
        else:
            uncached.append((i, model_path, cache_file))

    if not uncached:
        print(
            f"[PrecomputeStats] All {len(reward_model_paths)} models cached, skipping dataset load"
        )
        return precomputed_statistics

    # Load raw dataset and build (prompt, completion) pairs only if needed
    print(
        f"[PrecomputeStats] {len(uncached)} models need computation, loading dataset..."
    )
    raw_ds = datasets.load_dataset(dataset_path, split="train")
    n = min(sample_size, len(raw_ds))
    sample = raw_ds.shuffle(seed=42).select(range(n))

    all_conversations = []
    for item in sample:
        # Add ground truth 'chosen' and 'rejected' (if it exists)
        all_conversations.append(item["chosen"])
        if "rejected" in item and item["rejected"]:
            all_conversations.append(item["rejected"])

    print(
        f"[PrecomputeStats] Collected {len(all_conversations)} full conversations "
        f"from {n} dataset items"
    )

    # Second pass: compute statistics for uncached models.
    for i, model_path, cache_file in uncached:
        print(f"[PrecomputeStats] RM {i} ({model_path}): computing...")
        reward_tokenizer = reward_tokenizers[i]
        reward_model = _load_reward_model(
            model_path, reward_tokenizer, trust_remote_code
        )
        reward_model = reward_model.cuda().eval()

        # Reasoning models produce zero-mean BT scores by construction
        if is_reasoning(reward_model):
            precomputed_statistics[model_path] = {
                "mean_reward": 0.0,
                "std_reward": 1.0,
            }
            print(
                f"[PrecomputeStats] RM {i} is reasoning model; "
                f"setting mean=0.0, std=1.0"
            )
            del reward_model
            gc.collect()
            torch.cuda.empty_cache()

            cache_data = {
                "version": REWARD_STATISTICS_CACHE_VERSION,
                "model_path": model_path,
                "mean_reward": 0.0,
                "std_reward": 1.0,
                "num_samples": 0,
                "dataset_path": dataset_path,
                "computed_at": datetime.now().isoformat(),
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            continue

        all_rewards_for_model = []
        for j in range(0, len(all_conversations), batch_size):
            batch = all_conversations[j : j + batch_size]

            # Format and validate full conversations with the RM tokenizer.
            reward_texts = []
            for sample_id, conv in enumerate(batch):
                _, full_text, _ = format_and_validate_preference_sample(
                    conv,
                    reward_tokenizer,
                    max_prompt_length=None,
                    max_conversation_length=None,
                    sample_id=sample_id,
                    context="GRPO RM statistics precompute",
                )
                reward_texts.append(full_text)

            scores = get_reward_rm(
                reward_model, reward_tokenizer, reward_texts, batch_size=batch_size
            ).cpu().float().numpy()
            all_rewards_for_model.extend(scores)

        mean_reward = float(np.mean(all_rewards_for_model))
        std_reward = float(np.std(all_rewards_for_model))
        if std_reward < RM_STD_EPS:
            print(
                f"[PrecomputeStats] RM {i} ({model_path}) has tiny std={std_reward:.8f}; "
                "using std=1.0 to avoid unstable scaling."
            )
            std_reward = 1.0

        precomputed_statistics[model_path] = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
        }

        print(
            f"[PrecomputeStats] RM {i} mean={mean_reward:.4f}, std={std_reward:.4f} "
            f"(n={len(all_rewards_for_model)}), cached to {cache_file}"
        )

        # Save cache
        cache_data = {
            "version": REWARD_STATISTICS_CACHE_VERSION,
            "model_path": model_path,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "num_samples": len(all_rewards_for_model),
            "dataset_path": dataset_path,
            "computed_at": datetime.now().isoformat(),
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)

        # Unload model
        del reward_model
        gc.collect()
        torch.cuda.empty_cache()

    return precomputed_statistics


def build_reward_function(
    reward_models,
    reward_tokenizers,
    script_args,
    controller: RewardController,
    policy_tokenizer=None,
    precomputed_statistics=None,
):
    def model_reward_func(prompts, completions, **kwargs):
        global rew_mean_sum, rew_mean_count, ensemble_metric_sum, ensemble_metric_count, _last_logged_step, _reward_buffer, _prev_batch_step
        current_global_step = controller.trainer.state.global_step
        should_log = (
            current_global_step > 0
            and current_global_step % controller.trainer.state.logging_steps == 0
            and _last_logged_step != current_global_step
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

        models_to_process = []
        for i in active_indices:
            models_to_process.append((reward_models[i], reward_tokenizers[i]))

        # --- Step 2: Calculate raw rewards and log ---
        all_rewards_raw = []
        rewards_dict = {}
        n_clipped = 0
        n_total = 0

        # Parallel streams in main thread
        streams = []
        pending_results = []
        for rm, rt in models_to_process:
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                result = get_reward(
                    rm,
                    rt,
                    prompts,
                    completions,
                    reward_controller=controller,
                    prompt_messages=kwargs.get("prompt_messages"),
                )
            streams.append(stream)
            pending_results.append((result, rm.config._name_or_path))

        # Synchronize all streams
        for stream in streams:
            stream.synchronize()
        results = pending_results

        for rew, model_name in results:
            rewards_dict[model_name] = rew.detach()
            model_metric_name = f"reward/{model_name}"

            # Normalize per-model rewards using fixed pre-computed statistics.
            # Important for "min" and "uwo", where raw offsets/scales can bias aggregation.
            if precomputed_statistics is not None:
                if model_name not in precomputed_statistics:
                    raise KeyError(
                        f"Missing precomputed statistics for {model_name}. "
                        f"Available keys: {list(precomputed_statistics.keys())}"
                    )
                stats = precomputed_statistics[model_name]
                rew = rew - stats["mean_reward"]
                if script_args.rm_scale_reward_by_std_per_model:
                    std_reward = float(stats["std_reward"])
                    if std_reward < RM_STD_EPS:
                        std_reward = 1.0
                    rew = rew / std_reward

            if script_args.clip_reward_max is not None:
                clip_val = script_args.clip_reward_max
                n_clipped += (rew > clip_val).sum().item()
                n_total += rew.numel()
                rew = torch.min(rew, torch.tensor(clip_val, device=rew.device))

            all_rewards_raw.append(rew)

            _accumulate_metric(
                rew_mean_sum,
                rew_mean_count,
                model_metric_name,
                rew.mean().item(),
            )

            if should_log and wandb.run is not None:
                _log_mean_metric(
                    rew_mean_sum,
                    rew_mean_count,
                    model_metric_name,
                    step=current_global_step,
                )

        if n_total > 0:
            _accumulate_metric(
                ensemble_metric_sum, ensemble_metric_count,
                "rewards/clipped_pct", n_clipped / n_total * 100,
            )
            if should_log and wandb.run is not None:
                _log_mean_metric(
                    ensemble_metric_sum, ensemble_metric_count,
                    "rewards/clipped_pct", step=current_global_step,
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
        per_step_ensemble_mean = rewards_tensor.mean(dim=1)
        per_step_ensemble_min = rewards_tensor.min(dim=1).values
        per_step_ensemble_max = rewards_tensor.max(dim=1).values
        per_step_ensemble_std = rewards_tensor.std(dim=1, unbiased=False)
        per_step_ensemble_metrics = {
            "rewards/ensemble_mean": per_step_ensemble_mean.mean().item(),
            "rewards/ensemble_min": per_step_ensemble_min.mean().item(),
            "rewards/ensemble_max": per_step_ensemble_max.mean().item(),
            "rewards/ensemble_std": per_step_ensemble_std.mean().item(),
            "rewards/ensemble_mean_minus_std": (
                per_step_ensemble_mean - per_step_ensemble_std
            ).mean().item(),
            "rewards/ensemble_range": (
                per_step_ensemble_max - per_step_ensemble_min
            ).mean().item(),
            "rewards/ensemble_active_rms": float(len(active_indices)),
        }
        for metric_name, metric_value in per_step_ensemble_metrics.items():
            _accumulate_metric(
                ensemble_metric_sum,
                ensemble_metric_count,
                metric_name,
                metric_value,
            )
        if should_log and wandb.run is not None:
            for metric_name in per_step_ensemble_metrics.keys():
                _log_mean_metric(
                    ensemble_metric_sum,
                    ensemble_metric_count,
                    metric_name,
                    step=current_global_step,
                )

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
        elif script_args.ensemble_aggregation == "uwo":
            # Uncertainty-Weighted Optimization (UWO) from Coste et al. (2310.02743)
            # r_UWO = mean - lambda * std
            # Penalizes high disagreement across reward models
            mean_reward = rewards_tensor.mean(dim=1)
            std_reward = rewards_tensor.std(dim=1, unbiased=False)
            reward = mean_reward - script_args.uwo_lambda * std_reward
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
            stop_token_ids = get_generation_stop_token_ids(policy_tokenizer)

            has_eos_list = []
            for c_ids in completion_ids:
                has_eos_list.append(
                    completion_has_stop_token(
                        c_ids,
                        stop_token_ids=stop_token_ids,
                    )
                )

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

        # Flush completed step's rewards and compute true batch stats
        if current_global_step != _prev_batch_step and _prev_batch_step >= 0 and len(_reward_buffer) > 0:
            all_step_rewards = torch.cat(_reward_buffer)
            batch_stats = {
                "rewards/batch_mean": all_step_rewards.mean().item(),
                "rewards/batch_min": all_step_rewards.min().item(),
                "rewards/batch_max": all_step_rewards.max().item(),
                "rewards/batch_std": all_step_rewards.std(unbiased=False).item(),
            }
            for metric_name, metric_value in batch_stats.items():
                _accumulate_metric(
                    ensemble_metric_sum, ensemble_metric_count, metric_name, metric_value
                )
            _reward_buffer.clear()

        _reward_buffer.append(reward.detach().cpu())
        _prev_batch_step = current_global_step

        _batch_metric_keys = ["rewards/batch_mean", "rewards/batch_min", "rewards/batch_max", "rewards/batch_std"]
        if should_log and wandb.run is not None:
            for metric_name in _batch_metric_keys:
                _log_mean_metric(
                    ensemble_metric_sum, ensemble_metric_count, metric_name, step=current_global_step,
                )

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
            _reset_metric_buffers(rew_mean_sum, rew_mean_count)
            _reset_metric_buffers(ensemble_metric_sum, ensemble_metric_count)
            _last_logged_step = current_global_step
        return reward.tolist()

    return model_reward_func
