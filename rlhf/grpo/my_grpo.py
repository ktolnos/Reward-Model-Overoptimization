import os
from dataclasses import dataclass, field
from typing import Optional, Union, List, Any, Mapping

import qwen35_vllm_patch  # noqa: F401 — must run before any vLLM/TRL code

from accelerate import Accelerator, DeepSpeedPlugin
import torch
from tqdm import tqdm
from accelerate.utils import set_seed
import numpy as np
import pandas as pd
import shutil

from transformers.tokenization_utils_base import (
    TextInput,
    PreTokenizedInput,
    EncodedInput,
    TruncationStrategy,
)
from transformers.utils import PaddingStrategy
from trl.models import prepare_deepspeed

from qrm_gemma_tokenizer import TokenizerWrapper
from data_utils import (
    setup_tokenizer,
    load_policy_and_tokenizer,
    get_generation_stop_token_ids,
    get_length_config,
    set_lengths_from_config,
    compute_max_prompt_length,
    build_train_eval_datasets,
    write_run_manifest,
    DATASET_LENGTH_CONFIGS,
)

tqdm.pandas()
from grpo_utils import (
    post_process_grpo_dataset,
    build_reward_function,
    precompute_reward_statistics,
    RewardController,
)
from online_pet import OnlinePETConfig, OnlinePETCallback

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    TensorType,
    BatchEncoding,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)

from trl import (
    ModelConfig,
    GRPOConfig,
    GRPOTrainer,
    get_kbit_device_map,
    get_peft_config,
)
from peft import get_peft_model
from pathlib import Path


class _MultiTokenEosId(int):
    """An int that compares equal (via ``==``) to any of several stop-token ids.

    Subclasses ``int`` so anything that consumes ``self.eos_token_id`` as a
    plain integer (e.g. ``tokenizer.eos_token_id`` round-tripping, transformers
    ``generate`` arguments, tensor scalar coercion) keeps seeing the primary id.
    The custom ``__eq__`` only changes Python list-membership semantics: when
    ``last_id in [self.eos_token_id, self.pad_token_id]`` is evaluated, Python
    uses reflected dispatch (right operand's type is a subclass of int), calls
    our ``__eq__``, and returns True for any token in the full stop set.

    Note: ``Tensor == _MultiTokenEosId(...)`` does NOT use this override --
    ``Tensor.__eq__`` runs first and coerces to the primary int, so the
    tensor-comparison EOS path (the transformers-generate branch of
    ``GRPOTrainer._generate``, ~line 1320 in TRL 0.29) sees only the primary
    stop id. There is no separate handling of that path. This only matters with
    ``use_vllm=False`` and ``mask_truncated_completions``: a completion ending
    in a *secondary* stop token would be masked as truncated. The default vLLM
    path (list membership, as documented above) is unaffected.
    """

    _stop_set: frozenset

    def __new__(cls, primary_id, all_stop_ids):
        instance = super().__new__(cls, int(primary_id))
        instance._stop_set = frozenset(int(t) for t in all_stop_ids)
        return instance

    def __eq__(self, other):
        try:
            return int(other) in self._stop_set
        except (TypeError, ValueError):
            return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __hash__(self):
        return int.__hash__(self)


class MyGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that recognizes the full stop-token set for EOS/truncation.

    After ``super().__init__`` runs, swap ``self.eos_token_id`` for a
    ``_MultiTokenEosId`` covering every stop token reported by
    ``get_generation_stop_token_ids``. This corrects the
    ``completions/clipped_ratio`` metric and ``mask_truncated_completions``
    behavior for chat-template models where multiple tokens (e.g. ``<|im_end|>``
    and ``<|endoftext|>``) can legitimately terminate a completion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        all_ids = get_generation_stop_token_ids(self.processing_class)
        if len(all_ids) > 1:
            # Multimodal processors (e.g. Gemma4Processor) keep the tokenizer on `.tokenizer`.
            raw_pc = getattr(self.processing_class, "tokenizer", self.processing_class)
            primary = raw_pc.eos_token_id
            if primary not in all_ids:
                raise ValueError(f"Primary EOS token {primary} not found in stop set {all_ids}")
            self.eos_token_id = _MultiTokenEosId(primary, all_ids)
            self._all_stop_token_ids = tuple(sorted(int(t) for t in all_ids))
            print(
                f"[MyGRPOTrainer] eos_token_id covers {sorted(all_ids)}; "
                f"primary={int(self.eos_token_id)}"
            )
        else:
            self._all_stop_token_ids = tuple(int(t) for t in all_ids)

        # Unique prompts the policy sees per global_step — generation cycle
        # produces generation_batch_size completions covering generation_batch_size
        # / num_generations unique prompts, spread over steps_per_generation *
        # num_iterations gradient steps. Lets wandb panels plot any metric vs
        # data consumed, which stays comparable across num_generations changes.
        a = self.args
        self._prompts_per_step = a.generation_batch_size // (
            a.num_generations * a.steps_per_generation * self.num_iterations
        )

    def log(self, logs, start_time=None):
        logs["prompts_consumed"] = self.state.global_step * self._prompts_per_step
        super().log(logs, start_time)


@dataclass
class MyGRPOScriptArguments:
    dataset_path: Optional[str] = field(
        default="", metadata={"help": "training dataset path"}
    )
    dbg: Optional[bool] = field(default=False)
    reward_model_paths: list[str] = field(
        default_factory=lambda: ["google/gemma-2b-it"],
        metadata={"help": "path to the reward model"},
    )
    rm_switch_strategy: Optional[str] = field(
        default="ensemble",
        metadata={
            "help": "Strategy for using multiple reward models. "
            "Options: ensemble, sequential, mix"
        },
    )
    mix_ensemble_size: Optional[int] = field(
        default=2, metadata={"help": "Size of the active ensemble in mix strategy"}
    )
    mix_strategy: Optional[str] = field(
        default="disjoint",
        metadata={"help": "Strategy for mix ensemble. Options: disjoint, sliding, random_disjoint"},
    )
    rm_switches_multiplier: Optional[int] = field(
        default=1, metadata={"help": "Number of times we will use each reward model"}
    )
    sigmoid_rewards: Optional[bool] = field(
        default=False, metadata={"help": "if True, use sigmoid to normalize rewards"}
    )
    reference_rewards: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if True, subtract reference policy rewards during training. sigmoid_rewards + reference_rewards = PAR"
        },
    )
    ensemble_aggregation: Optional[str] = field(
        default="min",
        metadata={
            "help": "how to aggregate rewards from multiple reward models. Options: mean, min, uwo"
        },
    )
    save_generations_path: Optional[str] = field(
        default=None, metadata={"help": "path to save generations and rewards"}
    )
    adv_rm_lambda: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "lambda from Adv-RM paper, 0.0 means no Adv-RM loss. "
            "The loss is r1 - lambda * r2 s.t. r1 > base reward."
        },
    )
    rm_subtract_mean_reward_per_model: Optional[bool] = field(
        default=True,
        metadata={
            "help": "whether to subtract mean reward per model."
        },
    )
    rm_scale_reward_by_std_per_model: Optional[bool] = field(
        default=True,
        metadata={
            "help": "whether to divide by per-model reward std after mean subtraction."
        },
    )
    clip_reward_max: Optional[float] = field(
        default=None,
        metadata={
            "help": "clip normalized per-model rewards to [-clip_reward_max, clip_reward_max]. "
            "Requires rm_subtract_mean_reward_per_model and rm_scale_reward_by_std_per_model."
        },
    )
    penalize_no_eos: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if True, penalize completions that do not contain an EOS token. "
            "Uses soft penalty (DAPO-style) controlled by penalize_no_eos_soft_fraction "
            "and penalize_no_eos_max_penalty."
        },
    )
    penalize_no_eos_soft_fraction: Optional[float] = field(
        default=0.8,
        metadata={
            "help": "Fraction of max_completion_length at which the no-EOS penalty begins "
            "ramping up (soft cap). Penalty linearly increases from 0 at soft_cap to "
            "max_penalty at max_completion_length. Set to 1.0 to revert to hard penalty."
        },
    )
    penalize_no_eos_max_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Maximum penalty subtracted from reward at or beyond max_completion_length."
        },
    )
    penalize_no_eos_power: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Exponent for the penalty ramp between soft_cap and max_completion_length. "
            "1.0 = linear (default, DAPO-style). 2.0 = quadratic (gentler near soft_cap, "
            "steeper near max_len). Penalty is 0 at soft_cap and max_penalty at max_len "
            "regardless of this value."
        },
    )
    uwo_lambda: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "lambda parameter for UWO (Uncertainty-Weighted Optimization). "
            "Controls the penalty for disagreement across reward models. "
            "reward_uwo = mean_reward - lambda * std_reward"
        },
    )
    uwo_use_variance: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, UWO uses variance instead of std: "
            "reward = mean - lambda * var(rewards). "
            "Set True to match Coste et al. (2310.02743) paper formula."
        },
    )
    length_config: Optional[str] = field(
        default="default",
        metadata={
            "help": "Name of the length config from DATASET_LENGTH_CONFIGS. "
            "Controls max_prompt_length, max_completion_length, vllm_max_model_length. "
            "Use 'alpacafarm_paper' for the paper comparison (520/256/776)."
        },
    )
    auto_prompt_length: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Measure actual max prompt length from the dataset with the "
            "policy tokenizer and use it for vLLM memory allocation. "
            "max_completion_length still comes from --length_config."
        },
    )
    skip_length_validation: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Skip token-length validation during dataset processing. "
            "Use when the policy tokenizer differs from the one used to filter the dataset."
        },
    )


if __name__ == "__main__":
    parser = HfArgumentParser(
        (MyGRPOScriptArguments, GRPOConfig, ModelConfig, OnlinePETConfig)
    )
    script_args, training_args, model_args, pet_config = (
        parser.parse_args_into_dataclasses()
    )
    # Apply length config from DATASET_LENGTH_CONFIGS.
    # vLLM memory sizing is deferred until the policy tokenizer is loaded
    # when --auto_prompt_length is set (see Dataset section).
    length_cfg = set_lengths_from_config(
        training_args, script_args.length_config, trainer_type="grpo"
    )

    if script_args.clip_reward_max is not None and (
        not script_args.rm_subtract_mean_reward_per_model
        or not script_args.rm_scale_reward_by_std_per_model
    ):
        raise ValueError(
            "clip_reward_max requires both rm_subtract_mean_reward_per_model "
            "and rm_scale_reward_by_std_per_model to be enabled."
        )

    if pet_config.online_pet_enabled:
        assert (
            len(script_args.reward_model_paths) == 1
        ), "Online PET is currently only supported for a single reward model."

    # Write the run manifest into the checkpoints dir so evaluate_policy.py
    # defaults its config (dataset, training RM, KL base, temperature) to what
    # this run actually used. Explicit eval CLI flags still override it.
    # wandb is initialized here so the run id/name land in the same write; the
    # trainer's WandbCallback reuses an already-active run.
    if os.environ.get("RANK", "0") == "0":
        wandb_fields = {}
        if "wandb" in (training_args.report_to or []):
            import wandb
            if wandb.run is None:
                wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "huggingface"),
                    name=training_args.run_name,
                )
            wandb_fields = {
                "wandb_run_id": wandb.run.id,
                "wandb_run_name": wandb.run.name,
                "wandb_project": wandb.run.project,
                "wandb_url": wandb.run.url,
            }
        write_run_manifest(training_args.output_dir, {
            **wandb_fields,
            "model_name_or_path": model_args.model_name_or_path,
            "dataset_path": script_args.dataset_path,
            "temperature": training_args.temperature,
            "reward_model_paths": list(script_args.reward_model_paths),
            "rm_switch_strategy": script_args.rm_switch_strategy,
            "ensemble_aggregation": script_args.ensemble_aggregation,
            "beta": training_args.beta,
            "length_config": script_args.length_config,
            "max_completion_length": training_args.max_completion_length,
        })

    ################
    # Model & Tokenizer
    ################
    peft_config = get_peft_config(model_args)

    reward_models = []
    reward_tokenizers = []

    # Load all tokenizers first, as they are small and needed for all strategies.
    for reward_model_path in script_args.reward_model_paths:
        tokenizer = AutoTokenizer.from_pretrained(
            reward_model_path,
            trust_remote_code=model_args.trust_remote_code,
        )
        setup_tokenizer(tokenizer)
        if "QRM" in reward_model_path:
            print("wrapping QRM tokenizer")
            tokenizer = TokenizerWrapper(tokenizer, reward_model_path)

        reward_tokenizers.append(tokenizer)

    # We always initialize with None to allow on-demand loading in build_reward_function
    reward_models = [None] * len(script_args.reward_model_paths)

    # Pre-compute per-model reward statistics before loading the policy model (to keep GPU free).
    # Load only the policy tokenizer first for dataset processing.
    policy_tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    setup_tokenizer(policy_tokenizer)
    policy_stop_token_ids = get_generation_stop_token_ids(policy_tokenizer)

    precomputed_statistics = None
    if (script_args.rm_subtract_mean_reward_per_model):
        precomputed_statistics = precompute_reward_statistics(
            reward_model_paths=script_args.reward_model_paths,
            reward_tokenizers=reward_tokenizers,
            dataset_path=script_args.dataset_path,
            output_dir=str(Path(training_args.output_dir).parent),
            trust_remote_code=model_args.trust_remote_code,
        )

    policy, policy_tokenizer = load_policy_and_tokenizer(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Re-point tokenizer.eos_token_id to <|im_end|> when available. Chat-template-
    # trained models almost always emit <|im_end|> as the turn-end token, while base
    # tokenizers leave eos_token_id pointing at <|endoftext|>. TRL reads
    # tokenizer.eos_token_id as a single int in its truncation checks
    # (completions/clipped_ratio, mask_truncated_completions), so picking the
    # token the model actually emits avoids mis-classifying clean completions
    # as truncated. <|endoftext|> remains in get_generation_stop_token_ids so it
    # still counts as a valid stop token everywhere we use the full set.
    im_end_id = policy_tokenizer.convert_tokens_to_ids("<|im_end|>")
    unk_id = getattr(policy_tokenizer, "unk_token_id", None)
    if (
        im_end_id is not None
        and im_end_id != unk_id
        and policy_tokenizer.eos_token_id != im_end_id
    ):
        prev_id = policy_tokenizer.eos_token_id
        prev_token = policy_tokenizer.eos_token
        policy_tokenizer.eos_token_id = im_end_id
        policy_tokenizer.eos_token = "<|im_end|>"
        print(
            f"[my_grpo] Repointing tokenizer.eos_token_id "
            f"{prev_id} ({prev_token!r}) -> {im_end_id} ('<|im_end|>')."
        )
        if hasattr(policy, "generation_config"):
            policy.generation_config.eos_token_id = get_generation_stop_token_ids(
                policy_tokenizer
            )

    policy_stop_token_ids = get_generation_stop_token_ids(policy_tokenizer)

    # Ensure vLLM stops on ALL relevant EOS tokens. vLLM picks up
    # tokenizer.eos_token_id (now <|im_end|>) from the model config; the others
    # (e.g. <|endoftext|>) must be passed explicitly via stop_token_ids.
    extra_stop_ids = [
        tid for tid in policy_stop_token_ids
        if tid != policy_tokenizer.eos_token_id
    ]
    if extra_stop_ids:
        if training_args.generation_kwargs is None:
            training_args.generation_kwargs = {}
        existing = training_args.generation_kwargs.get("stop_token_ids", [])
        merged = list(dict.fromkeys(existing + extra_stop_ids))  # deduplicate, preserve order
        training_args.generation_kwargs["stop_token_ids"] = merged
        print(f"[my_grpo] Injected vLLM stop_token_ids: {merged}")

    ################
    # Dataset
    ################

    # Resolve vLLM memory sizing: auto_prompt_length measures the dataset,
    # otherwise use the hardcoded value from length_config.
    if script_args.auto_prompt_length:
        measured_prompt = compute_max_prompt_length(
            script_args.dataset_path, policy_tokenizer
        )
        training_args.max_prompt_length = measured_prompt
        training_args.vllm_max_model_length = measured_prompt + training_args.max_completion_length
    else:
        training_args.max_prompt_length = length_cfg["max_prompt_tokens"]
        training_args.vllm_max_model_length = length_cfg["max_conversation_tokens"]

    train_dataset, eval_dataset = build_train_eval_datasets(
        script_args.dataset_path,
        policy_tokenizer,
        post_process_fn=post_process_grpo_dataset,
        eval_proportion=0.1,
        size=100 if script_args.dbg else None,
        length_config=script_args.length_config,
        skip_length_validation=script_args.skip_length_validation,
    )
    print(f"Size of the train set: {len(train_dataset)}, eval set: {len(eval_dataset)}")

    for prompt in train_dataset["prompt"][:5]:
        print(f"Sample prompt: \n{prompt}")

    avg_len = np.mean(
        [len(policy_tokenizer.encode(prompt)) for prompt in train_dataset["prompt"]]
    )
    max_len = max(
        [len(policy_tokenizer.encode(prompt)) for prompt in train_dataset["prompt"]]
    )
    print(f"Average length of prompts: {avg_len}, Max length of prompts: {max_len}")

    trainer = None

    reward_controller = RewardController(
        save_path=script_args.save_generations_path,
        k_top_responses=(
            pet_config.k_top_responses if pet_config.online_pet_enabled else 0
        ),
    )
    reward_fn = build_reward_function(
        reward_models,
        reward_tokenizers,
        script_args,
        reward_controller,
        policy_tokenizer,
        precomputed_statistics=precomputed_statistics,
    )

    pet_callback = OnlinePETCallback(
        pet_config=pet_config,
        accelerator=None,  # Will be set by the trainer
        reward_models=reward_models,
        reward_tokenizers=reward_tokenizers,
        reward_controller=reward_controller,
        policy_tokenizer=policy_tokenizer,
        model_name=model_args.model_name_or_path,
    )

    callbacks = []
    if pet_config.online_pet_enabled:
        callbacks.append(pet_callback)
    ################
    # Training
    ################
    trainer = MyGRPOTrainer(
        args=training_args,
        # reward_processing_classes=reward_tokenizer,
        model=policy,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        reward_funcs=reward_fn,
        callbacks=callbacks,
    )
    pet_callback.accelerator = trainer.accelerator
    reward_controller.trainer = trainer

    trainer.train()

    # Save model
    trainer.save_model(training_args.output_dir)
