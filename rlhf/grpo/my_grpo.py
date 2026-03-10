import os
from dataclasses import dataclass, field
from typing import Optional, Union, List, Any, Mapping
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
    DATASET_LENGTH_CONFIGS,
)

tqdm.pandas()
from grpo_utils import (
    build_train_eval_datasets,
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
            "help": "if True, penalize completions that do not contain an EOS token"
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


if __name__ == "__main__":
    parser = HfArgumentParser(
        (MyGRPOScriptArguments, GRPOConfig, ModelConfig, OnlinePETConfig)
    )
    script_args, training_args, model_args, pet_config = (
        parser.parse_args_into_dataclasses()
    )
    # Apply length config from DATASET_LENGTH_CONFIGS.
    length_cfg = get_length_config(script_args.length_config)
    training_args.max_prompt_length = length_cfg["max_prompt_tokens"]
    if training_args.max_completion_length != 256:
        raise ValueError(
            f"max_completion_length is overridden on the command line. "
            f"Use --length_config instead (active config '{script_args.length_config}' "
            f"sets max_response_tokens={length_cfg['max_response_tokens']})."
        )
    training_args.max_completion_length = length_cfg["max_response_tokens"]
    if training_args.vllm_max_model_length is not None:
        raise ValueError(
            f"vllm_max_model_length is overridden on the command line. "
            f"Use --length_config instead (active config '{script_args.length_config}' "
            f"sets max_conversation_tokens={length_cfg['max_conversation_tokens']})."
        )
    training_args.vllm_max_model_length = length_cfg["max_conversation_tokens"]

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
    policy_stop_token_ids = get_generation_stop_token_ids(policy_tokenizer)

    ################
    # Dataset
    ################

    train_dataset, eval_dataset = build_train_eval_datasets(
        script_args.dataset_path,
        policy_tokenizer,
        eval_proportion=0.1,
        size=100 if script_args.dbg else None,
        length_config=script_args.length_config,
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
    trainer = GRPOTrainer(
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
