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

from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, EncodedInput, TruncationStrategy
from transformers.utils import PaddingStrategy
from trl.models import prepare_deepspeed

from qrm_gemma_tokenizer import TokenizerWrapper

tqdm.pandas()
from grpo_utils import (build_train_eval_datasets, build_reward_function, RewardController)
from online_pet import OnlinePETConfig, OnlinePETCallback

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser, PreTrainedTokenizerBase, TensorType, BatchEncoding,
    TrainerCallback, TrainerState, TrainerControl,
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
    reward_model_paths: list[str] = field(default='google/gemma-2b-it', metadata={'help': 'path to the reward model'})
    sigmoid_rewards: Optional[bool] = field(default=False,
                                            metadata={'help': 'if True, use sigmoid to normalize rewards'})
    reference_rewards: Optional[bool] = field(default=False, metadata={
        'help': 'if True, subtract reference policy rewards during training. sigmoid_rewards + reference_rewards = PAR'})
    ensemble_aggregation: Optional[str] = field(default='min',
                                                metadata={
                                                    'help': 'how to aggregate rewards from multiple reward models. Options: mean, min'}
                                                )
    save_generations_path: Optional[str] = field(default=None, metadata={'help': 'path to save generations and rewards'})
    adv_rm_lambda: Optional[float] = field(default=0.0,
                                           metadata={'help': 'lambda from Adv-RM paper, 0.0 means no Adv-RM loss. '
                                                             'The loss is r1 - lambda * r2 s.t. r1 > base reward.'})
    resume_from_checkpoint: Optional[str] = field(default="", metadata={'help': 'path to checkpoint from which to resume training'})


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, GRPOConfig, ModelConfig, OnlinePETConfig))
    script_args, training_args, model_args, pet_config = parser.parse_args_into_dataclasses()

    if pet_config.online_pet_enabled:
        assert len(script_args.reward_model_paths) == 1, "Online PET is currently only supported for a single reward model."

    ################
    # Model & Tokenizer
    ################
    peft_config = get_peft_config(model_args)

    reward_models = []
    reward_tokenizers = []
    for reward_model_path in script_args.reward_model_paths:
        if 'RRM' in reward_model_path:
            reward_model = AutoModelForCausalLM.from_pretrained(
                reward_model_path,
                trust_remote_code=model_args.trust_remote_code,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        else:
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                reward_model_path, trust_remote_code=model_args.trust_remote_code,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path,
                                                         trust_remote_code=model_args.trust_remote_code,
                                                         padding_side="left",
                                                         )
        if reward_tokenizer.pad_token is None:
            reward_tokenizer.pad_token = reward_tokenizer.eos_token
        reward_tokenizer.max_length = script_args.max_length
        reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

        reward_models.append(reward_model)

        if 'QRM' in reward_model_path:
            print('wrapping QRM tokenizer')
            reward_tokenizer = TokenizerWrapper(reward_tokenizer, reward_model_path)

        reward_tokenizers.append(reward_tokenizer)

    policy = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    policy_tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )

    ################
    # Dataset
    ################

    train_dataset, eval_dataset = build_train_eval_datasets(
        script_args.dataset_path, policy_tokenizer, eval_proportion=0.1, size=100 if script_args.dbg else None,
        max_length=training_args.max_prompt_length,
    )
    print(f"Size of the train set: {len(train_dataset)}, eval set: {len(eval_dataset)}")

    for prompt in train_dataset['prompt'][:5]:
        print(f"Sample prompt: \n{prompt}")

    avg_len = np.mean([len(policy_tokenizer.encode(prompt)) for prompt in train_dataset['prompt']])
    max_len = max([len(policy_tokenizer.encode(prompt)) for prompt in train_dataset['prompt']])
    print(f"Average length of prompts: {avg_len}, Max length of prompts: {max_len}")

    trainer = None

    reward_controller = RewardController(
        save_path=script_args.save_generations_path,
        k_top_responses=pet_config.k_top_responses if pet_config.online_pet_enabled else 0
    )
    reward_fn = build_reward_function(reward_models, reward_tokenizers, script_args, reward_controller)

    pet_callback = OnlinePETCallback(
        pet_config=pet_config,
        accelerator=None,  # Will be set later
        reward_models=reward_models,
        reward_tokenizers=reward_tokenizers,
        reward_controller=reward_controller,
        policy_tokenizer=policy_tokenizer,
        model_name=model_args.model_name_or_path
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
        callbacks=callbacks
    )
    if pet_config.online_pet_enabled:
        pet_callback.set_accelerator(Accelerator()) # we need to create it after the trainer is created
    reward_controller.trainer = trainer

    logging_steps = int(training_args.logging_steps * len(train_dataset))
    print("Logging steps:", logging_steps)
    reward_controller.logging_steps = logging_steps
    if pet_config.online_pet_enabled:
        for i, reward_model in enumerate(reward_models):
            reward_models[i] = pet_callback.accelerator.prepare(reward_model)
    else:
        for i, reward_model in enumerate(reward_models):
            reward_models[i] = trainer.accelerator.prepare_model(reward_model, evaluation_mode=True, device_placement=True)
    resume_from_checkpoint = None
    if script_args.resume_from_checkpoint:
        resume_from_checkpoint = script_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


