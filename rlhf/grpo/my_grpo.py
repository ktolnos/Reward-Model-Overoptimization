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
    reward_model_paths: list[str] = field(default='google/gemma-2b-it', metadata={'help': 'path to the reward model'})
    sigmoid_rewards: Optional[bool] = field(default=False, metadata={'help': 'if True, use sigmoid to normalize rewards'})
    reference_rewards: Optional[bool] = field(default=False, metadata={'help': 'if True, subtract reference policy rewards during training. sigmoid_rewards + reference_rewards = PAR'})
    ensemble_aggregation: Optional[str] = field(default='min',
        metadata={'help': 'how to aggregate rewards from multiple reward models. Options: mean, min'}
    )

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    ################
    # Model & Tokenizer
    ################
    peft_config = get_peft_config(model_args)

    reward_models = []
    reward_tokenizers = []
    for reward_model_path in script_args.reward_model_paths:
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
        reward_tokenizers.append(reward_tokenizer)

        if 'QRM' in reward_model_path:
            reward_tokenizer = TokenizerWrapper(reward_tokenizer)

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


    def prepare_input(data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": reward_models[0].device}
            if trainer.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update({"dtype": torch.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data


    def model_reward_func(prompts, completions, **kwargs):
        texts = [p + c for p, c in zip(prompts, completions)]
        rewards = []
        for reward_model, reward_tokenizer in zip(reward_models, reward_tokenizers):
            reward_inputs = reward_tokenizer(
                text=texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False,
            )
            reward_inputs = prepare_input(reward_inputs)
            with torch.inference_mode():
               reward = reward_model(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            if script_args.reference_rewards:
                raise NotImplementedError("Reference rewards are not implemented yet.")
            if script_args.sigmoid_rewards:
                reward = torch.sigmoid(reward)
            rewards.append(reward)

        rewards = torch.stack(rewards, dim=1)  # Shape (B*G, N)
        if script_args.ensemble_aggregation == 'mean':
            reward = rewards.mean(dim=1)
        elif script_args.ensemble_aggregation == 'min':
            reward = rewards.min(dim=1)
        else:
            raise ValueError(f"Unknown ensemble aggregation method: {script_args.ensemble_aggregation}")
        return reward.tolist()


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
        for reward_model in reward_models:
            prepare_deepspeed(reward_model, trainer.accelerator)
    else:
        for reward_model in reward_models:
            trainer.accelerator.prepare_model(
                reward_model, evaluation_mode=True, device_placement=True
            )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()