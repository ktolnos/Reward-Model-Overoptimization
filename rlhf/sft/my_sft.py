from dataclasses import dataclass, field
from typing import Optional
import torch
import datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)

from data_utils import (
    format_and_validate_preference_sample,
    tokenize_for_sft,
    setup_tokenizer,
    load_policy_and_tokenizer,
    get_length_config,
    DATASET_LENGTH_CONFIGS,
)

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
)

@dataclass
class ScriptArguments:
    length_config: str = field(metadata={
        'help': f'Name of the length config from DATASET_LENGTH_CONFIGS. Available: {list(DATASET_LENGTH_CONFIGS.keys())}'
    })
    dataset_path: Optional[str] = field(default='', metadata={'help': 'training dataset path'})
    dbg: Optional[bool] = field(default=False)

def build_train_eval_datasets(data_path_train, tokenizer, *, length_config, eval_proportion, size=None):
    ds = datasets.load_dataset(data_path_train, split="train")
    if size is not None:
        ds = ds.select(range(0, size))
    ds_dict = ds.train_test_split(test_size=eval_proportion, seed=42)
    ds_train = ds_dict['train']
    ds_eval = ds_dict['test']
    ds_train = post_process_common_dataset(ds_train, tokenizer, length_config=length_config)
    ds_eval = post_process_common_dataset(ds_eval, tokenizer, length_config=length_config)
    return ds_train, ds_eval


def build_dataset_common(data_path, tokenizer, *, length_config, split='', size=None):
    ds = datasets.load_dataset(data_path, split=split)

    if size is not None:
        ds = ds.select(range(0, size))

    ds = post_process_common_dataset(ds, tokenizer, length_config=length_config)
    return ds

def post_process_common_dataset(ds, tokenizer, *, length_config):
    def formatting_func(example):
        chosen_messages = example["chosen"]

        prompt_text, full_text, _ = format_and_validate_preference_sample(
            chosen_messages,
            tokenizer,
            length_config=length_config,
            sample_id=example.get("id"),
            context="SFT",
        )

        tokens_full = tokenize_for_sft(full_text, tokenizer)
        input_ids = tokens_full["input_ids"][0]
        # Use prompt token count as the mask boundary. Tokenizing the prompt
        # separately may produce slightly different tokens at the join point
        # (BPE boundary effect), but the count is correct or off by at most 1
        # token — negligible for training. The text content is already
        # validated by format_and_validate_preference_sample.
        prompt_ids = tokenize_for_sft(prompt_text, tokenizer)["input_ids"][0]
        prompt_len = len(prompt_ids)
        if prompt_len >= len(input_ids):
            raise ValueError(
                f"Invalid sample: prompt_len ({prompt_len}) must be smaller than full sequence length ({len(input_ids)})."
            )
        completion_mask = torch.zeros_like(input_ids)
        completion_mask[prompt_len:] = 1

        return {
            "input_ids": input_ids.tolist(),
            "completion_mask": completion_mask.tolist(),
        }

    ds = ds.map(formatting_func,
                remove_columns=ds.column_names,
                batched=False, num_proc=10)
    return ds

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    ################
    # Model & Tokenizer
    ################
    peft_config = get_peft_config(model_args)
    model, tokenizer = load_policy_and_tokenizer(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer.padding_side = "right"  # SFTTrainer requires right padding to avoid fp16 overflow

    # Enforce a chat template to avoid formatting drift.
    if tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer '{model_args.model_name_or_path}' has no chat_template. "
            "SFT requires a tokenizer with a chat template to keep "
            "formatting consistent across the pipeline."
        )
    
    ################
    # Dataset
    ################
    train_dataset, eval_dataset = build_train_eval_datasets(
        script_args.dataset_path, tokenizer,
        length_config=script_args.length_config,
        eval_proportion=0.1,
        size=100 if script_args.dbg else None
    )
    print(f"Size of the train set: {len(train_dataset)}, eval set: {len(eval_dataset)}")
    
    ################
    # Training
    ################
    if training_args.completion_only_loss is False:
        raise ValueError("SFT requires completion_only_loss=True to train only on the last chosen response.")
    training_args.completion_only_loss = True
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    # Train the model
    trainer.train()
    
    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()
