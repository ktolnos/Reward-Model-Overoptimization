from dataclasses import dataclass, field
from typing import Optional
import os
import torch
from transformers import HfArgumentParser

from data_utils import (
    format_and_validate_preference_sample,
    tokenize_for_sft,
    setup_tokenizer,
    load_policy_and_tokenizer,
    build_train_eval_datasets,
    set_lengths_from_config,
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
    skip_length_validation: Optional[bool] = field(
        default=False,
        metadata={
            'help': 'Skip token-length validation. Use when the SFT tokenizer differs '
                    'from the one used to filter the dataset.'
        },
    )

def post_process_sft_dataset(ds, tokenizer, *, length_config, skip_length_validation=False):
    def formatting_func(example):
        chosen_messages = example["chosen"]

        prompt_text, full_text, _ = format_and_validate_preference_sample(
            chosen_messages,
            tokenizer,
            length_config=length_config,
            skip_validation=skip_length_validation,
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

    # Run manifest + provenance, written before anything can fail, so this run
    # and its slurm job stay recoverable from the checkpoints dir alone. GRPO
    # records that dir as model_name_or_path, so eval links back here as
    # related/base_policy. wandb is initialized here rather than by the trainer's
    # WandbCallback so the run id lands in the same write.
    if os.environ.get("RANK", "0") == "0":
        from data_utils import write_run_manifest
        from run_provenance import (
            attach_to_wandb, manifest_slurm_fields, slurm_fields,
            wandb_manifest_fields,
        )
        _wandb_fields = {}
        if "wandb" in (training_args.report_to or []):
            import wandb
            if wandb.run is None:
                wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "huggingface"),
                    name=training_args.run_name,
                )
            _wandb_fields = wandb_manifest_fields()
            attach_to_wandb(slurm_fields())
        write_run_manifest(training_args.output_dir, {
            **_wandb_fields,
            **manifest_slurm_fields(),
            "component": "sft",
            "model_name_or_path": model_args.model_name_or_path,
            "dataset_path": script_args.dataset_path,
            "length_config": script_args.length_config,
        })

    
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
    set_lengths_from_config(
        training_args, script_args.length_config, trainer_type="sft"
    )
    train_dataset, eval_dataset = build_train_eval_datasets(
        script_args.dataset_path, tokenizer,
        post_process_fn=post_process_sft_dataset,
        dedupe_by_prompt=True,
        length_config=script_args.length_config,
        eval_proportion=0.1,
        size=100 if script_args.dbg else None,
        skip_length_validation=script_args.skip_length_validation,
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
