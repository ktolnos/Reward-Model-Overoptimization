"""Direct Preference Optimization (DPO) training script.

Supports standard DPO (sigmoid), APO-zero, APO-down, IPO, and other loss
variants available in TRL's DPOTrainer.

Usage:
    accelerate launch rlhf/dpo/my_dpo.py \
        --model_name_or_path <sft_checkpoint> \
        --dataset_path <preference_dataset> \
        --loss_type sigmoid \
        --beta 0.1 \
        ...

For APO-zero (recommended by HuggingFace for SmolLM3):
    --loss_type apo_zero --beta 0.05
"""

from dataclasses import dataclass, field
from typing import Optional

from data_utils import (
    setup_tokenizer,
    load_policy_and_tokenizer,
    build_train_eval_datasets,
    set_lengths_from_config,
    DATASET_LENGTH_CONFIGS,
)
from dpo_utils import post_process_dpo_dataset

from transformers import HfArgumentParser

from trl import (
    ModelConfig,
    DPOConfig,
    DPOTrainer,
    get_peft_config,
)


@dataclass
class ScriptArguments:
    dataset_path: Optional[str] = field(
        default="", metadata={"help": "training dataset path"}
    )
    length_config: str = field(
        default="default",
        metadata={
            "help": f"Name of the length config from DATASET_LENGTH_CONFIGS. "
            f"Available: {list(DATASET_LENGTH_CONFIGS.keys())}"
        },
    )
    skip_length_validation: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Skip token-length validation during dataset processing. "
            "Use when the policy tokenizer differs from the one used to filter the dataset."
        },
    )
    dbg: Optional[bool] = field(default=False)


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # Derive max_length from the length config (raises if --max_length
    # was set on the CLI to a conflicting value).
    set_lengths_from_config(
        training_args, script_args.length_config, trainer_type="dpo"
    )

    ################
    # Model & Tokenizer
    ################
    peft_config = get_peft_config(model_args)
    model, tokenizer = load_policy_and_tokenizer(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    if tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer '{model_args.model_name_or_path}' has no chat_template. "
            "DPO requires a tokenizer with a chat template to keep "
            "formatting consistent across the pipeline."
        )

    ################
    # Dataset
    ################
    train_dataset, eval_dataset = build_train_eval_datasets(
        script_args.dataset_path,
        tokenizer,
        post_process_fn=post_process_dpo_dataset,
        dedupe_by_prompt=False,  # DPO's signal IS the per-prompt response pairs
        eval_proportion=0.1,
        size=100 if script_args.dbg else None,
        length_config=script_args.length_config,
        skip_length_validation=script_args.skip_length_validation,
    )
    print(f"Size of the train set: {len(train_dataset)}, eval set: {len(eval_dataset)}")

    ################
    # Training
    ################
    # DPOTrainer creates the reference model automatically from the initial
    # model weights when ref_model=None.  Setting precompute_ref_log_probs=True
    # computes reference log-probs once before training, avoiding keeping a
    # second copy of the model in GPU memory.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # Save model
    trainer.save_model(training_args.output_dir)
