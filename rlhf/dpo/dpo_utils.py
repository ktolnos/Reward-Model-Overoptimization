"""DPO dataset utilities.

Provides the DPO-specific post-processing function.  The actual
load-split-postprocess skeleton lives in ``data_utils.build_train_eval_datasets``.
"""

from data_utils import format_and_validate_preference_sample


def post_process_dpo_dataset(ds, tokenizer, *, length_config, skip_length_validation=False):
    """Convert raw preference data to DPO conversational format.

    TRL DPOTrainer (v0.29) with conversational data expects:
      - prompt:   list[dict] -- the user turns
      - chosen:   list[dict] -- the preferred assistant response
      - rejected: list[dict] -- the rejected assistant response

    The trainer auto-applies the tokenizer's chat template internally.
    We still call format_and_validate_preference_sample to enforce length
    constraints (it's a no-op when skip_validation=True).
    """
    def formatting_func(example):
        chosen_messages = example["chosen"]
        rejected_messages = example.get("rejected")

        if rejected_messages is None:
            raise ValueError(
                "DPO requires both 'chosen' and 'rejected' columns in the dataset."
            )

        # Validate format and lengths (shared with SFT/GRPO).
        format_and_validate_preference_sample(
            chosen_messages,
            tokenizer,
            rejected_messages=rejected_messages,
            length_config=length_config,
            skip_validation=skip_length_validation,
            sample_id=example.get("id"),
            context="DPO",
        )

        # Extract prompt (all messages except the last assistant turn).
        prompt_msgs = chosen_messages[:-1]
        chosen_msg = [chosen_messages[-1]]
        rejected_msg = [rejected_messages[-1]]

        return {
            "prompt": prompt_msgs,
            "chosen": chosen_msg,
            "rejected": rejected_msg,
            # Qwen3.5 chat templates default to enable_thinking=True, which
            # causes a tokenization mismatch: add_generation_prompt produces
            # "<think>\n" while the full conversation has "<think>\n\n</think>".
            # Passing enable_thinking=False makes both paths emit the complete
            # empty-think block so the token prefix matches exactly.
            "chat_template_kwargs": {"enable_thinking": False},
        }

    ds = ds.map(
        formatting_func,
        remove_columns=ds.column_names,
        batched=False,
        num_proc=10,
    )
    return ds
