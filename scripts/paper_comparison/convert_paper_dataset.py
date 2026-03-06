#!/usr/bin/env python3
"""Convert the paper's AlpacaFarm datasets to our messages format.

Converts the following datasets:
1. tlc4418/1.4b-policy_preference_data_gold_labelled (46K preference pairs) -> RM training
2. AlpacaFarm "unlabeled" split (20K prompts) -> GRPO training
3. AlpacaFarm "val" split (2K) -> Evaluation
4. AlpacaFarm "sft" split (10K) -> SFT training

All output in our standard messages format:
  {chosen: [{role: "user", ...}, {role: "assistant", ...}], rejected: [...]}
or for prompt-only splits:
  {chosen: [{role: "user", ...}, {role: "assistant", content: ""}]}
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _combine_instruction_input(instruction: str, input_text: str) -> str:
    """Combine instruction and input fields into a single user message."""
    if input_text and input_text.strip():
        return f"{instruction}\n{input_text}"
    return instruction


def convert_preference_dataset(hf_token: str | None = None) -> DatasetDict:
    """Convert tlc4418/1.4b-policy_preference_data_gold_labelled.

    The dataset has columns: instruction, input, answers (list of 2 strings),
    preference (0-indexed: 0 or 1 indicating which answer is preferred).
    """
    print("Loading preference dataset: tlc4418/1.4b-policy_preference_data_gold_labelled")
    ds = load_dataset(
        "tlc4418/1.4b-policy_preference_data_gold_labelled",
        split="train",
        token=hf_token,
    )
    print(f"  Loaded {len(ds)} rows")

    def convert_pref(example, idx):
        user_content = _combine_instruction_input(
            example["instruction"], example.get("input", "")
        )
        user_msg = {"role": "user", "content": user_content}

        # The tlc4418 dataset stores responses as a list in "answers" and uses
        # 0-indexed preference (0 → first answer preferred, 1 → second preferred).
        answers = example["answers"]
        pref = example["preference"]
        if pref not in (0, 1):
            raise ValueError(f"Invalid preference value {pref} at index {idx}")
        chosen_response = answers[pref]
        rejected_response = answers[1 - pref]

        return {
            "chosen": [user_msg, {"role": "assistant", "content": chosen_response}],
            "rejected": [user_msg, {"role": "assistant", "content": rejected_response}],
        }

    converted = ds.map(
        convert_pref,
        with_indices=True,
        remove_columns=ds.column_names,
        num_proc=8,
    )
    return DatasetDict({"train": converted})


def convert_alpacafarm_split(
    split_name: str,
    *,
    is_prompt_only: bool = False,
    hf_token: str | None = None,
) -> Dataset:
    """Convert an AlpacaFarm split to our messages format.

    For 'sft' split: has instruction, input, output -> single chosen conversation.
    For 'unlabeled'/'val': instruction, input -> prompt-only (with empty assistant).
    """
    print(f"Loading AlpacaFarm split: {split_name}")
    # Must specify the "alpaca_instructions" config; the splits sft/unlabeled/val
    # live there, not in the default config.
    ds = load_dataset("tatsu-lab/alpaca_farm", "alpaca_instructions", split=split_name, token=hf_token)
    print(f"  Loaded {len(ds)} rows")

    def convert_sft(example):
        user_content = _combine_instruction_input(
            example["instruction"], example.get("input", "")
        )
        user_msg = {"role": "user", "content": user_content}
        assistant_msg = {"role": "assistant", "content": example.get("output", "")}
        return {"chosen": [user_msg, assistant_msg]}

    def convert_prompt_only(example):
        user_content = _combine_instruction_input(
            example["instruction"], example.get("input", "")
        )
        user_msg = {"role": "user", "content": user_content}
        # Include a dummy assistant message so the format matches expectations.
        # GRPO only uses the prompt portion (chosen[:-1]).
        assistant_msg = {"role": "assistant", "content": ""}
        return {"chosen": [user_msg, assistant_msg]}

    convert_fn = convert_prompt_only if is_prompt_only else convert_sft
    return ds.map(convert_fn, remove_columns=ds.column_names, num_proc=8)


def main():
    parser = argparse.ArgumentParser(
        description="Convert paper datasets to our messages format and push to HF"
    )
    parser.add_argument(
        "--hf-org",
        default="ktolnos",
        help="HuggingFace org/user namespace for output datasets",
    )
    parser.add_argument("--private", action="store_true", default=False)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Convert locally but don't push to HF Hub",
    )
    args = parser.parse_args()

    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    push_kwargs = {"private": args.private}
    if hf_token:
        push_kwargs["token"] = hf_token

    # 1. Preference dataset (46K) for RM training
    pref_ds = convert_preference_dataset(hf_token=hf_token)
    pref_name = f"{args.hf_org}/alpacafarm_paper_preference_messages"
    print(f"\nPreference dataset: {len(pref_ds['train'])} rows")
    print(f"  Sample: {pref_ds['train'][0]}")
    if not args.dry_run:
        pref_ds.push_to_hub(pref_name, **push_kwargs)
        print(f"  Pushed to {pref_name}")

    # 2. GRPO prompts (unlabeled, 20K)
    unlabeled_ds = convert_alpacafarm_split(
        "unlabeled", is_prompt_only=True, hf_token=hf_token
    )
    unlabeled_dict = DatasetDict({"train": unlabeled_ds})
    grpo_name = f"{args.hf_org}/alpacafarm_paper_grpo_prompts"
    print(f"\nGRPO prompts: {len(unlabeled_ds)} rows")
    print(f"  Sample: {unlabeled_ds[0]}")
    if not args.dry_run:
        unlabeled_dict.push_to_hub(grpo_name, **push_kwargs)
        print(f"  Pushed to {grpo_name}")

    # 3. Eval prompts (val, 2K)
    val_ds = convert_alpacafarm_split("val", is_prompt_only=True, hf_token=hf_token)
    val_dict = DatasetDict({"train": val_ds})
    eval_name = f"{args.hf_org}/alpacafarm_paper_eval_prompts"
    print(f"\nEval prompts: {len(val_ds)} rows")
    print(f"  Sample: {val_ds[0]}")
    if not args.dry_run:
        val_dict.push_to_hub(eval_name, **push_kwargs)
        print(f"  Pushed to {eval_name}")

    # 4. SFT data (sft split, 10K)
    sft_ds = convert_alpacafarm_split("sft", is_prompt_only=False, hf_token=hf_token)
    sft_dict = DatasetDict({"train": sft_ds})
    sft_name = f"{args.hf_org}/alpacafarm_paper_sft_messages"
    print(f"\nSFT data: {len(sft_ds)} rows")
    print(f"  Sample: {sft_ds[0]}")
    if not args.dry_run:
        sft_dict.push_to_hub(sft_name, **push_kwargs)
        print(f"  Pushed to {sft_name}")

    print("\nConversion complete!")
    if args.dry_run:
        print("(Dry run -- nothing was pushed to HF Hub)")


if __name__ == "__main__":
    main()
