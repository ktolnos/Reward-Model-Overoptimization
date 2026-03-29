#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Ensure repo-root imports work even when this script is launched directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils import (
    format_and_validate_preference_sample,
    get_length_config,
    setup_tokenizer,
    tokenize_for_rm,
)
from reward_utils import load_reward_model
from scripts.dataset_pipeline.pipeline_common import (
    clear_hf_dataset_cache,
    ensure_dataset_dict,
    validate_preference_example_structure,
)


def _parse_args() -> argparse.Namespace:
    _default_cfg = get_length_config("default")
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3: annotate a filtered preference dataset with reward model scores "
            "and upload the annotated dataset to Hugging Face."
        )
    )
    parser.add_argument("--source-dataset", required=True, help="HF filtered dataset")
    parser.add_argument("--output-dataset", required=True, help="HF annotated dataset")
    parser.add_argument(
        "--reward-model",
        default="",
        help="Reward model name/path used for annotation (required unless --skip-annotation)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        default="",
        help="Device for reward model, e.g. cuda, cuda:0, cpu. Default: auto",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create destination dataset as private",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=_default_cfg["max_prompt_tokens"],
        help="Validation max prompt tokens",
    )
    parser.add_argument(
        "--max-conversation-tokens",
        type=int,
        default=_default_cfg["max_conversation_tokens"],
        help="Validation max conversation tokens",
    )
    parser.add_argument(
        "--validation-tokenizer-name",
        default="",
        help=(
            "Tokenizer used only for strict length validation. "
            "If omitted, Stage 3 uses the reward model tokenizer."
        ),
    )
    parser.add_argument(
        "--skip-annotation",
        action="store_true",
        default=False,
        help=(
            "Skip gold RM annotation entirely.  The dataset is passed through "
            "with original chosen/rejected ordering.  Columns chosen_reward, "
            "rejected_reward, and does_gold_agree_with_original are not added.  "
            "--reward-model is not required when this flag is set."
        ),
    )
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading validation tokenizer (default).",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code when loading validation tokenizer.",
    )
    return parser.parse_args()


def _annotate_split(
    split_name: str,
    split_data,
    reward_model,
    reward_tokenizer,
    validation_tokenizer,
    rm_device,
    *,
    batch_size: int,
    max_prompt_tokens: int,
    max_conversation_tokens: int,
) -> Dataset:
    if len(split_data) == 0:
        return split_data

    results: list[dict] = []

    for start in tqdm(
        range(0, len(split_data), batch_size),
        desc=f"Annotating {split_name}",
    ):
        batch = split_data[start : start + batch_size]
        batch_size_actual = len(batch["chosen"])

        all_texts: list[str] = []
        batch_original_rows: list[dict] = []

        for offset in range(batch_size_actual):
            global_idx = start + offset
            sample = {col: batch[col][offset] for col in batch.keys()}

            validate_preference_example_structure(
                sample,
                split_name=split_name,
                idx=global_idx,
            )

            # Re-run strict length validation with the configured validation tokenizer.
            format_and_validate_preference_sample(
                sample["chosen"],
                validation_tokenizer,
                rejected_messages=sample["rejected"],
                length_config="default",
                sample_id=global_idx,
                context=f"Stage3-{split_name}",
            )

            # Use the shared formatter for RM texts as well to avoid formatting drift.
            _, chosen_text, rejected_text = format_and_validate_preference_sample(
                sample["chosen"],
                reward_tokenizer,
                rejected_messages=sample["rejected"],
                length_config="default",
                skip_validation=True,
                sample_id=global_idx,
                context=f"Stage3RMFormat-{split_name}",
            )
            if rejected_text is None:
                raise ValueError(
                    f"Stage3RMFormat-{split_name} missing rejected text "
                    f"(sample_id={global_idx})."
                )

            all_texts.append(chosen_text)
            all_texts.append(rejected_text)
            batch_original_rows.append(sample)

        inputs = tokenize_for_rm(all_texts, reward_tokenizer).to(rm_device)

        with torch.no_grad():
            outputs = reward_model(**inputs)
            all_rewards = outputs.logits.squeeze(-1).cpu().float().numpy()

        for offset, sample in enumerate(batch_original_rows):
            chosen_idx = offset * 2
            rejected_idx = chosen_idx + 1

            chosen_reward = float(all_rewards[chosen_idx])
            rejected_reward = float(all_rewards[rejected_idx])

            does_gold_agree_with_original = chosen_reward > rejected_reward

            annotated = dict(sample)
            if does_gold_agree_with_original:
                annotated["chosen"] = sample["chosen"]
                annotated["rejected"] = sample["rejected"]
            else:
                annotated["chosen"] = sample["rejected"]
                annotated["rejected"] = sample["chosen"]
                chosen_reward, rejected_reward = rejected_reward, chosen_reward

            annotated["chosen_reward"] = chosen_reward
            annotated["rejected_reward"] = rejected_reward
            annotated["does_gold_agree_with_original"] = does_gold_agree_with_original
            results.append(annotated)

    return Dataset.from_list(results)


def main() -> None:
    args = _parse_args()

    if not args.skip_annotation and not args.reward_model:
        raise ValueError("--reward-model is required unless --skip-annotation is set.")

    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

    dataset_dict = ensure_dataset_dict(load_dataset(args.source_dataset))

    if args.skip_annotation:
        print("--skip-annotation: passing through dataset without gold RM scoring.")
        output_dataset = dataset_dict
    else:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading reward model: {args.reward_model} on {device}")
        reward_model, reward_tokenizer = load_reward_model(
            args.reward_model,
            reasoning=False,
            device=device,
        )
        setup_tokenizer(reward_tokenizer, model_name=args.reward_model)

        validation_tokenizer = reward_tokenizer
        if args.validation_tokenizer_name:
            print(
                "Loading validation tokenizer for Stage 3 length checks: "
                f"{args.validation_tokenizer_name}"
            )
            validation_tokenizer = AutoTokenizer.from_pretrained(
                args.validation_tokenizer_name,
                trust_remote_code=args.trust_remote_code,
            )
            setup_tokenizer(
                validation_tokenizer,
                model_name=args.validation_tokenizer_name,
            )
        else:
            print("Using reward tokenizer for Stage 3 length checks.")

        annotated_splits = {}
        for split_name, split_data in dataset_dict.items():
            annotated_splits[split_name] = _annotate_split(
                split_name,
                split_data,
                reward_model,
                reward_tokenizer,
                validation_tokenizer,
                torch.device(device),
                batch_size=args.batch_size,
                max_prompt_tokens=args.max_prompt_tokens,
                max_conversation_tokens=args.max_conversation_tokens,
            )

        output_dataset = DatasetDict(annotated_splits)

    print(f"Uploading dataset to {args.output_dataset}")
    push_kwargs = {"private": args.private}
    if hf_token:
        push_kwargs["token"] = hf_token
    output_dataset.push_to_hub(args.output_dataset, **push_kwargs)
    print("Upload complete.")
    clear_hf_dataset_cache(context="Stage3")


if __name__ == "__main__":
    main()
