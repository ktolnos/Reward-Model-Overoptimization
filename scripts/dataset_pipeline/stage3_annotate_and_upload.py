#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

from data_utils import (
    DEFAULT_MAX_CONVERSATION_TOKENS,
    DEFAULT_MAX_PROMPT_TOKENS,
    format_and_validate_preference_sample,
    format_conversation,
    tokenize_for_rm,
)
from reward_utils import load_reward_model
from scripts.dataset_pipeline.pipeline_common import (
    ensure_dataset_dict,
    validate_preference_example_structure,
)


def parse_args() -> argparse.Namespace:
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
        required=True,
        help="Reward model name/path used for annotation",
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
        default=DEFAULT_MAX_PROMPT_TOKENS,
        help="Validation max prompt tokens",
    )
    parser.add_argument(
        "--max-conversation-tokens",
        type=int,
        default=DEFAULT_MAX_CONVERSATION_TOKENS,
        help="Validation max conversation tokens",
    )
    return parser.parse_args()


def annotate_split(
    split_name: str,
    split_data,
    reward_model,
    reward_tokenizer,
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

        all_conversations = []
        batch_original_rows: list[dict] = []

        for offset in range(batch_size_actual):
            global_idx = start + offset
            sample = {col: batch[col][offset] for col in batch.keys()}

            validate_preference_example_structure(
                sample,
                split_name=split_name,
                idx=global_idx,
            )

            # Re-run strict length validation with RM tokenizer for fail-fast consistency.
            format_and_validate_preference_sample(
                sample["chosen"],
                reward_tokenizer,
                rejected_messages=sample["rejected"],
                max_prompt_length=max_prompt_tokens,
                max_conversation_length=max_conversation_tokens,
                sample_id=global_idx,
                context=f"Stage3-{split_name}",
            )

            all_conversations.append(sample["chosen"])
            all_conversations.append(sample["rejected"])
            batch_original_rows.append(sample)

        formatted_texts = [format_conversation(conv, reward_tokenizer) for conv in all_conversations]
        inputs = tokenize_for_rm(formatted_texts, reward_tokenizer).to(reward_model.device)

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
    args = parse_args()

    dataset_dict = ensure_dataset_dict(load_dataset(args.source_dataset))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading reward model: {args.reward_model} on {device}")
    reward_model, reward_tokenizer = load_reward_model(
        args.reward_model,
        reasoning=False,
        device=device,
    )

    annotated_splits = {}
    for split_name, split_data in dataset_dict.items():
        annotated_splits[split_name] = annotate_split(
            split_name,
            split_data,
            reward_model,
            reward_tokenizer,
            batch_size=args.batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_conversation_tokens=args.max_conversation_tokens,
        )

    annotated_dataset = DatasetDict(annotated_splits)

    print(f"Uploading annotated dataset to {args.output_dataset}")
    push_kwargs = {"private": args.private}
    annotated_dataset.push_to_hub(args.output_dataset, **push_kwargs)
    print("Upload complete.")


if __name__ == "__main__":
    main()
