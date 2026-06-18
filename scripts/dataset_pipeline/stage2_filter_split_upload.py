#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from collections import Counter

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from data_utils import (
    count_tokens_with_special_tokens,
    format_and_validate_preference_sample,
    get_length_config,
    setup_tokenizer,
    tokenize_text_with_special_tokens,
)
from scripts.dataset_pipeline.pipeline_common import (
    assert_splits_disjoint,
    clear_hf_dataset_cache,
    ensure_dataset_dict,
    split_four_way,
    validate_preference_example_structure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2: validate+filter a preference dataset by prompt/response/conversation "
            "token lengths, split into train/select/validation/test, and upload to HF."
        )
    )
    parser.add_argument("--source-dataset", required=True, help="HF source dataset name/path")
    parser.add_argument("--output-dataset", required=True, help="HF destination dataset repo")
    parser.add_argument(
        "--tokenizer-name",
        default="Qwen/Qwen3-0.6B",
        help="Tokenizer used for length checks",
    )
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading tokenizer (default).",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code when loading tokenizer.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--select-ratio", type=float, default=0.05)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    _default_cfg = get_length_config("default")
    parser.add_argument("--max-prompt-tokens", type=int, default=_default_cfg["max_prompt_tokens"])
    parser.add_argument("--max-response-tokens", type=int, default=_default_cfg["max_response_tokens"])
    parser.add_argument(
        "--max-conversation-tokens",
        type=int,
        default=_default_cfg["max_conversation_tokens"],
    )
    parser.add_argument("--max-errors", type=int, default=20)
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create destination dataset as private",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        default=False,
        help="Do not upload dataset (useful for dry runs)",
    )
    parser.add_argument(
        "--merge-splits",
        action="store_true",
        default=False,
        help=(
            "For multi-split sources, merge all source splits into one base pool "
            "before the four-way split, instead of carving from 'train' and dropping "
            "the rest. Use when the source splits are a row-level partition of one "
            "population (same distribution), e.g. re-splitting an old three-way "
            "train/test/heldout dataset, so no datapoints are dropped."
        ),
    )
    parser.add_argument(
        "--skip-prefix-check",
        action="store_true",
        default=False,
        help=(
            "Skip the check that the tokenized prompt is a prefix of the tokenized "
            "chosen/rejected conversations. The check catches BPE boundary "
            "inconsistencies that can cause distribution shift between SFT and GRPO, "
            "but may produce false positives with some tokenizers (e.g. Qwen3)."
        ),
    )
    return parser.parse_args()


def _compute_token_lengths(text: str, tokenizer) -> tuple[int, list[int]]:
    tokenized = tokenize_text_with_special_tokens(text, tokenizer)
    ids = tokenized["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(token_id) for token_id in ids]
    return len(ids), ids


def _empty_like(dataset: Dataset) -> Dataset:
    return dataset.select(range(0))


def _filter_split(
    split_name: str,
    split_data: Dataset,
    tokenizer,
    args: argparse.Namespace,
) -> tuple[Dataset, Counter[str], list[str]]:
    keep_indices: list[int] = []
    drop_reasons: Counter[str] = Counter()
    errors: list[str] = []

    iterator = tqdm(
        enumerate(split_data),
        total=len(split_data),
        desc=f"Filtering {split_name}",
    )
    for idx, sample in iterator:
        try:
            validate_preference_example_structure(
                sample,
                split_name=split_name,
                idx=idx,
                require_different_last_assistant=False,
            )

            # Filter out preference pairs where the final assistant responses are identical.
            if sample["chosen"][-1]["content"] == sample["rejected"][-1]["content"]:
                drop_reasons["same_last_assistant"] += 1
                continue

            prompt_text, chosen_text, rejected_text = format_and_validate_preference_sample(
                sample["chosen"],
                tokenizer,
                rejected_messages=sample["rejected"],
                length_config="default",
                skip_validation=True,
                sample_id=idx,
                context=f"Stage2Format-{split_name}",
            )

            prompt_len, prompt_ids = _compute_token_lengths(prompt_text, tokenizer)
            chosen_len, chosen_ids = _compute_token_lengths(chosen_text, tokenizer)
            rejected_len, rejected_ids = _compute_token_lengths(rejected_text, tokenizer)

            prefix_len = len(prompt_ids)
            if not args.skip_prefix_check:
                if chosen_ids[:prefix_len] != prompt_ids:
                    drop_reasons["prefix_mismatch"] += 1
                    continue
                if rejected_ids[:prefix_len] != prompt_ids:
                    drop_reasons["prefix_mismatch"] += 1
                    continue

            chosen_response_len = chosen_len - prefix_len
            rejected_response_len = rejected_len - prefix_len

            keep = True
            if prompt_len > args.max_prompt_tokens:
                keep = False
                drop_reasons["prompt"] += 1

            if (
                chosen_len > args.max_conversation_tokens
                or rejected_len > args.max_conversation_tokens
            ):
                keep = False
                drop_reasons["conversation"] += 1

            if (
                chosen_response_len > args.max_response_tokens
                or rejected_response_len > args.max_response_tokens
            ):
                keep = False
                drop_reasons["response"] += 1

            if keep:
                # Keep parity with validate_length_or_fail semantics on text.
                _ = count_tokens_with_special_tokens(prompt_text, tokenizer)
                keep_indices.append(idx)

        except Exception as exc:  # noqa: BLE001
            errors.append(f"{split_name}[{idx}]: {exc}")
            if len(errors) >= args.max_errors:
                break

    filtered_split = split_data.select(keep_indices) if keep_indices else _empty_like(split_data)
    print(
        f"{split_name}: {len(split_data)} -> {len(filtered_split)} "
        f"(dropped {len(split_data) - len(filtered_split)})"
    )
    if drop_reasons:
        print(f"{split_name} drop reasons: {dict(drop_reasons)}")
    return filtered_split, drop_reasons, errors


def main() -> None:
    args = parse_args()
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

    print(f"Loading dataset: {args.source_dataset}")
    dataset_dict = ensure_dataset_dict(load_dataset(args.source_dataset))
    split_names = list(dataset_dict.keys())
    print(f"Loaded splits: {split_names}")

    print(f"Loading tokenizer for filtering: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        trust_remote_code=args.trust_remote_code,
    )
    setup_tokenizer(tokenizer, model_name=args.tokenizer_name)

    filtered_splits: dict[str, Dataset] = {}
    all_errors: list[str] = []

    for split_name, split_data in dataset_dict.items():
        filtered_split, _, split_errors = _filter_split(
            split_name,
            split_data,
            tokenizer,
            args,
        )
        filtered_splits[split_name] = filtered_split
        all_errors.extend(split_errors)
        if len(all_errors) >= args.max_errors:
            break

    if all_errors:
        print("Encountered invalid examples while filtering. First errors:")
        for err in all_errors[:10]:
            print(f"  - {err}")
        raise SystemExit(1)

    # Carve all four splits from a single base population so the held-out pools
    # (select/validation/test) are same-distribution and mutually comparable.
    # When the source has multiple splits, the source "train" is the base; any
    # other source splits are intentionally dropped (mixing differently-
    # distributed source splits into validation/test would make them non-comparable).
    # Exception: --merge-splits concatenates all source splits into the base pool,
    # for sources whose splits are a same-distribution row-level partition (e.g.
    # re-splitting an old three-way train/test/heldout dataset).
    if args.merge_splits and len(filtered_splits) > 1:
        base = concatenate_datasets([filtered_splits[name] for name in filtered_splits])
        if len(base) == 0:
            raise ValueError("All samples were filtered out; nothing to split/upload.")
        print(
            f"Info: --merge-splits set; merged source splits "
            f"{list(filtered_splits)} into one base pool of {len(base)} rows."
        )
    elif len(filtered_splits) > 1:
        if "train" not in filtered_splits:
            raise ValueError(
                "Dataset has multiple splits but no 'train' split. "
                "Expected to carve train/select/validation/test from the 'train' split."
            )
        base = filtered_splits["train"]
        if len(base) == 0:
            raise ValueError("Filtered 'train' split is empty; cannot derive splits.")
        dropped = [name for name in filtered_splits if name != "train"]
        if dropped:
            print(
                f"Info: source dataset has multiple splits; dropping non-train source "
                f"splits {dropped} and carving all four splits from 'train'."
            )
    else:
        base = next(iter(filtered_splits.values()))
        if len(base) == 0:
            raise ValueError("All samples were filtered out; nothing to split/upload.")

    split_dict = split_four_way(
        base,
        train_ratio=args.train_ratio,
        select_ratio=args.select_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    assert_splits_disjoint(split_dict)

    print(
        "Split sizes: "
        f"train={len(split_dict['train'])}, "
        f"select={len(split_dict['select'])}, "
        f"validation={len(split_dict['validation'])}, "
        f"test={len(split_dict['test'])}"
    )

    if args.skip_upload:
        print("--skip-upload set; skipping push_to_hub.")
        return

    print(f"Uploading filtered dataset to {args.output_dataset}")
    push_kwargs = {"private": args.private}
    if hf_token:
        push_kwargs["token"] = hf_token
    split_dict.push_to_hub(args.output_dataset, **push_kwargs)
    print("Upload complete.")
    clear_hf_dataset_cache(context="Stage2")


if __name__ == "__main__":
    main()
