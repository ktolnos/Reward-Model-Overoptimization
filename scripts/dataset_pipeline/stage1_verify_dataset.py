#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from data_utils import setup_tokenizer
from scripts.dataset_pipeline.pipeline_common import (
    ensure_dataset_dict,
    validate_apply_chat_template_compatibility,
    validate_preference_example_structure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1: verify dataset has chosen/rejected preference format suitable "
            "for apply_chat_template and downstream preprocessing."
        )
    )
    parser.add_argument("--source-dataset", required=True, help="HF dataset name/path")
    parser.add_argument(
        "--tokenizer-name",
        default="Qwen/Qwen3-0.6B",
        help="Tokenizer used to verify apply_chat_template compatibility",
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
    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Abort after collecting this many validation errors",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="Optional JSON file path for a validation report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading dataset: {args.source_dataset}")
    dataset = ensure_dataset_dict(load_dataset(args.source_dataset))
    print(f"Loaded splits: {list(dataset.keys())}")

    print(f"Loading tokenizer for validation: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        trust_remote_code=args.trust_remote_code,
    )
    setup_tokenizer(tokenizer)

    errors: list[str] = []
    split_counts: dict[str, int] = {}
    validated = 0

    for split_name, split_data in dataset.items():
        split_counts[split_name] = len(split_data)
        iterator = tqdm(
            enumerate(split_data),
            total=len(split_data),
            desc=f"Validating {split_name}",
        )
        for idx, example in iterator:
            try:
                validate_preference_example_structure(
                    example,
                    split_name=split_name,
                    idx=idx,
                )
                validate_apply_chat_template_compatibility(
                    example,
                    tokenizer,
                    split_name=split_name,
                    idx=idx,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{split_name}[{idx}]: {exc}")
                if len(errors) >= args.max_errors:
                    break
            validated += 1
        if len(errors) >= args.max_errors:
            break

    report: dict[str, Any] = {
        "source_dataset": args.source_dataset,
        "tokenizer_name": args.tokenizer_name,
        "split_counts": split_counts,
        "validated_examples": validated,
        "error_count": len(errors),
        "errors": errors,
    }

    if args.report_path:
        with open(args.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote validation report to {args.report_path}")

    if errors:
        print("Validation failed. First errors:")
        for err in errors[:10]:
            print(f"  - {err}")
        raise SystemExit(1)

    print(
        "Validation succeeded: all samples have chosen/rejected message structures "
        "compatible with apply_chat_template."
    )


if __name__ == "__main__":
    main()
