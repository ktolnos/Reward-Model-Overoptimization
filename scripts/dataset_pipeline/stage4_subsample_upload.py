#!/usr/bin/env python3
from __future__ import annotations

import argparse

from datasets import DatasetDict, load_dataset

from scripts.dataset_pipeline.pipeline_common import ensure_dataset_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 4: download annotated dataset, subsample each split, "
            "and upload to a new HF dataset location."
        )
    )
    parser.add_argument("--source-dataset", required=True, help="HF annotated dataset")
    parser.add_argument("--output-dataset", required=True, help="HF destination dataset")
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (0.0 < args.fraction <= 1.0):
        raise ValueError(f"--fraction must be in (0, 1], got {args.fraction}")

    dataset_dict = ensure_dataset_dict(load_dataset(args.source_dataset))

    subsampled = {}
    for split_name, split_data in dataset_dict.items():
        if len(split_data) == 0:
            subsampled[split_name] = split_data
            continue

        shuffled = split_data.shuffle(seed=args.seed)
        n = int(len(shuffled) * args.fraction)
        if n == 0:
            n = 1

        subsampled_split = shuffled.select(range(n))
        print(f"{split_name}: {len(split_data)} -> {len(subsampled_split)}")
        subsampled[split_name] = subsampled_split

    subsampled_dataset = DatasetDict(subsampled)

    if args.skip_upload:
        print("--skip-upload set; skipping push_to_hub.")
        return

    print(f"Uploading subsampled dataset to {args.output_dataset}")
    push_kwargs = {"private": args.private}
    subsampled_dataset.push_to_hub(args.output_dataset, **push_kwargs)
    print("Upload complete.")


if __name__ == "__main__":
    main()
