from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from data_utils import format_and_validate_preference_sample


def sanitize_repo_id_component(value: str) -> str:
    sanitized = value.replace("/", "-").replace(":", "-")
    sanitized = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in sanitized)
    while "--" in sanitized:
        sanitized = sanitized.replace("--", "-")
    return sanitized.strip("-") or "dataset"


def ensure_dataset_dict(dataset_obj: Dataset | DatasetDict) -> DatasetDict:
    if isinstance(dataset_obj, DatasetDict):
        return dataset_obj
    if isinstance(dataset_obj, Dataset):
        return DatasetDict({"train": dataset_obj})
    raise TypeError(f"Unsupported dataset object type: {type(dataset_obj)!r}")


def get_hf_home_path() -> Path:
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser()
    return Path.home() / ".cache" / "huggingface"


def clear_hf_dataset_cache(*, context: str | None = None) -> None:
    """Clear local HF dataset caches so subsequent jobs fetch fresh revisions."""
    hf_home = get_hf_home_path()
    label = f"[{context}] " if context else ""
    print(f"{label}Clearing HF dataset cache under {hf_home}")

    hub_dir = hf_home / "hub"
    lock_dir = hub_dir / ".locks"
    targets = [hf_home / "datasets"]

    if hub_dir.exists():
        targets.extend(sorted(hub_dir.glob("datasets--*")))
    if lock_dir.exists():
        targets.extend(sorted(lock_dir.glob("datasets--*")))

    for target in targets:
        try:
            if target.is_dir() or target.is_symlink():
                shutil.rmtree(target, ignore_errors=True)
            elif target.exists():
                target.unlink()
        except Exception as exc:  # noqa: BLE001
            print(f"{label}Warning: failed to remove {target}: {exc}")

    print(f"{label}HF dataset cache cleared.")




def validate_preference_example_structure(
    example: dict[str, Any],
    *,
    split_name: str,
    idx: int,
    require_different_last_assistant: bool = True,
) -> None:
    if "chosen" not in example or "rejected" not in example:
        raise ValueError(
            f"Sample {split_name}[{idx}] must contain 'chosen' and 'rejected' columns."
        )

    chosen = example["chosen"]
    rejected = example["rejected"]

    if not isinstance(chosen, list) or not isinstance(rejected, list):
        raise ValueError(f"Sample {split_name}[{idx}] chosen/rejected must be lists of messages.")
    if len(chosen) == 0 or len(rejected) == 0:
        raise ValueError(f"Sample {split_name}[{idx}] chosen/rejected must be non-empty.")
    if len(chosen) != len(rejected):
        raise ValueError(
            f"Sample {split_name}[{idx}] chosen/rejected must have equal lengths; "
            f"got {len(chosen)} and {len(rejected)}."
        )

    for field_name, messages in (("chosen", chosen), ("rejected", rejected)):
        for msg_idx, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(
                    f"Sample {split_name}[{idx}] {field_name}[{msg_idx}] is not a dict message."
                )
            if "role" not in message or "content" not in message:
                raise ValueError(
                    f"Sample {split_name}[{idx}] {field_name}[{msg_idx}] must contain role/content keys."
                )
            if not isinstance(message["role"], str):
                raise ValueError(
                    f"Sample {split_name}[{idx}] {field_name}[{msg_idx}].role must be a string."
                )
            if not isinstance(message["content"], str):
                raise ValueError(
                    f"Sample {split_name}[{idx}] {field_name}[{msg_idx}].content must be a string."
                )

    if chosen[:-1] != rejected[:-1]:
        raise ValueError(
            f"Sample {split_name}[{idx}] chosen/rejected must share identical prompt messages (all but last)."
        )

    chosen_last = chosen[-1]
    rejected_last = rejected[-1]
    if chosen_last.get("role") != "assistant" or rejected_last.get("role") != "assistant":
        raise ValueError(
            f"Sample {split_name}[{idx}] last chosen/rejected messages must both have role='assistant'."
        )
    if require_different_last_assistant and chosen_last.get("content") == rejected_last.get("content"):
        raise ValueError(
            f"Sample {split_name}[{idx}] chosen/rejected last assistant messages must differ."
        )


def validate_apply_chat_template_compatibility(
    example: dict[str, Any],
    tokenizer,
    *,
    split_name: str,
    idx: int,
) -> None:
    # Validate formatting compatibility without doing length filtering here.
    format_and_validate_preference_sample(
        example["chosen"],
        tokenizer,
        rejected_messages=example["rejected"],
        length_config="default",
        skip_validation=True,
        sample_id=idx,
        context=f"{split_name}",
    )


def split_four_way(
    dataset: Dataset,
    *,
    train_ratio: float,
    select_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetDict:
    """Split a dataset into train/select/validation/test (BENCHMARK.md §3).

    - ``train``      — method training (SFT / RM / GRPO).
    - ``select``     — held-out prompts for the no-peek checkpoint-selection rule.
    - ``validation`` — held-out prompts for hyperparameter sweeps.
    - ``test``       — held-out prompts for final truth evaluation (never selection).

    Splitting is **by prompt group**, not by row. The dominant reason is
    duplication: in the official HelpSteer3 (preference) train split, **35% of
    rows are exact full-row duplicates** (identical context + responses + label +
    raw annotations; measured directly). 41.6% of prompts appear on >1 row, but
    most of that is pure duplication rather than genuinely distinct response-pairs
    (only ~776 prompts carry truly different pairs). The duplicates are confined
    within a single (domain, language) and occur at a near-uniform rate across
    subsets — i.e. a systematic row-level artifact, not per-annotator rows
    (annotators are already aggregated inside each row's ``individual_preference``)
    nor cross-subset resampling. The exact construction cause is **not documented
    in the paper** (arxiv 2505.11475 does not address per-prompt multiplicity or
    dedup); the paper's intended unit is nonetheless one aggregated row per sample.
    A row-level split would scatter identical rows across train and the held-out
    splits, leaking the same example and breaking the no-peek rule. Grouping by
    prompt keeps all copies in one split. Ratios apply to the prompt-group count,
    so row counts deviate slightly; ``test`` absorbs any remainder. Splits are
    prompt-disjoint by construction.

    NOTE: this prevents *cross-split* leakage but does NOT remove *within-split*
    duplication — train still carries the redundant copies and eval metrics still
    average over them. Exact-row dedup before splitting is the principled fix (see
    HANDOFF.md); not done here to keep dataset outputs stable.
    """
    ratio_sum = train_ratio + select_ratio + validation_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            "Ratios must sum to 1.0, got "
            f"train+select+validation+test={ratio_sum}."
        )

    # Group row indices by prompt hash, then shuffle the *groups* deterministically.
    groups: dict[str, list[int]] = {}
    for idx, ex in enumerate(dataset):
        groups.setdefault(_prompt_hash(ex), []).append(idx)

    import random

    group_keys = sorted(groups)  # deterministic order before shuffling
    random.Random(seed).shuffle(group_keys)
    n_groups = len(group_keys)

    train_end = int(n_groups * train_ratio)
    select_end = train_end + int(n_groups * select_ratio)
    validation_end = select_end + int(n_groups * validation_ratio)

    def _rows(keys: list[str]) -> list[int]:
        return [idx for k in keys for idx in groups[k]]

    train_split = dataset.select(_rows(group_keys[0:train_end]))
    select_split = dataset.select(_rows(group_keys[train_end:select_end]))
    validation_split = dataset.select(_rows(group_keys[select_end:validation_end]))
    test_split = dataset.select(_rows(group_keys[validation_end:n_groups]))

    return DatasetDict(
        {
            "train": train_split,
            "select": select_split,
            "validation": validation_split,
            "test": test_split,
        }
    )


def _prompt_hash(example: dict[str, Any]) -> str:
    """Stable hash of an example's prompt (the conversation minus the final answer)."""
    prompt = example["chosen"][:-1]
    blob = json.dumps(prompt, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def assert_splits_disjoint(
    split_dict: DatasetDict,
    splits: tuple[str, ...] = ("train", "select", "validation", "test"),
) -> None:
    """Raise if any pair of splits shares a prompt (contamination guard).

    Within-dataset disjointness is already guaranteed by construction in
    ``split_four_way`` (sequential slicing of one shuffle); this is a cheap guard
    against future refactors or accidental concatenation.
    """
    hashes = {
        name: {_prompt_hash(ex) for ex in split_dict[name]}
        for name in splits
        if name in split_dict
    }
    names = list(hashes)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = hashes[names[i]] & hashes[names[j]]
            if overlap:
                raise ValueError(
                    f"Contamination: {len(overlap)} shared prompt(s) between "
                    f"'{names[i]}' and '{names[j]}'."
                )
