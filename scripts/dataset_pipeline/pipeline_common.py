from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict

from data_utils import format_and_validate_preference_sample

MAX_FORMAT_VALIDATION_TOKENS = 10**9


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
    # Uses very large limits to validate formatting compatibility without doing length filtering here.
    format_and_validate_preference_sample(
        example["chosen"],
        tokenizer,
        rejected_messages=example["rejected"],
        max_prompt_length=MAX_FORMAT_VALIDATION_TOKENS,
        max_conversation_length=MAX_FORMAT_VALIDATION_TOKENS,
        sample_id=idx,
        context=f"{split_name}",
    )


def split_three_way(
    dataset: Dataset,
    *,
    train_ratio: float,
    test_ratio: float,
    heldout_ratio: float,
    seed: int,
) -> DatasetDict:
    ratio_sum = train_ratio + test_ratio + heldout_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            f"Ratios must sum to 1.0, got train+test+heldout={ratio_sum}."
        )

    shuffled = dataset.shuffle(seed=seed)
    total = len(shuffled)

    train_end = int(total * train_ratio)
    test_end = train_end + int(total * test_ratio)

    train_split = shuffled.select(range(0, train_end))
    test_split = shuffled.select(range(train_end, test_end))
    heldout_split = shuffled.select(range(test_end, total))

    return DatasetDict(
        {
            "train": train_split,
            "test": test_split,
            "heldout": heldout_split,
        }
    )
