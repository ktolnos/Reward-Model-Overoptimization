#!/usr/bin/env python3
"""Test the fixed two-step AlpacaFarm reward model weight-diff recovery.

Run on the cluster:
    python debug_recovery.py

This script:
1. Deletes any stale cached recovery.
2. Re-runs the two-step recovery (sft_wdiff + base → sft, then rm_wdiff + sft → rm).
3. Tests the recovered model on a known example from tlc4418/gold_labelled_gens.
"""

import os
import sys
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import transformers


def patch_alpaca_farm_imports():
    """Monkey-patch broken imports in alpaca_farm for modern transformers."""
    import transformers.trainer as _trainer_mod
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

    if not hasattr(_trainer_mod, "is_deepspeed_zero3_enabled"):
        _trainer_mod.is_deepspeed_zero3_enabled = is_deepspeed_zero3_enabled
    if not hasattr(_trainer_mod, "WEIGHTS_NAME"):
        _trainer_mod.WEIGHTS_NAME = "pytorch_model.bin"
    if "transformers.deepspeed" not in sys.modules:
        from transformers.integrations import deepspeed as _ds_module
        sys.modules["transformers.deepspeed"] = _ds_module


def main():
    patch_alpaca_farm_imports()

    from alpacafarm_reward_model import (
        recover_alpacafarm_reward_model,
        WDIFF_HUB_NAME,
        DEFAULT_LLAMA_7B,
    )
    import dataclasses
    from alpaca_farm.models.reward_model import (
        RewardModel as PkgRewardModel,
        RewardModelOutput as PkgRewardModelOutput,
    )
    if not dataclasses.is_dataclass(PkgRewardModelOutput):
        PkgRewardModelOutput = dataclasses.dataclass(PkgRewardModelOutput)
        import alpaca_farm.models.reward_model as _rm_mod
        _rm_mod.RewardModelOutput = PkgRewardModelOutput

    cache_dir = "/nas/ucb/eop/cache/alpaca-farm-reward-model-human-wdiff"

    # Step 1: Delete stale cache.
    if os.path.exists(cache_dir):
        print(f"Deleting stale cache at {cache_dir}")
        shutil.rmtree(cache_dir)
        print("  Deleted.")
    else:
        print(f"No existing cache at {cache_dir}")

    # Step 2: Re-run two-step recovery from scratch.
    print("\n=== Running fresh two-step recovery ===")
    recovered_dir = recover_alpacafarm_reward_model(
        output_dir=cache_dir,
        wdiff_name=WDIFF_HUB_NAME,
        base_model_name=DEFAULT_LLAMA_7B,
    )
    print(f"Recovery complete. Model at: {recovered_dir}")

    # Step 3: Load the recovered model (same way as reward_utils.py does).
    print("\n=== Loading recovered model ===")
    model = PkgRewardModel.from_pretrained(
        recovered_dir,
        flash_attn=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.to("cuda")
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(recovered_dir)

    # Step 4: Test on known examples from tlc4418/gold_labelled_gens.
    print("\n=== Testing on known examples ===")
    from datasets import load_dataset
    ds = load_dataset("tlc4418/gold_labelled_gens", split="validation")

    # Test on first 5 examples.
    n_test = min(5, len(ds))
    total_diff = 0.0
    for i in range(n_test):
        row = ds[i]
        expected_score = row["gold_scores"][0]
        instruction = row["instruction"]
        input_text = row.get("input", "")
        output_text = row["answers"][0]

        has_input = bool(input_text and input_text.strip())
        if has_input:
            preamble = (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. "
                "Write a response that appropriately completes the request."
            )
            text = (
                f"{preamble}\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output_text}"
            )
        else:
            preamble = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request."
            )
            text = (
                f"{preamble}\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output_text}"
            )

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=776)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.inference_mode():
            output = model(**inputs)
        score = output.rewards.item()
        diff = abs(score - expected_score)
        total_diff += diff
        status = "OK" if diff < 0.5 else "MISMATCH"
        print(f"  [{i}] score={score:.4f}  expected={expected_score:.4f}  diff={diff:.4f}  {status}")

    avg_diff = total_diff / n_test
    print(f"\nAverage absolute difference: {avg_diff:.4f}")
    if avg_diff < 0.5:
        print("SUCCESS: Recovery looks correct!")
    else:
        print("FAILURE: Scores still don't match expected values.")


if __name__ == "__main__":
    main()
