#!/usr/bin/env python3
"""Debug the AlpacaFarm reward model weight-diff recovery.

Run on the cluster:
    python debug_recovery.py

This script:
1. Deletes any stale cached recovery.
2. Re-runs recovery from scratch with diagnostic output.
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
        RewardModel,
        RewardConfig,
        RewardModelOutput,
        recover_alpacafarm_reward_model,
        WDIFF_HUB_NAME,
        DEFAULT_LLAMA_7B,
    )
    from alpaca_farm.models.reward_model import RewardModelOutput as PkgRewardModelOutput
    import dataclasses
    if not dataclasses.is_dataclass(PkgRewardModelOutput):
        PkgRewardModelOutput = dataclasses.dataclass(PkgRewardModelOutput)
        import alpaca_farm.models.reward_model as _rm_mod
        _rm_mod.RewardModelOutput = PkgRewardModelOutput

    cache_dir = "/nas/ucb/eop/cache/alpaca-farm-reward-model-human-wdiff"

    # # Step 1: Delete stale cache.
    # if os.path.exists(cache_dir):
    #     print(f"Deleting stale cache at {cache_dir}")
    #     shutil.rmtree(cache_dir)
    #     print("  Deleted.")
    # else:
    #     print(f"No existing cache at {cache_dir}")

    # Step 2: Re-run recovery from scratch.
    print("\n=== Running fresh recovery ===")
    recovered_dir = recover_alpacafarm_reward_model(
        output_dir=cache_dir,
        wdiff_name=WDIFF_HUB_NAME,
        base_model_name=DEFAULT_LLAMA_7B,
    )
    print(f"Recovery complete. Model at: {recovered_dir}")

    # Step 3: Load the recovered model (same way as reward_utils.py does).
    print("\n=== Loading recovered model ===")
    from alpaca_farm.models.reward_model import RewardModel as PkgRewardModel

    model = PkgRewardModel.from_pretrained(
        recovered_dir,
        flash_attn=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.to("cuda")
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(recovered_dir)

    # Step 4: Test on a known example from tlc4418/gold_labelled_gens.
    print("\n=== Testing on known example ===")
    from datasets import load_dataset
    ds = load_dataset("tlc4418/gold_labelled_gens", split="validation")
    row = ds[0]
    expected_score = row["gold_score"]

    # Build the Alpaca-formatted text.
    instruction = row["instruction"]
    input_text = row.get("input", "")
    output_text = row["output"]

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

    print(f"Input text (first 200 chars): {text[:200]}...")
    print(f"Expected gold_score: {expected_score}")

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=776)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    print(f"Token IDs shape: {inputs['input_ids'].shape}")

    with torch.inference_mode():
        output = model(**inputs)
    score = output.rewards.item()
    print(f"Model score: {score:.4f}")
    print(f"Expected:    {expected_score:.4f}")
    print(f"Difference:  {abs(score - expected_score):.4f}")

    if abs(score - expected_score) < 0.5:
        print("\nSUCCESS: Score is close to expected!")
    else:
        print("\nFAILURE: Score is far from expected. Investigating...")

        # Additional diagnostics: check a few weight values.
        print("\n--- Weight diagnostics ---")
        sd = model.state_dict()
        for key in ["reward_head.weight", "reward_head.bias",
                     "backbone_model.model.layers.0.self_attn.q_proj.weight"]:
            if key in sd:
                t = sd[key]
                print(f"  {key}: shape={tuple(t.shape)}, mean={t.float().mean():.6f}, std={t.float().std():.6f}")

        # Also test: load the model directly from the diff + base without saving/loading.
        print("\n--- Testing direct recovery (no save/reload) ---")
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file as load_safetensors
        import glob
        import json

        diff_dir = snapshot_download(WDIFF_HUB_NAME)
        safetensors_files = sorted(glob.glob(os.path.join(diff_dir, "model*.safetensors")))
        bin_files = sorted(glob.glob(os.path.join(diff_dir, "pytorch_model*.bin")))
        bin_files = [f for f in bin_files if "index" not in f]

        diff_state = {}
        if safetensors_files:
            for sf in safetensors_files:
                diff_state.update(load_safetensors(sf))
        elif bin_files:
            for bf in bin_files:
                diff_state.update(torch.load(bf, map_location="cpu", weights_only=True))

        base_model = transformers.LlamaForCausalLM.from_pretrained(
            DEFAULT_LLAMA_7B, torch_dtype=torch.float32,
        )
        base_state = base_model.state_dict()

        # Print some diff stats before recovery.
        rh_w = diff_state.get("reward_head.weight")
        if rh_w is not None:
            print(f"  reward_head.weight (diff): mean={rh_w.float().mean():.6f}, std={rh_w.float().std():.6f}")

        # Count how many keys get base added.
        added = 0
        skipped_shape = 0
        skipped_missing = 0
        for key in diff_state:
            base_key = key
            if key.startswith("backbone_model."):
                base_key = key[len("backbone_model."):]
            if base_key in base_state:
                if diff_state[key].shape == base_state[base_key].shape:
                    diff_state[key].add_(base_state[base_key])
                    added += 1
                else:
                    skipped_shape += 1
                    print(f"  Shape mismatch: {key} diff={tuple(diff_state[key].shape)} base={tuple(base_state[base_key].shape)}")
            else:
                skipped_missing += 1

        print(f"  Keys: {added} recovered, {skipped_shape} skipped (shape), {skipped_missing} skipped (missing in base)")

        # Build model directly and load.
        del base_model
        config = RewardConfig(backbone_model_name_or_path=DEFAULT_LLAMA_7B)
        direct_model = RewardModel(config)

        embed_key = "backbone_model.model.embed_tokens.weight"
        if embed_key in diff_state:
            diff_vocab = diff_state[embed_key].shape[0]
            curr_vocab = direct_model.backbone_model.model.embed_tokens.weight.shape[0]
            if diff_vocab != curr_vocab:
                direct_model.backbone_model.resize_token_embeddings(diff_vocab)

        direct_model.load_state_dict(diff_state, strict=False)
        direct_model = direct_model.to(dtype=torch.bfloat16, device="cuda")
        direct_model.eval()

        with torch.inference_mode():
            output2 = direct_model(**inputs)
        score2 = output2.rewards.item()
        print(f"  Direct recovery score: {score2:.4f}")
        print(f"  Expected:              {expected_score:.4f}")


if __name__ == "__main__":
    main()
