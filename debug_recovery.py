#!/usr/bin/env python3
"""Test the fixed two-step AlpacaFarm reward model weight-diff recovery.

Run on the cluster:
    python debug_recovery.py

This script does the two-step recovery in-memory (no save/load) and tests
directly, plus verifies checksums from model_sum.txt.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import tqdm
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


def load_wdiff_state_dict(hub_name):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file as load_safetensors
    import glob

    diff_dir = snapshot_download(hub_name)
    safetensors_files = sorted(glob.glob(os.path.join(diff_dir, "model*.safetensors")))
    bin_files = sorted(glob.glob(os.path.join(diff_dir, "pytorch_model*.bin")))
    bin_files = [f for f in bin_files if "index" not in f]

    state = {}
    if safetensors_files:
        for sf in safetensors_files:
            state.update(load_safetensors(sf))
    elif bin_files:
        for bf in bin_files:
            state.update(torch.load(bf, map_location="cpu", weights_only=True))
    else:
        raise FileNotFoundError(f"No model weights in {diff_dir}")

    # Read checksum if available.
    sum_file = os.path.join(diff_dir, "model_sum.txt")
    target_sum = None
    if os.path.exists(sum_file):
        with open(sum_file) as f:
            target_sum = float(f.read().strip())

    return state, diff_dir, target_sum


def compute_state_sum(state_dict):
    return sum(v.float().sum().item() for v in state_dict.values())


def apply_wdiff(diff_state, base_state, key_prefix=""):
    added = 0
    skipped_shape = 0
    skipped_missing = 0
    for key in diff_state:
        base_key = key
        if key_prefix and key.startswith(key_prefix):
            base_key = key[len(key_prefix):]
        if base_key in base_state:
            if diff_state[key].shape == base_state[base_key].shape:
                diff_state[key].add_(base_state[base_key])
                added += 1
            else:
                skipped_shape += 1
        else:
            skipped_missing += 1
    return added, skipped_shape, skipped_missing


def build_alpaca_text(instruction, input_text, output_text):
    has_input = bool(input_text and input_text.strip())
    if has_input:
        preamble = (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. "
            "Write a response that appropriately completes the request."
        )
        return (
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
        return (
            f"{preamble}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output_text}"
        )


def main():
    patch_alpaca_farm_imports()

    from alpacafarm_reward_model import RewardModel, RewardConfig, RewardModelOutput
    import dataclasses
    if not dataclasses.is_dataclass(RewardModelOutput):
        import alpacafarm_reward_model as _mod
        _mod.RewardModelOutput = dataclasses.dataclass(RewardModelOutput)

    SFT_WDIFF = "tatsu-lab/alpaca-farm-sft10k-wdiff"
    RM_WDIFF = "tatsu-lab/alpaca-farm-reward-model-human-wdiff"
    BASE_LLAMA = "baffo32/decapoda-research-llama-7B-hf"

    # ── Step 1: Recover SFT ──────────────────────────────────────────────
    print("=== Step 1: Recovering SFT model ===")
    sft_diff, sft_dir, sft_target_sum = load_wdiff_state_dict(SFT_WDIFF)
    print(f"  SFT wdiff: {len(sft_diff)} keys, target_sum={sft_target_sum}")

    base_model = transformers.LlamaForCausalLM.from_pretrained(BASE_LLAMA, torch_dtype=torch.float32)
    base_state = base_model.state_dict()
    print(f"  Base model: {len(base_state)} keys")

    added, sk_shape, sk_miss = apply_wdiff(sft_diff, base_state, key_prefix="")
    print(f"  SFT recovery: {added} added, {sk_shape} shape-skipped, {sk_miss} missing")

    sft_sum = compute_state_sum(sft_diff)
    print(f"  SFT recovered sum: {sft_sum:.4f}")
    if sft_target_sum is not None:
        print(f"  SFT target sum:    {sft_target_sum:.4f}")
        print(f"  SFT sum match:     {abs(sft_sum - sft_target_sum) < 1e-2}")
    del base_state, base_model

    # ── Step 2: Recover RM ───────────────────────────────────────────────
    print("\n=== Step 2: Recovering RM ===")
    rm_diff, rm_dir, rm_target_sum = load_wdiff_state_dict(RM_WDIFF)
    print(f"  RM wdiff: {len(rm_diff)} keys, target_sum={rm_target_sum}")

    added, sk_shape, sk_miss = apply_wdiff(rm_diff, sft_diff, key_prefix="backbone_model.")
    print(f"  RM recovery: {added} added, {sk_shape} shape-skipped, {sk_miss} missing")

    rm_sum = compute_state_sum(rm_diff)
    print(f"  RM recovered sum: {rm_sum:.4f}")
    if rm_target_sum is not None:
        print(f"  RM target sum:    {rm_target_sum:.4f}")
        print(f"  RM sum match:     {abs(rm_sum - rm_target_sum) < 1e-2}")
    del sft_diff

    # ── Step 3: Build in-memory model and test ───────────────────────────
    print("\n=== Step 3: Building in-memory model ===")
    config = RewardConfig(backbone_model_name_or_path=BASE_LLAMA)
    model = RewardModel(config)

    embed_key = "backbone_model.model.embed_tokens.weight"
    if embed_key in rm_diff:
        diff_vocab = rm_diff[embed_key].shape[0]
        curr_vocab = model.backbone_model.model.embed_tokens.weight.shape[0]
        if diff_vocab != curr_vocab:
            print(f"  Resizing embeddings: {curr_vocab} -> {diff_vocab}")
            model.backbone_model.resize_token_embeddings(diff_vocab)

    model.load_state_dict(rm_diff, strict=False)
    model = model.to(dtype=torch.bfloat16, device="cuda")
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(rm_dir)

    # ── Step 4: Test on known examples ───────────────────────────────────
    print("\n=== Step 4: Testing on known examples ===")
    from datasets import load_dataset
    ds = load_dataset("tlc4418/gold_labelled_gens", split="validation")

    n_test = min(10, len(ds))
    total_diff = 0.0
    for i in range(n_test):
        row = ds[i]
        expected_score = row["gold_scores"][0]
        text = build_alpaca_text(row["instruction"], row.get("input", ""), row["answers"][0])

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
        print("SUCCESS!")
    else:
        print("FAILURE: Scores don't match.")
        print("\nIf SFT sum doesn't match target, the base model (huggyllama/llama-7b) may be wrong.")
        print("Try: baffo32/decapoda-research-llama-7B-hf or another LLaMA-7B source.")


if __name__ == "__main__":
    main()
