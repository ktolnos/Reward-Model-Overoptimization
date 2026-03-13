"""Test the AlpacaFarm reward model recovered via scripts/recover_on_cluster.sh.

Compares correct Alpaca template vs tlc4418's buggy template across multiple
data points from tlc4418/gold_labelled_gens.

Run on a cluster node with a GPU:
    srun --mem=32G --gres=gpu:1 python test_recovery.py
"""

import sys
import torch
import math

# ── Config ────────────────────────────────────────────────────────────────
RECOVERED_DIR = "/nas/ucb/eop/cache/alpaca_farm_models/reward-model-human"
SFT_DIR = "/nas/ucb/eop/cache/alpaca_farm_models/sft10k"
NUM_SAMPLES = 20


def format_alpaca_correct(instruction, input_text, output_text):
    """Correct Alpaca prompt template."""
    if input_text:
        return (
            f"Below is an instruction that describes a task, paired with an input that provides "
            f"further context. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        )
    return (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
    )


def format_alpaca_tlc4418_bug(instruction, input_text, output_text):
    """Reproduces the bug in tlc4418's _parse_entry where start_prompt is truncated."""
    start_prompt = "Below is an instruction that describes a task, paired with an "
    return f"{start_prompt}{output_text}"


def main():
    # Patch deepspeed import for modern transformers
    import transformers.trainer as _trainer_mod
    if "transformers.deepspeed" not in sys.modules:
        from transformers.integrations import deepspeed as _ds_module
        sys.modules["transformers.deepspeed"] = _ds_module
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
    if not hasattr(_trainer_mod, "is_deepspeed_zero3_enabled"):
        _trainer_mod.is_deepspeed_zero3_enabled = is_deepspeed_zero3_enabled
    if not hasattr(_trainer_mod, "WEIGHTS_NAME"):
        _trainer_mod.WEIGHTS_NAME = "pytorch_model.bin"

    from alpaca_farm.models.reward_model import RewardModel, RewardModelOutput, RewardConfig
    import dataclasses
    if not dataclasses.is_dataclass(RewardModelOutput):
        RewardModelOutput = dataclasses.dataclass(RewardModelOutput)
        import alpaca_farm.models.reward_model as _rm_mod
        _rm_mod.RewardModelOutput = RewardModelOutput

    import transformers
    import glob, os
    from datasets import load_dataset

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading model from {RECOVERED_DIR}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(RECOVERED_DIR)
    config = RewardConfig.from_pretrained(RECOVERED_DIR)
    model = RewardModel(config, flash_attn=False, torch_dtype=torch.float32)

    weight_files = sorted(glob.glob(os.path.join(RECOVERED_DIR, "pytorch_model*.bin")))
    state_dict = {}
    for wf in weight_files:
        state_dict.update(torch.load(wf, map_location="cpu", weights_only=True))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"Loading tlc4418/gold_labelled_gens dataset...")
    ds = load_dataset("tlc4418/gold_labelled_gens", split="validation")

    correct_scores = []
    bug_scores = []
    expected_scores = []

    print(f"\n{'idx':>3}  {'expected':>8}  {'correct':>8}  {'buggy':>8}  {'bug_diff':>8}  {'cor_diff':>8}  instruction (truncated)")
    print("-" * 100)

    for i in range(min(NUM_SAMPLES, len(ds))):
        row = ds[i]
        instruction = row["instruction"]
        input_text = row.get("input", "")
        answer = row["answers"][0]
        expected = row["gold_scores"][0]

        # Score with both templates
        scores = {}
        for name, fmt_fn in [("correct", format_alpaca_correct), ("bug", format_alpaca_tlc4418_bug)]:
            text = fmt_fn(instruction, input_text, answer)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                output = model(**inputs)
            scores[name] = output.rewards.item()

        correct_scores.append(scores["correct"])
        bug_scores.append(scores["bug"])
        expected_scores.append(expected)

        bug_diff = scores["bug"] - expected
        cor_diff = scores["correct"] - expected
        instr_short = instruction[:40]
        print(f"{i:3d}  {expected:8.4f}  {scores['correct']:8.4f}  {scores['bug']:8.4f}  {bug_diff:+8.4f}  {cor_diff:+8.4f}  {instr_short}")

    # ── Summary stats ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    bug_diffs = [b - e for b, e in zip(bug_scores, expected_scores)]
    cor_diffs = [c - e for c, e in zip(correct_scores, expected_scores)]

    bug_mae = sum(abs(d) for d in bug_diffs) / len(bug_diffs)
    cor_mae = sum(abs(d) for d in cor_diffs) / len(cor_diffs)
    bug_rmse = math.sqrt(sum(d**2 for d in bug_diffs) / len(bug_diffs))
    cor_rmse = math.sqrt(sum(d**2 for d in cor_diffs) / len(cor_diffs))

    bug_mean = sum(bug_scores) / len(bug_scores)
    cor_mean = sum(correct_scores) / len(correct_scores)
    exp_mean = sum(expected_scores) / len(expected_scores)

    print(f"                     {'Buggy tmpl':>12}  {'Correct tmpl':>12}  {'Expected':>12}")
    print(f"  Mean score:        {bug_mean:12.4f}  {cor_mean:12.4f}  {exp_mean:12.4f}")
    print(f"  MAE vs expected:   {bug_mae:12.4f}  {cor_mae:12.4f}")
    print(f"  RMSE vs expected:  {bug_rmse:12.4f}  {cor_rmse:12.4f}")

    # Correlation
    def pearson_r(xs, ys):
        n = len(xs)
        mx, my = sum(xs)/n, sum(ys)/n
        cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        sx = math.sqrt(sum((x-mx)**2 for x in xs))
        sy = math.sqrt(sum((y-my)**2 for y in ys))
        return cov / (sx * sy) if sx > 0 and sy > 0 else 0

    print(f"  Pearson r (bug):   {pearson_r(bug_scores, expected_scores):12.4f}")
    print(f"  Pearson r (correct):{pearson_r(correct_scores, expected_scores):11.4f}")


if __name__ == "__main__":
    main()
