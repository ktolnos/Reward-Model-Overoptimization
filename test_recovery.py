"""Test the AlpacaFarm reward model recovery with correct base LLaMA-7B weights.

Run on a cluster node with >=64GB RAM:
    srun --mem=64G python test_recovery.py

Steps:
  1. Delete stale cached recovery (if any)
  2. Run recovery with the correctly-converted base LLaMA-7B
  3. Verify checksum passes
  4. Score a known example and compare to expected value
"""

import os
import shutil
import sys
import torch

# ── 0. Config ─────────────────────────────────────────────────────────────
RECOVERED_DIR = "/nas/ucb/eop/cache/alpaca-farm-reward-model-human-wdiff"
BASE_LLAMA = "ktolnos/llama-7b-hf-converted"

# Known-good test case from tlc4418/gold_labelled_gens dataset (row 0)
EXPECTED_SCORE = 0.8320  # gold_scores[0] for row 0
TEST_INSTRUCTION = "What are the names of some famous combative animals that have been involved in sports or entertainment?"
TEST_INPUT = ""
TEST_OUTPUT = "Some famous animals involved in sports or entertainment include:\n\n1. Seabiscuit - A famous racehorse known for his underdog story during the Great Depression era.\n2. Secretariat - Considered one of the greatest racehorses of all time, winning the Triple Crown in 1973.\n3. Balto - A Siberian Husky sled dog who played a crucial role in the 1925 serum run to Nome, Alaska.\n4. Rin Tin Tin - A German Shepherd dog rescued from a World War I battlefield who went on to become an international star in motion pictures.\n5. Punxsutawney Phil - A groundhog who is the subject of the U.S. holiday tradition of Groundhog Day, held on February 2nd.\n6. Togo - Another Siberian Husky sled dog who played a key role in the 1925 serum run to Nome, Alaska, alongside Balto."

def format_alpaca(instruction, input_text, output_text):
    """Format as Alpaca prompt (matching the template used by evaluate_policy.py)."""
    if input_text:
        prompt = (
            f"Below is an instruction that describes a task, paired with an input that provides "
            f"further context. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        )
    else:
        prompt = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
        )
    return prompt


def main():
    # ── 1. Delete stale cache ─────────────────────────────────────────────
    if os.path.exists(RECOVERED_DIR):
        print(f"Deleting stale cache at {RECOVERED_DIR}")
        shutil.rmtree(RECOVERED_DIR)

    # ── 2. Run recovery ───────────────────────────────────────────────────
    sys.path.insert(0, os.path.dirname(__file__))
    from alpacafarm_reward_model import recover_alpacafarm_reward_model

    print(f"\n{'='*60}")
    print(f"Running recovery with base: {BASE_LLAMA}")
    print(f"{'='*60}\n")

    recover_alpacafarm_reward_model(
        output_dir=RECOVERED_DIR,
        base_model_name=BASE_LLAMA,
    )

    # ── 3. Load recovered model and score test example ────────────────────
    print(f"\n{'='*60}")
    print("Loading recovered model for scoring...")
    print(f"{'='*60}\n")

    # Patch deepspeed import for modern transformers
    if "transformers.deepspeed" not in sys.modules:
        from transformers.integrations import deepspeed as _ds_module
        sys.modules["transformers.deepspeed"] = _ds_module

    from alpaca_farm.models.reward_model import RewardModel, RewardModelOutput
    import dataclasses
    if not dataclasses.is_dataclass(RewardModelOutput):
        RewardModelOutput = dataclasses.dataclass(RewardModelOutput)
        import alpaca_farm.models.reward_model as _rm_mod
        _rm_mod.RewardModelOutput = RewardModelOutput

    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(RECOVERED_DIR)
    model = RewardModel.from_pretrained(
        RECOVERED_DIR,
        flash_attn=False,
        torch_dtype=torch.float32,
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Score the test example
    text = format_alpaca(TEST_INSTRUCTION, TEST_INPUT, TEST_OUTPUT)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        output = model(**inputs)
    score = output.rewards.item()

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Model score:    {score:.4f}")
    print(f"Expected score: {EXPECTED_SCORE:.4f}")
    print(f"Difference:     {abs(score - EXPECTED_SCORE):.4f}")
    if abs(score - EXPECTED_SCORE) < 0.1:
        print(">>> SUCCESS - scores match! <<<")
    else:
        print(">>> MISMATCH - scores don't match <<<")


if __name__ == "__main__":
    main()
