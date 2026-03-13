"""Test the AlpacaFarm reward model recovered via scripts/recover_on_cluster.sh.

Run on a cluster node with a GPU:
    srun --mem=32G --gres=gpu:1 python test_recovery.py
"""

import sys
import torch

# ── Config ────────────────────────────────────────────────────────────────
RECOVERED_DIR = "/nas/ucb/eop/cache/alpaca_farm_models/reward-model-human"
SFT_DIR = "/nas/ucb/eop/cache/alpaca_farm_models/sft10k"

# Known-good test case from tlc4418/gold_labelled_gens dataset (row 0)
EXPECTED_SCORE = 0.8320  # gold_scores[0] for row 0
TEST_INSTRUCTION = "What are the names of some famous combative animals that have been involved in sports or entertainment?"
TEST_INPUT = ""
TEST_OUTPUT = (
    "Some famous animals involved in sports or entertainment include:\n\n"
    "1. Seabiscuit - A famous racehorse known for his underdog story during the Great Depression era.\n"
    "2. Secretariat - Considered one of the greatest racehorses of all time, winning the Triple Crown in 1973.\n"
    "3. Balto - A Siberian Husky sled dog who played a crucial role in the 1925 serum run to Nome, Alaska.\n"
    "4. Rin Tin Tin - A German Shepherd dog rescued from a World War I battlefield who went on to become an international star in motion pictures.\n"
    "5. Punxsutawney Phil - A groundhog who is the subject of the U.S. holiday tradition of Groundhog Day, held on February 2nd.\n"
    "6. Togo - Another Siberian Husky sled dog who played a key role in the 1925 serum run to Nome, Alaska, alongside Balto."
)


def format_alpaca(instruction, input_text, output_text):
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

    from alpaca_farm.models.reward_model import RewardModel, RewardModelOutput
    import dataclasses
    if not dataclasses.is_dataclass(RewardModelOutput):
        RewardModelOutput = dataclasses.dataclass(RewardModelOutput)
        import alpaca_farm.models.reward_model as _rm_mod
        _rm_mod.RewardModelOutput = RewardModelOutput

    import transformers

    print(f"Loading model from {RECOVERED_DIR}")
    print(f"SFT backbone: {SFT_DIR}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(RECOVERED_DIR)
    # from_pretrained silently fails to load weights in newer transformers
    # (the "copying from non-meta parameter" warnings = no-op).
    # Instead: build the model, then manually load the state dict.
    from alpaca_farm.models.reward_model import RewardConfig
    import glob, os
    config = RewardConfig.from_pretrained(RECOVERED_DIR)
    model = RewardModel(config, flash_attn=False, torch_dtype=torch.float32)

    # Load saved weights manually
    weight_files = sorted(glob.glob(os.path.join(RECOVERED_DIR, "pytorch_model*.bin")))
    state_dict = {}
    for wf in weight_files:
        state_dict.update(torch.load(wf, map_location="cpu", weights_only=True))
    model.load_state_dict(state_dict, strict=False)

    print(f"reward_head.weight sum: {model.reward_head.weight.sum().item()}")
    print(f"reward_head.bias: {model.reward_head.bias.item()}")

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
