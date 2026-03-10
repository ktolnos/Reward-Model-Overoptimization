"""Quick smoke test for AlpacaFarm gold RM loading (recovery + inference)."""

import torch
from reward_utils import load_reward_model

MODEL_NAME = "tatsu-lab/alpaca-farm-reward-model-human-wdiff"

print("=== Loading AlpacaFarm gold RM ===")
model, tokenizer = load_reward_model(
    MODEL_NAME,
    reasoning=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_device_map=True,
)
print(f"Model type: {type(model)}")
print(f"Device: {next(model.parameters()).device}")

# Quick forward pass
test_text = "Human: What is 2+2?\nAssistant: 2+2 equals 4."
inputs = tokenizer(test_text, return_tensors="pt").to(next(model.parameters()).device)
with torch.no_grad():
    output = model(**inputs)
print(f"Reward: {output.rewards.item():.4f}")
print("=== PASSED ===")
