"""Vendored AlpacaFarm RewardModel class.

Extracted from the alpaca_farm package to avoid dependency on the full package,
which is incompatible with modern transformers (>= 4.36) due to moved imports
like ``from transformers.deepspeed import is_deepspeed_zero3_enabled``.

Only the minimal RewardModel, RewardConfig, and RewardModelOutput classes are
kept, plus the two helper functions they depend on (``make_generative_lm`` and
``get_transformer_hidden_size``).

Also includes ``recover_alpacafarm_reward_model`` which reconstructs full
weights from a HuggingFace weight-diff checkpoint + base LLaMA-7B weights,
since tatsu-lab distributes models as weight diffs for LLaMA licensing reasons.

Original source: https://github.com/tlc4418/alpaca_farm
License: Apache 2.0
"""

import os
from pathlib import Path

import torch
import tqdm
import transformers
from torch import Tensor, nn
from transformers.utils.generic import ModelOutput


def _make_generative_lm(model_name_or_path: str, **kwargs):
    """Load a generative LM backbone (always standard LlamaForCausalLM)."""
    return transformers.LlamaForCausalLM.from_pretrained(model_name_or_path, **kwargs)


def _get_transformer_hidden_size(model: transformers.PreTrainedModel) -> int:
    if isinstance(model, transformers.GPT2LMHeadModel):
        return model.config.n_embd
    elif isinstance(model, transformers.OPTForCausalLM):
        return model.config.word_embed_proj_dim
    elif isinstance(model, transformers.T5ForConditionalGeneration):
        return model.config.d_model
    else:
        llama_cls = getattr(
            transformers,
            "LLaMAForCausalLM" if hasattr(transformers, "LLaMAForCausalLM") else "LlamaForCausalLM",
        )
        if isinstance(model, llama_cls):
            return model.config.hidden_size
        raise ValueError(f"Unknown base_model type: {type(model)}")


class RewardConfig(transformers.PretrainedConfig):
    model_type = "reward_model"

    def __init__(self, backbone_model_name_or_path=None, **kwargs):
        super().__init__(**kwargs)
        self.backbone_model_name_or_path = backbone_model_name_or_path
        self._name_or_path = backbone_model_name_or_path


class RewardModelOutput(ModelOutput):
    rewards: Tensor = None


class RewardModel(transformers.PreTrainedModel):
    config_class = RewardConfig

    def __init__(self, config: RewardConfig, **kwargs):
        super().__init__(config)
        self.backbone_model = _make_generative_lm(config.backbone_model_name_or_path, **kwargs)
        hidden_size = _get_transformer_hidden_size(self.backbone_model)
        reward_head = nn.Linear(hidden_size, 1)
        torch.nn.init.zeros_(reward_head.bias)
        self.reward_head = reward_head.to(next(self.backbone_model.parameters()).device)

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        outputs = self.backbone_model.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs
        )
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state_at_the_end = last_hidden_state[:, -1, :]
        rewards = self.reward_head(last_hidden_state_at_the_end).squeeze(-1)
        return RewardModelOutput(rewards=rewards) if return_dict else (rewards,)


# -- Weight-diff recovery ---------------------------------------------------

# HuggingFace Hub name for the reward-model-human weight diff.
WDIFF_HUB_NAME = "tatsu-lab/alpaca-farm-reward-model-human-wdiff"

# Default base LLaMA-7B model on HuggingFace.
DEFAULT_LLAMA_7B = "huggyllama/llama-7b"


def recover_alpacafarm_reward_model(
    output_dir: str,
    wdiff_name: str = WDIFF_HUB_NAME,
    base_model_name: str = DEFAULT_LLAMA_7B,
) -> str:
    """Recover full AlpacaFarm reward model weights from a weight-diff checkpoint.

    tatsu-lab distributes AlpacaFarm models as weight diffs on top of LLaMA-7B
    for licensing reasons. This function:
      1. Downloads the weight-diff RewardModel from HuggingFace.
      2. Downloads the base LLaMA-7B model.
      3. Adds base weights to the diff: ``recovered = diff + base``.
      4. Saves the recovered model to ``output_dir``.

    Args:
        output_dir: Where to save the recovered full model.
        wdiff_name: HF Hub name for the weight-diff checkpoint.
        base_model_name: HF Hub name for the base LLaMA-7B model.

    Returns:
        The ``output_dir`` path (for convenience).
    """
    output_path = Path(output_dir)
    if output_path.exists() and (output_path / "config.json").exists():
        print(f"Recovered model already exists at {output_dir}, skipping recovery.")
        return output_dir

    print(f"Recovering AlpacaFarm reward model weights...")
    print(f"  Weight diff: {wdiff_name}")
    print(f"  Base model:  {base_model_name}")

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file as load_safetensors
    import json

    # 1. Download the weight-diff files (without loading as a model, since its
    #    config.backbone_model_name_or_path points to a Stanford-local path).
    diff_dir = snapshot_download(wdiff_name)
    diff_config_path = os.path.join(diff_dir, "config.json")
    with open(diff_config_path) as f:
        diff_config = json.load(f)

    # Load diff state dict from safetensors or pytorch bin.
    safetensors_file = os.path.join(diff_dir, "model.safetensors")
    bin_file = os.path.join(diff_dir, "pytorch_model.bin")
    if os.path.exists(safetensors_file):
        diff_state = load_safetensors(safetensors_file)
    elif os.path.exists(bin_file):
        diff_state = torch.load(bin_file, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No model weights found in {diff_dir}. "
            f"Expected model.safetensors or pytorch_model.bin"
        )

    # 2. Load the base LLaMA-7B model.
    base_model = transformers.LlamaForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float32,
    )
    base_state = base_model.state_dict()

    # 3. Add base weights to diff weights.
    # The reward model nests the LLaMA backbone under "backbone_model.",
    # so base keys need the prefix to match.
    for key in tqdm.tqdm(diff_state, desc="Recovering weights"):
        base_key = key
        # Strip "backbone_model." prefix to find the corresponding base key.
        if key.startswith("backbone_model."):
            base_key = key[len("backbone_model."):]

        if base_key in base_state:
            if diff_state[key].shape == base_state[base_key].shape:
                diff_state[key].add_(base_state[base_key])

    # 4. Build a RewardModel with the correct config pointing to our base model,
    #    load recovered weights, and save.
    del base_model
    config = RewardConfig(backbone_model_name_or_path=base_model_name)
    model = RewardModel(config)
    model.load_state_dict(diff_state, strict=False)

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)

    # Also save the tokenizer from the base model.
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_dir)

    print(f"Recovered model saved to {output_dir}")

    # Free memory.
    del model, diff_state, base_state

    return output_dir
