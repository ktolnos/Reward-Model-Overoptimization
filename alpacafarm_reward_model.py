"""Vendored AlpacaFarm RewardModel class.

Extracted from the alpaca_farm package to avoid dependency on the full package,
which is incompatible with modern transformers (>= 4.36) due to moved imports
like ``from transformers.deepspeed import is_deepspeed_zero3_enabled``.

Only the minimal RewardModel, RewardConfig, and RewardModelOutput classes are
kept, plus the two helper functions they depend on (``make_generative_lm`` and
``get_transformer_hidden_size``).

Original source: https://github.com/tlc4418/alpaca_farm
License: Apache 2.0
"""

import torch
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
