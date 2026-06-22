"""Gemma 4 sequence-classification (reward-model) support.

Transformers (as of 5.5.x / latest 5.12.x) ships ``Gemma4ForCausalLM`` and
``Gemma4ForConditionalGeneration`` but **no** ``Gemma4ForSequenceClassification``,
so ``AutoModelForSequenceClassification.from_pretrained("google/gemma-4-E4B-it")``
fails with "Unrecognized configuration class ... for AutoModelForSequenceClassification".
That blocks BT reward-model training/eval on Gemma 4.

This module ports the two classes from the (still-open, on-hold) upstream PR
https://github.com/huggingface/transformers/pull/45438 and registers them with
``AutoModelForSequenceClassification`` on import, so the rest of the pipeline
(``reward_models/run_reward_models_train.py``, ``reward_utils.load_reward_model``,
``rm_eval/eval.py``, GRPO) can use Gemma 4 as a reward model unchanged.

Import this module for its side effects *before* any
``AutoModelForSequenceClassification.from_pretrained`` call::

    import gemma4_sequence_classification  # noqa: F401

The classes are copied verbatim from the PR. The only deviation is that we set
``base_model_prefix = "model"`` directly on ``Gemma4ForSequenceClassification``
(the PR moves it onto ``Gemma4PreTrainedModel``); doing it on the subclass keeps
the change local and avoids altering checkpoint loading for the base/vision/audio
models, while still giving correct ``model.*`` -> ``self.model`` weight mapping.
"""

import torch
from torch import nn

from transformers import (
    AutoModelForSequenceClassification,
    Gemma4Config,
    Gemma4TextConfig,
)
from transformers.cache_utils import Cache
from transformers.modeling_layers import GenericForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.models.gemma4.modeling_gemma4 import Gemma4Model, Gemma4PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)

logger = logging.get_logger(__name__)


class Gemma4ForSequenceClassification(Gemma4PreTrainedModel):
    # ``self.model`` is the (multimodal) backbone; checkpoints store its weights
    # under the ``model.*`` prefix, so the base model prefix must be "model".
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Gemma4Model(config)
        self.score = nn.Linear(config.text_config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        image_position_ids: torch.LongTensor | None = None,
        video_position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutputWithPast:
        r"""
        input_features_mask (`torch.FloatTensor` of shape `(num_images, seq_length)`):
            The attention mask for the input audio.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*):
            2D patch position coordinates from the image processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        video_position_ids (`torch.LongTensor` of shape `(num_videos, num_frames, max_patches, 2)`, *optional*):
            2D patch position coordinates from the video processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        """

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            input_features=input_features,
            input_features_mask=input_features_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            image_position_ids=image_position_ids,
            video_position_ids=video_position_ids,
            return_dict=True,
            **kwargs,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.text_config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.text_config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.text_config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class Gemma4TextForSequenceClassification(GenericForSequenceClassification, Gemma4PreTrainedModel):
    """
    Gemma4TextForSequenceClassification is a text-only sequence classification model that works with Gemma4TextConfig.
    It uses the generic sequence classification implementation for efficiency and consistency.
    """

    config: Gemma4TextConfig
    input_modalities = ("text",)


def _register():
    """Register the classes with AutoModelForSequenceClassification (idempotent)."""
    for config_cls, model_cls in (
        (Gemma4Config, Gemma4ForSequenceClassification),
        (Gemma4TextConfig, Gemma4TextForSequenceClassification),
    ):
        try:
            AutoModelForSequenceClassification.register(config_cls, model_cls)
        except ValueError:
            # Already registered (e.g. module imported twice, or a future
            # transformers ships these natively) -- nothing to do.
            pass


_register()


__all__ = [
    "Gemma4ForSequenceClassification",
    "Gemma4TextForSequenceClassification",
]
