"""set_pad_token_id must reach the config transformers actually pools from.

Sequence-classification pooling reads ``config.get_text_config().pad_token_id``
(transformers modeling_layers.py). Nested configs -- Qwen3.5's Qwen3_5Config,
Gemma 4 -- keep that on a separate text sub-config, so setting only the
top-level config leaves it None and the forward raises "Cannot handle batch
sizes > 1 if no padding token is defined."
"""

from data_utils import set_pad_token_id


class _Config:
    """Flat config: get_text_config() returns itself, as transformers does."""

    def __init__(self):
        self.pad_token_id = None

    def get_text_config(self):
        return self


class _TextConfig:
    def __init__(self):
        self.pad_token_id = None


class _NestedConfig:
    """Nested config: the text sub-config is a distinct object."""

    def __init__(self):
        self.pad_token_id = None
        self.text_config = _TextConfig()

    def get_text_config(self):
        return self.text_config


class _Model:
    def __init__(self, config):
        self.config = config


class _Tokenizer:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id


def test_nested_config_gets_pad_on_text_config():
    model = _Model(_NestedConfig())
    set_pad_token_id(model, _Tokenizer(248044))
    # the one the pooling reads
    assert model.config.get_text_config().pad_token_id == 248044
    assert model.config.pad_token_id == 248044


def test_flat_config_still_set():
    model = _Model(_Config())
    set_pad_token_id(model, _Tokenizer(151643))
    assert model.config.pad_token_id == 151643
    assert model.config.get_text_config().pad_token_id == 151643


def test_generation_config_updated_when_present():
    model = _Model(_NestedConfig())
    model.generation_config = _TextConfig()
    set_pad_token_id(model, _Tokenizer(7))
    assert model.generation_config.pad_token_id == 7


def test_tokenizer_without_pad_token_is_left_alone():
    model = _Model(_NestedConfig())
    set_pad_token_id(model, _Tokenizer(None))
    assert model.config.get_text_config().pad_token_id is None
