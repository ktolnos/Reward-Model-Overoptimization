"""The trainer's EOS override must survive transformers' special-token plumbing.

TRL decides whether a completion terminated cleanly with

    ids[-1] not in [self._tokenizer.eos_token_id, self._tokenizer.pad_token_id]

(grpo_trainer.py, ``completions/clipped_ratio`` and
``mask_truncated_completions``). Chat models have more than one legitimate
terminator -- ``<|im_end|>`` and ``<|endoftext|>`` for Qwen -- and our Qwen3.5
checkpoints ship with ``eos == pad == <|endoftext|>``, so that pair does not
cover ``<|im_end|>``, the token the policy actually emits. ``MyGRPOTrainer``
widens the membership test with ``_MultiTokenEosId``.

Assigning it straight onto the tokenizer silently stopped working: transformers
5.x routes every ``*_token_id`` write through ``SpecialTokensMixin.__setattr__``,
which converts the id back to its token string, so the next read hands back a
plain int. Every completion ending in a secondary stop token was then counted as
truncated, and with ``mask_truncated_completions`` that zeroes the entire
batch's loss -- a run that trains on nothing while looking healthy.
"""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rlhf" / "grpo"))

from data_utils import get_generation_stop_token_ids  # noqa: E402

_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def helpers():
    from my_grpo import _MultiTokenEosId, _tokenizer_with_multi_eos

    return _MultiTokenEosId, _tokenizer_with_multi_eos


@pytest.fixture
def tokenizer():
    """A tokenizer shaped like our Qwen3.5 SFT checkpoints: eos == pad.

    Loaded fresh per test: ``copy.copy`` of a tokenizer shares
    ``_special_tokens_map``, so a test that writes a special token would leak
    into the next one.
    """
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(_MODEL)
    except Exception as exc:  # offline / no cache
        pytest.skip(f"{_MODEL} tokenizer unavailable: {exc}")
    tok.eos_token = "<|endoftext|>"
    tok.pad_token = "<|endoftext|>"
    assert tok.eos_token_id == tok.pad_token_id
    return tok


@pytest.fixture
def stop_ids(tokenizer):
    ids = get_generation_stop_token_ids(tokenizer)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert im_end in ids and im_end != tokenizer.eos_token_id
    return ids


def _trl_is_truncated(tok, last_id):
    """The exact expression TRL evaluates per completion."""
    return last_id not in [tok.eos_token_id, tok.pad_token_id]


def test_every_stop_token_counts_as_terminated(helpers, tokenizer, stop_ids):
    _, with_multi_eos = helpers
    tok = with_multi_eos(tokenizer, tokenizer.eos_token_id, stop_ids)
    for stop_id in stop_ids:
        assert not _trl_is_truncated(tok, stop_id), f"{stop_id} misread as truncated"


def test_ordinary_last_token_still_counts_as_truncated(helpers, tokenizer, stop_ids):
    _, with_multi_eos = helpers
    tok = with_multi_eos(tokenizer, tokenizer.eos_token_id, stop_ids)

    ordinary = tokenizer("hello world")["input_ids"][-1]
    assert ordinary not in stop_ids
    assert _trl_is_truncated(tok, ordinary)


def test_plain_assignment_is_swallowed(helpers, tokenizer, stop_ids):
    """Why the helper exists: transformers drops the subclass on assignment."""
    multi_cls, _ = helpers
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    tok = copy.copy(tokenizer)
    tok.eos_token_id = multi_cls(tokenizer.eos_token_id, stop_ids)

    assert type(tok.eos_token_id) is int
    assert _trl_is_truncated(tok, im_end)


def test_source_tokenizer_keeps_a_plain_int(helpers, tokenizer, stop_ids):
    """processing_class is shared with formatting, RM scoring and checkpointing."""
    _, with_multi_eos = helpers
    with_multi_eos(tokenizer, tokenizer.eos_token_id, stop_ids)

    assert type(tokenizer.eos_token_id) is int
    assert tokenizer.eos_token_id == tokenizer.convert_tokens_to_ids(tokenizer.eos_token)


def test_still_behaves_as_the_primary_int(helpers, tokenizer, stop_ids):
    """Consumers that want a plain id -- generate(), tensors -- must be unaffected."""
    _, with_multi_eos = helpers
    tok = with_multi_eos(tokenizer, tokenizer.eos_token_id, stop_ids)

    assert int(tok.eos_token_id) == tokenizer.eos_token_id
    assert tok.eos_token == tokenizer.eos_token
    assert tok.decode(tok("hi")["input_ids"]) == "hi"
