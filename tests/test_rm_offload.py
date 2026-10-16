"""--offload_reward_models parks frozen RMs in pinned host memory between scorings.

A 5x4B ensemble does not fit on one 80 GB card beside the policy and vLLM, so
``build_reward_function`` moves each RM onto the GPU only for its own scoring
call. The helpers repoint ``.data`` rather than calling ``.to()``, so these
check what that has to preserve: identical outputs, tied weights staying tied,
and the GPU copy actually being freed.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rlhf" / "grpo"))

from grpo_utils import (  # noqa: E402
    _offload_reward_model,
    _onload_reward_model,
    _park_reward_model_on_cpu,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class TiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(64, 256)
        self.body = torch.nn.Linear(256, 256)
        self.head = torch.nn.Linear(256, 64, bias=False)
        self.head.weight = self.embed.weight  # tied, like many LMs
        self.register_buffer("scale", torch.full((256,), 0.5))

    def forward(self, ids):
        return self.head(self.body(self.embed(ids)) * self.scale)


def _model_on_gpu():
    torch.manual_seed(0)
    return TiedModel().cuda().eval()


def test_park_moves_everything_to_pinned_host_memory():
    model = _model_on_gpu()
    _park_reward_model_on_cpu(model)
    for t in list(model.parameters()) + list(model.buffers()):
        assert t.device.type == "cpu"
        assert t.is_pinned()


def test_round_trip_reproduces_outputs():
    model = _model_on_gpu()
    ids = torch.arange(32, device="cuda")
    with torch.no_grad():
        expected = model(ids)

    _park_reward_model_on_cpu(model)
    for _ in range(2):  # a second round trip must not drift either
        _onload_reward_model(model, torch.device("cuda"))
        with torch.no_grad():
            got = model(ids)
        torch.cuda.current_stream().synchronize()
        _offload_reward_model(model)
        assert torch.equal(got, expected)


def test_tied_weights_stay_tied_on_gpu():
    model = _model_on_gpu()
    _park_reward_model_on_cpu(model)
    _onload_reward_model(model, torch.device("cuda"))
    assert model.head.weight is model.embed.weight
    assert model.head.weight.device.type == "cuda"


def test_offload_frees_the_gpu_copy():
    model = _model_on_gpu()
    _park_reward_model_on_cpu(model)
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()

    _onload_reward_model(model, torch.device("cuda"))
    assert torch.cuda.memory_allocated() > baseline
    _offload_reward_model(model)
    assert torch.cuda.memory_allocated() == baseline
