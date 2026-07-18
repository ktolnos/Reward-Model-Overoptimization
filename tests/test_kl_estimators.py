"""Tests for the per-sample KL estimators used by KLEvaluator.

Both token-logprob lists come from teacher-forcing the SAME token sequences
(policy weights, then base weights), so alignment is guaranteed by
construction and any length mismatch must be a hard error.
"""
import math
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from policy_eval.generation import kl_estimators_per_sample


def test_identical_logprobs_give_zero_kl():
    lps = [[-1.0, -2.0, -0.5], [-3.0]]
    k1, grpo = kl_estimators_per_sample(lps, lps)
    assert k1 == [0.0, 0.0]
    assert grpo == [0.0, 0.0]


def test_known_values():
    pol = [[-1.0, -2.0]]
    ref = [[-1.5, -2.5]]
    k1, grpo = kl_estimators_per_sample(pol, ref)
    # diff = ref - pol = -0.5 per token; k1 = mean(pol - ref) = 0.5
    assert k1 == [pytest.approx(0.5)]
    # grpo = exp(-0.5) - (-0.5) - 1
    assert grpo == [pytest.approx(math.exp(-0.5) - 0.5)]


def test_grpo_estimator_is_nonnegative():
    pol = [[-1.0, -4.0, -0.2]]
    ref = [[-2.5, -1.0, -0.2]]
    _, grpo = kl_estimators_per_sample(pol, ref)
    assert grpo[0] >= 0.0


def test_empty_sample_gives_zero():
    k1, grpo = kl_estimators_per_sample([[]], [[]])
    assert k1 == [0.0]
    assert grpo == [0.0]


def test_per_sample_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        kl_estimators_per_sample([[-1.0, -2.0]], [[-1.0]])


def test_sample_count_mismatch_raises():
    with pytest.raises(ValueError):
        kl_estimators_per_sample([[-1.0], [-2.0]], [[-1.0]])
