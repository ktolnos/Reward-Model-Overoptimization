"""--judge_selected_checkpoint_only: the judge runs on one checkpoint (L6).

The deferred (LLM-judge) phase is the expensive one, and only the selected
checkpoint's judge numbers are reported, so the flag trims the deferred cache
to the sibling-RM argmax — resolved from this run's metric rows, or (under
--load_generations) recomputed from the cached ``select`` per-example logs.
"""
import os
import sys
import types

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from policy_eval.eval_utils import (
    require_selection_available,
    restrict_deferred_cache,
    selected_checkpoint_from_cache,
    selected_checkpoint_from_rows,
)
from policy_eval.selection import SELECTION_METRIC, SELECTION_SCORE_COLUMN


def _rows():
    return [
        {"checkpoint": 100, SELECTION_METRIC: 0.1},
        {"checkpoint": 200, SELECTION_METRIC: 0.9},
        {"checkpoint": 300, SELECTION_METRIC: 0.5},
    ]


def _args(**kw):
    base = dict(judge_selected_checkpoint_only=True, load_generations=False,
                benchmarks="preference,select")
    base.update(kw)
    return types.SimpleNamespace(**base)


def _bench(name, *, judged=True):
    # Only benchmarks with a deferred evaluator are affected by the flag.
    return types.SimpleNamespace(name=name, deferred_evaluators=["judge"] if judged else [])


def test_selected_checkpoint_from_rows_picks_argmax():
    assert selected_checkpoint_from_rows(_rows()) == 200


def test_selected_checkpoint_from_rows_without_selection_metric_raises():
    with pytest.raises(ValueError, match=SELECTION_METRIC):
        selected_checkpoint_from_rows([{"checkpoint": 100, "gold_rm/mean": 1.0}])


def test_restrict_deferred_cache_keeps_only_selected():
    cache = {("preference", 100): "a", ("preference", 200): "b",
             ("arena_hard", 200): "c", ("arena_hard", 300): "d"}
    kept = restrict_deferred_cache(cache, 200)
    assert set(kept) == {("preference", 200), ("arena_hard", 200)}


def test_restrict_deferred_cache_missing_checkpoint_raises():
    with pytest.raises(ValueError, match="checkpoint-200"):
        restrict_deferred_cache({("preference", 100): "a"}, 200)


def test_require_selection_available_needs_select_benchmark():
    require_selection_available(
        _args(), [_bench("preference"), _bench("select", judged=False)])
    with pytest.raises(ValueError, match="select"):
        require_selection_available(_args(benchmarks="preference"), [_bench("preference")])


def test_require_selection_available_no_ops():
    # Flag off; score recomputed from cached logs; or no judge runs at all.
    require_selection_available(
        _args(judge_selected_checkpoint_only=False), [_bench("preference")])
    require_selection_available(_args(load_generations=True), [_bench("preference")])
    require_selection_available(
        _args(benchmarks="ifeval"), [_bench("ifeval", judged=False)])


def _write_select_log(dirpath, ckpt, scores):
    pd.DataFrame({SELECTION_SCORE_COLUMN: scores}).to_parquet(
        os.path.join(dirpath, f"select__checkpoint-{ckpt}.parquet"))


def test_selected_checkpoint_from_cache_recomputes_argmax(tmp_path):
    d = str(tmp_path)
    _write_select_log(d, 100, [0.0, 0.2])
    _write_select_log(d, 200, [1.0, 0.6])   # mean 0.8 -> argmax
    _write_select_log(d, 300, [0.5, 0.5])
    assert selected_checkpoint_from_cache(d) == 200


def test_selected_checkpoint_from_cache_without_logs_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="select"):
        selected_checkpoint_from_cache(str(tmp_path))


def test_selected_checkpoint_from_cache_without_score_column_raises(tmp_path):
    d = str(tmp_path)
    pd.DataFrame({"response_text": ["r"]}).to_parquet(
        os.path.join(d, "select__checkpoint-100.parquet"))
    with pytest.raises(ValueError, match=SELECTION_SCORE_COLUMN):
        selected_checkpoint_from_cache(d)
