"""--judge_selected_checkpoint_only: the judge runs on two checkpoints (L6).

The deferred (LLM-judge) phase is the expensive one (~2.8k games per checkpoint),
and only a couple of checkpoints' numbers are read, so the flag trims the
deferred cache to the sibling-RM argmax — from this run's metric rows, or under
--load_generations recomputed from the cached ``select`` per-example logs — plus
the final checkpoint, which is what would expose an overoptimized gold RM.

Also covers ``resolve_load_generations_source``: a judge-only pass must pin its
source dir and inherit the generating run's wandb id *before* wandb init, or its
metrics land on a run holding none of the curves they are read against.
"""
import json
import os
import sys
import types

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from policy_eval.eval_utils import (
    checkpoints_to_judge,
    require_selection_available,
    resolve_load_generations_source,
    restrict_deferred_cache,
    selected_checkpoint_from_cache,
    selected_checkpoint_from_rows,
    wandb_run_id_from_generations_dir,
)
from policy_eval.selection import SELECTION_METRIC, SELECTION_SCORE_COLUMN


def _rows():
    return [
        {"checkpoint": 100, SELECTION_METRIC: 0.1},
        {"checkpoint": 200, SELECTION_METRIC: 0.9},
        {"checkpoint": 300, SELECTION_METRIC: 0.5},
    ]


def _args(**kw):
    base = dict(judge_selected_checkpoint_only=True, judge_final_checkpoint=True,
                load_generations=False, benchmarks="preference,select")
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
    kept = restrict_deferred_cache(cache, [200])
    assert set(kept) == {("preference", 200), ("arena_hard", 200)}


def test_restrict_deferred_cache_keeps_every_requested_checkpoint():
    cache = {("preference", 100): "a", ("preference", 200): "b",
             ("preference", 300): "c", ("arena_hard", 300): "d"}
    kept = restrict_deferred_cache(cache, [200, 300])
    assert set(kept) == {("preference", 200), ("preference", 300),
                         ("arena_hard", 300)}


def test_restrict_deferred_cache_missing_checkpoint_raises():
    with pytest.raises(ValueError, match="checkpoint-200"):
        restrict_deferred_cache({("preference", 100): "a"}, [200])
    # One missing out of several is still a hard error: silently judging a
    # subset would report a number for a checkpoint that was never judged.
    with pytest.raises(ValueError, match="checkpoint-400"):
        restrict_deferred_cache({("preference", 100): "a"}, [100, 400])


def test_checkpoints_to_judge_adds_the_final_checkpoint():
    assert checkpoints_to_judge(200, [100, 200, 300], _args()) == [200, 300]


def test_checkpoints_to_judge_collapses_when_selected_is_final():
    assert checkpoints_to_judge(300, [100, 200, 300], _args()) == [300]


def test_checkpoints_to_judge_honours_the_flag():
    args = _args(judge_final_checkpoint=False)
    assert checkpoints_to_judge(200, [100, 200, 300], args) == [200]


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


# ---------------------------------------------------------------------------
# resolve_load_generations_source: pin the dir, inherit the wandb run
# ---------------------------------------------------------------------------

def _load_args(tmp_path, **kw):
    base = dict(
        load_generations=True, load_generations_dir="", wandb_run_id=None,
        disable_wandb=False, checkpoints_dir="/runs/grpo-1", debug=False,
        per_example_dir=str(tmp_path / "this_run_per_example"),
        output_file="evaluation_results.csv",
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _generations_dir(tmp_path, name="prior_per_example", *, manifest=None):
    d = tmp_path / name
    d.mkdir()
    pd.DataFrame({"response_text": ["r"]}).to_parquet(
        d / "preference__checkpoint-100.parquet")
    if manifest is not None:
        (d / "_manifest.json").write_text(json.dumps(manifest))
    return str(d)


def test_wandb_run_id_from_generations_dir_prefers_the_live_id(tmp_path):
    d = _generations_dir(tmp_path, manifest={
        "wandb_run_id": "live123", "args": {"wandb_run_id": "resumed456"}})
    assert wandb_run_id_from_generations_dir(d) == "live123"


def test_wandb_run_id_from_generations_dir_falls_back_to_args(tmp_path):
    # Manifest written before the top-level id existed, by a run that resumed.
    d = _generations_dir(tmp_path, manifest={"args": {"wandb_run_id": "resumed456"}})
    assert wandb_run_id_from_generations_dir(d) == "resumed456"


def test_wandb_run_id_from_generations_dir_missing_manifest(tmp_path):
    assert wandb_run_id_from_generations_dir(_generations_dir(tmp_path)) is None


def test_resolve_load_generations_source_pins_dir_and_resumes_run(tmp_path):
    d = _generations_dir(tmp_path, manifest={
        "wandb_run_id": "live123", "args": {"checkpoints_dir": "/runs/grpo-1"}})
    args = _load_args(tmp_path)
    assert resolve_load_generations_source(args, [_bench("preference")]) == d
    # Pinned, so the later load can't auto-discover a *different* dir.
    assert args.load_generations_dir == d
    assert args.wandb_run_id == "live123"


def test_resolve_load_generations_source_keeps_explicit_overrides(tmp_path):
    d = _generations_dir(tmp_path, manifest={"wandb_run_id": "live123"})
    args = _load_args(tmp_path, load_generations_dir=d, wandb_run_id="mine")
    assert resolve_load_generations_source(args, [_bench("preference")]) == d
    assert args.wandb_run_id == "mine"


def test_resolve_load_generations_source_without_a_recorded_id(tmp_path):
    d = _generations_dir(tmp_path)
    args = _load_args(tmp_path)
    assert resolve_load_generations_source(args, [_bench("preference")]) == d
    assert args.wandb_run_id is None   # a new run, announced but not fatal


def test_resolve_load_generations_source_defers_when_nothing_to_judge(tmp_path):
    # No deferred evaluator / no cached dir: run_load_generations raises the
    # actionable error, so this must not pre-empt it with a worse one.
    args = _load_args(tmp_path)
    assert resolve_load_generations_source(
        args, [_bench("preference", judged=False)]) is None
    assert resolve_load_generations_source(args, [_bench("preference")]) is None
    assert args.wandb_run_id is None
