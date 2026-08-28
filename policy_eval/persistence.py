"""Per-example persistence for policy evaluation.

Because the headline metric is chosen late and checkpoint weights are finite,
**every eval persists the per-example raw numbers** so any aggregation (panel
mean/min, win-rate vs any reference, style-controlled, length-controlled) can be
recomputed later without re-running generation or re-scoring with the reward
models.

One durable artifact is written per ``(benchmark, checkpoint)``, decoupled from
the checkpoint weights, with **one row per ``(prompt, response)``**:

    prompt_uid                       join key across evaluators / checkpoints
    sample_idx                       which of the n samples for this prompt
    response_text                    the (thinking-stripped) response
    response_raw_text                pre-strip response as returned by vLLM
    response_token_len               generated token count (truncation accounting)
    finish_reason                    "stop" | "length" | ...
    over_budget (bool)               response exceeded the response-token budget
    score__<evaluator>               one per RM / judge evaluator (policy side)
    chosen_or_baseline_score__<rm>   reference scores for win-rate
    ... (evaluator-specific extras: kl__*, ifeval_*, judge labels, battles)

Format: parquet by default (columnar, typed, fast to re-load with pandas);
``jsonl`` fallback. Files live under a per-run directory derived from
``--output_file`` (or ``--per_example_dir``), so they sit next to the run's CSV
and survive checkpoint deletion.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def example_uid(example) -> str:
    """Stable join key for one benchmark example.

    Prefers a natural id when the benchmark carries one (Arena-Hard ``uid``,
    IFEval ``key``); otherwise a content hash of the prompt messages. Hashing
    the prompt content makes the uid stable across checkpoints **and** across
    separate runs of the same prompt set, so per-example logs can always be
    joined back together.
    """
    md = getattr(example, "metadata", {}) or {}
    if md.get("uid") is not None:
        return str(md["uid"])
    ifeval = md.get("ifeval")
    if isinstance(ifeval, dict) and ifeval.get("key") is not None:
        return f"ifeval-{ifeval['key']}"
    h = hashlib.sha256()
    for m in example.prompt_messages:
        h.update((m.get("role", "") or "").encode("utf-8", "replace"))
        h.update(b"\x00")
        h.update((m.get("content", "") or "").encode("utf-8", "replace"))
        h.update(b"\x01")
    return "h-" + h.hexdigest()[:16]


class PerExampleRecorder:
    """Accumulate one row per ``(prompt, response)`` for a single
    ``(benchmark, checkpoint)`` and write it to disk.

    Row order matches the flattened ``GenerationResult.responses`` order:
    ``[prompt0/sample0, prompt0/sample1, ..., prompt1/sample0, ...]`` so row
    ``r`` corresponds to prompt ``r // n`` and sample ``r % n`` where ``n`` is
    ``n_responses_per_example``.

    Evaluators add columns during their ``evaluate`` call via the handle on
    ``EvalContext.recorder``:
      - ``add_response_column``: one value per response row (RM scores, KL, ...).
      - ``add_prompt_column``: one value per prompt, broadcast across its n
        samples (chosen/baseline scores, judge battles — these are per-prompt).
      - ``add_sparse_prompt_column``: per-prompt values for a subset of prompts
        (Arena-Hard per-category mode scores only a category's prompts).

    A missing column for some rows is left as NaN/None — re-aggregation code
    selects the rows it needs (e.g. ``over_budget == False``).
    """

    def __init__(
        self,
        *,
        benchmark_name: str,
        checkpoint_num: int,
        n_responses_per_example: int,
        n_examples: int,
    ):
        self.benchmark_name = benchmark_name
        self.checkpoint_num = checkpoint_num
        self.n = max(1, int(n_responses_per_example))
        self.n_examples = n_examples
        self.n_rows = n_examples * self.n
        # Two column spaces, broadcast/merged into one frame at write time:
        #   _response_cols: one value per response row (len n_rows)
        #   _prompt_cols:   one value per prompt (len n_examples), repeated x n
        # Insertion order is preserved across both for a stable column layout.
        self._response_cols: "OrderedDict[str, list]" = OrderedDict()
        self._prompt_cols: "OrderedDict[str, list]" = OrderedDict()
        self._order: List[tuple] = []  # (space, name) in insertion order

    # ------------------------------------------------------------------
    # Column adders
    # ------------------------------------------------------------------
    def add_response_column(self, name: str, values: Sequence[Any]) -> None:
        if len(values) != self.n_rows:
            raise ValueError(
                f"[per-example] column '{name}': {len(values)} values vs "
                f"{self.n_rows} response rows ({self.n_examples} prompts x {self.n})"
            )
        if name not in self._response_cols:
            self._order.append(("resp", name))
        self._response_cols[name] = [_to_py(v) for v in values]

    def add_prompt_column(self, name: str, values: Sequence[Any]) -> None:
        """Per-prompt values (len ``n_examples``); broadcast across n samples."""
        if len(values) != self.n_examples:
            raise ValueError(
                f"[per-example] prompt column '{name}': {len(values)} values vs "
                f"{self.n_examples} prompts"
            )
        if name not in self._prompt_cols:
            self._order.append(("prompt", name))
        self._prompt_cols[name] = [_to_py(v) for v in values]

    def add_sparse_prompt_column(
        self, name: str, indices: Sequence[int], values: Sequence[Any],
        *, fill: Any = None,
    ) -> None:
        """Per-prompt values for a subset of prompt indices.

        Other prompts get ``fill`` on first write. If the column already exists
        (e.g. Arena-Hard per-category mode contributing one category at a time),
        only the supplied indices are updated — existing values are preserved.
        """
        if len(indices) != len(values):
            raise ValueError(
                f"[per-example] sparse column '{name}': {len(indices)} indices "
                f"vs {len(values)} values"
            )
        if name in self._prompt_cols:
            per_prompt = self._prompt_cols[name]
        else:
            per_prompt = [fill] * self.n_examples
            self._prompt_cols[name] = per_prompt
            self._order.append(("prompt", name))
        for i, v in zip(indices, values):
            per_prompt[i] = _to_py(v)

    def has_column(self, name: str) -> bool:
        return name in self._response_cols or name in self._prompt_cols

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        data: "OrderedDict[str, list]" = OrderedDict()
        for space, name in self._order:
            if space == "resp":
                data[name] = self._response_cols[name]
            else:
                col: List[Any] = []
                for v in self._prompt_cols[name]:
                    col.extend([v] * self.n)
                data[name] = col
        df = pd.DataFrame(data)
        # Constant identity columns up front (handy when files are concatenated).
        df.insert(0, "checkpoint", self.checkpoint_num)
        df.insert(0, "benchmark", self.benchmark_name)
        return df

    def is_empty(self) -> bool:
        return self.n_rows == 0 or not self._order


def init_base_columns(
    recorder: PerExampleRecorder,
    examples,
    generation,
    *,
    response_token_budget: Optional[int],
) -> None:
    """Populate the base columns from a benchmark's generation result.

    Over-budget = response truncated by the generation cap (``finish_reason ==
    'length'``) or longer than ``response_token_budget`` (the 1024-token
    response budget; lets us flag over-budget even on benchmarks whose
    ``max_new_tokens`` is set higher, e.g. IFEval/Arena-Hard).
    """
    n = recorder.n
    n_rows = recorder.n_rows
    uids = [example_uid(ex) for ex in examples]
    recorder.add_prompt_column("prompt_uid", uids)
    # Full prompt conversation as JSON, so cached responses can be re-scored by a
    # new reward model offline (no checkpoint, no regeneration).
    recorder.add_prompt_column(
        "prompt_messages_json",
        [json.dumps(ex.prompt_messages, ensure_ascii=False) for ex in examples],
    )
    recorder.add_response_column("sample_idx", [r % n for r in range(n_rows)])
    recorder.add_response_column("response_text", list(generation.responses))
    recorder.add_response_column("response_raw_text", list(generation.raw_responses))

    token_lens = generation.response_token_lens
    if token_lens is None:
        token_lens = [None] * n_rows
    recorder.add_response_column("response_token_len", list(token_lens))
    recorder.add_response_column("finish_reason", list(generation.finish_reasons))

    budget = response_token_budget if (response_token_budget and response_token_budget > 0) else None
    over_budget = []
    for fr, tl in zip(generation.finish_reasons, token_lens):
        flag = fr == "length"
        if budget is not None and tl is not None:
            flag = flag or (tl > budget)
        over_budget.append(bool(flag))
    recorder.add_response_column("over_budget", over_budget)


def _to_py(v: Any) -> Any:
    """Coerce numpy scalars to native python so parquet/json don't choke."""
    if isinstance(v, (np.generic,)):
        return v.item()
    return v


# ---------------------------------------------------------------------------
# Run-level directory / manifest helpers
# ---------------------------------------------------------------------------

def resolve_per_example_dir(args) -> str:
    """Where per-example logs go. Always a real path — persistence is never off.

    Uses ``--per_example_dir`` when set, otherwise a default derived from
    ``--output_file`` so the logs sit next to the run's CSV
    (``<stem>_per_example/``). The ``--debug`` stem is mirrored from the CSV
    (``..._debug`` suffix on the output file).
    """
    raw = args.per_example_dir
    if raw:
        return raw
    out = args.output_file or "evaluation_results.csv"
    if args.debug and out.endswith(".csv"):
        out = out.replace(".csv", "_debug.csv")
    stem = os.path.splitext(out)[0] or "evaluation_results"
    return f"{stem}_per_example"


def recorder_path(per_example_dir: str, benchmark_name: str, checkpoint_num: int,
                  fmt: str = "parquet") -> str:
    safe_bench = benchmark_name.replace("/", "_")
    ext = "jsonl" if fmt == "jsonl" else "parquet"
    return os.path.join(
        per_example_dir, f"{safe_bench}__checkpoint-{checkpoint_num}.{ext}"
    )


def write_recorder(
    recorder: PerExampleRecorder, per_example_dir: str, *, fmt: str = "parquet",
) -> Optional[str]:
    """Write one recorder to disk; returns the path (or None if empty)."""
    if recorder.is_empty():
        return None
    os.makedirs(per_example_dir, exist_ok=True)
    path = recorder_path(per_example_dir, recorder.benchmark_name,
                         recorder.checkpoint_num, fmt=fmt)
    df = recorder.to_dataframe()
    if fmt == "jsonl":
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        df.to_parquet(path, index=False)
    print(f"[per-example] wrote {len(df)} rows -> {path}")
    return path


def redacted_args_dict(args) -> Dict[str, Any]:
    """``ScriptArguments`` as a dict with API keys blanked out.

    The manifest and the wandb run config both serialise every flag, so a key
    passed on the command line rather than through its env var would land on disk
    and in the wandb project, readable by anyone with access to the run.
    """
    out = dataclasses.asdict(args)
    for k, v in out.items():
        if k.endswith("_api_key") and v:
            out[k] = "<redacted>"
    return out


def write_manifest(
    per_example_dir: str, args, benchmarks, *, wandb_run_id: Optional[str] = None,
) -> None:
    """Write a small JSON manifest describing the run, for later joins.

    Captures the inputs that determine what the scores *mean* (dataset, split,
    RM identities, decoding budgets) so re-aggregation knows the provenance even
    after the checkpoints and the wandb run are gone.

    ``wandb_run_id`` is the *live* run id, which differs from ``args`` whenever
    this run created a run rather than resuming one. A later judge-only pass over
    these generations reads it back to log onto the same run
    (``resolve_load_generations_source``).
    """
    os.makedirs(per_example_dir, exist_ok=True)
    manifest: Dict[str, Any] = {
        "git": _git_info(),
        "wandb_run_id": wandb_run_id,
        # The complete ScriptArguments — every flag the eval ran with, so the run
        # is fully reproducible from the manifest alone (API keys excepted).
        "args": redacted_args_dict(args),
        "benchmarks": {
            b.name: {
                "evaluators": [e.name for e in b.evaluators],
                "thinking": b.generation_config.thinking,
                "n_responses_per_example": b.generation_config.n_responses_per_example,
            }
            for b in benchmarks
        },
    }
    path = os.path.join(per_example_dir, "_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[per-example] wrote manifest -> {path}")


def _git_info() -> Dict[str, Any]:
    """Current repo commit / branch / dirty flag, for reproducibility.

    Returns ``{"available": False, "error": ...}`` if git isn't usable (e.g. the
    code was copied outside a checkout) rather than failing the eval.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _run(*cmd: str) -> str:
        return subprocess.run(
            cmd, cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout.strip()

    try:
        return {
            "available": True,
            "commit": _run("git", "rev-parse", "HEAD"),
            "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_run("git", "status", "--porcelain")),
        }
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        return {"available": False, "error": str(e)}
