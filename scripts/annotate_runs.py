#!/usr/bin/env python3
"""Annotate every wandb-run link in EXPERIMENTS.md with its base model, and
build RUN_LINKS.md showing the training chain (training run, SFT base, RMs)
for every GRPO/DPO/SFT run.

Reads cached data from wandb_cache/ — run download_run_data.py first.

Annotation rules (EXPERIMENTS.md):
  - "[name](link)" gets " {SHORT}" appended (existing "{...}" is replaced).
  - SHORT is the base model name with "-SFT" suffix if the policy starts from
    an SFT'd model.
  - Known short names:
      Qwen/Qwen3-0.6B -> 0.6B
      Qwen/Qwen3-4B   -> 3-4B
      Qwen/Qwen3.5-4B -> 3.5-4B
      etc.

RUN_LINKS.md per training run:
  ## [training run name](link) <- [sft run name](link) <- base model name
  RMs:
  - [RM name](link) <- base model name (Xeps), <dataset>
  ...
  (SFT or RMs sections are omitted when not applicable.)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "wandb_cache"
EXPERIMENTS_MD = ROOT / "EXPERIMENTS.md"
RUN_LINKS_MD = ROOT / "RUN_LINKS.md"

PROJECT_FILES = {
    "eval": "distill-llms_policy-evaluation.json",
    "grpo": "distill-llms_grpo.json",
    "dpo":  "distill-llms_dpo.json",
    "sft":  "distill-llms_sft.json",
    "rm":   "distill-llms_huggingface.json",
}
PROJECT_PATHS = {
    "eval": "distill-llms/policy-evaluation",
    "grpo": "distill-llms/grpo",
    "dpo":  "distill-llms/dpo",
    "sft":  "distill-llms/sft",
    "rm":   "distill-llms/huggingface",
}

# Convention:
#   Qwen3 family    — Instruct is the default; Base variants get a "-Base" suffix.
#   Qwen3.5 family  — Base is the default; Instruct variants get a "-Instruct" suffix.
MODEL_SHORT_NAMES = {
    "Qwen/Qwen3-0.6B":          "0.6B",
    "Qwen/Qwen3-0.6B-Base":     "0.6B-Base",
    "Qwen/Qwen3-1.7B":          "1.7B",
    "Qwen/Qwen3-1.7B-Base":     "1.7B-Base",
    "Qwen/Qwen3-4B":            "3-4B",
    "Qwen/Qwen3-4B-Base":       "3-4B-Base",
    "Qwen/Qwen3-4B-Instruct":   "3-4B",
    "Qwen/Qwen3-8B":            "8B",
    "Qwen/Qwen3-8B-Base":       "8B-Base",
    "Qwen/Qwen3.5-4B-Base":     "3.5-4B",
    "Qwen/Qwen3.5-4B":          "3.5-4B-Instruct",
    "Qwen/Qwen3.5-4B-Instruct": "3.5-4B-Instruct",
}

# Config keys that may hold the upstream model identifier — checked in order.
MODEL_PATH_KEYS = (
    "model_name_or_path",
    "_name_or_path",
    "base_model_name_or_path",
    "model_path",
)


def upstream_model_path(cfg: dict) -> str | None:
    for key in MODEL_PATH_KEYS:
        v = cfg.get(key)
        if isinstance(v, str) and v:
            return v
    return None

# [text](https://wandb.ai/<entity>/<project>/runs/<id>) optionally followed by
# " {anything}" — the optional group lets us replace existing annotations.
LINK_RE = re.compile(
    r"\[([^\]]+)\]\(https://wandb\.ai/([^/)]+/[^/)]+)/runs/([A-Za-z0-9_\-]+)\)"
    r"(?:[ \t]*\{[^}\n]*\})?"
)


# ---------- cache loading ------------------------------------------------

def load_cache(kind: str) -> list[dict]:
    path = CACHE_DIR / PROJECT_FILES[kind]
    if not path.exists():
        print(f"warning: missing {path.name} — run download_run_data.py first.",
              file=sys.stderr)
        return []
    return json.loads(path.read_text())


# ---------- path matching helpers ----------------------------------------

def _output_dirs(runs: Iterable[dict]) -> dict[str, dict]:
    """Map output_dir (without trailing slash) -> run."""
    out = {}
    for r in runs:
        cfg = r.get("config") or {}
        d = cfg.get("output_dir") or cfg.get("logging_dir")
        if isinstance(d, str) and d:
            out[d.rstrip("/")] = r
    return out


def find_run_for_path(path: str | None, indices: dict[str, dict[str, dict]]
                      ) -> tuple[str | None, dict | None]:
    """Given a checkpoint or output path, find which run produced it.

    Tries exact match on each project's output_dir, then prefix match
    (for paths that include /checkpoint-N).
    Returns (kind, run) or (None, None).
    """
    if not path or not isinstance(path, str):
        return None, None
    p = path.rstrip("/")
    # exact match first across all kinds
    for kind in ("grpo", "dpo", "sft", "rm"):
        if p in indices[kind]:
            return kind, indices[kind][p]
    # prefix match: try the longest matching output_dir
    best: tuple[int, str | None, dict | None] = (-1, None, None)
    for kind in ("grpo", "dpo", "sft", "rm"):
        for out_dir, run in indices[kind].items():
            if (p == out_dir
                    or p.startswith(out_dir + "/")
                    or out_dir.startswith(p + "/")):
                if len(out_dir) > best[0]:
                    best = (len(out_dir), kind, run)
    return best[1], best[2]


def short_for_model(name: str | None) -> str | None:
    """Map a HF model id to its short name. Returns None for unknown HF ids
    or for filesystem paths — callers must resolve paths through the run cache
    rather than relying on a path-component fallback that produces gibberish
    like '20260318_161126_1069463'."""
    if not isinstance(name, str) or not name:
        return None
    if name in MODEL_SHORT_NAMES:
        return MODEL_SHORT_NAMES[name]
    # Filesystem path — caller must resolve via run cache.
    if name.startswith("/") or name.startswith("./"):
        return None
    # Unknown HF id — fall back to "<name>" without org prefix so it's still
    # readable in output (won't appear in EXPERIMENTS.md annotations because
    # those only get set when resolve_base returns a known model).
    if "/" in name:
        return name.split("/", 1)[1]
    return name or None


# ---------- chain resolution ---------------------------------------------

def resolve_base(model_path: str | None,
                 indices: dict[str, dict[str, dict]],
                 _seen: set[str] | None = None
                 ) -> tuple[str | None, bool, str]:
    """Walk model_name_or_path until we hit a HuggingFace base model.

    Returns (short_name, sft_in_chain, debug_trail).
    """
    if _seen is None:
        _seen = set()
    if not model_path or not isinstance(model_path, str):
        return None, False, "empty"
    if model_path in _seen:
        return short_for_model(model_path), False, f"cycle@{model_path}"
    _seen.add(model_path)

    # HF id (no leading slash) — terminal base model.
    if not model_path.startswith("/") and not model_path.startswith("./"):
        return short_for_model(model_path), False, f"hf:{model_path}"

    # Filesystem path — must resolve via the run cache. If we can't, refuse to
    # guess a label rather than producing a wrong one.
    kind, run = find_run_for_path(model_path, indices)
    if not run:
        return None, False, f"no-run-for-path:{model_path}"
    upstream = upstream_model_path(run.get("config") or {})
    short, sft, trail = resolve_base(upstream, indices, _seen)
    if kind == "sft":
        sft = True
    return short, sft, f"{kind}:{run.get('name') or run['id']} -> {trail}"


def annotation_for_eval_run(eval_run: dict,
                            indices: dict[str, dict[str, dict]]
                            ) -> tuple[str | None, str]:
    """Return (annotation, reason). When the policy can't be unambiguously
    identified through `checkpoints_dir -> training run -> upstream chain`,
    refuses to annotate (returns None) — fallbacks like `kl_base_model_path`
    or `baseline_model_path` are deliberately not used because the KL-base or
    eval-baseline isn't necessarily the policy's base."""
    cfg = eval_run.get("config") or {}
    ckpt = cfg.get("checkpoints_dir")
    if not ckpt:
        return None, "no checkpoints_dir on eval run"
    kind, train_run = find_run_for_path(ckpt, indices)
    if train_run is None:
        return None, (f"checkpoints_dir {ckpt!r} doesn't match any GRPO/DPO/SFT/"
                      f"RM run in the cache (deleted or not yet downloaded?)")
    starting = upstream_model_path(train_run.get("config") or {})
    short, sft, trail = resolve_base(starting, indices)
    if kind == "sft":
        sft = True
    if not short:
        return None, (f"can't resolve upstream of {kind}:"
                      f"{train_run.get('name') or train_run['id']} "
                      f"(starting={starting!r}); trail={trail}")
    ann = f"{short}-SFT" if sft else short
    return ann, f"via {kind}:{train_run.get('name') or train_run['id']} -> {trail}"


# ---------- EXPERIMENTS.md annotation ------------------------------------

def annotate_experiments(text: str, eval_index: dict[str, dict],
                         indices: dict[str, dict[str, dict]]
                         ) -> tuple[str, int, list[tuple[str, str, str]]]:
    """Replace every wandb link's annotation with the recomputed one.

    Returns (new_text, n_annotated, skips). Each skip entry is
    (run_id, link_text, reason).
    """
    n_ann = 0
    skips: list[tuple[str, str, str]] = []

    def repl(m: re.Match) -> str:
        nonlocal n_ann
        link_text = m.group(1)
        proj = m.group(2)
        run_id = m.group(3)
        base = f"[{link_text}](https://wandb.ai/{proj}/runs/{run_id})"
        if proj != PROJECT_PATHS["eval"]:
            skips.append((run_id, link_text,
                          f"link target project {proj!r} is not the eval project"))
            return m.group(0)
        run = eval_index.get(run_id)
        if run is None:
            skips.append((run_id, link_text,
                          "run id not in eval cache (deleted on W&B?)"))
            return base
        try:
            ann, reason = annotation_for_eval_run(run, indices)
        except Exception as e:  # noqa: BLE001
            skips.append((run_id, link_text, f"exception: {e!r}"))
            return base
        if ann is None:
            skips.append((run_id, link_text, reason))
            return base
        n_ann += 1
        return f"{base} {{{ann}}}"

    return LINK_RE.sub(repl, text), n_ann, skips


# ---------- RM epoch / dataset extraction --------------------------------

def rm_dataset_label(rm_run: dict) -> str | None:
    cfg = rm_run.get("config") or {}
    for key in ("train_dataset_name", "dataset_name", "train_dataset"):
        if cfg.get(key):
            return str(cfg[key])
    # Output dir often encodes the dataset like ".../data<name>/logs".
    out = (cfg.get("output_dir") or "").rstrip("/")
    m = re.search(r"_data([^/]+?)(?:/logs)?$", out)
    if m:
        return m.group(1)
    # Run name fallback: "..._<dataset_name>" or trailing token after RM path.
    name = rm_run.get("name") or ""
    m = re.search(r"_(helpsteer3[^_]*(?:_[^_]+)*)$", name)
    if m:
        return m.group(1)
    return None


def rm_epochs_for_path(rm_path: str, rm_run: dict) -> str | None:
    """Estimate epochs at the referenced checkpoint.

    rm_path is like .../output_dir[/checkpoint-N].
    Uses summary['train/global_step'] and summary['train/epoch'] to scale.
    """
    cfg = rm_run.get("config") or {}
    summ = rm_run.get("summary") or {}
    final_step = summ.get("train/global_step") or summ.get("global_step")
    final_epoch = summ.get("train/epoch") or summ.get("epoch")
    m = re.search(r"checkpoint-(\d+)", rm_path or "")
    ckpt_step = int(m.group(1)) if m else None
    if ckpt_step and final_step and final_epoch:
        try:
            ratio = float(ckpt_step) / float(final_step)
            return f"{ratio * float(final_epoch):.1f}"
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    if final_epoch is not None:
        try:
            return f"{float(final_epoch):.1f}"
        except (TypeError, ValueError):
            return str(final_epoch)
    if cfg.get("num_train_epochs") is not None:
        return str(cfg["num_train_epochs"])
    return None


# ---------- RUN_LINKS.md generation --------------------------------------

def wandb_url(kind: str, run_id: str) -> str:
    return f"https://wandb.ai/{PROJECT_PATHS[kind]}/runs/{run_id}"


def md_link(kind: str, run: dict) -> str:
    return f"[{run.get('name') or run['id']}]({wandb_url(kind, run['id'])})"


def find_sft_for_grpo(grpo_run: dict, sft_index: dict[str, dict]
                      ) -> dict | None:
    starting = upstream_model_path(grpo_run.get("config") or {})
    if not isinstance(starting, str) or not starting.startswith("/"):
        return None
    p = starting.rstrip("/")
    if p in sft_index:
        return sft_index[p]
    best_len, best_run = -1, None
    for out_dir, run in sft_index.items():
        if p == out_dir or p.startswith(out_dir + "/"):
            if len(out_dir) > best_len:
                best_len, best_run = len(out_dir), run
    return best_run


def _rm_paths_from_run(grpo_run: dict) -> list[str]:
    """RM paths used by a GRPO/DPO run.

    The training script logs per-RM stats as summary keys like
    "reward/<absolute path to checkpoint>", so we can recover the full RM list
    from there. Falls back to a config field if present.
    """
    cfg = grpo_run.get("config") or {}
    summary = grpo_run.get("summary") or {}

    # Primary source: summary keys.
    paths: list[str] = []
    seen: set[str] = set()
    for key in summary:
        if isinstance(key, str) and key.startswith("reward/") and "/" in key[7:]:
            p = key[len("reward/"):]
            if p and p not in seen:
                seen.add(p)
                paths.append(p)
    if paths:
        return paths

    # Fallback: explicit config field.
    raw = cfg.get("reward_model_paths") or cfg.get("reward_model_path") or []
    if isinstance(raw, str):
        if raw.startswith("[") and raw.endswith("]"):
            try:
                raw = json.loads(raw.replace("'", '"'))
            except Exception:
                raw = [raw]
        else:
            raw = [raw]
    return [p for p in raw if isinstance(p, str)]


def find_rms_for_grpo(grpo_run: dict, rm_index: dict[str, dict]
                      ) -> list[tuple[str, dict | None]]:
    """Return list of (rm_path_used, rm_run | None).

    Unmatched paths still appear (with None) so they're visible in the output
    instead of being silently dropped.
    """
    out: list[tuple[str, dict | None]] = []
    seen_keys: set[str] = set()
    for p in _rm_paths_from_run(grpo_run):
        cleaned = p.rstrip("/")
        match = None
        if cleaned in rm_index:
            match = rm_index[cleaned]
        else:
            best_len = -1
            for out_dir, run in rm_index.items():
                if cleaned == out_dir or cleaned.startswith(out_dir + "/"):
                    if len(out_dir) > best_len:
                        best_len, match = len(out_dir), run
        key = match["id"] if match else cleaned
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append((cleaned, match))
    return out


def render_links_block(kind: str, run: dict,
                       indices: dict[str, dict[str, dict]]) -> str:
    cfg = run.get("config") or {}
    lines: list[str] = []

    starting = upstream_model_path(cfg)
    base_short, _, _ = resolve_base(starting, indices)

    sft_run = None
    if kind in ("grpo", "dpo"):
        sft_run = find_sft_for_grpo(run, indices["sft"])

    header = f"## {md_link(kind, run)}"
    if sft_run is not None:
        sft_cfg = sft_run.get("config") or {}
        sft_base, _, _ = resolve_base(upstream_model_path(sft_cfg), indices)
        header += f" <- {md_link('sft', sft_run)}"
        header += f" <- {sft_base or 'unknown'}"
    else:
        header += f" <- {base_short or 'unknown'}"
    lines.append(header)

    if kind in ("grpo", "dpo"):
        rms = find_rms_for_grpo(run, indices["rm"])
        if rms:
            lines.append("RMs:")
            for rm_path, rm_run in rms:
                if rm_run is None:
                    lines.append(f"- `{rm_path}` (no matching RM run in cache)")
                    continue
                rm_cfg = rm_run.get("config") or {}
                rm_base, _, _ = resolve_base(
                    upstream_model_path(rm_cfg), indices)
                eps = rm_epochs_for_path(rm_path, rm_run)
                ds = rm_dataset_label(rm_run)
                bits = [rm_base or "unknown"]
                if eps:
                    bits[0] = f"{bits[0]} ({eps}eps)"
                if ds:
                    bits.append(ds)
                lines.append(f"- {md_link('rm', rm_run)} <- {', '.join(bits)}")
    return "\n".join(lines) + "\n"


def build_run_links(indices: dict[str, dict[str, dict]],
                    runs_by_kind: dict[str, list[dict]]) -> str:
    parts = ["# Training run chains\n",
             "Auto-generated by scripts/annotate_runs.py.\n"]
    for kind in ("grpo", "dpo", "sft"):
        runs = sorted(runs_by_kind.get(kind, []),
                      key=lambda r: r.get("created_at") or "",
                      reverse=True)
        if not runs:
            continue
        parts.append(f"\n## Project: {kind.upper()}\n")
        for run in runs:
            try:
                parts.append(render_links_block(kind, run, indices))
            except Exception as e:  # noqa: BLE001
                parts.append(f"## {md_link(kind, run)}  (error: {e})\n")
    return "\n".join(parts)


# ---------- main ---------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-experiments", action="store_true",
                        help="Skip annotating EXPERIMENTS.md.")
    parser.add_argument("--no-links", action="store_true",
                        help="Skip writing RUN_LINKS.md.")
    parser.add_argument("--experiments-file", default=str(EXPERIMENTS_MD))
    parser.add_argument("--links-file", default=str(RUN_LINKS_MD))
    args = parser.parse_args()

    if not CACHE_DIR.exists():
        print(f"error: {CACHE_DIR} not found. Run download_run_data.py first.",
              file=sys.stderr)
        return 2

    runs_by_kind = {kind: load_cache(kind) for kind in PROJECT_FILES}
    indices = {kind: _output_dirs(runs_by_kind[kind])
               for kind in ("grpo", "dpo", "sft", "rm")}
    eval_index = {r["id"]: r for r in runs_by_kind["eval"] if "id" in r}
    print(
        f"Loaded: {len(runs_by_kind['eval'])} eval, "
        f"{len(runs_by_kind['grpo'])} grpo, {len(runs_by_kind['dpo'])} dpo, "
        f"{len(runs_by_kind['sft'])} sft, {len(runs_by_kind['rm'])} rm.",
        file=sys.stderr,
    )

    if not args.no_experiments:
        md_path = Path(args.experiments_file)
        text = md_path.read_text()
        new_text, n_ann, skips = annotate_experiments(text, eval_index, indices)
        md_path.write_text(new_text)
        print(f"Annotated {n_ann} link(s) in {md_path.name}; "
              f"skipped {len(skips)}.", file=sys.stderr)
        if skips:
            print("Skip details:", file=sys.stderr)
            for run_id, link_text, reason in skips:
                print(f"  {run_id}  ({link_text})", file=sys.stderr)
                print(f"      {reason}", file=sys.stderr)

    if not args.no_links:
        out = build_run_links(indices, runs_by_kind)
        Path(args.links_file).write_text(out)
        print(f"Wrote {args.links_file}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
