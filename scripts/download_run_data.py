#!/usr/bin/env python3
"""Download all run metadata from the relevant W&B projects to a local JSON cache.

THIS MUST BE RUN MANUALLY before annotate_runs.py — it talks to W&B and is slow.

Projects fetched:
  - distill-llms/policy-evaluation  (eval runs)
  - distill-llms/grpo               (GRPO training runs)
  - distill-llms/dpo                (DPO training runs)
  - distill-llms/sft                (SFT training runs)
  - distill-llms/huggingface        (RM training runs)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import wandb

PROJECTS = [
    "distill-llms/policy-evaluation",
    "distill-llms/grpo",
    "distill-llms/dpo",
    "distill-llms/sft",
    "distill-llms/huggingface",
]
CACHE_DIR = Path(__file__).resolve().parent.parent / "wandb_cache"


def safe_dict(d):
    try:
        return {k: v for k, v in dict(d).items()}
    except Exception:
        return {}


def jsonable(v):
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    return str(v)


def serialize_run(rid, name, created, run):
    cfg = safe_dict(run.config)
    try:
        summary = {k: v for k, v in run.summary.items() if not k.startswith("_")}
    except Exception:
        summary = {}
    return {
        "id": rid,
        "name": name,
        "state": run.state,
        "created_at": created,
        "tags": list(run.tags or []),
        "config": {k: jsonable(v) for k, v in cfg.items()},
        "summary": {k: jsonable(v) for k, v in summary.items()},
    }


def fetch_project(api, project, save_path, force):
    if save_path.exists() and not force:
        print(f"[skip] {project}: {save_path.name} already exists "
              f"(use --force to overwrite)", file=sys.stderr)
        return
    print(f"[fetch] {project}", file=sys.stderr)
    runs_iter = api.runs(project, order="-created_at")
    ids = [(r.id, r.name, r.created_at) for r in runs_iter]
    print(f"  {len(ids)} runs", file=sys.stderr)
    rows = []
    for i, (rid, name, created) in enumerate(ids):
        try:
            run = api.run(f"{project}/{rid}")
            rows.append(serialize_run(rid, name, created, run))
        except Exception as e:
            print(f"  ! {rid}: {e}", file=sys.stderr)
            rows.append({"id": rid, "name": name, "created_at": created,
                         "error": str(e)})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ids)}", file=sys.stderr)
            save_path.write_text(json.dumps(rows, default=str))
    save_path.write_text(json.dumps(rows, default=str))
    print(f"  -> {save_path}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects", nargs="*", default=PROJECTS,
                        help="Subset of projects to fetch.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download projects whose cache already exists.")
    args = parser.parse_args()

    CACHE_DIR.mkdir(exist_ok=True)
    api = wandb.Api(timeout=60)
    for project in args.projects:
        save_path = CACHE_DIR / (project.replace("/", "_") + ".json")
        fetch_project(api, project, save_path, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
