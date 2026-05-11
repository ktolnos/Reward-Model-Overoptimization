#!/usr/bin/env python3
"""Download run metadata from the relevant W&B projects to a local JSON cache.

THIS MUST BE RUN MANUALLY before annotate_runs.py — it talks to W&B and is slow.

Default behavior is incremental: any project whose cache file already exists
is fetched only for IDs not present in the cache (plus any cached entries
that were saved as `error` placeholders on a prior crash). Use --force to
re-download every run in the listed projects.

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
    try:
        metadata = dict(run.metadata) if run.metadata else {}
    except Exception:
        metadata = {}
    return {
        "id": rid,
        "name": name,
        "state": run.state,
        "created_at": created,
        "tags": list(run.tags or []),
        "config": {k: jsonable(v) for k, v in cfg.items()},
        "summary": {k: jsonable(v) for k, v in summary.items()},
        "metadata": {k: jsonable(v) for k, v in metadata.items()},
    }


def fetch_project(api, project, save_path, force):
    existing: list[dict] = []
    cached_ids: set[str] = set()
    if save_path.exists() and not force:
        try:
            existing = json.loads(save_path.read_text())
            # Treat error placeholders as un-cached so they get retried.
            cached_ids = {r["id"] for r in existing
                          if isinstance(r, dict) and "id" in r and "error" not in r}
        except Exception as e:
            print(f"  warn: couldn't parse {save_path.name}: {e}; "
                  f"rebuilding from scratch", file=sys.stderr)
            existing = []
            cached_ids = set()

    mode = "incremental" if cached_ids else "full"
    print(f"[fetch] {project} ({mode})", file=sys.stderr)
    runs_iter = api.runs(project, order="-created_at")
    ids = [(r.id, r.name, r.created_at) for r in runs_iter]
    new_ids = [t for t in ids if t[0] not in cached_ids]
    print(f"  {len(ids)} remote, {len(cached_ids)} cached, "
          f"{len(new_ids)} to fetch", file=sys.stderr)

    if not new_ids:
        return

    new_rows: list[dict] = []

    def save() -> None:
        # New rows are newest-first (api.runs is ordered by -created_at) and
        # the existing cache is also newest-first — prepending preserves order.
        # Drop any existing entry whose id we just refetched (e.g. retried
        # error placeholders).
        new_row_ids = {nr.get("id") for nr in new_rows}
        merged = new_rows + [r for r in existing if r.get("id") not in new_row_ids]
        save_path.write_text(json.dumps(merged, default=str))

    for i, (rid, name, created) in enumerate(new_ids):
        try:
            run = api.run(f"{project}/{rid}")
            new_rows.append(serialize_run(rid, name, created, run))
        except Exception as e:
            print(f"  ! {rid}: {e}", file=sys.stderr)
            new_rows.append({"id": rid, "name": name, "created_at": created,
                             "error": str(e)})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(new_ids)}", file=sys.stderr)
            save()
    save()
    print(f"  -> {save_path} (+{len(new_rows)} fetched)", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects", nargs="*", default=PROJECTS,
                        help="Subset of projects to fetch.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download every run, not just new ones.")
    args = parser.parse_args()

    CACHE_DIR.mkdir(exist_ok=True)
    api = wandb.Api(timeout=60)
    for project in args.projects:
        save_path = CACHE_DIR / (project.replace("/", "_") + ".json")
        fetch_project(api, project, save_path, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
