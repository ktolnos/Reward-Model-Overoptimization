#!/usr/bin/env python3
"""Dump full metadata for every run in distill-llms/policy-evaluation to JSON.

Configs aren't auto-loaded when iterating ``api.runs()``; we re-fetch each run
by id with ``api.run(path)`` to populate them.
"""
import json
import sys
from pathlib import Path
import wandb

PROJECT = "distill-llms/policy-evaluation"
OUT = Path(__file__).resolve().parent.parent / "wandb_runs_dump.json"


def safe_dict(d):
    try:
        return {k: v for k, v in dict(d).items()}
    except Exception:
        return {}


def main():
    api = wandb.Api(timeout=60)
    runs_iter = api.runs(PROJECT, order="-created_at")
    ids = [(r.id, r.name, r.created_at) for r in runs_iter]
    print(f"Listing returned {len(ids)} runs.", file=sys.stderr)

    rows = []
    for i, (rid, name, created) in enumerate(ids):
        try:
            run = api.run(f"{PROJECT}/{rid}")
            cfg = safe_dict(run.config)
            try:
                summary = {k: v for k, v in run.summary.items() if not k.startswith("_")}
            except Exception:
                summary = {}
            rows.append({
                "id": rid,
                "name": name,
                "state": run.state,
                "created_at": created,
                "tags": list(run.tags or []),
                "notes": run.notes or "",
                "config": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                            for k, v in cfg.items()},
                "summary": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                            for k, v in summary.items()},
                "commit": getattr(run, "commit", None),
                "group": getattr(run, "group", None),
                "job_type": getattr(run, "job_type", None),
            })
        except Exception as e:
            print(f"  ! failed {rid}: {e}", file=sys.stderr)
            rows.append({"id": rid, "name": name, "created_at": created, "error": str(e)})
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(ids)} fetched", file=sys.stderr)
            OUT.write_text(json.dumps(rows, indent=2, default=str))
    OUT.write_text(json.dumps(rows, indent=2, default=str))
    print(f"Wrote {len(rows)} runs to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
