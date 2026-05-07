#!/usr/bin/env python3
"""Verify that every `[name](wandb-run-url)` entry in EXPERIMENTS.md uses the
correct run name.

Parses all markdown links in EXPERIMENTS.md whose target is a W&B run URL,
fetches each run from W&B, and reports entries whose link text doesn't match
the run's actual name. Also reports run IDs in the file that no longer exist
upstream.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import wandb

PROJECT = "distill-llms/policy-evaluation"
EXPERIMENTS_MD = Path(__file__).resolve().parent.parent / "EXPERIMENTS.md"

# Match [text](https://wandb.ai/<entity>/<project>/runs/<id>)
LINK_RE = re.compile(
    r"\[([^\]]+)\]\(https://wandb\.ai/([^/)]+/[^/)]+)/runs/([A-Za-z0-9_\-]+)\)"
)


def parse_links(text: str) -> list[tuple[str, str, str]]:
    """Return [(link_text, project, run_id), ...] for every wandb link."""
    return LINK_RE.findall(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default=PROJECT,
        help=f"Only verify links for this W&B project (default: {PROJECT}).",
    )
    parser.add_argument(
        "--file",
        default=str(EXPERIMENTS_MD),
        help="Markdown file to verify (default: EXPERIMENTS.md).",
    )
    args = parser.parse_args()

    md_path = Path(args.file)
    text = md_path.read_text()
    links = parse_links(text)
    print(f"Found {len(links)} wandb link(s) in {md_path.name}.")

    in_scope = [(lt, rid) for lt, proj, rid in links if proj == args.project]
    out_of_scope = len(links) - len(in_scope)
    if out_of_scope:
        print(f"Skipping {out_of_scope} link(s) outside project {args.project}.")

    # de-dup run IDs but keep all (link_text, run_id) pairs to flag every entry
    unique_ids = sorted({rid for _, rid in in_scope})
    print(f"Fetching {len(unique_ids)} unique run(s) from W&B…")

    api = wandb.Api()
    real_names: dict[str, str | None] = {}  # run_id -> name (None if missing)
    for rid in unique_ids:
        try:
            run = api.run(f"{args.project}/{rid}")
            real_names[rid] = run.name or ""
        except wandb.errors.CommError:
            real_names[rid] = None
        except Exception as e:  # noqa: BLE001
            print(f"  warning: failed to fetch {rid}: {e}", file=sys.stderr)
            real_names[rid] = None

    def normalize(s: str) -> str:
        return re.sub(r"\s+", "", s)

    mismatches: list[tuple[str, str, str]] = []  # (run_id, in_md, on_wandb)
    missing: list[str] = []
    for link_text, run_id in in_scope:
        actual = real_names.get(run_id)
        if actual is None:
            missing.append(run_id)
            continue
        if normalize(link_text) != normalize(actual):
            mismatches.append((run_id, link_text, actual))

    if mismatches:
        print(f"\n{len(mismatches)} name mismatch(es):")
        for run_id, in_md, on_wandb in mismatches:
            print(f"  {run_id}")
            print(f"    in MD : {in_md}")
            print(f"    on W&B: {on_wandb}")

    if missing:
        unique_missing = sorted(set(missing))
        print(f"\n{len(unique_missing)} run id(s) not found on W&B:")
        for rid in unique_missing:
            print(f"  {rid}")

    if not mismatches and not missing:
        print("All link names match W&B run names.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
