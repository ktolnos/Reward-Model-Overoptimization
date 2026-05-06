#!/usr/bin/env python3
"""Pull runs from W&B and prepend new ones to EXPERIMENTS.md.

Logic:
- Fetch all runs from ``distill-llms/policy-evaluation``.
- Read existing EXPERIMENTS.md (if any) and collect the run IDs that already
  appear anywhere in its text.
- If the file already has runs, find the oldest one that is still present and
  only consider W&B runs created strictly after that timestamp. This means
  manually-deleted runs do not get re-added.
- Skip any run whose ID is already somewhere in the file.
- Prepend the remaining new runs to the top of the file, ordered newest first.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import wandb

PROJECT = "distill-llms/policy-evaluation"
RUN_URL_TEMPLATE = "https://wandb.ai/{project}/runs/{run_id}"
EXPERIMENTS_MD = Path(__file__).resolve().parent.parent / "EXPERIMENTS.md"

# Matches /runs/<id> in any wandb URL form, with optional trailing chars.
RUN_ID_RE = re.compile(r"/runs/([A-Za-z0-9_\-]+)")


def existing_run_ids(text: str) -> set[str]:
    return set(RUN_ID_RE.findall(text))


def fetch_runs():
    api = wandb.Api()
    # order by created_at desc; keep the API call lazy-iterable
    return api.runs(PROJECT, order="-created_at")


def format_line(run) -> str:
    url = RUN_URL_TEMPLATE.format(project=PROJECT, run_id=run.id)
    date = (run.created_at or "")[:10]  # YYYY-MM-DD
    name = run.name or run.id
    return f"- {date} — [{name}]({url})"


def select_new_runs(runs: Iterable, existing_ids: set[str], cutoff: str | None):
    """Return runs that are not in the file and (if cutoff set) newer than it."""
    selected = []
    for run in runs:
        if run.id in existing_ids:
            continue
        if cutoff is not None and (run.created_at or "") <= cutoff:
            continue
        selected.append(run)
    return selected


def oldest_present_created_at(runs_in_order, existing_ids: set[str]) -> str | None:
    """Walk runs newest->oldest; the last one we see that's still in the file
    is the oldest still-present run. Returns its created_at or None."""
    last_seen = None
    for run in runs_in_order:
        if run.id in existing_ids:
            last_seen = run.created_at
    return last_seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be added without modifying the file.",
    )
    parser.add_argument(
        "--no-cutoff",
        action="store_true",
        help="Add every missing run, even those older than the oldest entry "
        "still in the file (default is to skip them so deleted runs stay gone).",
    )
    args = parser.parse_args()

    if EXPERIMENTS_MD.exists():
        existing_text = EXPERIMENTS_MD.read_text()
    else:
        existing_text = ""

    existing_ids = existing_run_ids(existing_text)
    print(f"Found {len(existing_ids)} run id(s) already in {EXPERIMENTS_MD.name}.")

    runs = list(fetch_runs())
    print(f"Fetched {len(runs)} run(s) from {PROJECT}.")

    cutoff: str | None = None
    if existing_ids and not args.no_cutoff:
        cutoff = oldest_present_created_at(runs, existing_ids)
        if cutoff:
            print(f"Cutoff: only adding runs newer than {cutoff}.")
        else:
            print(
                "Note: existing run ids in file weren't matched against W&B "
                "(maybe deleted upstream); falling back to no cutoff."
            )

    new_runs = select_new_runs(runs, existing_ids, cutoff)
    print(f"{len(new_runs)} new run(s) to add.")

    if not new_runs:
        return 0

    new_lines = [format_line(r) for r in new_runs]  # already newest-first
    new_block = "\n".join(new_lines) + "\n"

    if args.dry_run:
        print("--- would prepend ---")
        print(new_block, end="")
        return 0

    if existing_text and not existing_text.endswith("\n"):
        existing_text += "\n"
    EXPERIMENTS_MD.write_text(new_block + existing_text)
    print(f"Updated {EXPERIMENTS_MD}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
