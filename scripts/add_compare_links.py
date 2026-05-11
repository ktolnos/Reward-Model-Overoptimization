#!/usr/bin/env python3
"""Insert a `[Comparison](url)` link under every `### …` heading in EXPERIMENTS.md.

Each link points to a *saved* W&B workspace created via the `wandb-workspaces`
SDK. Going via the SDK (rather than constructing a `?filters=...` URL by hand)
is the only reliable way to share filtered views — W&B's UI ignores `filters=`
when `nw=` references a saved workspace, and without `nw=` it redirects to a
default workspace and drops the param.

Each generated workspace:
- has `auto_generate_panels=True` so all logged metrics show charts
- adds a pinned "Main" section at the top with the project's headline metrics
- filters the runset to just the runs in that section by display name

On re-runs the script reads the existing `?nw=<id>` URL from each section and
updates the saved workspace in place rather than creating duplicates.

Run with the venv Python so `wandb-workspaces` is on the path:

    venv/bin/python scripts/add_compare_links.py
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

EXPERIMENTS_MD = Path(__file__).resolve().parent.parent / "EXPERIMENTS.md"

# Headline metrics for the pinned "Main" section.
MAIN_METRICS = [
    "gold_rm/win_rate_vs_chosen",
    "secondary_rm/win_rate_vs_chosen",
    "gold_rm/sc_score",
    "secondary_rm/sc_score",
    "gold_rm/mean",
    "secondary_rm/mean",
    "ifeval/inst_strict_acc",
    "arena_hard/rm_gold_rm/hard_prompt/sc_score",
    "kl/grpo_mean",
]

LINK_RE = re.compile(
    r"\[([^\]]+)\]\(https://wandb\.ai/([^/)]+/[^/)]+)/runs/([A-Za-z0-9_\-]+)\)"
)
HEADING_RE = re.compile(r"^###\s.*$", re.MULTILINE)
COMPARE_LINE_RE = re.compile(
    r"^(?:\[Comparison\]\(https://wandb\.ai/[^)]*\?nw=[^)]*\)\s*)+\n",
    re.MULTILINE,
)
# Old formats from earlier versions of this script — strip on re-runs.
LEGACY_RES = [
    re.compile(r"^_Compare on W&B:[^\n]*\n", re.MULTILINE),
    re.compile(
        r"^(?:\[Comparison\]\(https://wandb\.ai/[^)]*\?filters=[^)]*\)\s*)+\n",
        re.MULTILINE,
    ),
]
EXISTING_LINK_RE = re.compile(
    r"\[Comparison\]\((https://wandb\.ai/([^/?]+/[^/?]+)\?nw=[^)]+)\)"
)


def make_main_section() -> ws.Section:
    return ws.Section(
        name="Main",
        panels=[wr.LinePlot(title=m, y=[m]) for m in MAIN_METRICS],
        is_open=True,
        pinned=True,
    )


def heading_to_workspace_name(heading: str) -> str:
    text = heading.lstrip("#").strip()
    text = re.sub(r"[^\x20-\x7E]", "", text)
    # `validate_no_emoji` rejects any char in Unicode symbol/surrogate
    # categories (Sk catches backticks, Sm catches some math symbols, etc).
    text = "".join(
        c for c in text
        if not unicodedata.category(c).startswith(("So", "Sk", "Sm", "Sc", "Cs"))
    )
    return ("auto: " + text)[:120]


def upsert_workspace(entity_project: str, heading: str, run_ids: list[str],
                     existing_url: str | None) -> str:
    """Create-or-update a saved workspace and return its share URL."""
    entity, project = entity_project.split("/", 1)
    # Filter on `ID` (the immutable run ID encoded in the URL) rather than
    # `Name` (the display name). Display names in EXPERIMENTS.md are often
    # manually trimmed for readability and don't exact-match the W&B run name,
    # which would otherwise silently drop runs from the saved comparison.
    filt = f"ID in {run_ids!r}"
    name = heading_to_workspace_name(heading)

    workspace = None
    if existing_url:
        try:
            workspace = ws.Workspace.from_url(existing_url)
        except Exception as e:  # noqa: BLE001
            print(f"  warn: from_url({existing_url}) failed: {e!r}; "
                  f"creating fresh", file=sys.stderr)
            workspace = None

    if workspace is None:
        workspace = ws.Workspace(
            entity=entity,
            project=project,
            name=name,
            auto_generate_panels=True,
            sections=[make_main_section()],
            runset_settings=ws.RunsetSettings(filters=filt),
        )
    else:
        # Refresh the filter and re-pin Main at the top. `from_url()` builds
        # the Workspace via `_from_model`, which does NOT propagate
        # `auto_generate_panels` — it silently resets to the dataclass default
        # (False) on every reload. The public `auto_generate_panels` is a
        # read-only @property, but the underlying private field is writable,
        # and `_to_model` reads from the property. Re-affirm via the private
        # field so save() doesn't strip auto-generated panels.
        workspace.name = name
        workspace.runset_settings = ws.RunsetSettings(filters=filt)
        workspace._auto_generate_panels = True
        sections = [s for s in workspace.sections if s.name != "Main"]
        workspace.sections = [make_main_section()] + sections

    return workspace.save().url


def parse_sections(text: str) -> list[tuple[int, int, str]]:
    """Return [(start, end, heading_line), ...] for every ### section."""
    headings = list(HEADING_RE.finditer(text))
    out = []
    for i, m in enumerate(headings):
        start = m.start()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        out.append((start, end, m.group(0)))
    return out


def process(text: str) -> tuple[str, int, int]:
    sections = parse_sections(text)
    n_done = 0
    # walk end → start so splicing keeps earlier offsets stable
    for sec_start, sec_end, heading in reversed(sections):
        section = text[sec_start:sec_end]

        # Look for an already-attached `[Comparison](...?nw=...)` to update.
        existing_url = None
        m = EXISTING_LINK_RE.search(section)
        if m:
            existing_url = m.group(1)

        # Strip every kind of compare line so we can reinsert cleanly.
        section_clean = COMPARE_LINE_RE.sub("", section)
        for r in LEGACY_RES:
            section_clean = r.sub("", section_clean)

        # Collect run IDs per project (deduped).
        by_proj: dict[str, list[str]] = defaultdict(list)
        seen: set[tuple[str, str]] = set()
        for lm in LINK_RE.finditer(section_clean):
            proj, rid = lm.group(2), lm.group(3)
            if (proj, rid) in seen:
                continue
            seen.add((proj, rid))
            by_proj[proj].append(rid)

        if not by_proj:
            text = text[:sec_start] + section_clean + text[sec_end:]
            continue

        print(f"[{n_done + 1}] {heading[:80]}", file=sys.stderr)
        links: list[str] = []
        for proj, ids in by_proj.items():
            url_existing = existing_url if (existing_url and existing_url.startswith(
                f"https://wandb.ai/{proj}?")) else None
            url = upsert_workspace(proj, heading, ids, url_existing)
            links.append(f"[Comparison]({url})")
        compare_line = " ".join(links) + "\n"

        heading_end = section_clean.find("\n") + 1
        new_section = (section_clean[:heading_end]
                       + compare_line
                       + section_clean[heading_end:])
        text = text[:sec_start] + new_section + text[sec_end:]
        n_done += 1

    return text, n_done, len(sections)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=str(EXPERIMENTS_MD))
    args = parser.parse_args()
    p = Path(args.file)
    text = p.read_text()
    new_text, n_done, n_sections = process(text)
    p.write_text(new_text)
    print(f"{p.name}: refreshed compare links for {n_done}/{n_sections} ### sections.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
