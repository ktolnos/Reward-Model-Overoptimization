#!/usr/bin/env python3
"""Turn bare run names in interesting_experiments.md into wandb links and add
a Comparison workspace under each `### ...` heading.

The file format is:

    ### Some interesting observation
    - run_name_1
    - run_name_2
    - [already-resolved-name](https://wandb.ai/.../runs/<id>)

Bare names are looked up against the cached wandb runs in `wandb_cache/`
(run `scripts/download_run_data.py` first) and rewritten as
`- [name](wandb-url)`. The script then reuses scripts/annotate_runs.py to
append a `{base-model}` annotation and scripts/add_compare_links.py to
create/refresh a `[Comparison](...)` link under the heading. Re-runs are
idempotent: resolved bullets stay resolved, workspaces are updated in place.

When a name matches in multiple projects (eval/grpo/dpo/sft/rm), the eval run
wins — comparison workspaces filter on eval-time metrics, so an evaluation of
a training run is more useful to compare than the training run itself.
Unresolved names are left as-is and reported on stderr.

Run with the venv Python so `wandb-workspaces` is on the path:

    venv/bin/python scripts/process_interesting_experiments.py
    venv/bin/python scripts/process_interesting_experiments.py --no-compare
    venv/bin/python scripts/process_interesting_experiments.py --file foo.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import annotate_runs as ar  # noqa: E402

ROOT = SCRIPTS_DIR.parent
INTERESTING_MD = ROOT / "interesting_experiments.md"

# Preference order when a name matches in multiple projects.
KIND_PRIORITY = ("eval", "grpo", "dpo", "sft", "rm")

# Bullet that's *just* a bare run name. The `[^\[\s]` first-char guard skips
# bullets that already contain a markdown link (those start with `[` after the
# `- `) and blank-looking bullets.
BARE_BULLET_RE = re.compile(
    r"^(?P<indent>[-*]\s+)(?P<name>[^\[\s][^\n]*?)\s*$",
    re.MULTILINE,
)


def build_name_index(runs_by_kind: dict[str, list[dict]]
                     ) -> dict[str, tuple[str, dict]]:
    """Map run name -> (kind, run). On collisions across kinds, prefer the
    higher-priority kind; within a kind, prefer the most recently created run.
    """
    name_index: dict[str, tuple[str, dict]] = {}
    for kind in KIND_PRIORITY:
        runs = sorted(
            runs_by_kind.get(kind, []),
            key=lambda r: r.get("created_at") or "",
            reverse=True,
        )
        for r in runs:
            name = r.get("name")
            if not name:
                continue
            name_index.setdefault(name, (kind, r))
    return name_index


def resolve_bare_bullets(text: str,
                         name_index: dict[str, tuple[str, dict]]
                         ) -> tuple[str, list[str], list[str]]:
    """Rewrite each bare-name bullet to a markdown link.

    Returns (new_text, resolved_names, unresolved_names).
    """
    resolved: list[str] = []
    unresolved: list[str] = []

    def repl(m: re.Match) -> str:
        indent = m.group("indent")
        name = m.group("name").strip()
        # Skip bullets that already contain a markdown link (resolved earlier).
        if "](" in name:
            return m.group(0)
        hit = name_index.get(name)
        if hit is None:
            unresolved.append(name)
            return m.group(0)
        kind, run = hit
        url = ar.wandb_url(kind, run["id"])
        resolved.append(name)
        return f"{indent}[{name}]({url})"

    return BARE_BULLET_RE.sub(repl, text), resolved, unresolved


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--file", default=str(INTERESTING_MD),
                        help="Markdown file to process (default: %(default)s).")
    parser.add_argument("--no-annotate", action="store_true",
                        help="Skip `{base-model}` annotations on the links.")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip creating/updating Comparison workspaces "
                             "(useful for a dry run without wandb writes).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the would-be output to stdout instead of "
                             "writing the file. Implies --no-compare.")
    args = parser.parse_args()
    if args.dry_run:
        args.no_compare = True

    if not ar.CACHE_DIR.exists():
        print(f"error: {ar.CACHE_DIR} not found. "
              f"Run scripts/download_run_data.py first.", file=sys.stderr)
        return 2

    runs_by_kind = {kind: ar.load_cache(kind) for kind in ar.PROJECT_FILES}
    indices = {kind: ar._output_dirs(runs_by_kind[kind])
               for kind in ("grpo", "dpo", "sft", "rm")}
    eval_index = {r["id"]: r for r in runs_by_kind["eval"] if "id" in r}
    name_index = build_name_index(runs_by_kind)
    print(
        f"Loaded: {len(runs_by_kind['eval'])} eval, "
        f"{len(runs_by_kind['grpo'])} grpo, {len(runs_by_kind['dpo'])} dpo, "
        f"{len(runs_by_kind['sft'])} sft, {len(runs_by_kind['rm'])} rm "
        f"({len(name_index)} unique run names).",
        file=sys.stderr,
    )

    p = Path(args.file)
    text = p.read_text()

    text, resolved, unresolved = resolve_bare_bullets(text, name_index)
    print(f"Resolved {len(resolved)} bare-name bullet(s); "
          f"{len(unresolved)} unresolved.", file=sys.stderr)
    for n in unresolved:
        print(f"  not in cache: {n}", file=sys.stderr)

    if not args.no_annotate:
        text, n_ann, skips = ar.annotate_experiments(text, eval_index, indices)
        print(f"Annotated {n_ann} link(s); skipped {len(skips)}.",
              file=sys.stderr)
        for run_id, link_text, reason in skips:
            print(f"  {run_id}  ({link_text}): {reason}", file=sys.stderr)

    if not args.no_compare:
        # Imported lazily so that --no-compare / --dry-run work without
        # `wandb-workspaces` installed.
        import add_compare_links as acl  # noqa: PLC0415
        text, n_done, n_sections = acl.process(text)
        print(f"{p.name}: refreshed compare links for {n_done}/{n_sections} "
              f"### section(s).", file=sys.stderr)

    if args.dry_run:
        sys.stdout.write(text)
    else:
        p.write_text(text)
        print(f"Wrote {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
