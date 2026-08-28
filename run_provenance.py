"""Cross-component run provenance: slurm job ids and links between wandb runs.

Every pipeline stage — SFT → RM training → GRPO → policy eval → the judge pass —
is its own slurm job with its own wandb run, so a number in an eval run is the
product of half a dozen jobs whose logs sit in half a dozen ``slurm-<id>.out``
files. Reconstructing that chain from timestamps goes wrong quietly.

So every stage records the same two things:

  1. **Its own slurm job**, under the *same* wandb field name in every project —
     ``slurm/job_id`` plus ``slurm/log`` (``slurm_fields``). A stage resuming
     another's run namespaces itself instead of clobbering the original (the
     judge pass logs under ``slurm/judge``).
  2. **The same ids in the run manifest** next to its checkpoints
     (``manifest_slurm_fields``), so a downstream stage holding only a checkpoint
     path can turn it back into a link to the run that produced it
     (``related_run_fields``).

All best-effort: off slurm, with wandb disabled, or against a legacy run with no
manifest, these return empty dicts. Provenance is never worth failing a run over.
"""
from __future__ import annotations

import os
import subprocess
from typing import Dict, Optional

# Keep stable: the whole value is that one filter works across every stage.
SLURM_PREFIX = "slurm"
RELATED_PREFIX = "related"


def slurm_job_id() -> Optional[str]:
    """This process's slurm job id, or None when not running under slurm."""
    return os.environ.get("SLURM_JOB_ID") or None


def slurm_log_path(job_id: Optional[str] = None) -> Optional[str]:
    """The StdOut path of a slurm job, via ``scontrol``.

    Answers "where are the full logs" directly instead of leaving the reader to
    guess which ``slurm-<id>.out`` in which submit dir. Works for pending jobs,
    so a queued judge job can be linked before it starts. None when scontrol is
    unavailable (off-cluster) or the job aged out of the controller's memory.
    """
    job_id = job_id or slurm_job_id()
    if not job_id:
        return None
    try:
        out = subprocess.run(
            ["scontrol", "show", "job", str(job_id)],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for token in out.split():
        if token.startswith("StdOut="):
            return token[len("StdOut="):] or None
    return None


def slurm_fields(
    prefix: str = SLURM_PREFIX, job_id: Optional[str] = None,
) -> Dict[str, str]:
    """Wandb fields describing a slurm job: ``<prefix>/job_id``, ``<prefix>/log``.

    The defaults are this run's own job, which is what every stage logs. Pass a
    prefix and an id to point at a *different* job — the queued judge pass —
    without overwriting the run's own.
    """
    job_id = job_id or slurm_job_id()
    if not job_id:
        return {}
    fields = {f"{prefix}/job_id": str(job_id)}
    log = slurm_log_path(job_id)
    if log:
        fields[f"{prefix}/log"] = log
    return fields


def manifest_slurm_fields(job_id: Optional[str] = None) -> Dict[str, str]:
    """The same ids in run-manifest spelling (no slashes), for ``write_run_manifest``.

    What makes ``related_run_fields`` work downstream: the manifest is the only
    thing surviving next to the checkpoints once the shell's environment is gone.
    """
    job_id = job_id or slurm_job_id()
    if not job_id:
        return {}
    out = {"slurm_job_id": str(job_id)}
    log = slurm_log_path(job_id)
    if log:
        out["slurm_log"] = log
    return out


def wandb_manifest_fields() -> Dict[str, str]:
    """Identity of the live wandb run, in run-manifest spelling.

    Written by each training stage so downstream stages can link back to it.
    Empty when wandb is not active (``--disable_wandb``, offline, non-main rank).
    """
    try:
        import wandb
    except ImportError:
        return {}
    run = getattr(wandb, "run", None)
    if run is None:
        return {}
    return {
        "wandb_run_id": run.id,
        "wandb_run_name": run.name,
        "wandb_project": run.project,
        "wandb_url": run.url,
    }


def related_run_fields(label: str, path: Optional[str]) -> Dict[str, str]:
    """Fields linking to the run that produced ``path``.

    ``path`` is an earlier stage's checkpoints dir (or a ``checkpoint-N`` inside
    one); its run manifest supplies the wandb url and slurm job. The path itself
    is always recorded — for a legacy run with no manifest it is all there is.
    """
    if not path:
        return {}
    fields = {f"{RELATED_PREFIX}/{label}/path": str(path)}
    try:
        from data_utils import read_run_manifest
        manifest = read_run_manifest(path) or {}
    except Exception as e:  # never fail a run over a provenance lookup
        print(f"[provenance] could not read run manifest for {label} at {path}: {e}")
        return fields
    for key in ("wandb_run_id", "wandb_run_name", "wandb_url",
                "slurm_job_id", "slurm_log"):
        value = manifest.get(key)
        if value:
            fields[f"{RELATED_PREFIX}/{label}/{key}"] = str(value)
    return fields


def attach_to_wandb(fields: Dict[str, str]) -> None:
    """Merge provenance fields into the live wandb run's config.

    ``allow_val_change`` because a resumed run (the judge pass) legitimately
    rewrites fields the original set. No-op when wandb is off.
    """
    if not fields:
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    wandb.run.config.update(fields, allow_val_change=True)
    print("[provenance] " + ", ".join(f"{k}={v}" for k, v in sorted(fields.items())))
