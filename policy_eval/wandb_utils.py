"""Wandb initialisation and logging helpers.

We use a **custom step axis** (``checkpoint``) via ``wandb.define_metric`` so
metrics can be logged at arbitrary checkpoint numbers regardless of order.
This is what makes resuming a run and adding new metric families work cleanly:
the new metric family gets its own x-axis, and log order doesn't matter.

Contract:
    - Call ``init_wandb(args)`` once at the start of the run. If ``disable_wandb``
      is set, this is a no-op.
    - Use ``log_metrics(metrics, checkpoint_num)`` to log per-checkpoint metrics.
      Do *not* pass ``step=`` to ``wandb.log`` directly — it defeats the purpose
      of the custom axis (wandb enforces monotonic step on its internal axis).
"""
from __future__ import annotations

import os
from typing import Optional

import wandb

from run_provenance import attach_to_wandb, related_run_fields, slurm_fields

from .persistence import redacted_args_dict


_STEP_AXIS = "checkpoint"
_INITIALIZED = False


def init_wandb(args) -> Optional[object]:
    """Initialise wandb (or resume) and set up the custom step axis.

    Returns the wandb run object, or None if wandb is disabled.
    """
    global _INITIALIZED
    if args.disable_wandb:
        return None

    init_kwargs = dict(
        project=args.wandb_project,
        config=redacted_args_dict(args),
        group=args.checkpoints_dir,
        job_type="evaluation",
    )

    if args.wandb_run_id:
        # Resume existing run. "must" fails loudly if the id doesn't exist,
        # which is what we want — silent creation of a new run on a typo is bad.
        init_kwargs["id"] = args.wandb_run_id
        init_kwargs["resume"] = "must"
    else:
        run_name = args.wandb_run_name or os.path.basename(
            os.path.normpath(args.checkpoints_dir)
        )
        if args.debug:
            run_name += "_debug"
        init_kwargs["name"] = run_name

    run = wandb.init(**init_kwargs)

    attach_to_wandb(eval_provenance_fields(args))

    # Register the custom step axis for all metrics logged through this helper.
    # On resume, this only affects metrics logged from *this* session onward;
    # previously-logged metrics keep their original x-axis.
    wandb.define_metric(_STEP_AXIS)
    wandb.define_metric("*", step_metric=_STEP_AXIS)

    _INITIALIZED = True
    return run


def eval_provenance_fields(args) -> dict:
    """Slurm jobs and links to the runs this eval depends on.

    Logged as config so a run can be *found* by them (filter on
    ``slurm/job_id``) and so every upstream artifact is one click away:

      - ``slurm/*``        this eval job (or ``slurm/judge/*`` for a judge-only
                           pass, which resumes the generating eval's run and
                           must not overwrite its job id)
      - ``slurm/judge/*``  the judge pass queued by evaluate_policy.sh, known
                           here because the shell submits it before starting
                           python and exports its id
      - ``related/*``      the GRPO run being evaluated, the SFT/base policy it
                           started from, and the RMs — resolved from each one's
                           run manifest, so they carry wandb urls and their own
                           slurm jobs

    Best-effort throughout: missing manifests and running off slurm just mean
    fewer fields.
    """
    own_prefix = "slurm/judge" if args.load_generations else "slurm"
    fields = dict(slurm_fields(prefix=own_prefix))
    # Only when the shell actually queued one: slurm_fields falls back to *this*
    # process's job id, which would label the eval job as the judge job.
    queued_judge = os.environ.get("JUDGE_SLURM_JOB_ID")
    if queued_judge:
        fields.update(slurm_fields(prefix="slurm/judge", job_id=queued_judge))
    fields.update(related_run_fields("policy", args.checkpoints_dir))
    fields.update(related_run_fields("base_policy", args.kl_base_model_path))
    fields.update(related_run_fields("training_rm", args.training_rm_path))
    fields.update(related_run_fields("sibling_rm", args.sibling_rm_path))
    return fields


def current_run_id() -> Optional[str]:
    """Id of the live wandb run, or None when wandb is disabled/uninitialised."""
    if not _INITIALIZED or wandb.run is None:
        return None
    return wandb.run.id


def log_metrics(metrics: dict, checkpoint_num: int, *, commit: bool = True) -> None:
    """Log per-checkpoint metrics against the ``checkpoint`` custom axis.

    Safe no-op when wandb is not initialised.
    """
    if not _INITIALIZED or wandb.run is None:
        return
    payload = dict(metrics)
    payload[_STEP_AXIS] = checkpoint_num
    wandb.log(payload, commit=commit)


def log_artifact(data: dict) -> None:
    """Log artifacts (tables, plots) not tied to a checkpoint step."""
    if not _INITIALIZED or wandb.run is None:
        return
    wandb.log(data)


def finish() -> None:
    global _INITIALIZED
    if _INITIALIZED and wandb.run is not None:
        wandb.finish()
    _INITIALIZED = False
