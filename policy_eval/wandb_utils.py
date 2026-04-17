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
        config=vars(args),
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

    # Register the custom step axis for all metrics logged through this helper.
    # On resume, this only affects metrics logged from *this* session onward;
    # previously-logged metrics keep their original x-axis.
    wandb.define_metric(_STEP_AXIS)
    wandb.define_metric("*", step_metric=_STEP_AXIS)

    _INITIALIZED = True
    return run


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
