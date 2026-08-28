"""Run provenance: every stage logs its slurm job under the same field name,
and downstream stages turn a checkpoint path back into a link to the run and
job that produced it.

The point of these tests is that provenance must never be able to fail a
training run: off slurm, without wandb, or against a legacy checkpoints dir with
no manifest, the helpers return fewer fields — never an exception.
"""
import json
import os
import subprocess
import sys
import types

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import run_provenance
from run_provenance import (
    manifest_slurm_fields,
    related_run_fields,
    slurm_fields,
    slurm_job_id,
    slurm_log_path,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("JUDGE_SLURM_JOB_ID", raising=False)


def _stub_scontrol(monkeypatch, stdout=None, exc=None):
    def fake_run(cmd, **kw):
        if exc is not None:
            raise exc
        return types.SimpleNamespace(stdout=stdout)
    monkeypatch.setattr(run_provenance.subprocess, "run", fake_run)


# ---------------------------------------------------------------------------
# Own job
# ---------------------------------------------------------------------------

def test_slurm_job_id_off_slurm_is_none():
    assert slurm_job_id() is None
    assert slurm_fields() == {}
    assert manifest_slurm_fields() == {}


def test_slurm_fields_uses_the_shared_field_name(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    _stub_scontrol(monkeypatch, stdout="JobId=4242 StdOut=/nas/logs/slurm-4242.out Foo=bar")
    assert slurm_fields() == {
        "slurm/job_id": "4242", "slurm/log": "/nas/logs/slurm-4242.out",
    }


def test_slurm_fields_namespaces_another_job(monkeypatch):
    # A stage that resumes another stage's run must not overwrite its job id.
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    _stub_scontrol(monkeypatch, stdout="StdOut=/nas/logs/slurm-99.out")
    assert slurm_fields(prefix="slurm/judge", job_id="99") == {
        "slurm/judge/job_id": "99", "slurm/judge/log": "/nas/logs/slurm-99.out",
    }


def test_slurm_log_path_survives_a_missing_scontrol(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    _stub_scontrol(monkeypatch, exc=FileNotFoundError("scontrol"))
    assert slurm_log_path() is None
    assert slurm_fields() == {"slurm/job_id": "4242"}   # id still recorded


def test_slurm_log_path_without_a_stdout_field(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    _stub_scontrol(monkeypatch, stdout="JobId=4242 JobState=PENDING")
    assert slurm_log_path() is None


def test_manifest_slurm_fields_use_manifest_spelling(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    _stub_scontrol(monkeypatch, stdout="StdOut=/nas/logs/slurm-4242.out")
    assert manifest_slurm_fields() == {
        "slurm_job_id": "4242", "slurm_log": "/nas/logs/slurm-4242.out",
    }


# ---------------------------------------------------------------------------
# Links to upstream runs
# ---------------------------------------------------------------------------

def _run_dir(tmp_path, name, **manifest):
    d = tmp_path / name
    d.mkdir(parents=True)
    if manifest:
        (d / "run_manifest.json").write_text(json.dumps(manifest))
    return str(d)


def test_related_run_fields_from_a_manifest(tmp_path):
    d = _run_dir(tmp_path, "grpo", wandb_run_id="abc123",
                 wandb_url="https://wandb.ai/x/grpo/runs/abc123",
                 slurm_job_id="777", dataset_path="ignored")
    fields = related_run_fields("policy", d)
    assert fields["related/policy/path"] == d
    assert fields["related/policy/wandb_run_id"] == "abc123"
    assert fields["related/policy/wandb_url"] == "https://wandb.ai/x/grpo/runs/abc123"
    assert fields["related/policy/slurm_job_id"] == "777"
    # Only the linking keys are lifted; the rest of the manifest stays put.
    assert not any("dataset_path" in k for k in fields)


def test_related_run_fields_resolves_from_a_checkpoint_subdir(tmp_path):
    d = _run_dir(tmp_path, "rm/logs", wandb_url="https://wandb.ai/x/rm/runs/z")
    ckpt = os.path.join(d, "checkpoint-142")
    os.makedirs(ckpt)
    # RM paths point at a checkpoint, but the manifest lives one level up.
    assert related_run_fields("training_rm", ckpt)["related/training_rm/wandb_url"] \
        == "https://wandb.ai/x/rm/runs/z"


def test_related_run_fields_without_a_manifest_still_records_the_path(tmp_path):
    d = _run_dir(tmp_path, "legacy")
    assert related_run_fields("policy", d) == {"related/policy/path": d}


def test_related_run_fields_tolerates_a_broken_manifest(tmp_path):
    d = _run_dir(tmp_path, "broken", ok=1)
    (tmp_path / "broken" / "run_manifest.json").write_text("{not json")
    assert related_run_fields("policy", d) == {"related/policy/path": d}


def test_related_run_fields_with_no_path():
    assert related_run_fields("training_rm", None) == {}
    assert related_run_fields("training_rm", "") == {}


# ---------------------------------------------------------------------------
# What the eval run actually logs
# ---------------------------------------------------------------------------

def _eval_args(**kw):
    base = dict(checkpoints_dir="", kl_base_model_path=None, training_rm_path="",
                sibling_rm_path="", load_generations=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_eval_provenance_links_every_upstream_stage(tmp_path, monkeypatch):
    from policy_eval.wandb_utils import eval_provenance_fields

    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.setenv("JUDGE_SLURM_JOB_ID", "101")
    _stub_scontrol(monkeypatch, exc=FileNotFoundError("scontrol"))
    fields = eval_provenance_fields(_eval_args(
        checkpoints_dir=_run_dir(tmp_path, "grpo", wandb_run_id="g1"),
        kl_base_model_path=_run_dir(tmp_path, "sft", wandb_run_id="s1"),
        training_rm_path=_run_dir(tmp_path, "rm0", wandb_run_id="r0"),
        sibling_rm_path=_run_dir(tmp_path, "rm1", wandb_run_id="r1"),
    ))
    assert fields["slurm/job_id"] == "100"
    assert fields["slurm/judge/job_id"] == "101"
    assert fields["related/policy/wandb_run_id"] == "g1"
    assert fields["related/base_policy/wandb_run_id"] == "s1"
    assert fields["related/training_rm/wandb_run_id"] == "r0"
    assert fields["related/sibling_rm/wandb_run_id"] == "r1"


def test_eval_provenance_judge_pass_does_not_clobber_the_eval_job(tmp_path, monkeypatch):
    # The judge pass resumes the eval's wandb run: its own job belongs under
    # slurm/judge, leaving the generating job's slurm/job_id intact.
    from policy_eval.wandb_utils import eval_provenance_fields

    monkeypatch.setenv("SLURM_JOB_ID", "101")
    _stub_scontrol(monkeypatch, exc=FileNotFoundError("scontrol"))
    fields = eval_provenance_fields(_eval_args(load_generations=True))
    assert fields == {"slurm/judge/job_id": "101"}
