"""Tests for the training-run manifest: write in training, default eval args.

Precedence contract: explicit CLI flag > run manifest > ScriptArguments default.
"""
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_utils import RUN_MANIFEST_FILENAME, read_run_manifest, write_run_manifest


def _write_manifest(run_dir, **overrides):
    manifest = {
        "model_name_or_path": "logs_sft/checkpoint-744",
        "dataset_path": "org/preference-dataset",
        "temperature": 0.9,
        "reward_model_paths": ["save_reward_models/rm-a", "save_reward_models/rm-b"],
    }
    manifest.update(overrides)
    write_run_manifest(str(run_dir), manifest)
    return manifest


class TestReadWriteRunManifest:
    def test_roundtrip_adds_version_and_timestamp(self, tmp_path):
        _write_manifest(tmp_path)
        manifest = read_run_manifest(str(tmp_path))
        assert manifest["dataset_path"] == "org/preference-dataset"
        assert manifest["manifest_version"] == 1
        assert "created_at" in manifest

    def test_resolves_from_single_checkpoint_subdir(self, tmp_path):
        _write_manifest(tmp_path)
        ckpt = tmp_path / "checkpoint-100"
        ckpt.mkdir()
        manifest = read_run_manifest(str(ckpt))
        assert manifest is not None
        assert manifest["temperature"] == 0.9
        # Trailing separator resolves the same way.
        assert read_run_manifest(str(ckpt) + os.sep) is not None

    def test_missing_manifest_returns_none(self, tmp_path):
        assert read_run_manifest(str(tmp_path)) is None

    def test_written_file_is_valid_json(self, tmp_path):
        _write_manifest(tmp_path)
        with open(tmp_path / RUN_MANIFEST_FILENAME) as f:
            assert json.load(f)["reward_model_paths"][0] == "save_reward_models/rm-a"


class TestApplyRunManifestDefaults:
    def _args(self, tmp_path, **kwargs):
        from evaluate_policy import ScriptArguments
        return ScriptArguments(checkpoints_dir=str(tmp_path), **kwargs)

    def test_manifest_fills_unset_fields(self, tmp_path):
        from policy_eval.eval_utils import apply_run_manifest_defaults
        _write_manifest(tmp_path)
        args = self._args(tmp_path)
        apply_run_manifest_defaults(args, argv=["--checkpoints_dir", str(tmp_path)])
        assert args.dataset_name == "org/preference-dataset"
        assert args.training_rm_path == "save_reward_models/rm-a"
        assert args.kl_base_model_path == "logs_sft/checkpoint-744"
        # eval_temperature is untied from the manifest: it keeps its default
        # even when the manifest records a different training temperature.
        assert args.eval_temperature == self._args(tmp_path).eval_temperature

    def test_explicit_cli_flags_override_manifest(self, tmp_path):
        from policy_eval.eval_utils import apply_run_manifest_defaults
        _write_manifest(tmp_path)
        args = self._args(
            tmp_path, dataset_name="cli/dataset", eval_temperature=0.5,
        )
        apply_run_manifest_defaults(args, argv=[
            "--checkpoints_dir", str(tmp_path),
            "--dataset_name", "cli/dataset",
            "--eval_temperature=0.5",  # --flag=value form must also count
        ])
        assert args.dataset_name == "cli/dataset"
        assert args.eval_temperature == 0.5
        # Fields not given on the CLI still come from the manifest.
        assert args.training_rm_path == "save_reward_models/rm-a"
        assert args.kl_base_model_path == "logs_sft/checkpoint-744"

    def test_dashed_and_abbreviated_flags_count_as_explicit(self, tmp_path):
        # HfArgumentParser accepts dashed aliases (--dataset-name) and argparse
        # prefix abbreviations (--eval_temp); both must count as explicit so
        # the manifest never overwrites a value the user actually passed.
        from policy_eval.eval_utils import apply_run_manifest_defaults
        _write_manifest(tmp_path)
        args = self._args(
            tmp_path, dataset_name="cli/dataset", eval_temperature=0.5,
        )
        apply_run_manifest_defaults(args, argv=[
            "--checkpoints_dir", str(tmp_path),
            "--dataset-name", "cli/dataset",
            "--eval_temp", "0.5",
        ])
        assert args.dataset_name == "cli/dataset"
        assert args.eval_temperature == 0.5
        assert args.training_rm_path == "save_reward_models/rm-a"

    def test_no_manifest_leaves_args_untouched(self, tmp_path):
        from policy_eval.eval_utils import apply_run_manifest_defaults
        args = self._args(tmp_path)
        before = dict(vars(args))
        apply_run_manifest_defaults(args, argv=["--checkpoints_dir", str(tmp_path)])
        assert vars(args) == before

    def test_training_wandb_id_does_not_leak_into_eval_resume_flag(self, tmp_path):
        # The manifest's wandb_run_id identifies the *training* run; eval's
        # --wandb_run_id resumes the *eval* run. They must stay independent.
        from policy_eval.eval_utils import apply_run_manifest_defaults
        _write_manifest(tmp_path, wandb_run_id="train123", wandb_run_name="train_run")
        args = self._args(tmp_path)
        apply_run_manifest_defaults(args, argv=["--checkpoints_dir", str(tmp_path)])
        assert args.wandb_run_id is None
        assert args.wandb_run_name is None

    def test_partial_manifest_applies_only_present_keys(self, tmp_path):
        from policy_eval.eval_utils import apply_run_manifest_defaults
        write_run_manifest(str(tmp_path), {"dataset_path": "org/only-dataset"})
        args = self._args(tmp_path)
        default_temperature = args.eval_temperature
        apply_run_manifest_defaults(args, argv=["--checkpoints_dir", str(tmp_path)])
        assert args.dataset_name == "org/only-dataset"
        assert args.training_rm_path == ""
        assert args.eval_temperature == default_temperature
