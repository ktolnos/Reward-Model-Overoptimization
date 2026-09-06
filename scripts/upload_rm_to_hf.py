"""Upload the recovered AlpacaFarm reward-model-human to a private HF repo.

Patches config.json to remove cluster-local backbone_model_name_or_path
before uploading so from_pretrained works on any machine.

Run on cluster: python scripts/upload_rm_to_hf.py
"""
import getpass
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

# Must match MODELS_DIR in scripts/recover_on_cluster.sh, which writes to the
# 60-day-TTL share by default. Override with ALPACA_FARM_DIR.
_ALPACA_FARM_DIR = os.environ.get(
    "ALPACA_FARM_DIR", os.environ.get("NAS_TTL", f"/nas/ttl=60d/{getpass.getuser()}")
)
RECOVERED_DIR = f"{_ALPACA_FARM_DIR}/alpaca_farm_models/reward-model-human"
REPO_ID = "ktolnos/alpaca-farm-reward-model-human"

# Create a temporary copy with patched config
with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir) / "model"
    shutil.copytree(RECOVERED_DIR, tmp_path)

    # Patch config.json: clear backbone_model_name_or_path (cluster-local path)
    config_file = tmp_path / "config.json"
    config = json.loads(config_file.read_text())
    config["backbone_model_name_or_path"] = None
    config["_name_or_path"] = REPO_ID
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Patched config: backbone_model_name_or_path = None")

    api = HfApi()
    api.create_repo(REPO_ID, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(tmp_path),
        repo_id=REPO_ID,
        commit_message="AlpacaFarm reward-model-human (recovered with original tatsu-lab script)",
    )

print(f"Uploaded to {REPO_ID}")
