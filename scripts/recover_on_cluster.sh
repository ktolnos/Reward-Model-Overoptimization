#!/bin/bash
# Run on cluster with: srun --mem=64G bash scripts/recover_on_cluster.sh
# Recovers AlpacaFarm reward-model-human using the original tatsu-lab script.
set -e

WORK_DIR="/nas/ucb/eop/Reward-Model-Overoptimization"
NAS_DIR="/nas/ucb/eop"
MODELS_DIR="$NAS_DIR/cache/alpaca_farm_models"
RAW_LLAMA_DIR="$NAS_DIR/cache/llama-7b-raw"
LLAMA_HF_DIR="$NAS_DIR/cache/llama-7b-hf-f32"
VENV_DIR="$NAS_DIR/recover_venv"

cd "$WORK_DIR"

# ── 0. Create a clean venv with transformers 4.29.2 ────────────────
echo "=== Step 0: Setting up venv with transformers 4.29.2 ==="
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install -q torch transformers==4.29.2 alpaca-farm sentencepiece protobuf huggingface_hub
else
    source "$VENV_DIR/bin/activate"
fi
echo "transformers=$(python -c 'import transformers; print(transformers.__version__)')"

# ── 1. Download raw Meta LLaMA weights ──────────────────────────────
echo ""
echo "=== Step 1: Download raw Meta LLaMA-7B weights ==="
if [ ! -f "$RAW_LLAMA_DIR/consolidated.00.pth" ]; then
    mkdir -p "$RAW_LLAMA_DIR"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('ktolnos/llama-7b-raw-meta', local_dir='$RAW_LLAMA_DIR')
"
fi
echo "Raw weights ready at $RAW_LLAMA_DIR"

# Set up directory structure expected by conversion script:
# input_dir/7B/consolidated.00.pth + input_dir/tokenizer.model
LLAMA_STRUCT="$NAS_DIR/cache/llama-raw-structured"
mkdir -p "$LLAMA_STRUCT/7B"
ln -sf "$RAW_LLAMA_DIR/consolidated.00.pth" "$LLAMA_STRUCT/7B/consolidated.00.pth"
ln -sf "$RAW_LLAMA_DIR/params.json" "$LLAMA_STRUCT/7B/params.json"
cp -f "$RAW_LLAMA_DIR/tokenizer.model" "$LLAMA_STRUCT/tokenizer.model"

# ── 2. Convert to HuggingFace format with float32 ──────────────────
echo ""
echo "=== Step 2: Convert to HuggingFace format (float32) ==="
if [ ! -f "$LLAMA_HF_DIR/config.json" ]; then
    python -c "
import torch, types, inspect
import transformers.models.llama.convert_llama_weights_to_hf as mod

src = inspect.getsource(mod)
src = src.replace('torch.float16', 'torch.float32')

patched = types.ModuleType('patched')
patched.__dict__.update(mod.__dict__)
exec(compile(src, '<patched>', 'exec'), patched.__dict__)

patched.write_model(
    model_path='$LLAMA_HF_DIR',
    input_base_path='$LLAMA_STRUCT/7B',
    model_size='7B',
)
print('Conversion done!')
"
else
    echo "Already converted at $LLAMA_HF_DIR"
fi

python -c "
import json
cfg = json.load(open('$LLAMA_HF_DIR/config.json'))
print(f'  transformers={cfg[\"transformers_version\"]}, dtype={cfg.get(\"torch_dtype\", \"unknown\")}')
"

# ── 3. Clone alpaca_farm repo (for recovery script) ────────────────
echo ""
echo "=== Step 3: Recover models using original alpaca_farm script ==="
ALPACA_FARM_REPO="$NAS_DIR/cache/alpaca_farm_repo"
if [ ! -d "$ALPACA_FARM_REPO" ]; then
    git clone --depth 1 https://github.com/tatsu-lab/alpaca_farm.git "$ALPACA_FARM_REPO"
fi
cd "$ALPACA_FARM_REPO"

# ── 4. Recover sft10k first (needed as backbone for RM) ────────────
echo ""
echo "--- Recovering sft10k ---"
if [ ! -f "$MODELS_DIR/sft10k/config.json" ]; then
    python -m pretrained_models.recover_model_weights \
        --llama-7b-hf-dir "$LLAMA_HF_DIR" \
        --alpaca-farm-model-name sft10k \
        --models-save-dir "$MODELS_DIR"
else
    echo "sft10k already recovered"
fi

# ── 5. Recover reward-model-human ──────────────────────────────────
echo ""
echo "--- Recovering reward-model-human ---"
if [ ! -f "$MODELS_DIR/reward-model-human/config.json" ]; then
    python -m pretrained_models.recover_model_weights \
        --llama-7b-hf-dir "$LLAMA_HF_DIR" \
        --alpaca-farm-model-name reward-model-human \
        --models-save-dir "$MODELS_DIR" \
        --path-to-sft10k "$MODELS_DIR/sft10k"
else
    echo "reward-model-human already recovered"
fi

echo ""
echo "=== Done! Models saved to $MODELS_DIR ==="
ls -la "$MODELS_DIR/"
