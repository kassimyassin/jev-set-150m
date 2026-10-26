#!/usr/bin/env bash
# scripts/run_colab_pro.sh
# Resumable launcher for JEV-Set-150M on Google Colab Pro (A100 80GB/40GB, L4 24GB, or T4 16GB)
#
# Usage:
#   bash scripts/run_colab_pro.sh            # FULL smoke pipeline (offline-capable, synthetic fallback allowed, mock teachers)
#   bash scripts/run_colab_pro.sh pilot      # PILOT: small REAL-data build, real teachers, 1 short epoch, measured Exp A/B
#   MODE=pilot bash scripts/run_colab_pro.sh # same as above
set -euo pipefail

MODE="${1:-${MODE:-full}}"
PILOT_ROWS_PER_GROUP="${PILOT_ROWS_PER_GROUP:-24}"      # rows pulled per dataset group in pilot
PILOT_MIN_REAL_GROUPS="${PILOT_MIN_REAL_GROUPS:-8}"     # fail loudly if fewer of the 16 groups yield real rows
PILOT_MIN_REAL_FRACTION="${PILOT_MIN_REAL_FRACTION:-0.95}"
PILOT_ABL_STEPS="${PILOT_ABL_STEPS:-40}"                # fine-tune steps per backbone in Exp A

echo "=================================================================="
echo " JEV-Set-150M | Colab Pro Pipeline (Decision Index 0.4.1) | MODE=${MODE}"
echo "=================================================================="

# ---- GPU detection -------------------------------------------------------------------------------
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
    echo "WARNING: No CUDA GPU detected; running in CPU mode (pilot will be very slow)."
    if [ "${MODE}" = "pilot" ] && [ "${ALLOW_CPU:-0}" != "1" ]; then
        echo "ERROR: pilot mode requires a GPU runtime (Runtime -> Change runtime type). Set ALLOW_CPU=1 to override."
        exit 2
    fi
fi

export HF_HUB_DISABLE_XET=1
export PYTHONPATH=".:src:external/decision-index"

# ---- Google Drive persistence (resume across Colab sessions) --------------------------------------
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/jev-set-150m}"
if [ -d "/content/drive/MyDrive" ]; then
    mkdir -p "${DRIVE_ROOT}/checkpoints" "${DRIVE_ROOT}/data/processed" data
    if [ ! -L "checkpoints" ]; then
        rm -rf checkpoints && ln -s "${DRIVE_ROOT}/checkpoints" checkpoints
    fi
    if [ ! -L "data/processed" ]; then
        rm -rf data/processed && ln -s "${DRIVE_ROOT}/data/processed" data/processed
    fi
    echo "Linked persistent Google Drive directory: ${DRIVE_ROOT}"
else
    echo "NOTE: Google Drive not mounted (/content/drive/MyDrive missing); nothing will persist across sessions."
fi

# =====================================================================================================
if [ "${MODE}" = "pilot" ]; then
    echo "[P0] Verifying real pretrained weights load + match transformers (needs network)..."
    python scripts/verify_real_weights.py

    echo "[P1] Param budget + test suite..."
    python scripts/verify_params.py
    pytest tests/ -q

    # Resume: skip the data build if a real-only summary already exists in persistent storage.
    if [ -f data/processed/data_build_summary.json ] && python - <<'PY'
import json, sys
s = json.load(open("data/processed/data_build_summary.json"))
sys.exit(0 if s.get("synthetic_rows_total", 1) == 0 and s.get("real_rows_total", 0) > 0 else 1)
PY
    then
        echo "[P2] Reusing existing REAL-only data build (delete data/processed/data_build_summary.json to rebuild)."
    else
        echo "[P2] Building small REAL-data pilot set (synthetic fallback DISABLED)..."
        rm -f data/processed/*.jsonl data/processed/data_build_summary.json
        python scripts/prepare_data.py --curate-all --commercial-clean \
            --rows-per-group "${PILOT_ROWS_PER_GROUP}" \
            --no-synthetic-fallback \
            --min-real-groups "${PILOT_MIN_REAL_GROUPS}" \
            --min-real-fraction "${PILOT_MIN_REAL_FRACTION}"
    fi
    echo "[P2b] Real-vs-synthetic summary:"
    python - <<'PY'
import json
s = json.load(open("data/processed/data_build_summary.json"))
print(f"real={s['real_rows_total']} synthetic={s['synthetic_rows_total']} groups_with_real={s['groups_with_real_data']} "
      f"groups_without_real={s['groups_without_real_data']}")
print("NOTE: O_val is a template-generated PLACEHOLDER (o_val_is_synthetic_placeholder=%s); do not read O_val numbers as OOD results." % s["o_val_is_synthetic_placeholder"])
assert s["synthetic_rows_total"] == 0, "synthetic rows present in pilot data"
assert s["real_rows_total"] > 0, "no real rows"
PY

    echo "[P3] Teacher logits (REAL teachers; --resume checkpointing)..."
    python scripts/generate_teacher_logits.py --input-jsonl data/processed/stage2_distill_train.jsonl \
        --output-jsonl data/processed/stage2_distill_train.jsonl --resume
    python scripts/generate_teacher_logits.py --input-jsonl data/processed/stage3_equilibrium_train.jsonl \
        --output-jsonl data/processed/stage3_equilibrium_train.jsonl --resume

    echo "[P4] Measured Exp A/B pilot (real data, real weights)..."
    python scripts/pilot_ablation.py --exp a b --steps "${PILOT_ABL_STEPS}" --require-real-data

    echo "[P5] One short epoch of Stage 1 (real data, --require-real-data smoke assertion active)..."
    python scripts/train.py --config configs/stage1_ood.yaml --load-pretrained --one-epoch --resume --require-real-data

    echo "=================================================================="
    echo " PILOT complete. Results: data/processed/pilot_ablation_results.json"
    echo " Checkpoint: checkpoints/jev-set-150m-stage1/model.safetensors"
    echo "=================================================================="
    exit 0
fi

# ---- FULL (smoke) pipeline ------------------------------------------------------------------------
echo "[1/7] Verifying 149,936,905 Parameter Budget & Permutation Equivariance..."
python scripts/verify_params.py

echo "[2/7] Running Automated Unit, Edge-Case & Phase 2 Verification Suite..."
pytest tests/ -v

echo "[3/7] Curating 16 Dataset Groups, 3-Tier Decontamination, 5-Fold S_val & 15-Task O_val..."
python scripts/prepare_data.py --curate-all --commercial-clean --rows-per-group 8

echo "[4/7] Extracting Multi-Teacher Ensemble Probabilities (with --resume checkpointing)..."
python scripts/generate_teacher_logits.py \
    --input-jsonl data/processed/stage2_distill_train.jsonl \
    --output-jsonl data/processed/stage2_distill_train.jsonl \
    --resume \
    --mock-teachers
python scripts/generate_teacher_logits.py \
    --input-jsonl data/processed/stage3_equilibrium_train.jsonl \
    --output-jsonl data/processed/stage3_equilibrium_train.jsonl \
    --resume \
    --mock-teachers

echo "[5/7] Experiment A-H gate EVALUATOR smoke (PLACEHOLDER score vectors, not measured results)..."
python scripts/train.py --ablation all

echo "[6/7] Running Resumable Staged Training (Auto-detecting Colab Pro GPU Profile)..."
python scripts/train.py --config configs/stage1_ood.yaml --max-steps 4 --resume
python scripts/train.py --config configs/stage2_distill.yaml --max-steps 4 --resume
python scripts/train.py --config configs/stage3_equilibrium.yaml --max-steps 4 --resume

echo "[7/7] Running Weight-Space Model Souping, L-BFGS Calibration & Official Validation..."
python scripts/soup_and_calibrate.py \
    --checkpoints checkpoints/jev-set-150m-stage1/model.safetensors checkpoints/jev-set-150m-stage2/model.safetensors checkpoints/jev-set-150m-stage3/model.safetensors \
    --weights 0.15 0.25 0.60 \
    --output checkpoints/jev-set-150m-final/model.safetensors \
    --demo-calibration

python scripts/evaluate_official.py --smoke-test
echo "=================================================================="
echo " Pipeline Verification Complete!"
echo "=================================================================="
