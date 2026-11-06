#!/usr/bin/env bash
# Preflight for 4B hist9 KV ABC gate (model, CUDA, patches, history window).
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=eval_env_defaults.sh
source "${_SCRIPT_DIR}/eval_env_defaults.sh"

HIST_N="${GEOPACT_EVAL_HISTORY_N:-9}"
MODEL_DIR="${MODEL_DIR:-models/GUI-Owl-1.5-4B-Instruct}"

cd "$REPO"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -f scripts/geopact_yolo_eval_env.sh ]]; then
  # shellcheck source=geopact_yolo_eval_env.sh
  source scripts/geopact_yolo_eval_env.sh
fi

echo "=== preflight 4B KV ABC gate hist=${HIST_N} ==="
echo "REPO=${REPO}"
echo "EVAL_PYTHON=${EVAL_PYTHON}"

ls -lh "${MODEL_DIR}/model-"*.safetensors
bash scripts/check_eval_cuda.sh
"$EVAL_PYTHON" scripts/restore_q3vl_geopact_patches.py
"$EVAL_PYTHON" scripts/verify_kv_history_dataset_bounds.py
GEOPACT_EVAL_HISTORY_N="${HIST_N}" PRUNE_TARGET=history "$EVAL_PYTHON" scripts/verify_hist9_abc_pipeline.py

GEOPACT_EVAL_HISTORY_N="${HIST_N}" "$EVAL_PYTHON" -c \
  "from lmms_eval.tasks.history_window import effective_history_slice_n; print('hist_n=', effective_history_slice_n())"

echo "OK preflight 4B hist${HIST_N}"
