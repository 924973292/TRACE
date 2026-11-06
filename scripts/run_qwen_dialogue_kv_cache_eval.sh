#!/usr/bin/env bash
# Full (or limited) dialogue-KV eval: task-once novideo, no vision embed cache.
# Baseline reference: task_once_vcache0 (see docs/qwen_dialogue_kv_eval_protocol.md).
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=eval_env_defaults.sh
source "${_SCRIPT_DIR}/eval_env_defaults.sh"
REPO="${REPO}"
EVAL="${EVAL_PYTHON}"
LIMIT="${LIMIT:-}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${REPO}/logs/dialogue_kv_cache_${TAG}}"
TASK="${TASK:-task_v2_qwen3_omni_novideo}"

cd "$REPO"
if [[ -f scripts/geopact_yolo_eval_env.sh ]]; then
  # shellcheck source=geopact_yolo_eval_env.sh
  source scripts/geopact_yolo_eval_env.sh
fi

export GEOPACT_COMPACT_BACKEND=source
export GEOPACT_VISION_EMBED_CACHE=0
export GEOPACT_SESSION_KV=1
export GEOPACT_SESSION_KV_MODE=dialogue
export GEOPACT_FORCE_EPISODE_ORDER=1
export GEOPACT_SCREENSHOT_ONLY_CACHE=1
export MIU_DISABLE_OCR="${MIU_DISABLE_OCR:-1}"
export LMMS_LIGHT_LOG="${LMMS_LIGHT_LOG:-1}"
unset GUIOWL_NATIVE_MULTITURN

MODEL_ARGS="${MODEL_ARGS:-pretrained=models/GUI-Owl-1.5-4B-Instruct,attn_implementation=flash_attention_2,device_map=auto,visual_prune_scope=image_sticky}"

mkdir -p "$OUT"
(
  flock -w 180 9
  "$EVAL" scripts/restore_q3vl_geopact_patches.py
) 9>"${REPO}/logs/.restore_q3vl_geopact.lock"

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

{
  echo "=== dialogue KV cache eval ==="
  echo "OUT=${OUT}"
  echo "TASK=${TASK} (task-once novideo)"
  echo "LIMIT=${LIMIT:-full}"
  echo "GEOPACT_SESSION_KV=${GEOPACT_SESSION_KV} MODE=${GEOPACT_SESSION_KV_MODE}"
  echo "GEOPACT_VISION_EMBED_CACHE=${GEOPACT_VISION_EMBED_CACHE}"
  echo "baseline_ref=logs/baseline_full_matrix_20260612_114748/task_once_vcache0"
} | tee "${OUT}/launcher.log"

"$EVAL" -m lmms_eval \
  --model gui_owl \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASK" \
  --batch_size 1 \
  "${LIMIT_ARGS[@]}" \
  --log_samples \
  --output_path "${OUT}/results" \
  2>&1 | tee "${OUT}/run.log"

echo "DONE ${OUT}"
