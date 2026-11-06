#!/usr/bin/env bash
# Run one cross-dataset track (baseline_full / baseline_kv / abc_rXX).
#
#   BENCHMARK=omnigui TASK=task_v2_qwen3_omni_novideo TRACK=abc_r40 \
#     EXP_ARM=18_prod_drop_scale_a02_sspc_zafd PRUNE_RATIO=0.40 OUT=... \
#     bash scripts/run_cross_dataset_track_one.sh
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=eval_env_defaults.sh
source "${_SCRIPT_DIR}/eval_env_defaults.sh"

BENCHMARK="${BENCHMARK:?BENCHMARK required}"
TASK="${TASK:?TASK required}"
TRACK="${TRACK:?TRACK required}"
EXP_ARM="${EXP_ARM:?EXP_ARM required}"
OUT="${OUT:?OUT required}"
MODEL_DIR="${MODEL_DIR:-models/GUI-Owl-1.5-4B-Instruct}"
LIMIT="${LIMIT:-}"
SKIP_ENSURE_DEPS="${SKIP_ENSURE_DEPS:-1}"
SKIP_VERIFY="${SKIP_VERIFY:-1}"

export GEOPACT_MODULE_A_CROSS_STEP_PREFETCH=0
export GEOPACT_MODULE_A_PREFETCH_REUSE=0
export GEOPACT_COMPACT_BACKEND=source
export GEOPACT_VISION_EMBED_CACHE=0
export GEOPACT_FORCE_EPISODE_ORDER=1
export GEOPACT_KV_SLIDE_ON_ROLL="${GEOPACT_KV_SLIDE_ON_ROLL:-1}"
export GEOPACT_MODULE_A_WORKER_YOLO_DEVICE="${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-0}"
export BILATERAL_DECODE_PIN="${BILATERAL_DECODE_PIN:-1}"
export GEOPACT_TIMING_CUDA_SYNC="${GEOPACT_TIMING_CUDA_SYNC:-0}"
export MIU_DISABLE_OCR="${MIU_DISABLE_OCR:-1}"
export LMMS_LIGHT_LOG="${LMMS_LIGHT_LOG:-1}"

cd "$REPO"
if [[ -f scripts/geopact_yolo_eval_env.sh ]]; then
  # shellcheck source=geopact_yolo_eval_env.sh
  source scripts/geopact_yolo_eval_env.sh
fi

mkdir -p "$(dirname "$OUT")"
"${EVAL_PYTHON}" scripts/restore_q3vl_geopact_patches.py

extra_env=()
case "$TRACK" in
  baseline_full)
    extra_env+=(GEOPACT_SESSION_KV=0)
    unset GEOPACT_SESSION_KV_MODE GEOPACT_SCREENSHOT_ONLY_CACHE || true
    ;;
  baseline_kv|abc_r*)
    extra_env+=(
      GEOPACT_SESSION_KV=1
      GEOPACT_SESSION_KV_MODE=dialogue
      GEOPACT_SCREENSHOT_ONLY_CACHE=1
      GEOPACT_KV_HISTORY_COMPRESS_LEGACY=0
      # Preserve-if-set: the deletion-approximation control needs to pass
      # GEOPACT_KV_HISTORY_REWRITE=0 through the job EXTRA_ENV column. An
      # unconditional 1 here silently overrode that arm and produced two
      # identical runs (W49 batch 3, discarded).
      "GEOPACT_KV_HISTORY_REWRITE=${GEOPACT_KV_HISTORY_REWRITE:-1}"
    )
    ;;
esac

# Prune target by benchmark: multi-step (OmniGUI / AndroidControl) prune history,
# keeping the current screenshot full; single-step benchmarks prune the current
# frame (measures single-image perception under pruning).
case "$BENCHMARK" in
  omnigui|android_control|androidcontrol|ac|mind2web|mind2web_*|m2w|m2w_*)
    PRUNE_TARGET="${PRUNE_TARGET:-history}"
    ;;
  *)
    PRUNE_TARGET="${PRUNE_TARGET:-current}"
    ;;
esac

# Official grounding resolution for single-step benchmarks (Mobile-Agent-v3.5
# eval_grounding_benchmarks.py): min 196*32*32=200704 / max 9800*32*32=10035200.
# Checkpoint preprocessor default (~1.0MP) cripples ScreenSpot-Pro 4K shots.
case "$BENCHMARK" in
  screenspot*|mmbench*)
    VISION_MAX_PIXELS="${VISION_MAX_PIXELS:-10035200}"
    VISION_MIN_PIXELS="${VISION_MIN_PIXELS:-200704}"
    ;;
esac

echo "=== cross_dataset track benchmark=${BENCHMARK} task=${TASK} track=${TRACK} arm=${EXP_ARM} ==="
env \
  MODEL_DIR="$MODEL_DIR" \
  TASK="$TASK" \
  LIMIT="$LIMIT" \
  OUT="$OUT" \
  EXP_ARM="$EXP_ARM" \
  PRUNE_RATIO="${PRUNE_RATIO:-}" \
  PRUNE_TARGET="${PRUNE_TARGET}" \
  VISION_MAX_PIXELS="${VISION_MAX_PIXELS:-}" \
  VISION_MIN_PIXELS="${VISION_MIN_PIXELS:-}" \
  SKIP_ENSURE_DEPS="$SKIP_ENSURE_DEPS" \
  SKIP_VERIFY="$SKIP_VERIFY" \
  "${extra_env[@]}" \
  bash "${_SCRIPT_DIR}/run_qwen_a100_five_arm_one.sh" \
  2>&1 | tee "${OUT}.wrapper.log"

echo "DONE ${BENCHMARK}/${TRACK} -> ${OUT}"
