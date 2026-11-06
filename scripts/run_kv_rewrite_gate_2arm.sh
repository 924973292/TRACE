#!/usr/bin/env bash
# 2-arm gate: baseline_full vs baseline_kv_rewrite (incremental dialogue KV, no ABC).
#
# Usage:
#   bash scripts/run_kv_rewrite_gate_2arm.sh
#   LIMIT=10 TAG=mytag bash scripts/run_kv_rewrite_gate_2arm.sh
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=geopact_gate_common.sh
source "${_SCRIPT_DIR}/geopact_gate_common.sh"
geopact_gate_init

LIMIT="${LIMIT:-10}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
HIST_N="${GEOPACT_EVAL_HISTORY_N:-9}"
TOL_PP="${TOL_PP:-0.5}"
MODEL_DIR="${MODEL_DIR:-models/GUI-Owl-1.5-4B-Instruct}"
OUT_ROOT="${OUT_ROOT:-${REPO}/logs/kv_rewrite_gate_2arm_hist${HIST_N}_${TAG}}"
TASK="${TASK:-task_v2_qwen3_omni_novideo}"
BASE_MODEL_ARGS="$(geopact_gate_base_model_args "$MODEL_DIR")"

geopact_gate_export_base_env "$HIST_N"
mkdir -p "$OUT_ROOT"

echo "=== KV rewrite 2-arm gate hist=${HIST_N} LIMIT=${LIMIT} ===" | tee "${OUT_ROOT}/gate.log"
geopact_gate_verify_2arm "${OUT_ROOT}/gate.log"
geopact_gate_restore_patches

OUT_BASE="${OUT_ROOT}/baseline_full"
geopact_gate_run_baseline_full "$OUT_BASE" "$LIMIT" "$TASK" "$BASE_MODEL_ARGS" \
  2>&1 | tee "${OUT_BASE}.wrapper.log"

OUT_KV="${OUT_ROOT}/baseline_kv_rewrite"
geopact_gate_run_baseline_kv "$OUT_KV" "$LIMIT" "$TASK" "$BASE_MODEL_ARGS" \
  2>&1 | tee "${OUT_KV}.wrapper.log"

FULL_PROF="${OUT_BASE}/results/performance_profiles/step_profiles.jsonl"
KV_PROF="${OUT_KV}/results/performance_profiles/step_profiles.jsonl"
"$EVAL_PYTHON" scripts/verify_kv_rewrite_gate.py \
  --baseline "${OUT_BASE}/run.log" \
  --kv "${OUT_KV}/run.log" \
  --tol-pp "$TOL_PP" \
  --limit "$LIMIT" \
  --baseline-profile "${FULL_PROF}" \
  --kv-profile "${KV_PROF}" \
  --require-kv-faster \
  --min-speedup 1.05 \
  2>&1 | tee -a "${OUT_ROOT}/gate.log"

if [[ -f "$FULL_PROF" && -f "$KV_PROF" ]]; then
  echo "=== profile baseline_full vs baseline_kv_rewrite ===" | tee -a "${OUT_ROOT}/gate.log"
  "$EVAL_PYTHON" scripts/compare_step_latency_profiles.py "$FULL_PROF" "$KV_PROF" \
    2>&1 | tee -a "${OUT_ROOT}/profile_compare.log"
fi

echo "PASS KV rewrite 2-arm gate -> ${OUT_ROOT}" | tee -a "${OUT_ROOT}/gate.log"
