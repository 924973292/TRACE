#!/usr/bin/env bash
# 3-arm gate: baseline_full / baseline_kv / abc_r40_kv (hist9 + sliding rewrite).
#
# Usage:
#   bash scripts/run_kv_abc_gate_4b.sh
#   LIMIT=100 TAG=mytag bash scripts/run_kv_abc_gate_4b.sh
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=geopact_gate_common.sh
source "${_SCRIPT_DIR}/geopact_gate_common.sh"
geopact_gate_init

LIMIT="${LIMIT:-100}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
HIST_N="${GEOPACT_EVAL_HISTORY_N:-9}"
# Tolerances. hist2 forced-eviction runs carry inherent GP/EM jitter from the
# StreamingLLM-style eviction approximation (bilateral trace confirmed the
# position bookkeeping is exact; flips are coordinate-level fp jitter), so
# both gates default wider there; hist9 (production window) stays tight.
if [[ "${HIST_N}" -le 2 ]]; then
  TOL_PP="${TOL_PP:-1.0}"
  EM_TOL_PP="${EM_TOL_PP:-2.5}"
else
  TOL_PP="${TOL_PP:-0.5}"
  EM_TOL_PP="${EM_TOL_PP:-1.5}"
fi
ABC_ARM="${ABC_ARM:-18_prod_drop_scale_a02_sspc_zafd}"
PRUNE_TARGET="${PRUNE_TARGET:-history}"
MODEL_DIR="${MODEL_DIR:-models/GUI-Owl-1.5-4B-Instruct}"
OUT_ROOT="${OUT_ROOT:-${REPO}/logs/kv_abc_gate_4b_hist${HIST_N}_${TAG}}"
TASK="${TASK:-task_v2_qwen3_omni_novideo}"
BASE_MODEL_ARGS="$(geopact_gate_base_model_args "$MODEL_DIR")"

geopact_gate_export_base_env "$HIST_N"
geopact_gate_export_abc_env
mkdir -p "$OUT_ROOT"

echo "=== KV ABC gate 4B hist=${HIST_N} LIMIT=${LIMIT} arm=${ABC_ARM} ===" | tee "${OUT_ROOT}/gate.log"
geopact_gate_verify_3arm "${OUT_ROOT}/gate.log" "$HIST_N" "$PRUNE_TARGET"
geopact_gate_restore_patches

OUT_BASE="${OUT_ROOT}/baseline_full"
geopact_gate_run_baseline_full "$OUT_BASE" "$LIMIT" "$TASK" "$BASE_MODEL_ARGS" \
  2>&1 | tee "${OUT_BASE}.wrapper.log"

OUT_KV="${OUT_ROOT}/baseline_kv"
geopact_gate_run_baseline_kv "$OUT_KV" "$LIMIT" "$TASK" "$BASE_MODEL_ARGS" \
  2>&1 | tee "${OUT_KV}.wrapper.log"

OUT_ABC="${OUT_ROOT}/abc_r40_kv"
geopact_gate_run_abc_r40 "$OUT_ABC" "$LIMIT" "$TASK" "$MODEL_DIR" "$ABC_ARM" "$PRUNE_TARGET" \
  2>&1 | tee "${OUT_ABC}.wrapper.log"

KV_PROF="${OUT_KV}/results/performance_profiles/step_profiles.jsonl"
ABC_PROF="${OUT_ABC}/results/performance_profiles/step_profiles.jsonl"
SLIDING_GATE_EXTRA=()
if [[ -n "${ABC_PROFILE_BEFORE:-}" && -f "${ABC_PROFILE_BEFORE}" ]]; then
  SLIDING_GATE_EXTRA+=(--abc-profile-before "${ABC_PROFILE_BEFORE}")
fi
SLIDING_GATE_EXTRA+=(--max-abc-decode-per-token-ratio "${MAX_ABC_DECODE_PER_TOKEN_RATIO:-1.30}")

"$EVAL_PYTHON" scripts/verify_kv_abc_gate.py \
  --baseline "${OUT_BASE}/run.log" \
  --kv "${OUT_KV}/run.log" \
  --abc "${OUT_ABC}/run.log" \
  --tol-pp "$TOL_PP" \
  --em-tol-pp "$EM_TOL_PP" \
  --abc-drift-warn-only \
  --limit "$LIMIT" \
  --sliding-gate \
  --kv-profile "${KV_PROF}" \
  --abc-profile "${ABC_PROF}" \
  "${SLIDING_GATE_EXTRA[@]}" \
  2>&1 | tee -a "${OUT_ROOT}/gate.log"

if [[ -f "$KV_PROF" && -f "$ABC_PROF" ]]; then
  geopact_gate_profile_report "$KV_PROF" "$ABC_PROF" "${OUT_ROOT}/profile_compare.log"
fi

echo "PASS KV ABC gate 4B -> ${OUT_ROOT}" | tee -a "${OUT_ROOT}/gate.log"
