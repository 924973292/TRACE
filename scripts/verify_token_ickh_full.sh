#!/usr/bin/env bash
# Full verification for token-level ICKH (correctness + GPU bench + optional LIMIT=100 eval).
#
# Usage (DSW):
#   cd $REPO
#   bash scripts/verify_token_ickh_full.sh
#
# Correctness + bench only (skip eval):
#   SKIP_EVAL=1 bash scripts/verify_token_ickh_full.sh
#
# Run LIMIT=100 three-arm eval after checks:
#   TAG=token_ickh_r40_$(date +%Y%m%d) bash scripts/verify_token_ickh_full.sh
#
# Custom bench size (typical GUI screenshot ~800–1200 tokens):
#   ICKH_BENCH_N=1152 ICKH_BENCH_HIDDEN=2560 bash scripts/verify_token_ickh_full.sh
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=eval_env_defaults.sh
source "${_SCRIPT_DIR}/eval_env_defaults.sh"

REPO="${REPO}"
EVAL="${EVAL_PYTHON}"
SKIP_EVAL="${SKIP_EVAL:-0}"
ICKH_BENCH_N="${ICKH_BENCH_N:-1024}"
ICKH_BENCH_HIDDEN="${ICKH_BENCH_HIDDEN:-2560}"
ICKH_BENCH_RATIO="${ICKH_BENCH_RATIO:-0.40}"
TAG="${TAG:-token_ickh_r40_$(date +%Y%m%d_%H%M%S)}"

cd "${REPO}"

_log() {
  echo ""
  echo "========== $* =========="
}

_fail() {
  echo "ERROR: $*" >&2
  exit 1
}

_log "0) env"
echo "REPO=${REPO}"
echo "EVAL_PYTHON=${EVAL}"
echo "cluster=${EVAL_CLUSTER:-?} host=$(hostname 2>/dev/null || echo unknown)"
"${EVAL}" -c "import sys; print('python', sys.version.split()[0])"

_log "1) CUDA + flash_attn preflight"
bash "${_SCRIPT_DIR}/check_eval_cuda.sh"

_log "2) unit tests (policy routing + reference equivalence @ n=256 CPU/GPU)"
"${EVAL}" "${REPO}/scripts/verify_token_ickh_sticky.py"

_log "3) GPU micro-benchmark (fast vs reference @ n=${ICKH_BENCH_N}, r=${ICKH_BENCH_RATIO})"
"${EVAL}" "${REPO}/scripts/bench_token_ickh_kernel.py" \
  --device cuda \
  --n "${ICKH_BENCH_N}" \
  --hidden "${ICKH_BENCH_HIDDEN}" \
  --ratio "${ICKH_BENCH_RATIO}" \
  --warmup 8 \
  --repeats 40

if [[ "${SKIP_EVAL}" == "1" ]]; then
  _log "done (SKIP_EVAL=1, eval skipped)"
  echo ""
  echo "To run LIMIT=100 three-arm eval:"
  echo "  TAG=${TAG} bash scripts/run_qwen_token_ickh_r40_limit100.sh"
  exit 0
fi

MODEL_DIR="${MODELS_DIR}/GUI-Owl-1.5-4B-Instruct"
if ! compgen -G "${MODEL_DIR}/model-*.safetensors" >/dev/null; then
  _fail "4B model missing: ${MODEL_DIR}/model-*.safetensors"
fi

_log "4) LIMIT=100 three-arm eval (uniform vs token_ickh diversity vs relevance)"
export TAG LIMIT=100
export OUT_ROOT="${OUT_ROOT:-${REPO}/logs/kv_prune_token_ickh_r40_limit100_${TAG}}"
echo "OUT_ROOT=${OUT_ROOT}"
bash "${REPO}/scripts/run_qwen_token_ickh_r40_limit100.sh"

_log "5) summary"
if [[ -f "${OUT_ROOT}/token_ickh_r40_summary.md" ]]; then
  cat "${OUT_ROOT}/token_ickh_r40_summary.md"
else
  echo "MISSING ${OUT_ROOT}/token_ickh_r40_summary.md"
fi

echo ""
echo "=== verify_token_ickh_full DONE ==="
echo "OUT_ROOT=${OUT_ROOT}"
echo "Per-arm logs:"
for arm in 04_dialogue_kv_r40 04b_dialogue_kv_r40_tok_div 04c_dialogue_kv_r40_tok_rel; do
  echo "  ${OUT_ROOT}/${arm}/run.log"
done
