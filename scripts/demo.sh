#!/usr/bin/env bash
# Quick-start gate for TRACE. CPU path always runs; GPU smoke is optional.
#
#   bash scripts/demo.sh
#   bash scripts/demo.sh --gpu    # LIMIT=8 ScreenSpot-v2 if CUDA + weights exist
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=eval_env_defaults.sh
source "${_SCRIPT_DIR}/eval_env_defaults.sh"
cd "${REPO}"

WANT_GPU=0
if [[ "${1:-}" == "--gpu" ]]; then
  WANT_GPU=1
fi

PY="${EVAL_PYTHON}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

echo "== TRACE demo =="
echo "REPO=${REPO}"
echo "EVAL_PYTHON=${PY}"

echo "[1/4] import lmms_eval + geopact hook"
"${PY}" -c "
import lmms_eval
from lmms_eval.models.geopact_nap import nap_contract_mode, nap_anticipate_enabled, ratio_current
from lmms_eval.models.geopact_crs import crs_enabled
print('lmms_eval ok')
print('code defaults: NAP_CONTRACT=%s ANTICIPATE=%s RATIO_CURRENT=%.2f CRS=%s'
      % (nap_contract_mode(), nap_anticipate_enabled(), ratio_current(), crs_enabled()))
print('paper jobs override these via EXTRA_ENV; do not trust function defaults')
"

echo "[2/4] invariant tests (CPU, no weights)"
"${PY}" scripts/verify_sliding_rewrite_invariants.py
"${PY}" scripts/verify_kv_incremental_flow.py
"${PY}" scripts/_w60_selfcond_unit.py

echo "[3/4] regenerate published table snapshot (not a GPU rerun)"
"${PY}" scripts/gen_tables.py

echo "[4/4] GPU smoke"
HAS_CUDA=0
if "${PY}" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  HAS_CUDA=1
fi
MODEL="${MODEL_DIR:-${REPO}/models/GUI-Owl-1.5-8B-Instruct}"
WEIGHTS="${OMNIPARSER_ICON_WEIGHTS:-}"

if [[ "${WANT_GPU}" != "1" ]]; then
  echo "  skipped (pass --gpu to attempt LIMIT=8 ScreenSpot-v2)"
elif [[ "${HAS_CUDA}" != "1" ]]; then
  echo "  skipped: no CUDA visible on this interpreter"
elif [[ ! -d "${MODEL}" ]]; then
  echo "  skipped: checkpoint not found at ${MODEL}"
elif [[ -z "${WEIGHTS}" || ! -f "${WEIGHTS}" ]]; then
  echo "  skipped: set OMNIPARSER_ICON_WEIGHTS to OmniParser v2 icon_detect/model.pt"
else
  export MODEL_DIR="${MODEL}"
  export GEOPACT_RATIO_CURRENT=0.10
  export GEOPACT_CRS=1
  export GEOPACT_CRS_REP=stride
  export GEOPACT_CRS_SKEL_FRAC=0.50
  export GEOPACT_BAM_ALPHA_CAL=1
  export GEOPACT_NAP_ANTICIPATE=0
  LIMIT=8 BENCHMARK=screenspot_v2 TASK=task_screenspot_v2_guiowl \
    TRACK=abc_r10 EXP_ARM=v3_nap PRUNE_RATIO=0.10 \
    OUT=logs/smoke/ssv2_r10 \
    bash scripts/run_cross_dataset_track_one.sh
  echo "  GPU smoke finished; see logs/smoke/ssv2_r10"
fi

echo "=== demo complete ==="
echo "CPU checks confirm the harness and TRACE modules import and that table"
echo "snapshots regenerate. They do NOT reproduce paper GPU numbers."
