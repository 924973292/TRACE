#!/usr/bin/env bash
# Verify the two GUI-Owl limit-500 baselines on remote:
#   1) imageonly_a  (current image only, no history screenshots)
#   2) hist_k5      (FIFO K=5 history screenshots)
#
# Usage:
#   cd "${REPO:-.}"
#   bash scripts/verify_guiowl_limit500_experiments.sh
#   bash scripts/verify_guiowl_limit500_experiments.sh --append-summary

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${OMNIGUI_PYTHON:-${EVAL_PYTHON:-python3}}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3 || command -v python)"
fi

echo "[verify] repo=$ROOT python=$PYTHON"
"$PYTHON" scripts/verify_guiowl_limit500_experiments.py "$@"

echo
echo "[dimension breakdown] imageonly_a"
"$PYTHON" scripts/compute_omnigui_dimension_breakdown.py \
  --samples "$(ls -t outputs/guiowl_imageonly_a_limit500/models__*/**/*_samples_task_v2_guiowl_imageonly_a.jsonl 2>/dev/null | head -1)"

echo
echo "[dimension breakdown] hist_k5"
"$PYTHON" scripts/compute_omnigui_dimension_breakdown.py \
  --samples "$(ls -t outputs/guiowl_imageonly_a_hist_k5_limit500/models__*/**/*_samples_task_v2_guiowl_imageonly_a_hist_k5.jsonl 2>/dev/null | head -1)"
