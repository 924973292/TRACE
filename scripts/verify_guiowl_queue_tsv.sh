#!/usr/bin/env bash
# Verify queue TSV column alignment for guiowl hist_k5 full ablation.
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
QUEUE="${REPO}/scripts/guiowl_hist_k5_full_ablation_queue.tsv"
echo "=== TSV field check (row 2 = baseline, row 3 = random_r05) ==="
awk -F'\t' 'NR==2{printf "baseline limit=[%s] ratio=[%s] scope=[%s] label=[%s]\n", $5, $6, $8, $10}' "$QUEUE"
awk -F'\t' 'NR==3{printf "random_r05 limit=[%s] ratio=[%s] policy=[%s] scope=[%s]\n", $5, $6, $7, $8}' "$QUEUE"
echo "=== Active workers ==="
ps aux | grep run_geopact_ablation_job | grep -v grep || echo none
echo "=== Active locks ==="
ls -1 "${REPO}/logs/.geopact_queue_locks/"*.lock 2>/dev/null | xargs -n1 basename || echo none
