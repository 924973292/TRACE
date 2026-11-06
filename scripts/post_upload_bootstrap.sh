#!/usr/bin/env bash
# Post-upload bootstrap: build manifests, patch paths, strict verify, offline-matrix preflight.
# Assumes REPO already contains data/ + models/ (merged from E: upload).
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_SCRIPT_DIR}/eval_env_defaults.sh"

REPO="${REPO}"
EVAL="${EVAL_PYTHON:-python3}"
export OMINI_REPO="${OMINI_REPO:-$REPO}"

cd "$REPO"
echo "=== post_upload_bootstrap REPO=${REPO} ==="

"${EVAL}" scripts/build_screenspot_v2_full_manifest.py
"${EVAL}" scripts/build_screenspot_pro_manifest.py
"${EVAL}" scripts/build_mmbench_gui_manifest.py
"${EVAL}" scripts/build_android_control_manifest.py

# Tier2: patch eval_steps image_path if screenshots already extracted (no-op if complete).
"${EVAL}" scripts/download_android_control_images.py --repo-root "$REPO" --skip-download

"${EVAL}" scripts/verify_cross_benchmark_data.py --strict --repo-root "$REPO"
"${EVAL}" scripts/preflight_cross_benchmark.py --require-8b --offline-matrix

echo "=== post_upload_bootstrap OK — run: bash scripts/run_cross_benchmark_offline_matrix_8b.sh ==="
