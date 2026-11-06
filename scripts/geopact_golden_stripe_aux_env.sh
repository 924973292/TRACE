#!/usr/bin/env bash
# Legacy LIMIT=100 gate stack (stripe_aux GP 42.19% @ r=0.4).
# Production default: scripts/geopact_production_full_env.sh (full-fleet optimal).
# Source after eval_env_defaults.sh:
#   source scripts/geopact_golden_stripe_aux_env.sh
#
# Causal ladder B→A ablations use the same detect/KV stack unless noted.

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=geopact_module_a_cpu_prefetch_env.sh
source "${_SCRIPT_DIR}/geopact_module_a_cpu_prefetch_env.sh"

export GEOPACT_SESSION_KV=1
export GEOPACT_SESSION_KV_MODE=dialogue
export GEOPACT_SCREENSHOT_ONLY_CACHE=1
unset GEOPACT_COORD_EPHEMERAL
export GEOPACT_COORD_METADATA_ONLY=1
export GEOPACT_CDCT_ALPHA=0.1
export GEOPACT_YOLO_CIRCULAR_AUX=1
export GEOPACT_YOLO_STRIPE_AUX=1
export GEOPACT_YOLO_OCR_AUX=0
export MIU_DISABLE_OCR=1
export GEOPACT_MODULE_A_CACHE=0
export GEOPACT_MODULE_A_PIL_SHORT=0
export GEOPACT_MODULE_A_WORKER_YOLO_DEVICE="${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-0}"
unset GEOPACT_MODULE_A_DETECT_PARALLEL
unset GEOPACT_MODULE_A_CV_PARALLEL
unset BLOB_AUX_USE_FULL_RES
unset OMNIPARSER_ICON_CONF
