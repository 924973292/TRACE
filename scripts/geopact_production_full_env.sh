#!/usr/bin/env bash
# Production stack: factorial R3 champion (2572 steps, r=40% GP=28.04%).
# EXP_ARM=07_prod_drop_scale_a02  (drop_S + circular ON + CDCT α=0.2)
#
#   source scripts/geopact_production_full_env.sh
#   PRUNE_RATIO=0.40 EXP_ARM=07_prod_drop_scale_a02 bash scripts/run_qwen_a100_five_arm_one.sh

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=geopact_module_a_cpu_prefetch_env.sh
source "${_SCRIPT_DIR}/geopact_module_a_cpu_prefetch_env.sh"

export GEOPACT_SESSION_KV=1
export GEOPACT_SESSION_KV_MODE=dialogue
export GEOPACT_SCREENSHOT_ONLY_CACHE=1
unset GEOPACT_COORD_EPHEMERAL
export GEOPACT_COORD_METADATA_ONLY=1
export GEOPACT_CDCT_ALPHA=0.2
export GEOPACT_YOLO_CIRCULAR_AUX=1
unset GEOPACT_YOLO_STRIPE_AUX
export GEOPACT_YOLO_OCR_AUX=0
export MIU_DISABLE_OCR=1
export GEOPACT_MODULE_A_CACHE=0
export GEOPACT_MODULE_A_PIL_SHORT=0
export GEOPACT_MODULE_A_WORKER_YOLO_DEVICE="${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-0}"
unset GEOPACT_MODULE_A_DETECT_PARALLEL
unset GEOPACT_MODULE_A_CV_PARALLEL
unset BLOB_AUX_USE_FULL_RES
unset OMNIPARSER_ICON_CONF

# Default production arm (override via EXP_ARM= if needed)
# v0 ZAFD champion: EXP_ARM=18_prod_drop_scale_a02_sspc_zafd (+0.44pp @ 2572)
# IPCC v2 candidate: source scripts/geopact_ipcc_v2_env.sh (19_prod_drop_scale_a02_sspc_ipcc)
export EXP_ARM="${EXP_ARM:-07_prod_drop_scale_a02}"

# Policy: oas_ickh_cdct_no_coord_yolo_v2_drop_pent_scale (set by 07_prod_drop_scale_a02 arm)
