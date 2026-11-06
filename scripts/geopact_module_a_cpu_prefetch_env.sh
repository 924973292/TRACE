#!/usr/bin/env bash
# Shared Module A (YOLO/OmniParser) CPU prefetch stack for A+B eval arms.
# Source after eval_env_defaults.sh:
#   source scripts/geopact_module_a_cpu_prefetch_env.sh
#
# Ensures MIU-based B policies (04g/04x/04n) all run the same detector path before prune.

unset GEOPACT_MODULE_A_SIDELOAD
export GEOPACT_MODULE_A_PREFETCH=cpu
export GEOPACT_MODULE_A_SUBPROCESS=1
export GEOPACT_MODULE_A_SUBPROCESS_POOL="${GEOPACT_MODULE_A_SUBPROCESS_POOL:-1}"
export GEOPACT_MODULE_A_SUBPROCESS_PYTHON="${GEOPACT_MODULE_A_SUBPROCESS_PYTHON:-${EVAL_PYTHON}}"
export GEOPACT_MODULE_A_PREFETCH_ASYNC="${GEOPACT_MODULE_A_PREFETCH_ASYNC:-1}"
export GEOPACT_MODULE_A_WORKER_SHM="${GEOPACT_MODULE_A_WORKER_SHM:-1}"
export GEOPACT_MODULE_A_SUBPROCESS_PREWARM="${GEOPACT_MODULE_A_SUBPROCESS_PREWARM:-1}"
# Single job per GPU (eval default): YOLO in subprocess on the same visible device for
# ~150–200ms sticky_keep. Set GEOPACT_MODULE_A_WORKER_YOLO_DEVICE=cpu when multiple
# processes share one GPU and ViT/YOLO contend.
export GEOPACT_MODULE_A_WORKER_YOLO_DEVICE="${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-0}"
# Module A content / PIL short-circuit cache is forbidden in production eval.
export GEOPACT_MODULE_A_CACHE=0
export GEOPACT_MODULE_A_PIL_SHORT=0
export MIU_YOLO_DEVICE=cpu
export OMNIPARSER_ICON_MAX_EDGE="${OMNIPARSER_ICON_MAX_EDGE:-1536}"
export MIU_YOLO_MAX_EDGE="${MIU_YOLO_MAX_EDGE:-1536}"
