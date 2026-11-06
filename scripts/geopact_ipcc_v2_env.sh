#!/usr/bin/env bash
# IPCC v2 candidate production stack (Module C: donor-norm + hybrid kernel + adaptive σ).
# Run full 2572 gate before switching default EXP_ARM from 07_prod.
#
#   source scripts/geopact_ipcc_v2_env.sh
#   PRUNE_RATIO=0.40 bash scripts/run_qwen_a100_five_arm_one.sh

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=geopact_production_full_env.sh
source "${_SCRIPT_DIR}/geopact_production_full_env.sh"

export EXP_ARM="${EXP_ARM:-19_prod_drop_scale_a02_sspc_ipcc}"
