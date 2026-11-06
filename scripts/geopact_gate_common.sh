#!/usr/bin/env bash
# Shared helpers for GeoPACT gates: baseline_full / baseline_kv / abc_r40_kv.
# Source from gate scripts; do not execute directly.
set -euo pipefail

geopact_gate_init() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
  # shellcheck source=eval_env_defaults.sh
  source "${script_dir}/eval_env_defaults.sh"
  if [[ -f "${script_dir}/geopact_yolo_eval_env.sh" ]]; then
    # shellcheck source=geopact_yolo_eval_env.sh
    source "${script_dir}/geopact_yolo_eval_env.sh"
  fi
  cd "$REPO"
  mkdir -p "$REPO/logs"
}

geopact_gate_export_base_env() {
  local hist_n="${1:-9}"
  export GEOPACT_EVAL_HISTORY_N="${hist_n}"
  export GEOPACT_COMPACT_BACKEND=source
  export GEOPACT_VISION_EMBED_CACHE=0
  export GEOPACT_FORCE_EPISODE_ORDER=1
  export GEOPACT_KV_SLIDE_ON_ROLL="${GEOPACT_KV_SLIDE_ON_ROLL:-1}"
  export GEOPACT_KV_HISTORY_COMPRESS_LEGACY=0
  export GEOPACT_KV_HISTORY_REWRITE="${GEOPACT_KV_HISTORY_REWRITE:-1}"
  export MIU_DISABLE_OCR="${MIU_DISABLE_OCR:-1}"
  export LMMS_LIGHT_LOG="${LMMS_LIGHT_LOG:-1}"
  unset GUIOWL_NATIVE_MULTITURN
}

geopact_gate_export_abc_env() {
  export GEOPACT_MODULE_A_CROSS_STEP_PREFETCH="${GEOPACT_MODULE_A_CROSS_STEP_PREFETCH:-0}"
  export GEOPACT_MODULE_A_PREFETCH_REUSE="${GEOPACT_MODULE_A_PREFETCH_REUSE:-0}"
  export GEOPACT_MODULE_A_WORKER_YOLO_DEVICE="${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-0}"
  # Prod ABC: pure ZAFD dispersion (no spatial skeleton anchors in Module B budget).
  export GEOPACT_SSPC="${GEOPACT_SSPC:-1}"
  export GEOPACT_SSPC_SKELETON="${GEOPACT_SSPC_SKELETON:-0}"
  export GEOPACT_SSPC_ZAFD="${GEOPACT_SSPC_ZAFD:-1}"
  export GEOPACT_SSPC_BUDGET="${GEOPACT_SSPC_BUDGET:-in_budget}"
  export GEOPACT_ZAFD_OMEGA_NORM="${GEOPACT_ZAFD_OMEGA_NORM:-survivor}"
}

geopact_gate_base_model_args() {
  local model_dir="${1:-models/GUI-Owl-1.5-4B-Instruct}"
  echo "pretrained=${model_dir},attn_implementation=flash_attention_2,device_map=cuda:0,visual_prune_scope=image_sticky"
}

geopact_gate_restore_patches() {
  (
    flock -w 180 9
    "$EVAL_PYTHON" scripts/restore_q3vl_geopact_patches.py
  ) 9>"${REPO}/logs/.restore_q3vl_geopact.lock"
}

geopact_gate_verify_2arm() {
  local log="${1:-/dev/stdout}"
  "$EVAL_PYTHON" scripts/verify_kv_incremental_flow.py 2>&1 | tee -a "$log"
  "$EVAL_PYTHON" scripts/verify_kv_crop_no_mutation.py 2>&1 | tee -a "$log"
  "$EVAL_PYTHON" scripts/verify_kv_step0_native_routing.py 2>&1 | tee -a "$log"
  "$EVAL_PYTHON" scripts/verify_baseline_generate_payload.py 2>&1 | tee -a "$log"
  "$EVAL_PYTHON" scripts/verify_dialogue_mrope_boundary.py 2>&1 | tee -a "$log"
  "$EVAL_PYTHON" scripts/verify_sliding_rewrite_invariants.py 2>&1 | tee -a "$log"
  "$EVAL_PYTHON" scripts/verify_decode_mrope_contract.py 2>&1 | tee -a "$log"
}

geopact_gate_verify_3arm() {
  local log="${1:-/dev/stdout}" hist_n="${2:-9}" prune_target="${3:-history}"
  "$EVAL_PYTHON" scripts/verify_kv_crop_no_mutation.py 2>&1 | tee -a "$log"
  "$EVAL_PYTHON" scripts/verify_module_a_prefetch_identity.py 2>&1 | tee -a "$log"
  "$EVAL_PYTHON" scripts/verify_kv_history_dataset_bounds.py 2>&1 | tee -a "$log"
  GEOPACT_EVAL_HISTORY_N="${hist_n}" PRUNE_TARGET="${prune_target}" \
    "$EVAL_PYTHON" scripts/verify_hist9_abc_pipeline.py 2>&1 | tee -a "$log"
  "$EVAL_PYTHON" scripts/verify_sliding_rewrite_invariants.py 2>&1 | tee -a "$log"
  "$EVAL_PYTHON" scripts/verify_baseline_generate_payload.py 2>&1 | tee -a "$log"
  "$EVAL_PYTHON" scripts/verify_dialogue_mrope_boundary.py 2>&1 | tee -a "$log"
}

geopact_gate_run_baseline_full() {
  local out="$1" limit="$2" task="$3" model_args="$4"
  export GEOPACT_SESSION_KV=0
  unset GEOPACT_SESSION_KV_MODE GEOPACT_SCREENSHOT_ONLY_CACHE
  MODEL_ARGS="$model_args" LIMIT="$limit" OUT="$out" TASK="$task" \
    bash scripts/run_qwen_task_once_baseline_eval.sh
}

geopact_gate_run_baseline_kv() {
  local out="$1" limit="$2" task="$3" model_args="$4"
  export GEOPACT_SESSION_KV=1
  export GEOPACT_SESSION_KV_MODE=dialogue
  export GEOPACT_SCREENSHOT_ONLY_CACHE=1
  MODEL_ARGS="$model_args" LIMIT="$limit" OUT="$out" TASK="$task" \
    bash scripts/run_qwen_dialogue_kv_cache_eval.sh
}

geopact_gate_run_abc_r40() {
  local out="$1" limit="$2" task="$3" model_dir="$4" abc_arm="$5" prune_target="$6"
  export GEOPACT_SESSION_KV=1
  export GEOPACT_SESSION_KV_MODE=dialogue
  export GEOPACT_SCREENSHOT_ONLY_CACHE=1
  MODEL_DIR="$model_dir" LIMIT="$limit" TASK="$task" \
    EXP_ARM="$abc_arm" PRUNE_RATIO=0.40 OUT="$out" \
    PRUNE_TARGET="$prune_target" \
    SKIP_ENSURE_DEPS=1 SKIP_VERIFY=1 \
    bash scripts/run_qwen_a100_five_arm_one.sh
}

geopact_gate_profile_report() {
  local kv_prof="$1" abc_prof="$2" report="${3:-}"
  if [[ ! -f "$kv_prof" || ! -f "$abc_prof" ]]; then
    echo "WARN: missing profiles for report" >&2
    return 0
  fi
  local sink=(cat)
  if [[ -n "$report" ]]; then
    sink=(tee "$report")
  fi
  {
    echo "=== diagnose_abc_decode_latency ==="
    "$EVAL_PYTHON" scripts/diagnose_abc_decode_latency.py "$kv_prof" "$abc_prof"
    echo ""
    echo "=== compare_step_latency_profiles ==="
    "$EVAL_PYTHON" scripts/compare_step_latency_profiles.py "$kv_prof" "$abc_prof"
    echo ""
    echo "=== abc_latency_next_target ==="
    "$EVAL_PYTHON" scripts/abc_latency_next_target.py "$kv_prof" "$abc_prof"
  } | "${sink[@]}"
}
