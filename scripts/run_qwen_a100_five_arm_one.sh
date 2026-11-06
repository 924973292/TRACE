#!/usr/bin/env bash
# Single arm for KV+prune sweep (no vision embed cache).
#
# Arms (EXP_ARM): baseline + dialogue SESSION_KV + uniform keep 99/95/40/20/10/5%.
#
# Example:
#   LIMIT=100 EXP_ARM=03_dialogue_kv_r95 TAG=kv_prune_20260618 \
#     bash scripts/run_qwen_a100_five_arm_one.sh
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=eval_env_defaults.sh
source "${_SCRIPT_DIR}/eval_env_defaults.sh"
# shellcheck source=kv_prune_sweep_arms.sh
source "${_SCRIPT_DIR}/kv_prune_sweep_arms.sh"
REPO="${REPO}"
EVAL="${EVAL_PYTHON}"
LIMIT="${LIMIT:-}"
TASK="${TASK:-task_v2_qwen3_omni_novideo}"
EXP_ARM="${EXP_ARM:?EXP_ARM required — see kv_prune_sweep_arms.sh}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
if [[ -n "$LIMIT" ]]; then
  OUT_ROOT="${OUT_ROOT:-${REPO}/logs/kv_prune_sweep_limit${LIMIT}_${TAG}}"
else
  OUT_ROOT="${OUT_ROOT:-${REPO}/logs/kv_prune_sweep_full_${TAG}}"
fi
OUT="${OUT:-${OUT_ROOT}/${EXP_ARM}}"

cd "$REPO"
if [[ -f scripts/geopact_yolo_eval_env.sh ]]; then
  # shellcheck source=geopact_yolo_eval_env.sh
  source scripts/geopact_yolo_eval_env.sh
fi

echo "EVAL_PYTHON=${EVAL_PYTHON}"
"${EVAL}" -c "import sys; print('sys.executable=', sys.executable)"
if [[ "${SKIP_ENSURE_DEPS:-0}" != "1" ]]; then
  bash "${_SCRIPT_DIR}/ensure_omini_eval_deps.sh" --check-only || bash "${_SCRIPT_DIR}/ensure_omini_eval_deps.sh"
fi

bash "${_SCRIPT_DIR}/check_eval_cuda.sh"

export GEOPACT_COMPACT_BACKEND=source
export GEOPACT_VISION_EMBED_CACHE=0
export GEOPACT_FORCE_EPISODE_ORDER=1
export MIU_DISABLE_OCR="${MIU_DISABLE_OCR:-1}"
export LMMS_LIGHT_LOG="${LMMS_LIGHT_LOG:-1}"
export BILATERAL_DECODE_PIN="${BILATERAL_DECODE_PIN:-1}"
export GEOPACT_MODULE_A_CROSS_STEP_PREFETCH="${GEOPACT_MODULE_A_CROSS_STEP_PREFETCH:-0}"
export GEOPACT_MODULE_A_PREFETCH_REUSE="${GEOPACT_MODULE_A_PREFETCH_REUSE:-0}"
export GEOPACT_TIMING_CUDA_SYNC="${GEOPACT_TIMING_CUDA_SYNC:-0}"
MODEL_DIR="${MODEL_DIR:-models/GUI-Owl-1.5-8B-Instruct}"
# Episode-ordered eval + dialogue KV suffix alignment for official native_mt4 OmniGUI.
if [[ "${TASK}" == *native_mt4* ]] || [[ "${TASK}" == task_guiowl_omnigui* ]]; then
  export GUIOWL_NATIVE_MULTITURN=1
else
  unset GUIOWL_NATIVE_MULTITURN
fi
BASE_MODEL_ARGS="pretrained=${MODEL_DIR},attn_implementation=flash_attention_2,device_map=auto,visual_prune_scope=image_sticky"
# Vision resolution override (official grounding eval uses min=196*32*32=200704,
# max=9800*32*32=10035200; checkpoint preprocessor default caps at ~1.0MP which
# cripples ScreenSpot-Pro 4K screenshots).
if [[ -n "${VISION_MAX_PIXELS:-}" ]]; then
  BASE_MODEL_ARGS="${BASE_MODEL_ARGS},max_pixels=${VISION_MAX_PIXELS}"
fi
if [[ -n "${VISION_MIN_PIXELS:-}" ]]; then
  BASE_MODEL_ARGS="${BASE_MODEL_ARGS},min_pixels=${VISION_MIN_PIXELS}"
fi
# Prune target: current (default, single-image perception) vs history (multi-step:
# keep the current screenshot full, prune only frames that leave the current slot).
PRUNE_TARGET="${PRUNE_TARGET:-current}"
if [[ "${PRUNE_TARGET}" == "history" ]]; then
  BASE_MODEL_ARGS="${BASE_MODEL_ARGS},visual_prune_target=history"
fi
PRUNE_SUFFIX=",visual_prune_policy=uniform"
FLEET_PRUNE_RATIO="${PRUNE_RATIO:-}"
PRUNE_RATIO=""

_apply_fleet_prune_ratio() {
  if [[ -z "${FLEET_PRUNE_RATIO}" ]]; then
    return 0
  fi
  PRUNE_RATIO="${FLEET_PRUNE_RATIO}"
  if [[ -n "${MODEL_ARGS:-}" ]] && [[ "${MODEL_ARGS}" == *visual_prune_ratio=* ]]; then
    MODEL_ARGS="$(echo "${MODEL_ARGS}" | sed -E "s/visual_prune_ratio=[^,]*/visual_prune_ratio=${PRUNE_RATIO}/")"
  fi
}

_setup_module_a_cdct_attr_arm() {
  local policy_suffix="$1"
  local arm_label="$2"
  export GEOPACT_SESSION_KV=1
  export GEOPACT_SESSION_KV_MODE=dialogue
  export GEOPACT_SCREENSHOT_ONLY_CACHE=1
  unset GEOPACT_COORD_EPHEMERAL
  unset GEOPACT_CORRECTION
  export GEOPACT_COORD_METADATA_ONLY=1
  # shellcheck source=geopact_module_a_cpu_prefetch_env.sh
  source "${_SCRIPT_DIR}/geopact_module_a_cpu_prefetch_env.sh"
  export GEOPACT_CDCT_ALPHA=0.1
  PRUNE_RATIO="0.40"
  local pol="oas_ickh_cdct_no_coord_yolo"
  if [[ -n "${policy_suffix}" ]]; then
    pol="${pol}_${policy_suffix}"
  fi
  MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=${pol},visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
  ARM_LABEL="${arm_label}"
}

_setup_module_a_v2_arm() {
  local policy_suffix="$1"
  local arm_label="$2"
  local circular_aux="${3:-1}"
  export GEOPACT_SESSION_KV=1
  export GEOPACT_SESSION_KV_MODE=dialogue
  export GEOPACT_SCREENSHOT_ONLY_CACHE=1
  unset GEOPACT_COORD_EPHEMERAL
  unset GEOPACT_CORRECTION
  export GEOPACT_COORD_METADATA_ONLY=1
  # shellcheck source=geopact_module_a_cpu_prefetch_env.sh
  source "${_SCRIPT_DIR}/geopact_module_a_cpu_prefetch_env.sh"
  export GEOPACT_CDCT_ALPHA=0.1
  if [[ "${circular_aux}" == "0" ]]; then
    export GEOPACT_YOLO_CIRCULAR_AUX=0
  else
    export GEOPACT_YOLO_CIRCULAR_AUX=1
  fi
  PRUNE_RATIO="0.40"
  local pol="oas_ickh_cdct_no_coord_yolo_v2"
  if [[ -n "${policy_suffix}" ]]; then
    pol="${pol}_${policy_suffix}"
  fi
  # W49: energy-term knockouts (drop_pent_entropy/contrast/containment/
  # resonance) and the with-S control are policy tokens, not env flags, so the
  # queue needs a way to append them without cloning this arm. Setting
  # GEOPACT_POLICY_EXTRA=none yields the bare v2 policy, which restores S.
  if [[ -n "${GEOPACT_POLICY_EXTRA:-}" ]]; then
    if [[ "${GEOPACT_POLICY_EXTRA}" == "none" ]]; then
      pol="oas_ickh_cdct_no_coord_yolo_v2"
    else
      pol="${pol}_${GEOPACT_POLICY_EXTRA}"
    fi
  fi
  MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=${pol},visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
  ARM_LABEL="${arm_label}"
}

_setup_module_c_geom_v2_arm() {
  local policy_suffix="$1"
  local arm_label="$2"
  local geom_mode="$3"
  local circular_aux="${4:-1}"
  _setup_module_a_v2_arm "${policy_suffix}" "${arm_label}" "${circular_aux}"
  export GEOPACT_CDCT_ALPHA=0.2
  unset GEOPACT_COORD_METADATA_ONLY
  unset GEOPACT_SPATIAL_RPB
  unset GEOPACT_CDCT_RESIDUE
  unset GEOPACT_RPB_SHUFFLE_POS
  unset GEOPACT_RPB_SHUFFLE_MIU
  unset GEOPACT_RPB_IOU
  unset GEOPACT_GAEC_SHUFFLE_POS
  unset GEOPACT_GAEC_V2
  unset GEOPACT_GAEC_TOPK
  export GEOPACT_COORD_MODE=text
  unset GEOPACT_COORD_EPHEMERAL
  unset GEOPACT_COORD_TOPK
  unset GEOPACT_COORD_FORMAT
  unset GEOPACT_COORD_SLOT

  local pol="oas_ickh_cdct_no_coord_yolo_v2"
  case "${geom_mode}" in
    rpb)
      export GEOPACT_SPATIAL_RPB=1
      ;;
    rpb_residue)
      export GEOPACT_SPATIAL_RPB=1
      export GEOPACT_CDCT_RESIDUE=1
      ;;
    rpb_shuffle)
      export GEOPACT_SPATIAL_RPB=1
      export GEOPACT_RPB_SHUFFLE_POS=1
      ;;
    rpb_iou)
      export GEOPACT_SPATIAL_RPB=1
      export GEOPACT_RPB_IOU=1
      ;;
    rpb_iou_residue)
      export GEOPACT_SPATIAL_RPB=1
      export GEOPACT_RPB_IOU=1
      export GEOPACT_CDCT_RESIDUE=1
      ;;
    rpb_iou_shuffle)
      export GEOPACT_SPATIAL_RPB=1
      export GEOPACT_RPB_IOU=1
      export GEOPACT_RPB_SHUFFLE_POS=1
      ;;
    gaec)
      export GEOPACT_COORD_MODE=gaec
      export GEOPACT_COORD_EPHEMERAL=1
      export GEOPACT_COORD_TOPK=3
      export GEOPACT_COORD_SLOT=before_gen_tail
      pol="oas_ickh_cdct_coord_sdb_yolo_v2"
      ;;
    gaec_shuffle)
      export GEOPACT_COORD_MODE=gaec
      export GEOPACT_COORD_EPHEMERAL=1
      export GEOPACT_COORD_TOPK=3
      export GEOPACT_COORD_SLOT=before_gen_tail
      export GEOPACT_GAEC_SHUFFLE_POS=1
      pol="oas_ickh_cdct_coord_sdb_yolo_v2"
      ;;
    gaec_v2)
      export GEOPACT_SPATIAL_RPB=0
      export GEOPACT_COORD_MODE=gaec
      export GEOPACT_COORD_EPHEMERAL=1
      export GEOPACT_GAEC_V2=1
      export GEOPACT_GAEC_TOPK=1
      export GEOPACT_COORD_TOPK=1
      export GEOPACT_COORD_SLOT=before_gen_tail
      pol="oas_ickh_cdct_coord_sdb_yolo_v2"
      ;;
    abc_iou)
      export GEOPACT_SPATIAL_RPB=1
      export GEOPACT_RPB_IOU=1
      export GEOPACT_COORD_MODE=gaec
      export GEOPACT_COORD_EPHEMERAL=1
      export GEOPACT_GAEC_V2=1
      export GEOPACT_GAEC_TOPK=1
      export GEOPACT_COORD_TOPK=1
      export GEOPACT_COORD_SLOT=before_gen_tail
      pol="oas_ickh_cdct_coord_sdb_yolo_v2"
      ;;
    *)
      echo "Unknown geom_mode: ${geom_mode}" >&2
      exit 1
      ;;
  esac
  if [[ -n "${policy_suffix}" ]]; then
    pol="${pol}_${policy_suffix}"
  fi
  if [[ -n "${MODEL_ARGS:-}" ]] && [[ "${MODEL_ARGS}" == *visual_prune_policy=* ]]; then
    MODEL_ARGS="$(echo "${MODEL_ARGS}" | sed -E "s/visual_prune_policy=[^,]*/visual_prune_policy=${pol}/")"
  fi
  ARM_LABEL="${arm_label}"
}

_setup_module_c_parb_arm() {
  local policy_suffix="$1"
  local arm_label="$2"
  local parb_mode="$3"
  local circular_aux="${4:-1}"
  _setup_module_a_v2_arm "${policy_suffix}" "${arm_label}" "${circular_aux}"
  export GEOPACT_CDCT_ALPHA=0.2
  unset GEOPACT_SPATIAL_RPB
  unset GEOPACT_CDCT_RESIDUE
  unset GEOPACT_COORD_EPHEMERAL
  unset GEOPACT_COORD_MODE
  export GEOPACT_PARB=1
  unset GEOPACT_PARB_VV
  unset GEOPACT_PARB_TV
  unset GEOPACT_PARB_DENSITY
  unset GEOPACT_PARB_IOU_PHI
  case "${parb_mode}" in
    parb_eager_vv)
      export GEOPACT_PARB_MODE=eager
      export GEOPACT_PARB_VV=1
      export GEOPACT_PARB_TV=0
      ;;
    parb_eager_tv)
      export GEOPACT_PARB_MODE=eager
      export GEOPACT_PARB_VV=0
      export GEOPACT_PARB_TV=1
      ;;
    parb_eager_full)
      export GEOPACT_PARB_MODE=eager
      export GEOPACT_PARB_VV=1
      export GEOPACT_PARB_TV=1
      ;;
    parb_hybrid_full)
      export GEOPACT_PARB_MODE=hybrid
      export GEOPACT_PARB_VV=1
      export GEOPACT_PARB_TV=1
      ;;
    parb_hybrid_vv)
      export GEOPACT_PARB_MODE=hybrid
      export GEOPACT_PARB_VV=1
      export GEOPACT_PARB_TV=0
      ;;
    parb_hybrid_tv)
      export GEOPACT_PARB_MODE=hybrid
      export GEOPACT_PARB_VV=0
      export GEOPACT_PARB_TV=1
      ;;
    parb_hybrid_nodensity)
      export GEOPACT_PARB_MODE=hybrid
      export GEOPACT_PARB_VV=1
      export GEOPACT_PARB_TV=1
      export GEOPACT_PARB_DENSITY=0
      ;;
    parb_hybrid_no_phi)
      export GEOPACT_PARB_MODE=hybrid
      export GEOPACT_PARB_VV=1
      export GEOPACT_PARB_TV=1
      export GEOPACT_PARB_IOU_PHI=0
      ;;
    *)
      echo "Unknown parb_mode: ${parb_mode}" >&2
      exit 1
      ;;
  esac
  local pol="oas_ickh_cdct_no_coord_yolo_v2"
  if [[ -n "${policy_suffix}" ]]; then
    pol="${pol}_${policy_suffix}"
  fi
  if [[ -n "${MODEL_ARGS:-}" ]] && [[ "${MODEL_ARGS}" == *visual_prune_policy=* ]]; then
    MODEL_ARGS="$(echo "${MODEL_ARGS}" | sed -E "s/visual_prune_policy=[^,]*/visual_prune_policy=${pol}/")"
  fi
  ARM_LABEL="${arm_label}"
}

_setup_module_c_parb_lift_arm() {
  local policy_suffix="$1"
  local arm_label="$2"
  local lift_mode="$3"
  local circular_aux="${4:-1}"
  _setup_module_a_v2_arm "${policy_suffix}" "${arm_label}" "${circular_aux}"
  export GEOPACT_CDCT_ALPHA=0.2
  unset GEOPACT_SPATIAL_RPB
  unset GEOPACT_CDCT_RESIDUE
  unset GEOPACT_COORD_EPHEMERAL
  unset GEOPACT_COORD_MODE
  export GEOPACT_PARB=1
  export GEOPACT_PARB_LIFT=1
  unset GEOPACT_PARB_MODE
  unset GEOPACT_PARB_VV
  unset GEOPACT_PARB_TV
  unset GEOPACT_PARB_DENSITY
  unset GEOPACT_PARB_IOU_PHI
  unset GEOPACT_PARB_LIFT_RANK
  case "${lift_mode}" in
    lift_decomposed_full)
      export GEOPACT_PARB_LIFT_MODE=decomposed
      export GEOPACT_PARB_VV=1
      export GEOPACT_PARB_TV=1
      ;;
    lift_decomposed_tv)
      export GEOPACT_PARB_LIFT_MODE=decomposed
      export GEOPACT_PARB_VV=0
      export GEOPACT_PARB_TV=1
      ;;
    lift_svd_full)
      export GEOPACT_PARB_LIFT_MODE=svd
      export GEOPACT_PARB_VV=1
      export GEOPACT_PARB_TV=1
      export GEOPACT_PARB_LIFT_RANK=16
      ;;
    *)
      echo "Unknown lift_mode: ${lift_mode}" >&2
      exit 1
      ;;
  esac
  local pol="oas_ickh_cdct_no_coord_yolo_v2"
  if [[ -n "${policy_suffix}" ]]; then
    pol="${pol}_${policy_suffix}"
  fi
  if [[ -n "${MODEL_ARGS:-}" ]] && [[ "${MODEL_ARGS}" == *visual_prune_policy=* ]]; then
    MODEL_ARGS="$(echo "${MODEL_ARGS}" | sed -E "s/visual_prune_policy=[^,]*/visual_prune_policy=${pol}/")"
  fi
  ARM_LABEL="${arm_label}"
}

_setup_module_c_trp_arm() {
  local policy_suffix="$1"
  local arm_label="$2"
  local trp_mode_arg="$3"
  local circular_aux="${4:-1}"
  _setup_module_a_v2_arm "${policy_suffix}" "${arm_label}" "${circular_aux}"
  export GEOPACT_CDCT_ALPHA=0.2
  unset GEOPACT_SPATIAL_RPB
  unset GEOPACT_CDCT_RESIDUE
  unset GEOPACT_COORD_EPHEMERAL
  unset GEOPACT_COORD_MODE
  unset GEOPACT_PARB
  unset GEOPACT_PARB_LIFT
  export GEOPACT_TRP=1
  unset GEOPACT_TRP_ETA
  unset GEOPACT_TRP_REGION
  unset GEOPACT_TRP_PSI
  unset GEOPACT_TRP_LAYERS
  case "${trp_mode_arg}" in
    trp_hybrid)
      export GEOPACT_TRP_MODE=hybrid
      ;;
    trp_lift)
      export GEOPACT_TRP_MODE=lift
      ;;
    trp_lift_no_psi)
      export GEOPACT_TRP_MODE=lift
      export GEOPACT_TRP_PSI=0
      ;;
    trp_lift_no_region)
      export GEOPACT_TRP_MODE=lift
      export GEOPACT_TRP_REGION=0
      ;;
    trp_lift_l0)
      export GEOPACT_TRP_MODE=lift
      export GEOPACT_TRP_LAYERS=0
      ;;
    *)
      echo "Unknown trp_mode: ${trp_mode_arg}" >&2
      exit 1
      ;;
  esac
  local pol="oas_ickh_cdct_no_coord_yolo_v2"
  if [[ -n "${policy_suffix}" ]]; then
    pol="${pol}_${policy_suffix}"
  fi
  if [[ -n "${MODEL_ARGS:-}" ]] && [[ "${MODEL_ARGS}" == *visual_prune_policy=* ]]; then
    MODEL_ARGS="$(echo "${MODEL_ARGS}" | sed -E "s/visual_prune_policy=[^,]*/visual_prune_policy=${pol}/")"
  fi
  ARM_LABEL="${arm_label}"
}

_setup_module_c_gsc_arm() {
  local policy_suffix="$1"
  local arm_label="$2"
  local gsc_arm_arg="$3"
  local circular_aux="${4:-1}"
  _setup_module_a_v2_arm "${policy_suffix}" "${arm_label}" "${circular_aux}"
  export GEOPACT_CDCT_ALPHA=0.2
  unset GEOPACT_SPATIAL_RPB
  unset GEOPACT_CDCT_RESIDUE
  unset GEOPACT_COORD_EPHEMERAL
  unset GEOPACT_COORD_MODE
  unset GEOPACT_PARB
  unset GEOPACT_PARB_LIFT
  unset GEOPACT_TRP
  export GEOPACT_GSC=1
  export GEOPACT_GSC_MODE=fa
  unset GEOPACT_GSC_VFS
  unset GEOPACT_GSC_SPF
  unset GEOPACT_GSC_ETA0
  unset GEOPACT_GSC_TAU
  unset GEOPACT_GSC_SCHEDULE
  unset GEOPACT_GSC_QUERY_ADAPT
  unset GEOPACT_GSC_LAYERS
  case "${gsc_arm_arg}" in
    gsc_fa)
      export GEOPACT_GSC_VFS=1
      export GEOPACT_GSC_SPF=center
      ;;
    gsc_fa_lpr_only)
      export GEOPACT_GSC_VFS=1
      export GEOPACT_GSC_SPF=off
      ;;
    gsc_fa_schedule)
      export GEOPACT_GSC_VFS=1
      export GEOPACT_GSC_SPF=center
      export GEOPACT_GSC_SCHEDULE=1
      ;;
    gsc_fa_query_adapt)
      export GEOPACT_GSC_VFS=1
      export GEOPACT_GSC_SPF=full
      export GEOPACT_GSC_QUERY_ADAPT=1
      ;;
    gsc_spf_center)
      export GEOPACT_GSC_MODE=lift
      export GEOPACT_GSC_VFS=0
      export GEOPACT_GSC_SPF=center
      ;;
    gsc_vfs_only)
      export GEOPACT_GSC_MODE=lift
      export GEOPACT_GSC_VFS=1
      export GEOPACT_GSC_SPF=off
      ;;
    gsc_vfs_spf)
      export GEOPACT_GSC_MODE=lift
      export GEOPACT_GSC_VFS=1
      export GEOPACT_GSC_SPF=center
      ;;
    gsc_vfs_spf_schedule)
      export GEOPACT_GSC_MODE=lift
      export GEOPACT_GSC_VFS=1
      export GEOPACT_GSC_SPF=center
      export GEOPACT_GSC_SCHEDULE=1
      ;;
    gsc_full_query_adapt)
      export GEOPACT_GSC_MODE=lift
      export GEOPACT_GSC_VFS=1
      export GEOPACT_GSC_SPF=full
      export GEOPACT_GSC_QUERY_ADAPT=1
      ;;
    gsc_spf_full)
      export GEOPACT_GSC_MODE=lift
      export GEOPACT_GSC_VFS=0
      export GEOPACT_GSC_SPF=full
      ;;
    gsc_hybrid)
      export GEOPACT_GSC_MODE=hybrid
      export GEOPACT_GSC_VFS=1
      export GEOPACT_GSC_SPF=center
      ;;
    *)
      echo "Unknown gsc_arm: ${gsc_arm_arg}" >&2
      exit 1
      ;;
  esac
  local pol="oas_ickh_cdct_no_coord_yolo_v2"
  if [[ -n "${policy_suffix}" ]]; then
    pol="${pol}_${policy_suffix}"
  fi
  if [[ -n "${MODEL_ARGS:-}" ]] && [[ "${MODEL_ARGS}" == *visual_prune_policy=* ]]; then
    MODEL_ARGS="$(echo "${MODEL_ARGS}" | sed -E "s/visual_prune_policy=[^,]*/visual_prune_policy=${pol}/")"
  fi
  ARM_LABEL="${arm_label}"
}

_setup_module_c_correction_arm() {
  local policy_suffix="$1"
  local arm_label="$2"
  local correction_arm_arg="$3"
  local circular_aux="${4:-1}"
  _setup_module_a_v2_arm "${policy_suffix}" "${arm_label}" "${circular_aux}"
  export GEOPACT_CDCT_ALPHA=0.2
  unset GEOPACT_SPATIAL_RPB
  unset GEOPACT_CDCT_RESIDUE
  unset GEOPACT_COORD_EPHEMERAL
  unset GEOPACT_COORD_MODE
  unset GEOPACT_PARB
  unset GEOPACT_PARB_LIFT
  unset GEOPACT_TRP
  unset GEOPACT_GSC
  export GEOPACT_CORRECTION=1
  unset GEOPACT_CORRECTION_LPR
  unset GEOPACT_CORRECTION_TV
  unset GEOPACT_CORRECTION_REGION
  unset GEOPACT_CORRECTION_PSI
  unset GEOPACT_CORRECTION_ETA
  unset GEOPACT_CORRECTION_TAU
  unset GEOPACT_CORRECTION_SCHEDULE
  unset GEOPACT_CORRECTION_LAYERS
  case "${correction_arm_arg}" in
    dmtc_lift_lpr)
      export GEOPACT_CORRECTION_LPR=1
      export GEOPACT_CORRECTION_TV=1
      ;;
    dmtc_lift_no_lpr)
      export GEOPACT_CORRECTION_LPR=0
      export GEOPACT_CORRECTION_TV=1
      ;;
    dmtc_lift_no_tv)
      export GEOPACT_CORRECTION_LPR=1
      export GEOPACT_CORRECTION_TV=0
      ;;
    dmtc_lift_schedule)
      export GEOPACT_CORRECTION_LPR=1
      export GEOPACT_CORRECTION_TV=1
      export GEOPACT_CORRECTION_SCHEDULE=1
      ;;
    dmtc_lift_no_region)
      export GEOPACT_CORRECTION_LPR=1
      export GEOPACT_CORRECTION_TV=1
      export GEOPACT_CORRECTION_REGION=0
      ;;
    dmtc_lift_no_psi)
      export GEOPACT_CORRECTION_LPR=1
      export GEOPACT_CORRECTION_TV=1
      export GEOPACT_CORRECTION_PSI=0
      ;;
    *)
      echo "Unknown correction_arm: ${correction_arm_arg}" >&2
      exit 1
      ;;
  esac
  local pol="oas_ickh_cdct_no_coord_yolo_v2"
  if [[ -n "${policy_suffix}" ]]; then
    pol="${pol}_${policy_suffix}"
  fi
  if [[ -n "${MODEL_ARGS:-}" ]] && [[ "${MODEL_ARGS}" == *visual_prune_policy=* ]]; then
    MODEL_ARGS="$(echo "${MODEL_ARGS}" | sed -E "s/visual_prune_policy=[^,]*/visual_prune_policy=${pol}/")"
  fi
  ARM_LABEL="${arm_label}"
}

_setup_module_c_vsas_arm() {
  local policy_suffix="$1"
  local arm_label="$2"
  local vsas_arm_arg="$3"
  local circular_aux="${4:-1}"
  _setup_module_a_v2_arm "${policy_suffix}" "${arm_label}" "${circular_aux}"
  export GEOPACT_CDCT_ALPHA=0.2
  unset GEOPACT_SPATIAL_RPB
  unset GEOPACT_CDCT_RESIDUE
  unset GEOPACT_COORD_EPHEMERAL
  unset GEOPACT_COORD_MODE
  unset GEOPACT_PARB
  unset GEOPACT_PARB_LIFT
  unset GEOPACT_TRP
  unset GEOPACT_GSC
  unset GEOPACT_CORRECTION
  unset GEOPACT_CORRECTION_LPR
  unset GEOPACT_CORRECTION_TV
  unset GEOPACT_CORRECTION_REGION
  unset GEOPACT_CORRECTION_PSI
  unset GEOPACT_CORRECTION_ETA
  unset GEOPACT_CORRECTION_TAU
  unset GEOPACT_CORRECTION_SCHEDULE
  unset GEOPACT_CORRECTION_LAYERS
  export GEOPACT_VSAS=1
  unset GEOPACT_VSAS_MODE
  unset GEOPACT_VSAS_SIGMA
  unset GEOPACT_VSAS_ANCHORS
  unset GEOPACT_VSAS_VISION_ONLY
  unset GEOPACT_VSAS_DENOM_ONLY
  unset GEOPACT_VSAS_LAYERS
  export GEOPACT_VSAS_VISION_ONLY=1
  case "${vsas_arm_arg}" in
    vsas_internal_9)
      export GEOPACT_VSAS_MODE=internal
      export GEOPACT_VSAS_ANCHORS=nine_grid
      ;;
    vsas_internal_4)
      export GEOPACT_VSAS_MODE=internal
      export GEOPACT_VSAS_ANCHORS=four_corners
      ;;
    vsas_internal_l0)
      export GEOPACT_VSAS_MODE=internal
      export GEOPACT_VSAS_ANCHORS=nine_grid
      export GEOPACT_VSAS_LAYERS=0
      ;;
    vsas_internal_denom)
      export GEOPACT_VSAS_MODE=internal
      export GEOPACT_VSAS_ANCHORS=nine_grid
      export GEOPACT_VSAS_DENOM_ONLY=1
      ;;
    vsas_seq_9)
      export GEOPACT_VSAS_MODE=seq
      export GEOPACT_VSAS_ANCHORS=nine_grid
      ;;
    vsas_seq_4)
      export GEOPACT_VSAS_MODE=seq
      export GEOPACT_VSAS_ANCHORS=four_corners
      ;;
    *)
      echo "Unknown vsas_arm: ${vsas_arm_arg}" >&2
      exit 1
      ;;
  esac
  local pol="oas_ickh_cdct_no_coord_yolo_v2"
  if [[ -n "${policy_suffix}" ]]; then
    pol="${pol}_${policy_suffix}"
  fi
  if [[ -n "${MODEL_ARGS:-}" ]] && [[ "${MODEL_ARGS}" == *visual_prune_policy=* ]]; then
    MODEL_ARGS="$(echo "${MODEL_ARGS}" | sed -E "s/visual_prune_policy=[^,]*/visual_prune_policy=${pol}/")"
  fi
  ARM_LABEL="${arm_label}"
}

_setup_module_sspc_arm() {
  local policy_suffix="$1"
  local arm_label="$2"
  local sspc_arm_arg="$3"
  local circular_aux="${4:-1}"
  _setup_module_a_v2_arm "${policy_suffix}" "${arm_label}" "${circular_aux}"
  export GEOPACT_CDCT_ALPHA=0.2
  unset GEOPACT_COORD_EPHEMERAL
  unset GEOPACT_COORD_MODE
  unset GEOPACT_PARB
  unset GEOPACT_PARB_LIFT
  unset GEOPACT_TRP
  unset GEOPACT_GSC
  unset GEOPACT_CORRECTION
  unset GEOPACT_VSAS
  export GEOPACT_SSPC=1
  unset GEOPACT_SSPC_SKELETON
  unset GEOPACT_SSPC_ZAFD
  unset GEOPACT_SSPC_RANDOM_EXTRA
  unset GEOPACT_SSPC_RANDOM_EXTRA_K
  unset GEOPACT_SSPC_SIGMA
  unset GEOPACT_SSPC_K_GRID
  unset GEOPACT_SSPC_AXIS
  unset GEOPACT_SSPC_BUDGET
  unset GEOPACT_ZAFD_OMEGA_NORM
  unset GEOPACT_ZAFD_GAMMA
  unset GEOPACT_ZAFD_ADAPTIVE_SIGMA
  unset GEOPACT_ZAFD_SIGMA_BETA
  case "${sspc_arm_arg}" in
    sspc_full)
      export GEOPACT_SSPC_SKELETON=1
      export GEOPACT_SSPC_ZAFD=1
      export GEOPACT_SSPC_BUDGET=in_budget
      ;;
    sspc_zafd)
      export GEOPACT_SSPC_SKELETON=0
      export GEOPACT_SSPC_ZAFD=1
      export GEOPACT_SSPC_BUDGET=in_budget
      export GEOPACT_ZAFD_OMEGA_NORM=survivor
      ;;
    sspc_skel)
      export GEOPACT_SSPC_SKELETON=1
      export GEOPACT_SSPC_ZAFD=0
      export GEOPACT_SSPC_BUDGET=in_budget
      ;;
    sspc_random_extra)
      export GEOPACT_SSPC_SKELETON=0
      export GEOPACT_SSPC_ZAFD=0
      export GEOPACT_SSPC_RANDOM_EXTRA=1
      export GEOPACT_SSPC_RANDOM_EXTRA_K=9
      export GEOPACT_SSPC_BUDGET=in_budget
      ;;
    sspc_full_k25)
      export GEOPACT_SSPC_SKELETON=1
      export GEOPACT_SSPC_ZAFD=1
      export GEOPACT_SSPC_K_GRID=5
      export GEOPACT_SSPC_AXIS=linspace
      export GEOPACT_SSPC_BUDGET=additive
      ;;
    sspc_full_k49)
      export GEOPACT_SSPC_SKELETON=1
      export GEOPACT_SSPC_ZAFD=1
      export GEOPACT_SSPC_K_GRID=7
      export GEOPACT_SSPC_AXIS=linspace
      export GEOPACT_SSPC_BUDGET=additive
      ;;
    sspc_skel_k25)
      export GEOPACT_SSPC_SKELETON=1
      export GEOPACT_SSPC_ZAFD=0
      export GEOPACT_SSPC_K_GRID=5
      export GEOPACT_SSPC_AXIS=linspace
      export GEOPACT_SSPC_BUDGET=in_budget
      ;;
    sspc_rand_k25)
      export GEOPACT_SSPC_SKELETON=0
      export GEOPACT_SSPC_ZAFD=0
      export GEOPACT_SSPC_RANDOM_EXTRA=1
      export GEOPACT_SSPC_RANDOM_EXTRA_K=25
      export GEOPACT_SSPC_BUDGET=in_budget
      ;;
    sspc_rand_k49)
      export GEOPACT_SSPC_SKELETON=0
      export GEOPACT_SSPC_ZAFD=0
      export GEOPACT_SSPC_RANDOM_EXTRA=1
      export GEOPACT_SSPC_RANDOM_EXTRA_K=49
      export GEOPACT_SSPC_BUDGET=in_budget
      ;;
    sspc_full_g5_s02)
      export GEOPACT_SSPC_SKELETON=1
      export GEOPACT_SSPC_ZAFD=1
      export GEOPACT_SSPC_K_GRID=5
      export GEOPACT_SSPC_AXIS=linspace
      export GEOPACT_SSPC_SIGMA=0.2
      export GEOPACT_SSPC_BUDGET=in_budget
      ;;
    sspc_full_g5_s06)
      export GEOPACT_SSPC_SKELETON=1
      export GEOPACT_SSPC_ZAFD=1
      export GEOPACT_SSPC_K_GRID=5
      export GEOPACT_SSPC_AXIS=linspace
      export GEOPACT_SSPC_SIGMA=0.6
      export GEOPACT_SSPC_BUDGET=in_budget
      ;;
    sspc_full_g*)
      export GEOPACT_SSPC_SKELETON=1
      export GEOPACT_SSPC_ZAFD=1
      export GEOPACT_SSPC_K_GRID="${sspc_arm_arg#sspc_full_g}"
      export GEOPACT_SSPC_AXIS=linspace
      export GEOPACT_SSPC_BUDGET=in_budget
      ;;
    sspc_skel_g*)
      export GEOPACT_SSPC_SKELETON=1
      export GEOPACT_SSPC_ZAFD=0
      export GEOPACT_SSPC_K_GRID="${sspc_arm_arg#sspc_skel_g}"
      export GEOPACT_SSPC_AXIS=linspace
      export GEOPACT_SSPC_BUDGET=in_budget
      ;;
    sspc_rand_g*)
      export GEOPACT_SSPC_SKELETON=0
      export GEOPACT_SSPC_ZAFD=0
      export GEOPACT_SSPC_RANDOM_EXTRA=1
      export GEOPACT_SSPC_K_GRID="${sspc_arm_arg#sspc_rand_g}"
      export GEOPACT_SSPC_RANDOM_EXTRA_K=0
      export GEOPACT_SSPC_AXIS=linspace
      export GEOPACT_SSPC_BUDGET=in_budget
      ;;
    sspc_zafd_s02)
      export GEOPACT_SSPC_SKELETON=0
      export GEOPACT_SSPC_ZAFD=1
      export GEOPACT_SSPC_SIGMA=0.2
      export GEOPACT_SSPC_BUDGET=in_budget
      export GEOPACT_ZAFD_OMEGA_NORM=survivor
      ;;
    sspc_zafd_s06)
      export GEOPACT_SSPC_SKELETON=0
      export GEOPACT_SSPC_ZAFD=1
      export GEOPACT_SSPC_SIGMA=0.6
      export GEOPACT_SSPC_BUDGET=in_budget
      export GEOPACT_ZAFD_OMEGA_NORM=survivor
      ;;
    sspc_zafd_cons)
      export GEOPACT_SSPC_SKELETON=0
      export GEOPACT_SSPC_ZAFD=1
      export GEOPACT_SSPC_BUDGET=in_budget
      export GEOPACT_ZAFD_OMEGA_NORM=donor
      export GEOPACT_ZAFD_GAMMA=0
      ;;
    sspc_zafd_hybrid_g025)
      export GEOPACT_SSPC_SKELETON=0
      export GEOPACT_SSPC_ZAFD=1
      export GEOPACT_SSPC_BUDGET=in_budget
      export GEOPACT_ZAFD_OMEGA_NORM=donor
      export GEOPACT_ZAFD_GAMMA=0.25
      ;;
    sspc_zafd_hybrid_g05)
      export GEOPACT_SSPC_SKELETON=0
      export GEOPACT_SSPC_ZAFD=1
      export GEOPACT_SSPC_BUDGET=in_budget
      export GEOPACT_ZAFD_OMEGA_NORM=donor
      export GEOPACT_ZAFD_GAMMA=0.5
      ;;
    sspc_zafd_adaptive)
      export GEOPACT_SSPC_SKELETON=0
      export GEOPACT_SSPC_ZAFD=1
      export GEOPACT_SSPC_BUDGET=in_budget
      export GEOPACT_ZAFD_OMEGA_NORM=donor
      export GEOPACT_ZAFD_GAMMA=0.25
      export GEOPACT_ZAFD_ADAPTIVE_SIGMA=1
      export GEOPACT_ZAFD_SIGMA_BETA=1.0
      ;;
    sspc_ipcc)
      export GEOPACT_SSPC_SKELETON=0
      export GEOPACT_SSPC_ZAFD=1
      export GEOPACT_SSPC_BUDGET=in_budget
      export GEOPACT_ZAFD_OMEGA_NORM=donor
      export GEOPACT_ZAFD_GAMMA=0.25
      export GEOPACT_ZAFD_ADAPTIVE_SIGMA=1
      export GEOPACT_ZAFD_SIGMA_BETA=1.0
      ;;
  esac
  export GEOPACT_MODULE_A_WORKER_YOLO_DEVICE="${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-0}"
  export GEOPACT_SSPC_STRICT="${GEOPACT_SSPC_STRICT:-1}"
  local pol="oas_ickh_cdct_no_coord_yolo_v2"
  if [[ -n "${policy_suffix}" ]]; then
    pol="${pol}_${policy_suffix}"
  fi
  if [[ -n "${MODEL_ARGS:-}" ]] && [[ "${MODEL_ARGS}" == *visual_prune_policy=* ]]; then
    MODEL_ARGS="$(echo "${MODEL_ARGS}" | sed -E "s/visual_prune_policy=[^,]*/visual_prune_policy=${pol}/")"
  fi
  ARM_LABEL="${arm_label}"
}

_setup_module_c_prod_v2_arm() {
  local policy_suffix="$1"
  local arm_label="$2"
  local coord_format="$3"
  local circular_aux="${4:-1}"
  local topk="${5:-3}"
  local coord_policy_tag="${6:-coord_sdb}"
  _setup_module_a_v2_arm "${policy_suffix}" "${arm_label}" "${circular_aux}"
  export GEOPACT_CDCT_ALPHA=0.2
  export GEOPACT_COORD_EPHEMERAL=1
  unset GEOPACT_COORD_METADATA_ONLY
  export GEOPACT_COORD_MODE=text
  export GEOPACT_COORD_TOPK="${topk}"
  export GEOPACT_COORD_FORMAT="${coord_format}"
  export GEOPACT_COORD_SLOT=before_gen_tail
  local pol="oas_ickh_cdct_${coord_policy_tag}_yolo_v2"
  if [[ -n "${policy_suffix}" ]]; then
    pol="${pol}_${policy_suffix}"
  fi
  if [[ -n "${MODEL_ARGS:-}" ]] && [[ "${MODEL_ARGS}" == *visual_prune_policy=* ]]; then
    MODEL_ARGS="$(echo "${MODEL_ARGS}" | sed -E "s/visual_prune_policy=[^,]*/visual_prune_policy=${pol}/")"
  fi
  ARM_LABEL="${arm_label}"
}

_setup_module_a_v2_det_arm() {
  local arm_label="$1"
  local miu_detector="${2:-omniparser}"
  export GEOPACT_SESSION_KV=1
  export GEOPACT_SESSION_KV_MODE=dialogue
  export GEOPACT_SCREENSHOT_ONLY_CACHE=1
  unset GEOPACT_COORD_EPHEMERAL
  export GEOPACT_COORD_METADATA_ONLY=1
  # shellcheck source=geopact_module_a_cpu_prefetch_env.sh
  source "${_SCRIPT_DIR}/geopact_module_a_cpu_prefetch_env.sh"
  export GEOPACT_CDCT_ALPHA=0.1
  export GEOPACT_YOLO_CIRCULAR_AUX=1
  unset GEOPACT_YOLO_OCR_AUX
  unset GEOPACT_YOLO_STRIPE_AUX
  unset BLOB_AUX_USE_FULL_RES
  unset OMNIPARSER_ICON_CONF
  PRUNE_RATIO="0.40"
  local pol="oas_ickh_cdct_no_coord_yolo_v2"
  shift 2
  while [[ $# -gt 0 ]]; do
    case "$1" in
      *=*)
        export "$1"
        ;;
      *)
        echo "Unknown det arm token: $1" >&2
        exit 1
        ;;
    esac
    shift
  done
  MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=${pol},visual_prune_ratio=${PRUNE_RATIO},miu_detector=${miu_detector}}"
  ARM_LABEL="${arm_label}"
}

case "$EXP_ARM" in
  00_baseline)
    export GEOPACT_SESSION_KV=0
    unset GEOPACT_SESSION_KV_MODE
    unset GEOPACT_SCREENSHOT_ONLY_CACHE
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS}}"
    ARM_LABEL="baseline: no SESSION_KV, no vision cache, no pruning"
    ;;
  01_dialogue_kv_noprune)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS}}"
    ARM_LABEL="dialogue SESSION_KV, no vision cache, no pruning"
    ;;
  02_dialogue_kv_r99)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    PRUNE_RATIO="0.99"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS}${PRUNE_SUFFIX},visual_prune_ratio=${PRUNE_RATIO}}"
    ARM_LABEL="dialogue SESSION_KV + uniform keep 99% (no vision cache)"
    ;;
  03_dialogue_kv_r95)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    PRUNE_RATIO="0.95"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS}${PRUNE_SUFFIX},visual_prune_ratio=${PRUNE_RATIO}}"
    ARM_LABEL="dialogue SESSION_KV + uniform keep 95% (no vision cache)"
    ;;
  04_dialogue_kv_r40)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS}${PRUNE_SUFFIX},visual_prune_ratio=${PRUNE_RATIO}}"
    ARM_LABEL="dialogue SESSION_KV + uniform keep 40% (no vision cache)"
    ;;
  04a_dialogue_kv_r40_random)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=random,visual_prune_ratio=${PRUNE_RATIO}}"
    ARM_LABEL="dialogue SESSION_KV + random keep 40% (no vision cache)"
    ;;
  04b_dialogue_kv_r40_tok_div)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=token_ickh_diversity,visual_prune_ratio=${PRUNE_RATIO}}"
    ARM_LABEL="dialogue SESSION_KV + token ICKH diversity keep 40%"
    ;;
  04c_dialogue_kv_r40_tok_rel)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=token_ickh_relevance,visual_prune_ratio=${PRUNE_RATIO}}"
    ARM_LABEL="dialogue SESSION_KV + token ICKH relevance-only keep 40%"
    ;;
  04d_dialogue_kv_r40_tok_instr_topk)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=token_instr_topk,visual_prune_ratio=${PRUNE_RATIO}}"
    ARM_LABEL="dialogue SESSION_KV + token instruction top-K keep 40%"
    ;;
  04e_dialogue_kv_r40_tok_div_topk)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=token_div_topk,visual_prune_ratio=${PRUNE_RATIO}}"
    ARM_LABEL="dialogue SESSION_KV + token FPS diversity top-K keep 40%"
    ;;
  04f_dialogue_kv_r40_oas_ickh_yolo)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_coord_center_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="dialogue SESSION_KV + oas ICKH YOLO keep 40% (Module A on sticky path, AB-only)"
    ;;
  04g_dialogue_kv_r40_oas_ickh_nd_yolo)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    unset GEOPACT_COORD_EPHEMERAL
    export GEOPACT_COORD_METADATA_ONLY=1
    # shellcheck source=geopact_module_a_cpu_prefetch_env.sh
    source "${_SCRIPT_DIR}/geopact_module_a_cpu_prefetch_env.sh"
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_no_density_no_coord_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="A+B: Module A CPU prefetch YOLO + B selects MIU/control (whole-box ICKH, no backfill) r=40%"
    ;;
  04x_dialogue_kv_r40_miu_first_backfill_yolo)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    unset GEOPACT_COORD_EPHEMERAL
    export GEOPACT_COORD_METADATA_ONLY=1
    # shellcheck source=geopact_module_a_cpu_prefetch_env.sh
    source "${_SCRIPT_DIR}/geopact_module_a_cpu_prefetch_env.sh"
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_miu_first_token_backfill_no_density_no_coord_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="A+B: Module A CPU prefetch YOLO + B selects MIU/control first, token ICKH backfill r=40%"
    ;;
  04h_dialogue_kv_r40_oas_ickh_cdct_yolo)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_cdct_no_coord_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="dialogue SESSION_KV + CDCT token ICKH + YOLO rho field keep 40% (A+CDCT AB-only)"
    ;;
  04h_cpu_dialogue_kv_r40_oas_ickh_cdct_yolo)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    export MIU_YOLO_DEVICE=cpu
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_cdct_no_coord_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="dialogue SESSION_KV + CDCT token ICKH + YOLO CPU keep 40% (A+CDCT AB-only)"
    ;;
  04i_dialogue_kv_r40_tok_rel_yolo)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    export GEOPACT_MODULE_A_SIDELOAD=1
    export GEOPACT_MODULE_A_SIDELOAD_MODE=full
    export MIU_YOLO_DEVICE="${MIU_YOLO_DEVICE:-cpu}"
    export GEOPACT_MODULE_A_SIDELOAD_TIMING="${GEOPACT_MODULE_A_SIDELOAD_TIMING:-eval_end}"
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=token_ickh_relevance,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="dialogue SESSION_KV + token ICKH relevance + Module A sideload full (post-decode, pure B prune)"
    ;;
  04i_gpu_dialogue_kv_r40_tok_rel_yolo)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    export GEOPACT_MODULE_A_SIDELOAD=1
    export GEOPACT_MODULE_A_SIDELOAD_MODE=full
    export MIU_YOLO_DEVICE=0
    export GEOPACT_MODULE_A_SIDELOAD_TIMING="${GEOPACT_MODULE_A_SIDELOAD_TIMING:-eval_end}"
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=token_ickh_relevance,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="dialogue SESSION_KV + token ICKH relevance + Module A sideload full GPU eval_end (pure B prune)"
    ;;
  04i_min_dialogue_kv_r40_tok_rel_yolo)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    export GEOPACT_MODULE_A_SIDELOAD=1
    export GEOPACT_MODULE_A_SIDELOAD_MODE=yolo_only
    export MIU_YOLO_DEVICE="${MIU_YOLO_DEVICE:-cpu}"
    export GEOPACT_MODULE_A_SIDELOAD_TIMING="${GEOPACT_MODULE_A_SIDELOAD_TIMING:-eval_end}"
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=token_ickh_relevance,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="dialogue SESSION_KV + token ICKH relevance + Module A sideload YOLO-only (post-decode, pure B prune)"
    ;;
  04j_dialogue_kv_r40_tok_rel_yolo_export)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    export GEOPACT_MODULE_A_SIDELOAD=1
    export GEOPACT_MODULE_A_SIDELOAD_MODE=full
    export GEOPACT_MODULE_A_EXPORT_WEIGHTS=1
    export GEOPACT_MODULE_A_EXPORT_PATH="${OUT}/module_a_weights.jsonl"
    export MIU_YOLO_DEVICE="${MIU_YOLO_DEVICE:-cpu}"
    export GEOPACT_MODULE_A_SIDELOAD_TIMING="${GEOPACT_MODULE_A_SIDELOAD_TIMING:-eval_end}"
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=token_ickh_relevance,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="dialogue SESSION_KV + token ICKH relevance + Module A sideload full + CDCT weight export (pure B prune)"
    ;;
  04k_cpu_prefetch_cdct_a0_dialogue_kv_r40)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    unset GEOPACT_MODULE_A_SIDELOAD
    export GEOPACT_MODULE_A_PREFETCH=cpu
    export GEOPACT_CDCT_ALPHA=0
    export GEOPACT_MODULE_A_SUBPROCESS=1
    export GEOPACT_MODULE_A_SUBPROCESS_POOL="${GEOPACT_MODULE_A_SUBPROCESS_POOL:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PYTHON="${GEOPACT_MODULE_A_SUBPROCESS_PYTHON:-${EVAL_PYTHON}}"
    export GEOPACT_MODULE_A_PREFETCH_ASYNC="${GEOPACT_MODULE_A_PREFETCH_ASYNC:-1}"
    export GEOPACT_MODULE_A_WORKER_SHM="${GEOPACT_MODULE_A_WORKER_SHM:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PREWARM="${GEOPACT_MODULE_A_SUBPROCESS_PREWARM:-1}"
    # Worker GPU YOLO (bench: ~54% faster hot path vs cpu on same card)
    export GEOPACT_MODULE_A_WORKER_YOLO_DEVICE="${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-cpu}"
    export GEOPACT_MODULE_A_CACHE=0
    export MIU_YOLO_DEVICE=cpu
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_cdct_no_coord_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="dialogue SESSION_KV + subprocess Module A + CDCT alpha=0 (spatial prior computed, B=w_j=r_j)"
    ;;
  04l_cpu_prefetch_cdct_a1_dialogue_kv_r40)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    unset GEOPACT_MODULE_A_SIDELOAD
    export GEOPACT_MODULE_A_PREFETCH=cpu
    export GEOPACT_CDCT_ALPHA=1
    export GEOPACT_MODULE_A_SUBPROCESS=1
    export GEOPACT_MODULE_A_SUBPROCESS_POOL="${GEOPACT_MODULE_A_SUBPROCESS_POOL:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PYTHON="${GEOPACT_MODULE_A_SUBPROCESS_PYTHON:-${EVAL_PYTHON}}"
    export GEOPACT_MODULE_A_PREFETCH_ASYNC="${GEOPACT_MODULE_A_PREFETCH_ASYNC:-1}"
    export GEOPACT_MODULE_A_WORKER_SHM="${GEOPACT_MODULE_A_WORKER_SHM:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PREWARM="${GEOPACT_MODULE_A_SUBPROCESS_PREWARM:-1}"
    export GEOPACT_MODULE_A_WORKER_YOLO_DEVICE="${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-cpu}"
    export GEOPACT_MODULE_A_CACHE=0
    export MIU_YOLO_DEVICE=cpu
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_cdct_no_coord_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="dialogue SESSION_KV + CPU prefetch Module A + CDCT alpha=1 (spatial prior active)"
    ;;
  04m_cpu_prefetch_cdct_a1_dialogue_kv_r40_fullres_yolo)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    unset GEOPACT_MODULE_A_SIDELOAD
    export GEOPACT_MODULE_A_PREFETCH=cpu
    export GEOPACT_CDCT_ALPHA=1
    export GEOPACT_MODULE_A_SUBPROCESS=1
    export GEOPACT_MODULE_A_SUBPROCESS_POOL="${GEOPACT_MODULE_A_SUBPROCESS_POOL:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PYTHON="${GEOPACT_MODULE_A_SUBPROCESS_PYTHON:-${EVAL_PYTHON}}"
    export GEOPACT_MODULE_A_PREFETCH_ASYNC="${GEOPACT_MODULE_A_PREFETCH_ASYNC:-1}"
    export GEOPACT_MODULE_A_WORKER_SHM="${GEOPACT_MODULE_A_WORKER_SHM:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PREWARM="${GEOPACT_MODULE_A_SUBPROCESS_PREWARM:-1}"
    export GEOPACT_MODULE_A_WORKER_YOLO_DEVICE="${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-cpu}"
    export GEOPACT_MODULE_A_CACHE=0
    export MIU_YOLO_DEVICE=cpu
    # 0 = no longest-edge downscale before YOLO (full input resolution)
    export OMNIPARSER_ICON_MAX_EDGE=0
    export MIU_YOLO_MAX_EDGE=0
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_cdct_no_coord_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="dialogue SESSION_KV + subprocess Module A + CDCT alpha=1 + YOLO full-res (no max_edge)"
    ;;
  04n_cpu_prefetch_cdct_a01_maxedge1536_dialogue_kv_r40)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    unset GEOPACT_COORD_EPHEMERAL
    export GEOPACT_COORD_METADATA_ONLY=1
    # shellcheck source=geopact_module_a_cpu_prefetch_env.sh
    source "${_SCRIPT_DIR}/geopact_module_a_cpu_prefetch_env.sh"
    export GEOPACT_CDCT_ALPHA=0.1
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_cdct_no_coord_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="A+B: Module A CPU prefetch YOLO + B selects token (CDCT alpha=0.1 soft prior) r=40%"
    ;;
  04p_cpu_prefetch_cdct_a01_fullres_dialogue_kv_r40)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    unset GEOPACT_MODULE_A_SIDELOAD
    export GEOPACT_MODULE_A_PREFETCH=cpu
    export GEOPACT_CDCT_ALPHA=0.1
    export GEOPACT_MODULE_A_SUBPROCESS=1
    export GEOPACT_MODULE_A_SUBPROCESS_POOL="${GEOPACT_MODULE_A_SUBPROCESS_POOL:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PYTHON="${GEOPACT_MODULE_A_SUBPROCESS_PYTHON:-${EVAL_PYTHON}}"
    export GEOPACT_MODULE_A_PREFETCH_ASYNC="${GEOPACT_MODULE_A_PREFETCH_ASYNC:-1}"
    export GEOPACT_MODULE_A_WORKER_SHM="${GEOPACT_MODULE_A_WORKER_SHM:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PREWARM="${GEOPACT_MODULE_A_SUBPROCESS_PREWARM:-1}"
    export GEOPACT_MODULE_A_WORKER_YOLO_DEVICE="${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-cpu}"
    export GEOPACT_MODULE_A_CACHE=0
    export MIU_YOLO_DEVICE=cpu
    export OMNIPARSER_ICON_MAX_EDGE=0
    export MIU_YOLO_MAX_EDGE=0
    unset GEOPACT_COORD_EPHEMERAL
    export GEOPACT_COORD_METADATA_ONLY=1
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_cdct_no_coord_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="dialogue SESSION_KV + subprocess Module A + CDCT alpha=0.1 + YOLO full-res (A+B only)"
    ;;
  04o_cpu_prefetch_cdct_a01_coord_ephemeral_fullres_r40)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    unset GEOPACT_MODULE_A_SIDELOAD
    export GEOPACT_MODULE_A_PREFETCH=cpu
    export GEOPACT_CDCT_ALPHA=0.1
    export GEOPACT_MODULE_A_SUBPROCESS=1
    export GEOPACT_MODULE_A_SUBPROCESS_POOL="${GEOPACT_MODULE_A_SUBPROCESS_POOL:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PYTHON="${GEOPACT_MODULE_A_SUBPROCESS_PYTHON:-${EVAL_PYTHON}}"
    export GEOPACT_MODULE_A_PREFETCH_ASYNC="${GEOPACT_MODULE_A_PREFETCH_ASYNC:-1}"
    export GEOPACT_MODULE_A_WORKER_SHM="${GEOPACT_MODULE_A_WORKER_SHM:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PREWARM="${GEOPACT_MODULE_A_SUBPROCESS_PREWARM:-1}"
    export GEOPACT_MODULE_A_WORKER_YOLO_DEVICE="${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-cpu}"
    export GEOPACT_MODULE_A_CACHE=0
    export MIU_YOLO_DEVICE=cpu
    export OMNIPARSER_ICON_MAX_EDGE=0
    export MIU_YOLO_MAX_EDGE=0
    export GEOPACT_COORD_EPHEMERAL=1
    export GEOPACT_COORD_METADATA_ONLY=0
    export GEOPACT_COORD_TOPK=5
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_cdct_coord_ephemeral_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="dialogue SESSION_KV + subprocess Module A + CDCT alpha=0.1 + ephemeral Module C top5 + YOLO full-res"
    ;;
  04o_cpu_prefetch_cdct_a01_coord_ephemeral_maxedge1536_r40)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    unset GEOPACT_MODULE_A_SIDELOAD
    export GEOPACT_MODULE_A_PREFETCH=cpu
    export GEOPACT_CDCT_ALPHA=0.1
    export GEOPACT_MODULE_A_SUBPROCESS=1
    export GEOPACT_MODULE_A_SUBPROCESS_POOL="${GEOPACT_MODULE_A_SUBPROCESS_POOL:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PYTHON="${GEOPACT_MODULE_A_SUBPROCESS_PYTHON:-${EVAL_PYTHON}}"
    export GEOPACT_MODULE_A_PREFETCH_ASYNC="${GEOPACT_MODULE_A_PREFETCH_ASYNC:-1}"
    export GEOPACT_MODULE_A_WORKER_SHM="${GEOPACT_MODULE_A_WORKER_SHM:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PREWARM="${GEOPACT_MODULE_A_SUBPROCESS_PREWARM:-1}"
    export GEOPACT_MODULE_A_WORKER_YOLO_DEVICE="${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-cpu}"
    export GEOPACT_MODULE_A_CACHE=0
    export MIU_YOLO_DEVICE=cpu
    export OMNIPARSER_ICON_MAX_EDGE=1536
    export MIU_YOLO_MAX_EDGE=1536
    export GEOPACT_COORD_EPHEMERAL=1
    export GEOPACT_COORD_METADATA_ONLY=0
    export GEOPACT_COORD_TOPK=5
    export GEOPACT_COORD_FORMAT=legacy
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_cdct_coord_ephemeral_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="dialogue SESSION_KV + subprocess Module A + CDCT alpha=0.1 + ephemeral Module C top5 + YOLO max_edge=1536"
    ;;
  04q_cpu_prefetch_cdct_a01_coord_sdb_maxedge1536_r40)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    unset GEOPACT_MODULE_A_SIDELOAD
    export GEOPACT_MODULE_A_PREFETCH=cpu
    export GEOPACT_CDCT_ALPHA=0.1
    export GEOPACT_MODULE_A_SUBPROCESS=1
    export GEOPACT_MODULE_A_SUBPROCESS_POOL="${GEOPACT_MODULE_A_SUBPROCESS_POOL:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PYTHON="${GEOPACT_MODULE_A_SUBPROCESS_PYTHON:-${EVAL_PYTHON}}"
    export GEOPACT_MODULE_A_PREFETCH_ASYNC="${GEOPACT_MODULE_A_PREFETCH_ASYNC:-1}"
    export GEOPACT_MODULE_A_WORKER_SHM="${GEOPACT_MODULE_A_WORKER_SHM:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PREWARM="${GEOPACT_MODULE_A_SUBPROCESS_PREWARM:-1}"
    export GEOPACT_MODULE_A_WORKER_YOLO_DEVICE="${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-cpu}"
    export GEOPACT_MODULE_A_CACHE=0
    export MIU_YOLO_DEVICE=cpu
    export OMNIPARSER_ICON_MAX_EDGE=1536
    export MIU_YOLO_MAX_EDGE=1536
    export GEOPACT_COORD_EPHEMERAL=1
    export GEOPACT_COORD_METADATA_ONLY=0
    export GEOPACT_COORD_TOPK=3
    export GEOPACT_COORD_FORMAT=sdb
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_cdct_coord_sdb_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="dialogue SESSION_KV + CDCT alpha=0.1 + SDB Module C (deprecated) top3 + max_edge=1536"
    ;;
  04r_cpu_prefetch_cdct_a01_coord_spatial_plain_maxedge1536_r40)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    unset GEOPACT_MODULE_A_SIDELOAD
    export GEOPACT_MODULE_A_PREFETCH=cpu
    export GEOPACT_CDCT_ALPHA=0.1
    export GEOPACT_MODULE_A_SUBPROCESS=1
    export GEOPACT_MODULE_A_SUBPROCESS_POOL="${GEOPACT_MODULE_A_SUBPROCESS_POOL:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PYTHON="${GEOPACT_MODULE_A_SUBPROCESS_PYTHON:-${EVAL_PYTHON}}"
    export GEOPACT_MODULE_A_PREFETCH_ASYNC="${GEOPACT_MODULE_A_PREFETCH_ASYNC:-1}"
    export GEOPACT_MODULE_A_WORKER_SHM="${GEOPACT_MODULE_A_WORKER_SHM:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PREWARM="${GEOPACT_MODULE_A_SUBPROCESS_PREWARM:-1}"
    export GEOPACT_MODULE_A_WORKER_YOLO_DEVICE="${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-cpu}"
    export GEOPACT_MODULE_A_CACHE=0
    export MIU_YOLO_DEVICE=cpu
    export OMNIPARSER_ICON_MAX_EDGE=1536
    export MIU_YOLO_MAX_EDGE=1536
    export GEOPACT_COORD_EPHEMERAL=1
    export GEOPACT_COORD_METADATA_ONLY=0
    export GEOPACT_COORD_TOPK=3
    export GEOPACT_COORD_FORMAT=spatial_plain
    export GEOPACT_COORD_SLOT=before_gen_tail
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_cdct_coord_sdb_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="dialogue SESSION_KV + CDCT alpha=0.1 + UDSI spatial_plain Module C top3 + max_edge=1536"
    ;;
  04s_cpu_prefetch_cdct_a01_coord_spatial_ranked_maxedge1536_r40)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    unset GEOPACT_MODULE_A_SIDELOAD
    export GEOPACT_MODULE_A_PREFETCH=cpu
    export GEOPACT_CDCT_ALPHA=0.1
    export GEOPACT_MODULE_A_SUBPROCESS=1
    export GEOPACT_MODULE_A_SUBPROCESS_POOL="${GEOPACT_MODULE_A_SUBPROCESS_POOL:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PYTHON="${GEOPACT_MODULE_A_SUBPROCESS_PYTHON:-${EVAL_PYTHON}}"
    export GEOPACT_MODULE_A_PREFETCH_ASYNC="${GEOPACT_MODULE_A_PREFETCH_ASYNC:-1}"
    export GEOPACT_MODULE_A_WORKER_SHM="${GEOPACT_MODULE_A_WORKER_SHM:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PREWARM="${GEOPACT_MODULE_A_SUBPROCESS_PREWARM:-1}"
    export GEOPACT_MODULE_A_WORKER_YOLO_DEVICE="${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-cpu}"
    export GEOPACT_MODULE_A_CACHE=0
    export MIU_YOLO_DEVICE=cpu
    export OMNIPARSER_ICON_MAX_EDGE=1536
    export MIU_YOLO_MAX_EDGE=1536
    export GEOPACT_COORD_EPHEMERAL=1
    export GEOPACT_COORD_METADATA_ONLY=0
    export GEOPACT_COORD_TOPK=3
    export GEOPACT_COORD_FORMAT=spatial_ranked
    export GEOPACT_COORD_SLOT=before_gen_tail
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_cdct_coord_sdb_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="dialogue SESSION_KV + CDCT alpha=0.1 + UDSI spatial_ranked Module C top3 + max_edge=1536"
    ;;
  04t_cpu_prefetch_cdct_a01_coord_sidecar_bbox_maxedge1536_r40)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    unset GEOPACT_MODULE_A_SIDELOAD
    export GEOPACT_MODULE_A_PREFETCH=cpu
    export GEOPACT_CDCT_ALPHA=0.1
    export GEOPACT_MODULE_A_SUBPROCESS=1
    export GEOPACT_MODULE_A_SUBPROCESS_POOL="${GEOPACT_MODULE_A_SUBPROCESS_POOL:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PYTHON="${GEOPACT_MODULE_A_SUBPROCESS_PYTHON:-${EVAL_PYTHON}}"
    export GEOPACT_MODULE_A_PREFETCH_ASYNC="${GEOPACT_MODULE_A_PREFETCH_ASYNC:-1}"
    export GEOPACT_MODULE_A_WORKER_SHM="${GEOPACT_MODULE_A_WORKER_SHM:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PREWARM="${GEOPACT_MODULE_A_SUBPROCESS_PREWARM:-1}"
    export GEOPACT_MODULE_A_WORKER_YOLO_DEVICE="${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-cpu}"
    export GEOPACT_MODULE_A_CACHE=0
    export MIU_YOLO_DEVICE=cpu
    export OMNIPARSER_ICON_MAX_EDGE=1536
    export MIU_YOLO_MAX_EDGE=1536
    export GEOPACT_COORD_EPHEMERAL=1
    export GEOPACT_COORD_METADATA_ONLY=0
    export GEOPACT_COORD_TOPK=3
    export GEOPACT_COORD_FORMAT=sidecar_bbox
    export GEOPACT_COORD_SLOT=before_gen_tail
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_cdct_coord_sdb_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="GATS L0 sidecar_bbox only + max_edge=1536"
    ;;
  04u_cpu_prefetch_cdct_a01_coord_sidecar_bbox_role_maxedge1536_r40)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    unset GEOPACT_MODULE_A_SIDELOAD
    export GEOPACT_MODULE_A_PREFETCH=cpu
    export GEOPACT_CDCT_ALPHA=0.1
    export GEOPACT_MODULE_A_SUBPROCESS=1
    export GEOPACT_MODULE_A_SUBPROCESS_POOL="${GEOPACT_MODULE_A_SUBPROCESS_POOL:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PYTHON="${GEOPACT_MODULE_A_SUBPROCESS_PYTHON:-${EVAL_PYTHON}}"
    export GEOPACT_MODULE_A_PREFETCH_ASYNC="${GEOPACT_MODULE_A_PREFETCH_ASYNC:-1}"
    export GEOPACT_MODULE_A_WORKER_SHM="${GEOPACT_MODULE_A_WORKER_SHM:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PREWARM="${GEOPACT_MODULE_A_SUBPROCESS_PREWARM:-1}"
    export GEOPACT_MODULE_A_WORKER_YOLO_DEVICE="${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-cpu}"
    export GEOPACT_MODULE_A_CACHE=0
    export MIU_YOLO_DEVICE=cpu
    export OMNIPARSER_ICON_MAX_EDGE=1536
    export MIU_YOLO_MAX_EDGE=1536
    export GEOPACT_COORD_EPHEMERAL=1
    export GEOPACT_COORD_METADATA_ONLY=0
    export GEOPACT_COORD_TOPK=3
    export GEOPACT_COORD_FORMAT=sidecar_bbox_role
    export GEOPACT_COORD_SLOT=before_gen_tail
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_cdct_coord_sdb_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="GATS L1 sidecar_bbox+role + max_edge=1536"
    ;;
  04v_cpu_prefetch_cdct_a01_coord_sidecar_bbox_role_text_maxedge1536_r40)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    unset GEOPACT_MODULE_A_SIDELOAD
    export GEOPACT_MODULE_A_PREFETCH=cpu
    export GEOPACT_CDCT_ALPHA=0.1
    export GEOPACT_MODULE_A_SUBPROCESS=1
    export GEOPACT_MODULE_A_SUBPROCESS_POOL="${GEOPACT_MODULE_A_SUBPROCESS_POOL:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PYTHON="${GEOPACT_MODULE_A_SUBPROCESS_PYTHON:-${EVAL_PYTHON}}"
    export GEOPACT_MODULE_A_PREFETCH_ASYNC="${GEOPACT_MODULE_A_PREFETCH_ASYNC:-1}"
    export GEOPACT_MODULE_A_WORKER_SHM="${GEOPACT_MODULE_A_WORKER_SHM:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PREWARM="${GEOPACT_MODULE_A_SUBPROCESS_PREWARM:-1}"
    export GEOPACT_MODULE_A_WORKER_YOLO_DEVICE="${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-cpu}"
    export GEOPACT_MODULE_A_CACHE=0
    export MIU_YOLO_DEVICE=cpu
    export OMNIPARSER_ICON_MAX_EDGE=1536
    export MIU_YOLO_MAX_EDGE=1536
    export GEOPACT_COORD_EPHEMERAL=1
    export GEOPACT_COORD_METADATA_ONLY=0
    export GEOPACT_COORD_TOPK=3
    export GEOPACT_COORD_FORMAT=sidecar_bbox_role_text
    export GEOPACT_COORD_SLOT=before_gen_tail
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_cdct_coord_sdb_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="GATS L2 sidecar_bbox+role+text + max_edge=1536"
    ;;
  04w_cpu_prefetch_cdct_a01_coord_sidecar_keep_maxedge1536_r40)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    unset GEOPACT_MODULE_A_SIDELOAD
    export GEOPACT_MODULE_A_PREFETCH=cpu
    export GEOPACT_CDCT_ALPHA=0.1
    export GEOPACT_MODULE_A_SUBPROCESS=1
    export GEOPACT_MODULE_A_SUBPROCESS_POOL="${GEOPACT_MODULE_A_SUBPROCESS_POOL:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PYTHON="${GEOPACT_MODULE_A_SUBPROCESS_PYTHON:-${EVAL_PYTHON}}"
    export GEOPACT_MODULE_A_PREFETCH_ASYNC="${GEOPACT_MODULE_A_PREFETCH_ASYNC:-1}"
    export GEOPACT_MODULE_A_WORKER_SHM="${GEOPACT_MODULE_A_WORKER_SHM:-1}"
    export GEOPACT_MODULE_A_SUBPROCESS_PREWARM="${GEOPACT_MODULE_A_SUBPROCESS_PREWARM:-1}"
    export GEOPACT_MODULE_A_WORKER_YOLO_DEVICE="${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-cpu}"
    export GEOPACT_MODULE_A_CACHE=0
    export MIU_YOLO_DEVICE=cpu
    export OMNIPARSER_ICON_MAX_EDGE=1536
    export MIU_YOLO_MAX_EDGE=1536
    export GEOPACT_COORD_EPHEMERAL=1
    export GEOPACT_COORD_METADATA_ONLY=0
    export GEOPACT_COORD_TOPK=3
    export GEOPACT_COORD_FORMAT=sidecar_keep
    export GEOPACT_COORD_SLOT=before_gen_tail
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_cdct_coord_sdb_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="GATS L3 sidecar_keep full + max_edge=1536"
    ;;
  05_dialogue_kv_r20)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    PRUNE_RATIO="0.20"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS}${PRUNE_SUFFIX},visual_prune_ratio=${PRUNE_RATIO}}"
    ARM_LABEL="dialogue SESSION_KV + uniform keep 20% (no vision cache)"
    ;;
  06_dialogue_kv_r10)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    PRUNE_RATIO="0.10"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS}${PRUNE_SUFFIX},visual_prune_ratio=${PRUNE_RATIO}}"
    ARM_LABEL="dialogue SESSION_KV + uniform keep 10% (no vision cache)"
    ;;
  07_dialogue_kv_r05)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    PRUNE_RATIO="0.05"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS}${PRUNE_SUFFIX},visual_prune_ratio=${PRUNE_RATIO}}"
    ARM_LABEL="dialogue SESSION_KV + uniform keep 5% (no vision cache)"
    ;;
  08_dialogue_kv_r80)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    PRUNE_RATIO="0.80"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS}${PRUNE_SUFFIX},visual_prune_ratio=${PRUNE_RATIO}}"
    ARM_LABEL="dialogue SESSION_KV + uniform keep 80% (no vision cache)"
    ;;
  # Legacy arm IDs (old 5-arm layout); kept for resume on in-flight runs.
  03_dialogue_kv_r40)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS}${PRUNE_SUFFIX},visual_prune_ratio=${PRUNE_RATIO}}"
    ARM_LABEL="dialogue SESSION_KV + uniform keep 40% (legacy arm id 03)"
    ;;
  04_dialogue_kv_r20)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    PRUNE_RATIO="0.20"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS}${PRUNE_SUFFIX},visual_prune_ratio=${PRUNE_RATIO}}"
    ARM_LABEL="dialogue SESSION_KV + uniform keep 20% (legacy arm id 04)"
    ;;
  06_cdct_no_instr)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    unset GEOPACT_COORD_EPHEMERAL
    export GEOPACT_COORD_METADATA_ONLY=1
    # shellcheck source=geopact_module_a_cpu_prefetch_env.sh
    source "${_SCRIPT_DIR}/geopact_module_a_cpu_prefetch_env.sh"
    export GEOPACT_CDCT_ALPHA=0.1
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_cdct_no_coord_yolo_no_instr_rel,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="CDCT + YOLO, no instruction text relevance (no_instr_rel)"
    ;;
  06_cdct_alpha02)
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    unset GEOPACT_COORD_EPHEMERAL
    export GEOPACT_COORD_METADATA_ONLY=1
    # shellcheck source=geopact_module_a_cpu_prefetch_env.sh
    source "${_SCRIPT_DIR}/geopact_module_a_cpu_prefetch_env.sh"
    export GEOPACT_CDCT_ALPHA=0.2
    PRUNE_RATIO="0.40"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=oas_ickh_cdct_no_coord_yolo,visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="CDCT alpha=0.2 + CPU prefetch YOLO"
    ;;
  07_prod_drop_scale_no_circular)
    _setup_module_a_v2_arm "drop_pent_scale" "Factorial: v2 drop scale, circular OFF, alpha=0.1" 0
    ;;
  07_prod_drop_scale_no_circular_a02)
    _setup_module_a_v2_arm "drop_pent_scale" "Factorial: v2 drop scale, circular OFF, alpha=0.2" 0
    export GEOPACT_CDCT_ALPHA=0.2
    ;;
  07_prod_drop_scale_a02)
    _setup_module_a_v2_arm "drop_pent_scale" "Factorial: v2 drop scale, circular ON, alpha=0.2" 1
    export GEOPACT_CDCT_ALPHA=0.2
    ;;
  08_prod_drop_scale_a02_coord_plain)
    _setup_module_c_prod_v2_arm "drop_pent_scale" "Prod A+B+C: v2 drop scale alpha=0.2 + UDSI spatial_plain top3" "spatial_plain" 1 3
    ;;
  08_prod_drop_scale_a02_coord_ranked)
    _setup_module_c_prod_v2_arm "drop_pent_scale" "Prod A+B+C: v2 drop scale alpha=0.2 + UDSI spatial_ranked top3" "spatial_ranked" 1 3
    ;;
  08_prod_drop_scale_a02_coord_sidecar_keep)
    _setup_module_c_prod_v2_arm "drop_pent_scale" "Prod A+B+C: v2 drop scale alpha=0.2 + GATS sidecar_keep top3" "sidecar_keep" 1 3
    ;;
  08_prod_drop_scale_a02_coord_shuffle)
    _setup_module_c_prod_v2_arm "drop_pent_scale" "Prod A+B+C: v2 drop scale + coord_shuffle control" "spatial_plain" 1 3 "coord_shuffle"
    ;;
  09_prod_drop_scale_a02_rpb)
    _setup_module_c_geom_v2_arm "drop_pent_scale" "Prod A+B+ES-RPB (flash-safe embedding-space)" "rpb" 1
    ;;
  09_prod_drop_scale_a02_rpb_residue)
    _setup_module_c_geom_v2_arm "drop_pent_scale" "Prod A+B+ES-RPB+CDCT residue" "rpb_residue" 1
    ;;
  09_prod_drop_scale_a02_rpb_shuffle)
    _setup_module_c_geom_v2_arm "drop_pent_scale" "Prod A+B+ES-RPB shuffle control" "rpb_shuffle" 1
    ;;
  09_prod_drop_scale_a02_gaec)
    _setup_module_c_geom_v2_arm "drop_pent_scale" "Prod A+B+GAEC vision-MRoPE carriers" "gaec" 1
    ;;
  09_prod_drop_scale_a02_gaec_shuffle)
    _setup_module_c_geom_v2_arm "drop_pent_scale" "Prod A+B+GAEC shuffle control" "gaec_shuffle" 1
    ;;
  09_prod_drop_scale_a02_rpb_iou)
    _setup_module_c_geom_v2_arm "drop_pent_scale" "Prod A+B+G2-RPB MIU-IoU ES-RPB" "rpb_iou" 1
    ;;
  09_prod_drop_scale_a02_rpb_iou_residue)
    _setup_module_c_geom_v2_arm "drop_pent_scale" "Prod A+B+G2-RPB IoU+CDCT residue" "rpb_iou_residue" 1
    ;;
  09_prod_drop_scale_a02_rpb_iou_shuffle)
    _setup_module_c_geom_v2_arm "drop_pent_scale" "Prod A+B+G2-RPB IoU shuffle neg" "rpb_iou_shuffle" 1
    ;;
  09_prod_drop_scale_a02_gaec_v2)
    _setup_module_c_geom_v2_arm "drop_pent_scale" "Prod A+B+GAEC v2 origpos anchor" "gaec_v2" 1
    ;;
  09_prod_drop_scale_a02_abc_iou)
    _setup_module_c_geom_v2_arm "drop_pent_scale" "Prod A+B+C G2-RPB IoU+GAEC v2" "abc_iou" 1
    ;;
  10_prod_drop_scale_a02_parb_eager_vv)
    _setup_module_c_parb_arm "drop_pent_scale" "Prod A+B+PARB eager VV only" "parb_eager_vv" 1
    ;;
  10_prod_drop_scale_a02_parb_eager_tv)
    _setup_module_c_parb_arm "drop_pent_scale" "Prod A+B+PARB eager TV only" "parb_eager_tv" 1
    ;;
  10_prod_drop_scale_a02_parb_eager_full)
    _setup_module_c_parb_arm "drop_pent_scale" "Prod A+B+PARB eager VV+TV" "parb_eager_full" 1
    ;;
  10_prod_drop_scale_a02_parb_hybrid_full)
    _setup_module_c_parb_arm "drop_pent_scale" "Prod A+B+PARB hybrid VV+TV" "parb_hybrid_full" 1
    ;;
  10_prod_drop_scale_a02_parb_hybrid_vv)
    _setup_module_c_parb_arm "drop_pent_scale" "Prod A+B+PARB hybrid VV only" "parb_hybrid_vv" 1
    ;;
  10_prod_drop_scale_a02_parb_hybrid_tv)
    _setup_module_c_parb_arm "drop_pent_scale" "Prod A+B+PARB hybrid TV only" "parb_hybrid_tv" 1
    ;;
  10_prod_drop_scale_a02_parb_hybrid_nodensity)
    _setup_module_c_parb_arm "drop_pent_scale" "Prod A+B+PARB hybrid no density" "parb_hybrid_nodensity" 1
    ;;
  10_prod_drop_scale_a02_parb_hybrid_no_phi)
    _setup_module_c_parb_arm "drop_pent_scale" "Prod A+B+PARB hybrid no IoU phi" "parb_hybrid_no_phi" 1
    ;;
  11_prod_drop_scale_a02_parb_lift_decomposed_full)
    _setup_module_c_parb_lift_arm "drop_pent_scale" "Prod A+B+PARB-Lift decomposed VV+TV (FA)" "lift_decomposed_full" 1
    ;;
  11_prod_drop_scale_a02_parb_lift_decomposed_tv)
    _setup_module_c_parb_lift_arm "drop_pent_scale" "Prod A+B+PARB-Lift decomposed TV-only (FA)" "lift_decomposed_tv" 1
    ;;
  11_prod_drop_scale_a02_parb_lift_svd_full)
    _setup_module_c_parb_lift_arm "drop_pent_scale" "Prod A+B+PARB-Lift SVD full (FA)" "lift_svd_full" 1
    ;;
  12_prod_drop_scale_a02_trp_hybrid)
    _setup_module_c_trp_arm "drop_pent_scale" "Prod A+B+TRP hybrid (TV logit oracle)" "trp_hybrid" 1
    ;;
  12_prod_drop_scale_a02_trp_lift)
    _setup_module_c_trp_arm "drop_pent_scale" "Prod A+B+TRP lift (FA-native TV)" "trp_lift" 1
    ;;
  12_prod_drop_scale_a02_trp_lift_no_psi)
    _setup_module_c_trp_arm "drop_pent_scale" "Prod A+B+TRP lift no psi ablation" "trp_lift_no_psi" 1
    ;;
  12_prod_drop_scale_a02_trp_lift_no_region)
    _setup_module_c_trp_arm "drop_pent_scale" "Prod A+B+TRP lift no region ablation" "trp_lift_no_region" 1
    ;;
  12_prod_drop_scale_a02_trp_lift_l0)
    _setup_module_c_trp_arm "drop_pent_scale" "Prod A+B+TRP lift layer0 only" "trp_lift_l0" 1
    ;;
  13_prod_drop_scale_a02_gsc_fa)
    _setup_module_c_gsc_arm "drop_pent_scale" "Prod A+B+GSC FA ICC+LPR (G4 v2)" "gsc_fa" 1
    ;;
  13_prod_drop_scale_a02_gsc_fa_lpr_only)
    _setup_module_c_gsc_arm "drop_pent_scale" "Prod A+B+GSC FA LPR only (G3 v2)" "gsc_fa_lpr_only" 1
    ;;
  13_prod_drop_scale_a02_gsc_fa_schedule)
    _setup_module_c_gsc_arm "drop_pent_scale" "Prod A+B+GSC FA+schedule (G5 v2)" "gsc_fa_schedule" 1
    ;;
  13_prod_drop_scale_a02_gsc_fa_query_adapt)
    _setup_module_c_gsc_arm "drop_pent_scale" "Prod A+B+GSC FA query adapt (G6 v2)" "gsc_fa_query_adapt" 1
    ;;
  14_prod_drop_scale_a02_dmtc_lift_lpr)
    _setup_module_c_correction_arm "drop_pent_scale" "Prod A+B+DMTC lift+LPR (FA-native)" "dmtc_lift_lpr" 1
    ;;
  14_prod_drop_scale_a02_dmtc_lift_no_lpr)
    _setup_module_c_correction_arm "drop_pent_scale" "Prod A+B+DMTC lift no LPR" "dmtc_lift_no_lpr" 1
    ;;
  14_prod_drop_scale_a02_dmtc_lift_no_tv)
    _setup_module_c_correction_arm "drop_pent_scale" "Prod A+B+DMTC LPR only" "dmtc_lift_no_tv" 1
    ;;
  14_prod_drop_scale_a02_dmtc_lift_schedule)
    _setup_module_c_correction_arm "drop_pent_scale" "Prod A+B+DMTC lift+LPR schedule" "dmtc_lift_schedule" 1
    ;;
  14_prod_drop_scale_a02_dmtc_lift_no_region)
    _setup_module_c_correction_arm "drop_pent_scale" "Prod A+B+DMTC no region" "dmtc_lift_no_region" 1
    ;;
  14_prod_drop_scale_a02_dmtc_lift_no_psi)
    _setup_module_c_correction_arm "drop_pent_scale" "Prod A+B+DMTC no psi" "dmtc_lift_no_psi" 1
    ;;
  15_prod_drop_scale_a02_vsas_internal_9)
    _setup_module_c_vsas_arm "drop_pent_scale" "Prod A+B+VSAS internal K=9" "vsas_internal_9" 1
    ;;
  15_prod_drop_scale_a02_vsas_internal_4)
    _setup_module_c_vsas_arm "drop_pent_scale" "Prod A+B+VSAS internal K=4" "vsas_internal_4" 1
    ;;
  15_prod_drop_scale_a02_vsas_internal_l0)
    _setup_module_c_vsas_arm "drop_pent_scale" "Prod A+B+VSAS internal layer0" "vsas_internal_l0" 1
    ;;
  15_prod_drop_scale_a02_vsas_internal_denom)
    _setup_module_c_vsas_arm "drop_pent_scale" "Prod A+B+VSAS internal denom-only" "vsas_internal_denom" 1
    ;;
  15_prod_drop_scale_a02_vsas_seq_9)
    _setup_module_c_vsas_arm "drop_pent_scale" "Prod A+B+VSAS seq splice K=9" "vsas_seq_9" 1
    ;;
  15_prod_drop_scale_a02_vsas_seq_4)
    _setup_module_c_vsas_arm "drop_pent_scale" "Prod A+B+VSAS seq splice K=4" "vsas_seq_4" 1
    ;;
  16_prod_drop_scale_a02_sspc_full)
    _setup_module_sspc_arm "drop_pent_scale" "Prod A+B+SSPC skeleton+ZAFD" "sspc_full" 1
    ;;
  16_prod_drop_scale_a02_sspc_zafd)
    _setup_module_sspc_arm "drop_pent_scale" "Prod A+B+SSPC ZAFD only" "sspc_zafd" 1
    ;;
  16_prod_drop_scale_a02_sspc_skel)
    _setup_module_sspc_arm "drop_pent_scale" "Prod A+B+SSPC skeleton only" "sspc_skel" 1
    ;;
  16_prod_drop_scale_a02_sspc_random_extra)
    _setup_module_sspc_arm "drop_pent_scale" "Prod A+B+SSPC random extra K=9" "sspc_random_extra" 1
    ;;
  17_prod_drop_scale_a02_sspc_full_k25)
    _setup_module_sspc_arm "drop_pent_scale" "Prod A+B+SSPC v2 full k25+ZAFD" "sspc_full_k25" 1
    ;;
  17_prod_drop_scale_a02_sspc_full_k49)
    _setup_module_sspc_arm "drop_pent_scale" "Prod A+B+SSPC v2 full k49+ZAFD" "sspc_full_k49" 1
    ;;
  17_prod_drop_scale_a02_sspc_skel_k25)
    _setup_module_sspc_arm "drop_pent_scale" "Prod A+B+SSPC v2 skeleton k25 only" "sspc_skel_k25" 1
    ;;
  17_prod_drop_scale_a02_sspc_rand_k25)
    _setup_module_sspc_arm "drop_pent_scale" "Prod A+B+SSPC v2 random extra K=25" "sspc_rand_k25" 1
    ;;
  17_prod_drop_scale_a02_sspc_rand_k49)
    _setup_module_sspc_arm "drop_pent_scale" "Prod A+B+SSPC v2 random extra K=49" "sspc_rand_k49" 1
    ;;
  18_prod_drop_scale_a02_sspc_full_g1)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 in-budget full g1 K=1" "sspc_full_g1" 1
    ;;
  18_prod_drop_scale_a02_sspc_full_g2)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 in-budget full g2 K=4" "sspc_full_g2" 1
    ;;
  18_prod_drop_scale_a02_sspc_full_g3)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 in-budget full g3 K=9" "sspc_full_g3" 1
    ;;
  18_prod_drop_scale_a02_sspc_full_g4)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 in-budget full g4 K=16" "sspc_full_g4" 1
    ;;
  18_prod_drop_scale_a02_sspc_full_g5)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 in-budget full g5 K=25" "sspc_full_g5" 1
    ;;
  18_prod_drop_scale_a02_sspc_full_g6)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 in-budget full g6 K=36" "sspc_full_g6" 1
    ;;
  18_prod_drop_scale_a02_sspc_full_g7)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 in-budget full g7 K=49" "sspc_full_g7" 1
    ;;
  18_prod_drop_scale_a02_sspc_full_g8)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 in-budget full g8 K=64" "sspc_full_g8" 1
    ;;
  18_prod_drop_scale_a02_sspc_full_g9)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 in-budget full g9 K=81" "sspc_full_g9" 1
    ;;
  18_prod_drop_scale_a02_sspc_zafd)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 in-budget ZAFD only" "sspc_zafd" 1
    ;;
  xb1_sspc_zafd_uniform)
    # B1 cross arm: prod scaffold (SSPC in-budget + ZAFD rectify) with the
    # selector swapped to uniform — isolates selector vs pipeline extras.
    _setup_module_sspc_arm "drop_pent_scale" "XB1 SSPC-ZAFD scaffold + uniform selector" "sspc_zafd" 1
    MODEL_ARGS="$(echo "${MODEL_ARGS}" | sed -E 's/visual_prune_policy=[^,]*/visual_prune_policy=uniform/')"
    ARM_LABEL="XB1 SSPC-ZAFD scaffold + uniform selector"
    ;;
  xb1_sspc_skel_uniform)
    # B1 cross arm: in-budget uniform skeleton anchor + uniform fill (pure
    # spatial scaffold, no relevance herding, no ZAFD).
    _setup_module_sspc_arm "drop_pent_scale" "XB1 skeleton anchor + uniform selector" "sspc_skel" 1
    MODEL_ARGS="$(echo "${MODEL_ARGS}" | sed -E 's/visual_prune_policy=[^,]*/visual_prune_policy=uniform/')"
    ARM_LABEL="XB1 skeleton anchor + uniform selector"
    ;;
  xb1_prod_fusion_fast)
    # B1 cross arm: full prod stack with Module A detector swapped
    # omniparser -> fusion_fast (detector attribution).
    _setup_module_sspc_arm "drop_pent_scale" "XB1 prod stack + fusion_fast detector" "sspc_zafd" 1
    MODEL_ARGS="$(echo "${MODEL_ARGS}" | sed -E 's/miu_detector=[^,]*/miu_detector=fusion_fast/')"
    ARM_LABEL="XB1 prod stack + fusion_fast detector"
    ;;
  xb1_prod_skel_additive)
    # B1 cross arm: prod selector + additive (extra-budget) skeleton + ZAFD —
    # the historically validated skeleton-anchoring configuration.
    _setup_module_sspc_arm "drop_pent_scale" "XB1 prod + additive skeleton" "sspc_zafd" 1
    export GEOPACT_SSPC_SKELETON=1
    export GEOPACT_SSPC_ZAFD=1
    export GEOPACT_SSPC_BUDGET=additive
    unset GEOPACT_ZAFD_OMEGA_NORM
    ARM_LABEL="XB1 prod + additive skeleton"
    ;;
  # ── XB2 family: two-stage geometric-floor budget + herding calibration ──
  # B-family isolation arms (no SSPC/ZAFD): A+B CDCT scaffold from
  # 07_prod_drop_scale_a02. Flags: GEOPACT_B_GEO_FLOOR / GEOPACT_B_GEO_BETA
  # (raster anchors k_g=floor(beta*k), herding fills the rest, budget exact)
  # and GEOPACT_ICKH_DENOM=t1 (exact equal-weight MMD greedy denominator).
  xb2_cdct_b_only)
    _setup_module_a_v2_arm "drop_pent_scale" "XB2 current CDCT-Herding, B-only" 1
    export GEOPACT_CDCT_ALPHA=0.2
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    ;;
  xb3_geo_floor)
    _setup_module_a_v2_arm "drop_pent_scale" "XB3 geo-floor(auto beta) + CDCT fill" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_GEO_FLOOR=1
    export GEOPACT_B_GEO_BETA=auto
    unset GEOPACT_ICKH_DENOM
    ;;
  xb3_geo_floor_b25|xb3_geo_floor_b50|xb3_geo_floor_b75)
    _setup_module_a_v2_arm "drop_pent_scale" "XB3 geo-floor(fixed beta) + CDCT fill" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_GEO_FLOOR=1
    case "$EXP_ARM" in
      *b25) export GEOPACT_B_GEO_BETA=0.25 ;;
      *b50) export GEOPACT_B_GEO_BETA=0.50 ;;
      *b75) export GEOPACT_B_GEO_BETA=0.75 ;;
    esac
    unset GEOPACT_ICKH_DENOM
    ARM_LABEL="XB3 geo-floor beta=${GEOPACT_B_GEO_BETA} + CDCT fill"
    ;;
  xb5_geo_floor_t1)
    _setup_module_a_v2_arm "drop_pent_scale" "XB5 geo-floor(auto) + CDCT fill, denom t+1" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_GEO_FLOOR=1
    export GEOPACT_B_GEO_BETA=auto
    export GEOPACT_ICKH_DENOM=t1
    ;;
  xb5_t1_only)
    _setup_module_a_v2_arm "drop_pent_scale" "XB5 CDCT-Herding denom t+1 (no floor)" 1
    export GEOPACT_CDCT_ALPHA=0.2
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA
    export GEOPACT_ICKH_DENOM=t1
    ;;
  xb4_prod_geo_floor)
    # Full prod stack (SSPC in-budget + ZAFD) + geometric floor: does the
    # two-stage budget also fix the integrated pipeline?
    _setup_module_sspc_arm "drop_pent_scale" "XB4 prod stack + geo-floor(auto)" "sspc_zafd" 1
    export GEOPACT_B_GEO_FLOOR=1
    export GEOPACT_B_GEO_BETA=auto
    unset GEOPACT_ICKH_DENOM
    ;;
  xb6_gf_zafd_b50|xb6_gf_zafd_b75)
    # Combined main candidate: geometric floor (fixed beta) + donor-norm
    # ZAFD with receiver clamp — tests whether the B fix and the C fix are
    # additive (B3xC3 interaction cell of the ablation matrix).
    _setup_module_sspc_arm "drop_pent_scale" "XB6 geo-floor + donor-clamp ZAFD" "sspc_zafd" 1
    export GEOPACT_B_GEO_FLOOR=1
    case "$EXP_ARM" in
      *b50) export GEOPACT_B_GEO_BETA=0.50 ;;
      *b75) export GEOPACT_B_GEO_BETA=0.75 ;;
    esac
    export GEOPACT_ZAFD_OMEGA_NORM=donor
    export GEOPACT_ZAFD_DELTA_CLAMP=0.5
    unset GEOPACT_ICKH_DENOM
    ARM_LABEL="XB6 geo-floor beta=${GEOPACT_B_GEO_BETA} + donor-clamp ZAFD"
    ;;
  # ── XC family: Module C (ZAFD) conservation-transport variants ──
  # C0 = xb2_cdct_b_only (B-only). C1 = 18_prod..._sspc_zafd (survivor-norm,
  # the historical production arm). New arms below.
  xc2_zafd_donor)
    # C2: donor-normalized (column-stochastic) Omega — per-donor mass
    # conservation.
    _setup_module_sspc_arm "drop_pent_scale" "XC2 ZAFD donor-norm" "sspc_zafd" 1
    export GEOPACT_ZAFD_OMEGA_NORM=donor
    ;;
  xc3_zafd_donor_clamp)
    # C3: donor-norm + receiver-side relative-norm clamp (main C candidate).
    _setup_module_sspc_arm "drop_pent_scale" "XC3 ZAFD donor-norm + clamp 0.5" "sspc_zafd" 1
    export GEOPACT_ZAFD_OMEGA_NORM=donor
    export GEOPACT_ZAFD_DELTA_CLAMP=0.5
    ;;
  xc4_zafd_shuffle_omega)
    # C4a negative control: Omega rows shuffled — spatial correspondence
    # destroyed, injected mass preserved. Any "gain" here is an artifact.
    _setup_module_sspc_arm "drop_pent_scale" "XC4 ZAFD omega-shuffle control" "sspc_zafd" 1
    export GEOPACT_ZAFD_OMEGA_NORM=donor
    export GEOPACT_ZAFD_SHUFFLE=omega
    ;;
  xc4_zafd_shuffle_donor)
    # C4b negative control: donor features shuffled against positions.
    _setup_module_sspc_arm "drop_pent_scale" "XC4 ZAFD donor-shuffle control" "sspc_zafd" 1
    export GEOPACT_ZAFD_OMEGA_NORM=donor
    export GEOPACT_ZAFD_SHUFFLE=donor
    ;;
  # ── V2 family: GeoPACT v2 (docs/geopact_v2_experiment_protocol.md) ──
  # Scaffold = xb2_cdct_b_only (A+B CDCT, alpha=0.2, no SSPC/ZAFD/geo-floor).
  # B-v1 reference arm for H1 is xb2_cdct_b_only itself.
  v2b_qd)
    # H1: Module B v2 quality-diversity selector (rank fusion + GUI kernel +
    # orthogonal-residual greedy), no C.
    _setup_module_a_v2_arm "drop_pent_scale" "V2 B-qd (QD selector, no C)" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_SELECTOR=qd
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    unset GEOPACT_B_COMPENSABILITY GEOPACT_C_RELAY GEOPACT_C_RELAY_RESID
    ;;
  v2b_qd_kappa)
    # H3 cell (kappa on, resid off, relay off): compensability discount only.
    # Needs the frame KV snapshot for probe keys even though relay stays off.
    _setup_module_a_v2_arm "drop_pent_scale" "V2 B-qd + compensability kappa" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_SELECTOR=qd
    export GEOPACT_B_COMPENSABILITY=1.0
    export GEOPACT_C_RELAY_KV_SNAPSHOT=1
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    unset GEOPACT_C_RELAY GEOPACT_C_RELAY_RESID
    ;;
  v3_bam)
    # v3 final config: BAM (matching-pursuit scoring, default) + token-level
    # turn query with the mean row prepended (qmix). This is the arm that won
    # the 500-step screen (r05: TM/EM/GP all above uniform AND random) and
    # the single-step full runs (SS-v2 51.65 vs 30.19 random @ r10).
    _setup_module_a_v2_arm "drop_pent_scale" "V3 BAM final (qmix, no C)" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_SELECTOR=bam
    export GEOPACT_TURN_QUERY_TOKENS=1
    export GEOPACT_TURN_QUERY_MEANROW=1
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    unset GEOPACT_B_COMPENSABILITY GEOPACT_C_RELAY GEOPACT_C_RELAY_RESID
    unset GEOPACT_BAM_NO_MASS GEOPACT_BAM_UNIFORM_A GEOPACT_BAM_SHARP_A
    ;;
  v3_bam_centry)
    # Single-step Module C variant: C-Entry summary tokens. B's budget is
    # reduced by R=min(16, k//16); the dropped set is partitioned with the
    # relay's spatial partition and each cluster's mean embedding rides at
    # its medoid slot (in-budget, FA-transparent, embedding-space QASMR).
    _setup_module_a_v2_arm "drop_pent_scale" "V3 BAM + C-Entry summaries" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_SELECTOR=bam
    export GEOPACT_TURN_QUERY_TOKENS=1
    export GEOPACT_TURN_QUERY_MEANROW=1
    export GEOPACT_C_ENTRY=1
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    unset GEOPACT_B_COMPENSABILITY GEOPACT_C_RELAY GEOPACT_C_RELAY_RESID
    unset GEOPACT_BAM_NO_MASS GEOPACT_BAM_UNIFORM_A GEOPACT_BAM_SHARP_A GEOPACT_TWO_STAGE
    ;;
  v3_nap)
    # NAP (Nested Anticipative Pruning): v3_bam config + two-stage nesting.
    # Stage 1 prunes the current screenshot at entry (r_cur, default 0.75)
    # and stores the BAM matching-pursuit pick sequence; stage 2 (slide into
    # history) contracts to the r_hist(=PRUNE_RATIO) prefix — no A/B re-run.
    # Requires PRUNE_TARGET=history in the job env.
    _setup_module_a_v2_arm "drop_pent_scale" "V3 NAP two-stage (BAM qmix)" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_SELECTOR=bam
    # Preserve-if-set: instruction-conditioning ablation arms pass
    # GEOPACT_TURN_QUERY_TOKENS=0 / MEANROW=0 through EXTRA_ENV; an
    # unconditional export here would silently neutralize them (same
    # failure mode as the round-4 blanket-unset bug).
    export GEOPACT_TURN_QUERY_TOKENS="${GEOPACT_TURN_QUERY_TOKENS:-1}"
    export GEOPACT_TURN_QUERY_MEANROW="${GEOPACT_TURN_QUERY_MEANROW:-1}"
    export GEOPACT_TWO_STAGE=1
    export GEOPACT_RATIO_CURRENT="${GEOPACT_RATIO_CURRENT:-0.75}"
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    unset GEOPACT_B_COMPENSABILITY GEOPACT_C_RELAY GEOPACT_C_RELAY_RESID
    # NOTE: BAM ablation flags (NO_MASS / UNIFORM_A / SHARP_A / NO_DIVERSITY)
    # arrive via EXTRA_ENV with a fresh queue env — do not unset here.
    ;;
  v3_nap_crelay)
    # NAP + C reconstruction: the middle layer (S_cur \ S_hist) was really
    # prefilled at stage 1, so its KV rows are lazily compressed into relays
    # straight from the dialogue cache at transition (no snapshot clone).
    _setup_module_a_v2_arm "drop_pent_scale" "V3 NAP + C cache-row relay" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_SELECTOR=bam
    export GEOPACT_TURN_QUERY_TOKENS=1
    export GEOPACT_TURN_QUERY_MEANROW=1
    export GEOPACT_TWO_STAGE=1
    export GEOPACT_RATIO_CURRENT="${GEOPACT_RATIO_CURRENT:-0.75}"
    export GEOPACT_C_RELAY=1
    export GEOPACT_C_RELAY_MODE=kv1
    export GEOPACT_C_RELAY_FROM_CACHE=1
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    unset GEOPACT_B_COMPENSABILITY GEOPACT_C_RELAY_RESID
    unset GEOPACT_BAM_NO_MASS GEOPACT_BAM_UNIFORM_A GEOPACT_BAM_SHARP_A
    ;;
  v3_nap_absorb)
    # NAP + Module C v3 (NAP-History Absorb): history-only, key-frozen value
    # absorption. The BAM ordering also fixes the rank-monotone absorption
    # tree at entry; at the history transition each dropped row's
    # *contextualized* value is merged into its surviving host row in place
    # (balanced loads => kappa -> 1, zero-order exact at kappa=1 / delta=0).
    # No extra attention rows, no bias, FA2-native, strict token budget.
    # Tunables via job EXTRA_ENV: GEOPACT_C_LAMBDA_HIST (default 1.0),
    # GEOPACT_C_ABSORB_ASSIGN=balanced|nearest, GEOPACT_C_LAMBDA_ENTRY
    # (independent ToMe-like entry-merge extension, default 0=off).
    _setup_module_a_v2_arm "drop_pent_scale" "V3 NAP + C history-absorb" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_SELECTOR=bam
    # Preserve-if-set: instruction-conditioning arms override via EXTRA_ENV.
    export GEOPACT_TURN_QUERY_TOKENS="${GEOPACT_TURN_QUERY_TOKENS:-1}"
    export GEOPACT_TURN_QUERY_MEANROW="${GEOPACT_TURN_QUERY_MEANROW:-1}"
    export GEOPACT_TWO_STAGE=1
    export GEOPACT_RATIO_CURRENT="${GEOPACT_RATIO_CURRENT:-0.75}"
    export GEOPACT_C_ABSORB=1
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    unset GEOPACT_B_COMPENSABILITY GEOPACT_C_RELAY GEOPACT_C_RELAY_RESID
    # NOTE: do NOT unset GEOPACT_BAM_NO_MASS / GEOPACT_BAM_UNIFORM_A /
    # GEOPACT_BAM_SHARP_A here — queue jobs run with a fresh env and pass
    # module-necessity ablation flags through EXTRA_ENV (round-4 bug: the
    # blanket unset silently neutralized the -A/-instruction arms).
    ;;
  nap_uniform)
    # Nested-uniform baseline: stride S_cur at entry, re-stride inside S_cur
    # at the history transition (honest nesting, no ordering information).
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    export GEOPACT_TWO_STAGE=1
    export GEOPACT_RATIO_CURRENT="${GEOPACT_RATIO_CURRENT:-0.75}"
    PRUNE_RATIO="${PRUNE_RATIO:-0.25}"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=uniform,visual_prune_ratio=${PRUNE_RATIO}}"
    ARM_LABEL="NAP nested-uniform baseline (two-stage stride)"
    ;;
  nap_random)
    # Nested-random baseline: deterministic per-frame permutation (sticky
    # RNG); S_cur = first k_cur of the permutation, S_hist = its k_hist prefix.
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    export GEOPACT_TWO_STAGE=1
    export GEOPACT_RATIO_CURRENT="${GEOPACT_RATIO_CURRENT:-0.75}"
    PRUNE_RATIO="${PRUNE_RATIO:-0.25}"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=random,visual_prune_ratio=${PRUNE_RATIO}}"
    ARM_LABEL="NAP nested-random baseline (two-stage permutation prefix)"
    ;;
  nap_pubsel)
    # Paradigm-transfer control: run a published
    # feature-level selector (GEOPACT_NAP_PUBSEL_POLICY, e.g. divprune) on
    # OUR kv_rewrite serving path with the same two-stage NAP nesting as
    # nap_uniform — isolates "selector quality" from "serving paradigm".
    export GEOPACT_SESSION_KV=1
    export GEOPACT_SESSION_KV_MODE=dialogue
    export GEOPACT_SCREENSHOT_ONLY_CACHE=1
    export GEOPACT_TWO_STAGE=1
    export GEOPACT_RATIO_CURRENT="${GEOPACT_RATIO_CURRENT:-0.75}"
    PRUNE_RATIO="${PRUNE_RATIO:-0.25}"
    _PUBSEL_POLICY="${GEOPACT_NAP_PUBSEL_POLICY:-divprune}"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=${_PUBSEL_POLICY},visual_prune_ratio=${PRUNE_RATIO}}"
    ARM_LABEL="NAP paradigm-transfer control (${_PUBSEL_POLICY} on kv_rewrite)"
    ;;
  pub_hybrid_vispruner|pub_hybrid_divprune|pub_hybrid_cdpruner|pub_hybrid_prunesid)
    # E3 hybrid attribution arms: published selector × Module A spatial-prior
    # mass (GEOPACT_PUB_HYBRID_MASS=1 in the hook). Same baseline_full
    # serving as pub_*; the detector is OmniParser so the grafted prior is
    # the exact field GeoPACT consumes. Alpha defaults to the main-table 2.0
    # (override via EXTRA_ENV GEOPACT_PUB_HYBRID_ALPHA).
    # 2026-07-31: extended from the original vispruner/divprune pair to also
    # cover CDPruner and PruneSID, so the parity control reaches the *strongest*
    # published baseline instead of only the two weaker ones.
    export GEOPACT_SESSION_KV=0
    unset GEOPACT_SESSION_KV_MODE
    unset GEOPACT_SCREENSHOT_ONLY_CACHE
    _PUB_POLICY="${EXP_ARM#pub_hybrid_}"
    PRUNE_RATIO="${PRUNE_RATIO:-0.25}"
    export GEOPACT_PUB_HYBRID_MASS=1
    export GEOPACT_PUB_HYBRID_ALPHA="${GEOPACT_PUB_HYBRID_ALPHA:-2.0}"
    case "${_PUB_POLICY}" in
      vispruner)
        # same fidelity ruling as pub_vispruner: real ViT attention-received
        export GEOPACT_VZ_ATTN_CAPTURE=1
        ;;
    esac
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=${_PUB_POLICY},visual_prune_ratio=${PRUNE_RATIO},miu_detector=omniparser}"
    ARM_LABEL="E3 hybrid ${_PUB_POLICY} x spatial prior (baseline_full)"
    ;;
  pub_divprune|pub_cdpruner|pub_visionzip|pub_vispruner|pub_prumergeplus|pub_trim|pub_visiontrim|pub_zooprune|pub_prunesid)
    # Published training-free baselines (feature-level, single stage) under the
    # baseline_full paradigm: full re-prefill each step, no KV rewrite.
    # Single-step: PRUNE_RATIO = keep ratio of the current image.
    # Multi-step (c/h): PRUNE_RATIO = history keep h, plus
    #   GEOPACT_PUBBASE_CURRENT=c (via EXTRA_ENV) to prune the current frame.
    export GEOPACT_SESSION_KV=0
    unset GEOPACT_SESSION_KV_MODE
    unset GEOPACT_SCREENSHOT_ONLY_CACHE
    _PUB_POLICY="${EXP_ARM#pub_}"
    PRUNE_RATIO="${PRUNE_RATIO:-0.25}"
    case "${_PUB_POLICY}" in
      visionzip|vispruner|prumergeplus|visiontrim)
        # 2026-07-20 fidelity ruling: scoring signal is method core — these
        # arms score by the model's REAL last-vision-block attention
        # received (streaming capture in the transformers patch); the
        # feature-space proxy is forbidden for paper-table arms.
        # (VisionTrim ICLR'26: DVTS global term uses the same no-CLS
        # attention-received mapping as the VisionZip family.)
        export GEOPACT_VZ_ATTN_CAPTURE=1
        ;;
      zooprune)
        # ZOO-Prune CVPR'26: zeroth-order projector sensitivity is the
        # method-core scoring signal; computed in the transformers patch
        # against the model's own PatchMerger (no proxy permitted).
        export GEOPACT_ZOO_SENS_CAPTURE=1
        ;;
    esac
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS},visual_prune_policy=${_PUB_POLICY},visual_prune_ratio=${PRUNE_RATIO}}"
    ARM_LABEL="published baseline ${_PUB_POLICY} (baseline_full, single-stage)"
    ;;
  pub_dart)
    # DART (official in-LLM protocol, 2026-07-20 fidelity decision): pivots =
    # top layer-(K-1) key-state L1 norms (4 image + 4 text, official Qwen2-VL
    # defaults), retained set built per-pivot by least cosine similarity,
    # pruned before layer K=2 — inline in the transformers patch like FastV.
    # PRUNE_RATIO = keep ratio of the current image; multi-step passes
    #   GEOPACT_DART_KEEP_HIST=h via EXTRA_ENV.
    # (The former feature-level dart variant is deprecated for paper tables;
    # its arms live under visual_prune_policy=dart and are no longer queued.)
    export GEOPACT_SESSION_KV=0
    unset GEOPACT_SESSION_KV_MODE
    unset GEOPACT_SCREENSHOT_ONLY_CACHE
    export GEOPACT_DART=1
    export GEOPACT_DART_DROP_K="${GEOPACT_DART_DROP_K:-2}"
    # PRUNE_RATIO was moved to FLEET_PRUNE_RATIO before the case dispatch.
    export GEOPACT_DART_KEEP_CUR="${GEOPACT_DART_KEEP_CUR:-${FLEET_PRUNE_RATIO:-0.25}}"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS}}"
    ARM_LABEL="published baseline DART (official in-LLM layer-${GEOPACT_DART_DROP_K}, baseline_full)"
    ;;
  pub_ivcprune)
    # IVC-Prune (ICLR'26, official FireRedTeam Qwen2.5-VL protocol): single
    # selection at intermediate layer k — RoPE-derived IVC tokens (cos/sin
    # per-token means, per-score fraction 0.2*r) + two-stage value-similarity
    # foreground (text->img softmax value attention, 1% seeds, seed->img) —
    # original position ids preserved, earlier-layer KV retro-pruned.
    # k default 22 = official 36-layer Qwen2.5-VL-3B setting (our LLM depth).
    # PRUNE_RATIO = keep ratio of the current image; multi-step passes
    #   GEOPACT_IVCP_KEEP_HIST=h via EXTRA_ENV.
    export GEOPACT_SESSION_KV=0
    unset GEOPACT_SESSION_KV_MODE
    unset GEOPACT_SCREENSHOT_ONLY_CACHE
    export GEOPACT_IVCP=1
    export GEOPACT_IVCP_K="${GEOPACT_IVCP_K:-22}"
    # PRUNE_RATIO was moved to FLEET_PRUNE_RATIO before the case dispatch.
    export GEOPACT_IVCP_KEEP_CUR="${GEOPACT_IVCP_KEEP_CUR:-${FLEET_PRUNE_RATIO:-0.25}}"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS}}"
    ARM_LABEL="published baseline IVC-Prune (official in-LLM layer-${GEOPACT_IVCP_K}, baseline_full)"
    ;;
  pub_fastv)
    # FastV (original protocol): layer-K attention pruning inline in the LLM
    # prefill, baseline_full paradigm, native vision path (no GeoPACT hook).
    # PRUNE_RATIO = keep ratio of the current image; multi-step passes
    #   GEOPACT_FASTV_KEEP_HIST=h via EXTRA_ENV.
    export GEOPACT_SESSION_KV=0
    unset GEOPACT_SESSION_KV_MODE
    unset GEOPACT_SCREENSHOT_ONLY_CACHE
    export GEOPACT_FASTV=1
    export GEOPACT_FASTV_DROP_K="${GEOPACT_FASTV_DROP_K:-2}"
    # PRUNE_RATIO was moved to FLEET_PRUNE_RATIO before the case dispatch.
    export GEOPACT_FASTV_KEEP_CUR="${GEOPACT_FASTV_KEEP_CUR:-${FLEET_PRUNE_RATIO:-0.25}}"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS}}"
    ARM_LABEL="published baseline FastV (layer-${GEOPACT_FASTV_DROP_K} attention, baseline_full)"
    ;;
  pub_pdrop)
    # PDrop / PyramidDrop (original protocol): multi-stage rank&drop inline in
    # the LLM prefill (S=4 equal stages, per-stage lambda = r^(1/3)),
    # baseline_full paradigm, native vision path (no GeoPACT hook).
    # PRUNE_RATIO = final-stage keep of the current image; multi-step passes
    #   GEOPACT_PDROP_KEEP_HIST=h via EXTRA_ENV.
    export GEOPACT_SESSION_KV=0
    unset GEOPACT_SESSION_KV_MODE
    unset GEOPACT_SCREENSHOT_ONLY_CACHE
    export GEOPACT_PDROP=1
    export GEOPACT_PDROP_KEEP_CUR="${GEOPACT_PDROP_KEEP_CUR:-${FLEET_PRUNE_RATIO:-0.25}}"
    MODEL_ARGS="${MODEL_ARGS:-${BASE_MODEL_ARGS}}"
    ARM_LABEL="published baseline PDrop (pyramid rank&drop, baseline_full)"
    ;;
  v2b_bam)
    # B v3 (BAM): parameter-free Bayes-attention matching pursuit — softmax
    # zscore likelihood x A mass-measure feature lengths x orthogonal residual
    # greedy. Zero hyperparameters, no C.
    _setup_module_a_v2_arm "drop_pent_scale" "V2 B-bam (param-free, no C)" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_SELECTOR=bam
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    unset GEOPACT_B_COMPENSABILITY GEOPACT_C_RELAY GEOPACT_C_RELAY_RESID
    unset GEOPACT_BAM_NO_MASS GEOPACT_BAM_UNIFORM_A
    ;;
  v2b_bam_soft)
    # BAM design variant: likelihood scored once (log a + log d^2, matching
    # pursuit reading) — halves the instruction sharpness vs the DPP square.
    _setup_module_a_v2_arm "drop_pent_scale" "V2 B-bam soft-likelihood" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_SELECTOR=bam
    export GEOPACT_BAM_SOFT_A=1
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    unset GEOPACT_B_COMPENSABILITY GEOPACT_C_RELAY GEOPACT_C_RELAY_RESID
    unset GEOPACT_BAM_NO_MASS GEOPACT_BAM_UNIFORM_A
    ;;
  v2b_bam_noA)
    # BAM ablation: force m=1 (Module A mass measure removed from B).
    _setup_module_a_v2_arm "drop_pent_scale" "V2 B-bam no-A-mass ablation" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_SELECTOR=bam
    export GEOPACT_BAM_NO_MASS=1
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    unset GEOPACT_B_COMPENSABILITY GEOPACT_C_RELAY GEOPACT_C_RELAY_RESID
    unset GEOPACT_BAM_UNIFORM_A
    ;;
  v2b_bam_div)
    # BAM ablation: force uniform likelihood (instruction guidance removed;
    # pure mass-weighted volume maximization).
    _setup_module_a_v2_arm "drop_pent_scale" "V2 B-bam uniform-a ablation" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_SELECTOR=bam
    export GEOPACT_BAM_UNIFORM_A=1
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    unset GEOPACT_B_COMPENSABILITY GEOPACT_C_RELAY GEOPACT_C_RELAY_RESID
    unset GEOPACT_BAM_NO_MASS
    ;;
  v2b_qd_noP)
    # QD ablation: beta_P=0 — Module A density prior removed from quality.
    _setup_module_a_v2_arm "drop_pent_scale" "V2 B-qd no-prior ablation" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_SELECTOR=qd
    export GEOPACT_QD_BETA_P=0
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    unset GEOPACT_B_COMPENSABILITY GEOPACT_C_RELAY GEOPACT_C_RELAY_RESID
    unset GEOPACT_QD_GAMMA
    ;;
  v2b_qd_noA)
    # QD ablation: beta_P=0 and gamma=0 — B fully decoupled from Module A
    # (quality = relevance rank only, kernel without the same-MIU boost).
    _setup_module_a_v2_arm "drop_pent_scale" "V2 B-qd no-A ablation" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_SELECTOR=qd
    export GEOPACT_QD_BETA_P=0
    export GEOPACT_QD_GAMMA=0
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    unset GEOPACT_B_COMPENSABILITY GEOPACT_C_RELAY GEOPACT_C_RELAY_RESID
    ;;
  xb2_herding_noA)
    # v1 herding ablation: alpha=0 — Module A density weighting removed.
    _setup_module_a_v2_arm "drop_pent_scale" "XB2 herding no-A (alpha=0)" 1
    export GEOPACT_CDCT_ALPHA=0
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    unset GEOPACT_B_SELECTOR GEOPACT_B_COMPENSABILITY
    unset GEOPACT_C_RELAY GEOPACT_C_RELAY_RESID
    ;;
  v2c_qd_qasmr)
    # H2 main: B-qd + QASMR kv1 relay (R=16, no coupling flags).
    _setup_module_a_v2_arm "drop_pent_scale" "V2 B-qd + QASMR kv1" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_SELECTOR=qd
    export GEOPACT_C_RELAY=1
    export GEOPACT_C_RELAY_MODE=kv1
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    unset GEOPACT_B_COMPENSABILITY GEOPACT_C_RELAY_RESID
    ;;
  v2c_qd_qasmr_resid)
    # H3 cell (resid on, kappa off): rho_m residual handoff -> relay budgets.
    _setup_module_a_v2_arm "drop_pent_scale" "V2 B-qd + QASMR kv1 + resid" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_SELECTOR=qd
    export GEOPACT_C_RELAY=1
    export GEOPACT_C_RELAY_MODE=kv1
    export GEOPACT_C_RELAY_RESID=1
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    unset GEOPACT_B_COMPENSABILITY
    ;;
  v2c_qd_qasmr_full)
    # H3 cell (kappa on, resid on): full B<->C coupling.
    _setup_module_a_v2_arm "drop_pent_scale" "V2 B-qd + QASMR kv1 + kappa + resid" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_SELECTOR=qd
    export GEOPACT_B_COMPENSABILITY=1.0
    export GEOPACT_C_RELAY=1
    export GEOPACT_C_RELAY_MODE=kv1
    export GEOPACT_C_RELAY_RESID=1
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    ;;
  v2c_herding_qasmr)
    # C attribution on B-v1: herding selector + QASMR (isolates the relay gain
    # from the selector swap).
    _setup_module_a_v2_arm "drop_pent_scale" "V2 B-herding + QASMR kv1" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_SELECTOR=herding
    export GEOPACT_C_RELAY=1
    export GEOPACT_C_RELAY_MODE=kv1
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    unset GEOPACT_B_COMPENSABILITY GEOPACT_C_RELAY_RESID
    ;;
  v2c_qd_clite)
    # H5 mechanism baseline: embedding-level C-Lite OT (no attention relay).
    _setup_module_a_v2_arm "drop_pent_scale" "V2 B-qd + C-Lite OT" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_SELECTOR=qd
    export GEOPACT_C_RELAY=1
    export GEOPACT_C_RELAY_MODE=lite
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    unset GEOPACT_B_COMPENSABILITY GEOPACT_C_RELAY_RESID
    ;;
  v2c_qd_kv2)
    # H5 error-order arm: second-order relay moments (diag), EQUAL-BYTE vs
    # the kv1 main path — diag moments double the per-relay bytes
    # (k̄,v̄ + Σkk,Σvk diag), so the per-frame budget is halved (R=8 vs
    # kv1's R=16). Protocol: kv2 may only enter the main path if it beats
    # kv1(R=16) on GP under this equal-byte budget AND passes decode gate G2
    # in scripts/_test_relay_fa2_gpu.py (2026-07-13 A100: G2 FAIL, +86-93%
    # marginal — analysis-only unless re-measured otherwise).
    _setup_module_a_v2_arm "drop_pent_scale" "V2 B-qd + QASMR kv2 diag eq-byte" 1
    export GEOPACT_CDCT_ALPHA=0.2
    export GEOPACT_B_SELECTOR=qd
    export GEOPACT_C_RELAY=1
    export GEOPACT_C_RELAY_MODE=kv2
    export GEOPACT_C_RELAY_R=8
    unset GEOPACT_B_GEO_FLOOR GEOPACT_B_GEO_BETA GEOPACT_ICKH_DENOM
    unset GEOPACT_B_COMPENSABILITY GEOPACT_C_RELAY_RESID
    ;;
  18_prod_drop_scale_a02_sspc_skel_g3)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 in-budget skel g3" "sspc_skel_g3" 1
    ;;
  18_prod_drop_scale_a02_sspc_skel_g5)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 in-budget skel g5" "sspc_skel_g5" 1
    ;;
  18_prod_drop_scale_a02_sspc_skel_g7)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 in-budget skel g7" "sspc_skel_g7" 1
    ;;
  18_prod_drop_scale_a02_sspc_rand_g3)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 in-budget rand g3" "sspc_rand_g3" 1
    ;;
  18_prod_drop_scale_a02_sspc_rand_g5)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 in-budget rand g5" "sspc_rand_g5" 1
    ;;
  18_prod_drop_scale_a02_sspc_rand_g7)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 in-budget rand g7" "sspc_rand_g7" 1
    ;;
  18_prod_drop_scale_a02_sspc_full_g5_s02)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 full g5 sigma=0.2" "sspc_full_g5_s02" 1
    ;;
  18_prod_drop_scale_a02_sspc_full_g5_s06)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 full g5 sigma=0.6" "sspc_full_g5_s06" 1
    ;;
  18_prod_drop_scale_a02_sspc_zafd_s02)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 zafd sigma=0.2" "sspc_zafd_s02" 1
    ;;
  18_prod_drop_scale_a02_sspc_zafd_s06)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 zafd sigma=0.6" "sspc_zafd_s06" 1
    ;;
  18_prod_drop_scale_a02_sspc_zafd_cons)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 IPCC donor-norm" "sspc_zafd_cons" 1
    ;;
  18_prod_drop_scale_a02_sspc_zafd_hybrid_g025)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 IPCC hybrid gamma=0.25" "sspc_zafd_hybrid_g025" 1
    ;;
  18_prod_drop_scale_a02_sspc_zafd_hybrid_g05)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 IPCC hybrid gamma=0.5" "sspc_zafd_hybrid_g05" 1
    ;;
  18_prod_drop_scale_a02_sspc_zafd_adaptive)
    _setup_module_sspc_arm "drop_pent_scale" "SSPC v3 IPCC adaptive sigma" "sspc_zafd_adaptive" 1
    ;;
  19_prod_drop_scale_a02_sspc_ipcc)
    _setup_module_sspc_arm "drop_pent_scale" "Prod A+B+IPCC v2" "sspc_ipcc" 1
    ;;
  13_prod_drop_scale_a02_gsc_spf_center)
    _setup_module_c_gsc_arm "drop_pent_scale" "Prod A+B+GSC SPF center only (G2)" "gsc_spf_center" 1
    ;;
  13_prod_drop_scale_a02_gsc_vfs_only)
    _setup_module_c_gsc_arm "drop_pent_scale" "Prod A+B+GSC VFS only (G3)" "gsc_vfs_only" 1
    ;;
  13_prod_drop_scale_a02_gsc_vfs_spf)
    _setup_module_c_gsc_arm "drop_pent_scale" "Prod A+B+GSC VFS+SPF center (G4)" "gsc_vfs_spf" 1
    ;;
  13_prod_drop_scale_a02_gsc_vfs_spf_schedule)
    _setup_module_c_gsc_arm "drop_pent_scale" "Prod A+B+GSC VFS+SPF+schedule (G5)" "gsc_vfs_spf_schedule" 1
    ;;
  13_prod_drop_scale_a02_gsc_full_query_adapt)
    _setup_module_c_gsc_arm "drop_pent_scale" "Prod A+B+GSC full+query adapt (G6)" "gsc_full_query_adapt" 1
    ;;
  13_prod_drop_scale_a02_gsc_spf_full)
    _setup_module_c_gsc_arm "drop_pent_scale" "Prod A+B+GSC SPF full ablation (G7)" "gsc_spf_full" 1
    ;;
  07_prod_drop_scale_a02_seqpos)
    _setup_module_a_v2_arm "drop_pent_scale_seqpos" "Factorial: v2 drop scale seqpos (no origpos)" 1
    export GEOPACT_CDCT_ALPHA=0.2
    ;;
  08_prod_drop_scale_a02_seqpos_coord_plain)
    _setup_module_c_prod_v2_arm "drop_pent_scale_seqpos" "Prod seqpos + UDSI spatial_plain" "spatial_plain" 1 3
    ;;
  07_prod_drop_contrast_no_circular)
    _setup_module_a_v2_arm "drop_pent_contrast" "Factorial: v2 drop contrast, circular OFF, alpha=0.1" 0
    ;;
  05a_r40_cdct_a01_baseline)
    _setup_module_a_cdct_attr_arm "" "Module A attr ablation A0 baseline (CDCT alpha=0.1)"
    ;;
  05a_r40_cdct_a01_no_info_density)
    _setup_module_a_cdct_attr_arm "no_info_density" "Module A attr A_null no_info_density"
    ;;
  05a_r40_cdct_a01_raw_energy)
    _setup_module_a_cdct_attr_arm "raw_energy" "Module A attr A_raw rho=E (no div token)"
    ;;
  05a_r40_cdct_a01_div_area_heuristic)
    _setup_module_a_cdct_attr_arm "div_area_heuristic" "Module A attr A_div_area area heuristic n"
    ;;
  05a_r40_cdct_a01_no_conf_gate)
    _setup_module_a_cdct_attr_arm "no_conf_gate" "Module A attr A_conf_add additive conf"
    ;;
  05a_r40_cdct_a01_conf_attr_only)
    _setup_module_a_cdct_attr_arm "conf_attr_only" "Module A attr A_conf_only f(det) only"
    ;;
  05a_r40_cdct_a01_no_conf)
    _setup_module_a_cdct_attr_arm "no_conf" "Module A attr A_conf_none no conf term"
    ;;
  05a_r40_cdct_a01_drop_pent_scale)
    _setup_module_a_cdct_attr_arm "drop_pent_scale" "Module A attr A_drop_S drop scale"
    ;;
  05a_r40_cdct_a01_drop_pent_entropy)
    _setup_module_a_cdct_attr_arm "drop_pent_entropy" "Module A attr A_drop_H drop entropy"
    ;;
  05a_r40_cdct_a01_drop_pent_contrast)
    _setup_module_a_cdct_attr_arm "drop_pent_contrast" "Module A attr A_drop_C drop contrast"
    ;;
  05a_r40_cdct_a01_drop_pent_containment)
    _setup_module_a_cdct_attr_arm "drop_pent_containment" "Module A attr A_drop_G drop containment"
    ;;
  05a_r40_cdct_a01_drop_pent_resonance)
    _setup_module_a_cdct_attr_arm "drop_pent_resonance" "Module A attr A_drop_R drop resonance"
    ;;
  05b_r40_cdct_v2)
    _setup_module_a_v2_arm "" "Module A v2.0 structural E=S+H+C+G+R (CDCT alpha=0.1)"
    ;;
  05b_r40_cdct_v2_drop_contrast)
    _setup_module_a_v2_arm "drop_pent_contrast" "Module A v2.1 drop contrast"
    ;;
  05b_r40_cdct_v2_drop_scale)
    _setup_module_a_v2_arm "drop_pent_scale" "Module A v2.2 drop scale"
    ;;
  05b_r40_cdct_v2_drop_scale_drop_contrast)
    _setup_module_a_v2_arm "drop_pent_scale_drop_pent_contrast" "Module A v2.3 drop scale+contrast"
    ;;
  05b_r40_cdct_legacy_baseline)
    _setup_module_a_cdct_attr_arm "" "Module A legacy baseline det×struct (05a A0 repro)"
    ;;
  05c_r40_cdct_v2_circular_on)
    _setup_module_a_v2_arm "" "Module A v2 + circular/ellipse aux ON (control)" 1
    ;;
  05c_r40_cdct_v2_no_circular_aux)
    _setup_module_a_v2_arm "" "Module A v2 + GEOPACT_YOLO_CIRCULAR_AUX=0" 0
    ;;
  05d_v2_det_baseline)
    _setup_module_a_v2_det_arm "Module A v2 det baseline (YOLO+circular)" omniparser
    ;;
  05d_v2_det_fullres_aux)
    _setup_module_a_v2_det_arm "Module A v2 det full-res circular aux" omniparser BLOB_AUX_USE_FULL_RES=1
    ;;
  05d_v2_det_conf003)
    _setup_module_a_v2_det_arm "Module A v2 det OMNIPARSER_ICON_CONF=0.03" omniparser OMNIPARSER_ICON_CONF=0.03
    ;;
  05d_v2_det_conf001)
    _setup_module_a_v2_det_arm "Module A v2 det OMNIPARSER_ICON_CONF=0.01" omniparser OMNIPARSER_ICON_CONF=0.01
    ;;
  05d_v2_det_ocr_fusion)
    _setup_module_a_v2_det_arm "Module A v2 det YOLO+circular+OCR fusion" omniparser GEOPACT_YOLO_OCR_AUX=1
    ;;
  05d_v2_det_ocr_only)
    _setup_module_a_v2_det_arm "Module A v2 det OCR-only+circular (Stack C)" ocr_only
    ;;
  05d_v2_det_stripe_aux)
    _setup_module_a_v2_det_arm "Module A v2 det stripe_row CV aux" omniparser GEOPACT_YOLO_STRIPE_AUX=1
    ;;
  05d_v2_det_ocr_fusion_fullres)
    _setup_module_a_v2_det_arm "Module A v2 det OCR fusion + full-res aux" omniparser \
      GEOPACT_YOLO_OCR_AUX=1 BLOB_AUX_USE_FULL_RES=1
    ;;
  05d_v2_det_ocr_stripe_combo)
    _setup_module_a_v2_det_arm "Module A v2 det OCR+circular+stripe triple combo" omniparser \
      GEOPACT_YOLO_OCR_AUX=1 GEOPACT_YOLO_STRIPE_AUX=1 MIU_DISABLE_OCR=0 MIU_OCR_ENGINE=rapidocr
    ;;
  *)
    echo "Unknown EXP_ARM=${EXP_ARM}" >&2
    exit 1
    ;;
esac

# Per-job escape hatch: GEOPACT_SESSION_KV_FORCE=0 (via queue EXTRA_ENV)
# disables the dialogue session-KV engine even for arms/tracks that export
# it, e.g. single-step protocols on episode-structured data (AC low-level
# alpha sweep) where the dialogue prefix machinery must never engage.
if [[ "${GEOPACT_SESSION_KV_FORCE:-}" == "0" ]]; then
  export GEOPACT_SESSION_KV=0
  unset GEOPACT_SESSION_KV_MODE
  echo "GEOPACT_SESSION_KV_FORCE=0: session KV disabled for this job"
fi

_apply_fleet_prune_ratio

mkdir -p "$OUT"
# lmms_eval wrapper switch: gui_owl (Qwen3-VL, default) or ui_tars (Qwen2/2.5-VL).
LMMS_MODEL="${LMMS_MODEL:-gui_owl}"
if [[ "${LMMS_MODEL}" == "ui_tars" ]]; then
  (
    flock -w 180 9
    "$EVAL" scripts/restore_q25vl_geopact_patches.py
  ) 9>"${REPO}/logs/.restore_q25vl_geopact.lock"
else
  (
    flock -w 180 9
    "$EVAL" scripts/restore_q3vl_geopact_patches.py
  ) 9>"${REPO}/logs/.restore_q3vl_geopact.lock"
fi

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

{
  echo "=== KV+prune five-arm: ${EXP_ARM} ==="
  echo "ARM=${ARM_LABEL}"
  echo "OUT=${OUT}"
  echo "TASK=${TASK}"
  echo "LIMIT=${LIMIT:-full}"
  echo "GEOPACT_SESSION_KV=${GEOPACT_SESSION_KV:-0}"
  echo "GEOPACT_SESSION_KV_MODE=${GEOPACT_SESSION_KV_MODE:-unset}"
  echo "GEOPACT_VISION_EMBED_CACHE=${GEOPACT_VISION_EMBED_CACHE:-0}"
  echo "PRUNE_RATIO=${PRUNE_RATIO:-none}"
  echo "GEOPACT_B_GEO_FLOOR=${GEOPACT_B_GEO_FLOOR:-unset}"
  echo "GEOPACT_B_GEO_BETA=${GEOPACT_B_GEO_BETA:-unset}"
  echo "GEOPACT_ICKH_DENOM=${GEOPACT_ICKH_DENOM:-unset}"
  echo "GEOPACT_ZAFD_OMEGA_NORM=${GEOPACT_ZAFD_OMEGA_NORM:-unset}"
  echo "GEOPACT_ZAFD_GAMMA=${GEOPACT_ZAFD_GAMMA:-unset}"
  echo "BILATERAL_DECODE_PIN=${BILATERAL_DECODE_PIN:-1}"
  echo "LMMS_MODEL=${LMMS_MODEL}"
  echo "MODEL_DIR=${MODEL_DIR}"
  echo "MODEL_ARGS=${MODEL_ARGS}"
  echo "GEOPACT_MODULE_A_PREFETCH=${GEOPACT_MODULE_A_PREFETCH:-unset}"
  echo "GEOPACT_MODULE_A_CACHE=${GEOPACT_MODULE_A_CACHE:-unset}"
  echo "GEOPACT_MODULE_A_PIL_SHORT=${GEOPACT_MODULE_A_PIL_SHORT:-unset}"
  echo "GEOPACT_MODULE_A_WORKER_YOLO_DEVICE=${GEOPACT_MODULE_A_WORKER_YOLO_DEVICE:-unset}"
  echo "GEOPACT_MODULE_A_PREFETCH_ASYNC=${GEOPACT_MODULE_A_PREFETCH_ASYNC:-unset}"
  echo "GEOPACT_MODULE_A_SUBPROCESS=${GEOPACT_MODULE_A_SUBPROCESS:-unset}"
  echo "MIU_YOLO_DEVICE=${MIU_YOLO_DEVICE:-unset}"
} | tee "${OUT}/launcher.log"

# Module A worker prewarm runs inside Qwen3VLGui.__init__ (same process as lmms_eval).
if [[ "${SKIP_VERIFY:-0}" != "1" ]]; then
  "${EVAL}" "${REPO}/scripts/verify_dialogue_kv_early_stop.py"
fi

"$EVAL" -m lmms_eval \
  --model "${LMMS_MODEL}" \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASK" \
  --batch_size 1 \
  "${LIMIT_ARGS[@]}" \
  --log_samples \
  --output_path "${OUT}/results" \
  2>&1 | tee "${OUT}/run.log"

grep -E '^OVERALL ALL|^LATENCY_STEP|^LATENCY_EPISODE' "${OUT}/run.log" | tee "${OUT}/summary.txt" || true
echo "  dialogue_kv_hit: $(grep -c 'dialogue_kv_hit.*True' "${OUT}/run.log" 2>/dev/null || echo 0)"
echo "  pruning_ms>0:    $(grep -c 'pruning_ms.: [1-9]' "${OUT}/run.log" 2>/dev/null || echo 0)"
echo "  vision_cache_on: $(grep -c 'vision_cache_enabled.: True' "${OUT}/run.log" 2>/dev/null || echo 0)"

PROF_JSONL="${OUT}/results/performance_profiles/step_profiles.jsonl"
if [[ ! -f "$PROF_JSONL" ]]; then
  "$EVAL" scripts/verify_performance_profiles.py --run-dir "${OUT}/results" --min-steps "${LIMIT:-1}" || true
fi
if [[ -f "$PROF_JSONL" ]]; then
  "$EVAL" scripts/verify_performance_profiles.py --run-dir "${OUT}/results" --min-steps "${LIMIT:-1}" || true
fi

echo "DONE ${OUT}"
