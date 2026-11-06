#!/usr/bin/env bash
# Shared env for GeoPACT full eval (YOLO / OmniParser Module A). Source from worker scripts.
# Do NOT enable debug/viz flags here — full fleet runs headless eval only.

_GEO_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=eval_env_defaults.sh
source "${_GEO_ENV_DIR}/eval_env_defaults.sh"

export MIU_DISABLE_OCR="${MIU_DISABLE_OCR:-1}"
export LMMS_LIGHT_LOG="${LMMS_LIGHT_LOG:-1}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

# Module A: in-process YOLO only (no silent subprocess / Canny fallback).
# OMNIPARSER_ICON_WEIGHTS / MIU_YOLO_WEIGHTS set by eval_env_defaults.sh
export MIU_YOLO_DEVICE="${MIU_YOLO_DEVICE:-0}"
export MIU_DISABLE_OMNIPARSER_SUBPROC_FALLBACK="${MIU_DISABLE_OMNIPARSER_SUBPROC_FALLBACK:-1}"

# Pre-generate compact (7B-style) before generate; set to in_forward to use legacy hook-only path.
export GEOPACT_COMPACT_BACKEND="${GEOPACT_COMPACT_BACKEND:-source}"

# Session eval: episode-ordered dialogue LLM KV reuse (default on).
export GEOPACT_SESSION_KV="${GEOPACT_SESSION_KV:-1}"
export GEOPACT_SESSION_KV_MODE="${GEOPACT_SESSION_KV_MODE:-dialogue}"
export GEOPACT_FORCE_EPISODE_ORDER="${GEOPACT_FORCE_EPISODE_ORDER:-1}"

# Per-image ViT+projector output cache (off when dialogue KV is primary).
export GEOPACT_VISION_EMBED_CACHE="${GEOPACT_VISION_EMBED_CACHE:-0}"

# Module C: metadata-only coord (no token splice; preserves exact origpos MRoPE).
export GEOPACT_COORD_METADATA_ONLY="${GEOPACT_COORD_METADATA_ONLY:-1}"
export OMNIGUI_COORD_METADATA_ONLY="${OMNIGUI_COORD_METADATA_ONLY:-${GEOPACT_COORD_METADATA_ONLY}}"

# Per-frame grid rewrite after prune (compact token count contract for MRoPE fallback).
export GEOPACT_REWRITE_GRID="${GEOPACT_REWRITE_GRID:-1}"

# Decode: early-stop on complete Omni action; fail loud if dialogue-KV prompt_len is wrong.
export GEOPACT_STRICT_EARLY_STOP="${GEOPACT_STRICT_EARLY_STOP:-1}"

unset GUIOWL_DEBUG_MESSAGES GUIOWL_DEBUG_VISION GEOPACT_ALLOW_STRIDE_FALLBACK 2>/dev/null || true
