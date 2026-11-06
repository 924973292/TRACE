#!/usr/bin/env bash
# Portable TRACE eval paths. Override with environment variables:
#   REPO, EVAL_PYTHON, MODELS_DIR, OMNIPARSER_ICON_WEIGHTS
#
# Source from worker scripts:
#   source "$(dirname "${BASH_SOURCE[0]}")/eval_env_defaults.sh"

if [[ -n "${EVAL_ENV_DEFAULTS_LOADED:-}" ]]; then
  return 0 2>/dev/null || true
fi
export EVAL_ENV_DEFAULTS_LOADED=1

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_DEFAULT="$(cd "${_SCRIPT_DIR}/.." && pwd)"

export REPO="${REPO:-${_REPO_DEFAULT}}"
export EVAL_PYTHON="${EVAL_PYTHON:-${PYTHON:-python3}}"
export MODELS_DIR="${MODELS_DIR:-${REPO}/models}"
export OMNIPARSER_ICON_WEIGHTS="${OMNIPARSER_ICON_WEIGHTS:-${REPO}/models/omniparser/icon_detect/model.pt}"
export MIU_YOLO_WEIGHTS="${MIU_YOLO_WEIGHTS:-${OMNIPARSER_ICON_WEIGHTS}}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

if [[ "${EVAL_PYTHON}" == /* || "${EVAL_PYTHON}" == [A-Za-z]:* ]]; then
  if [[ ! -x "${EVAL_PYTHON}" && ! -f "${EVAL_PYTHON}" ]]; then
    echo "WARNING: EVAL_PYTHON not found: ${EVAL_PYTHON}" >&2
    echo "Set EVAL_PYTHON to the Python interpreter of your eval environment." >&2
  fi
fi
