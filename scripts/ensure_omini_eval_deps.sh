#!/usr/bin/env bash
# Install missing pip packages into OminiEval site-packages (DSW partial venv copy).
#
#   bash scripts/ensure_omini_eval_deps.sh
#   bash scripts/ensure_omini_eval_deps.sh --check-only
#
# Always uses EVAL_PYTHON from eval_env_defaults.sh (DSW -> ${EVAL_PYTHON:-python3}).
# 禁止对 datasets 等做全量 pip（会升级 huggingface-hub 1.x 破坏 transformers）。
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=eval_env_defaults.sh
source "${_SCRIPT_DIR}/eval_env_defaults.sh"

CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1
fi

echo "OminiEval python: ${EVAL_PYTHON}"

_resolve_site_pkgs() {
  local exe="${EVAL_PYTHON}"
  local site
  site="$(dirname "$(dirname "${exe}")")/lib/python3.10/site-packages"
  if [[ -d "${site}" ]]; then
    echo "${site}"
    return 0
  fi
  site="$(dirname "$(dirname "$(readlink -f "${exe}" 2>/dev/null || echo "${exe}")")")/lib/python3.10/site-packages"
  if [[ -d "${site}" ]]; then
    echo "${site}"
    return 0
  fi
  return 1
}

_index_host() {
  local url="$1"
  url="${url#https://}"
  url="${url#http://}"
  echo "${url%%/*}"
}

_DEFAULT_INDEXES=(
  "https://mirrors.aliyun.com/pypi/simple/"
  "https://pypi.mirrors.ustc.edu.cn/simple/"
  "https://mirrors.cloud.tencent.com/pypi/simple/"
  "https://pypi.tuna.tsinghua.edu.cn/simple"
  "https://pypi.org/simple"
)

_pip_install_one() {
  local pkg="$1"
  local indexes=()
  if [[ -n "${PIP_INDEX_URL:-}" ]]; then
    indexes+=("${PIP_INDEX_URL}")
  fi
  local idx
  for idx in "${_DEFAULT_INDEXES[@]}"; do
    local seen=0
    for existing in "${indexes[@]:-}"; do
      if [[ "${existing}" == "${idx}" ]]; then
        seen=1
        break
      fi
    done
    if [[ "${seen}" -eq 0 ]]; then
      indexes+=("${idx}")
    fi
  done

  local host
  local args
  for idx in "${indexes[@]}"; do
    host="$(_index_host "${idx}")"
    echo "    index: ${idx}"
    args=(
      python3.10 -m pip install
      --no-cache-dir
      --no-deps
      --target="${SITE_PKGS}"
      --progress-bar on
      --index-url "${idx}"
      --trusted-host "${host}"
    )
    if "${args[@]}" "${pkg}"; then
      return 0
    fi
    echo "    WARN: failed on ${idx}, trying next mirror ..."
  done
  return 1
}

SITE_PKGS="$(_resolve_site_pkgs)" || {
  echo "ERROR: cannot find site-packages for EVAL_PYTHON=${EVAL_PYTHON}" >&2
  exit 1
}

# import_name:pip_name
REQUIRED_SPECS=(
  safetensors
  accelerate
  loguru
  einops
  pytz
  dateutil:python-dateutil
  six
  pandas
  tenacity
  huggingface_hub
  datasets
  yaml:pyyaml
  typing_inspection:typing-inspection
  annotated_types:annotated-types
  pydantic_core:pydantic-core==2.46.4
  pydantic:pydantic==2.13.4
  distro
  jiter
  sniffio
  httpcore
  httpx
  openai
)

# Extra --no-deps wheels needed for datasets (never pull huggingface-hub>=1.0).
_DATASETS_RUNTIME_PIP=(
  pyarrow
  dill
  multiprocess
  xxhash
  pyarrow-hotfix
)

missing_imports=()
missing_pip=()
for spec in "${REQUIRED_SPECS[@]}"; do
  mod="${spec%%:*}"
  pip_pkg="${spec#*:}"
  if [[ "${pip_pkg}" == "${spec}" ]]; then
    pip_pkg="${mod}"
  fi
  if ! "${EVAL_PYTHON}" -c "import ${mod}" 2>/dev/null; then
    missing_imports+=("${mod}")
    missing_pip+=("${pip_pkg}")
  fi
done

if [[ ${#missing_imports[@]} -eq 0 ]]; then
  echo "OminiEval deps: OK"
  "${EVAL_PYTHON}" -c "import sys; print('  executable:', sys.executable)"
  "${EVAL_PYTHON}" -c "import lmms_eval; print('  lmms_eval: OK')" 2>/dev/null || {
    echo "WARN: lmms_eval import failed — run: bash scripts/ensure_omini_eval_deps.sh" >&2
    exit 1
  }
  exit 0
fi

echo "OminiEval missing imports: ${missing_imports[*]}"
echo "  site-packages: ${SITE_PKGS}"

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo "Run: bash scripts/ensure_omini_eval_deps.sh" >&2
  exit 1
fi

if ! command -v python3.10 >/dev/null 2>&1; then
  echo "ERROR: system python3.10 not found for pip --target install" >&2
  exit 1
fi

if [[ -n "${PIP_INDEX_URL:-}" ]]; then
  echo "  PIP_INDEX_URL override: ${PIP_INDEX_URL}"
else
  echo "  pip: auto-try aliyun / ustc / tencent / tuna / pypi.org (--no-deps only)"
fi

echo "Installing into ${SITE_PKGS} ..."
echo ""

_install_list() {
  local pkg="$1"
  echo ">>> pip install ${pkg} (--no-deps) ..."
  if ! _pip_install_one "${pkg}"; then
    echo "ERROR: could not install ${pkg}" >&2
    exit 1
  fi
}

_install_pydantic_stack() {
  echo ">>> pydantic stack (pinned; replace pydantic_core tree) ..."
  rm -rf "${SITE_PKGS}/pydantic_core" "${SITE_PKGS}"/pydantic_core-*.dist-info
  _install_list "annotated-types"
  _install_list "typing-inspection"
  _install_list "pydantic-core==2.46.4"
  _install_list "pydantic==2.13.4"
}

for i in "${!missing_pip[@]}"; do
  pkg="${missing_pip[$i]}"
  mod="${missing_imports[$i]}"
  if [[ "${mod}" == "datasets" ]]; then
    _install_list "datasets==2.21.0"
    for dep in "${_DATASETS_RUNTIME_PIP[@]}"; do
      _install_list "${dep}"
    done
  elif [[ "${mod}" == "annotated_types" || "${mod}" == "typing_inspection" || "${mod}" == "pydantic_core" || "${mod}" == "pydantic" ]]; then
    : # installed via _install_pydantic_stack below
  else
    _install_list "${pkg}"
  fi
done

if printf '%s\n' "${missing_imports[@]}" | grep -qE '^(annotated_types|typing_inspection|pydantic_core|pydantic)$'; then
  _install_pydantic_stack
fi

for mod in "${missing_imports[@]}"; do
  if ! "${EVAL_PYTHON}" -c "import ${mod}" 2>/dev/null; then
    echo "ERROR: import ${mod} still fails after pip install" >&2
    echo "  If huggingface-hub/transformers conflict: bash scripts/repair_omini_eval_pip_damage.sh" >&2
    exit 1
  fi
  echo "    ${mod}: OK"
done

# Guard against broken pip upgrades from prior manual installs.
if [[ -d "${SITE_PKGS}/huggingface_hub-1.19.0.dist-info" ]] || compgen -G "${SITE_PKGS}/numpy-2.*.dist-info" >/dev/null; then
  echo "WARN: detected pip damage (hf-hub 1.x or numpy 2.x) — running repair ..."
  bash "${_SCRIPT_DIR}/repair_omini_eval_pip_damage.sh"
fi

echo ""
echo "OminiEval deps: installed ${missing_pip[*]}"
"${EVAL_PYTHON}" -c "import sys; print('  executable:', sys.executable)"
echo ">>> lmms_eval import smoke ..."
"${EVAL_PYTHON}" -c "import lmms_eval; print('  lmms_eval: OK')"
