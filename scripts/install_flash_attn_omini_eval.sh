#!/usr/bin/env bash
# Install flash-attn into OminiEval for torch 2.2.x + cu12 (prebuilt wheel).
#
#   bash scripts/install_flash_attn_omini_eval.sh
#   FLASH_ATTN_GH_MIRROR=https://ghfast.top bash scripts/install_flash_attn_omini_eval.sh
#   FLASH_ATTN_WHEEL_FILE=/path/to/local.whl bash scripts/install_flash_attn_omini_eval.sh
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=eval_env_defaults.sh
source "${_SCRIPT_DIR}/eval_env_defaults.sh"

SITE_PKGS="$(dirname "$(dirname "${EVAL_PYTHON}")")/lib/python3.10/site-packages"

echo "OminiEval: ${EVAL_PYTHON}"
echo "site-packages: ${SITE_PKGS}"

"${EVAL_PYTHON}" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

if "${EVAL_PYTHON}" -c "import flash_attn" 2>/dev/null; then
  "${EVAL_PYTHON}" -c "import flash_attn; print('flash_attn already:', flash_attn.__version__)"
  exit 0
fi

# torch 2.2.2+cu121 -> cu122torch2.2 wheel (pip 要求 PEP 427 文件名)
WHEEL_CANON_NAME="flash_attn-2.5.8+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
WHEEL_REL="/Dao-AILab/flash-attention/releases/download/v2.5.8/${WHEEL_CANON_NAME}"
CACHE_DIR="${FLASH_ATTN_CACHE_DIR:-${REPO}/.cache}"
WHEEL_FILE="${FLASH_ATTN_WHEEL_FILE:-${CACHE_DIR}/${WHEEL_CANON_NAME}}"
mkdir -p "$(dirname "${WHEEL_FILE}")"

# 手动上传时可能用了短文件名，复制为 pip 可识别的 canonical 名
_short="${CACHE_DIR}/flash_attn_torch22_cu122_cp310.whl"
if [[ ! -s "${WHEEL_FILE}" && -s "${_short}" ]]; then
  echo "Using uploaded ${_short} -> ${WHEEL_FILE}"
  cp -f "${_short}" "${WHEEL_FILE}"
fi

_download_with_curl() {
  local url="$1"
  echo "  try: ${url}"
  if curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 --max-time 7200 \
    -o "${WHEEL_FILE}.part" "${url}"; then
    mv -f "${WHEEL_FILE}.part" "${WHEEL_FILE}"
    return 0
  fi
  rm -f "${WHEEL_FILE}.part"
  return 1
}

if [[ ! -s "${WHEEL_FILE}" ]]; then
  echo "Downloading flash-attn wheel (~116MB) ..."
  echo "  (GitHub 直连很慢；自动尝试国内 gh 加速镜像)"
  mirrors=()
  if [[ -n "${FLASH_ATTN_GH_MIRROR:-}" ]]; then
    mirrors+=("${FLASH_ATTN_GH_MIRROR}")
  fi
  mirrors+=(
    "https://ghfast.top/https://github.com"
    "https://ghproxy.net/https://github.com"
    "https://mirror.ghproxy.com/https://github.com"
    "https://gh-proxy.com/https://github.com"
    "https://github.com"
  )
  ok=0
  for base in "${mirrors[@]}"; do
    if _download_with_curl "${base}${WHEEL_REL}"; then
      ok=1
      break
    fi
    echo "  WARN: failed, next mirror ..."
  done
  if [[ "${ok}" -eq 0 ]]; then
    echo "ERROR: all mirrors failed." >&2
    echo "  1) 本机下载后 scp 到 ${WHEEL_FILE}" >&2
    echo "  2) 或指定: FLASH_ATTN_GH_MIRROR=https://ghfast.top bash $0" >&2
    exit 1
  fi
  ls -lh "${WHEEL_FILE}"
fi

echo "pip install from ${WHEEL_FILE} (--no-deps; torch 已在 OminiEval)"
python3.10 -m pip install --no-cache-dir --no-deps --target="${SITE_PKGS}" "${WHEEL_FILE}"

"${EVAL_PYTHON}" -c "import flash_attn; print('flash_attn OK:', flash_attn.__version__)"
