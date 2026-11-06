#!/usr/bin/env bash
# Portable TRACE eval environment bootstrap (Linux + CUDA).
#
#   conda create -n trace python=3.10 -y
#   conda activate trace
#   bash scripts/setup_eval_env.sh
#
# Override EVAL_PYTHON / REPO via the environment. This script does not
# assume any author-machine path.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=eval_env_defaults.sh
source "${_SCRIPT_DIR}/eval_env_defaults.sh"

PY="${EVAL_PYTHON}"
echo "EVAL_PYTHON=${PY}"
echo "REPO=${REPO}"

echo "[1] PyTorch 2.2.2+cu121 (skip if already present)"
if ! "${PY}" -c "import torch; assert torch.__version__.startswith('2.2')" 2>/dev/null; then
  "${PY}" -m pip install torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121
fi
"${PY}" -c "import torch; print('torch', torch.__version__, 'cuda', getattr(torch.version, 'cuda', None))"

echo "[2] transformers>=4.57 + qwen-vl-utils + repo requirements"
"${PY}" -m pip install "transformers>=4.57.0,<5.0" "qwen-vl-utils>=0.0.10"
"${PY}" -m pip install -r "${REPO}/requirements.txt"
"${PY}" -m pip install -e "${REPO}/lmms_eval"

echo "[3] FlashAttention 2.5.8 (CUDA only)"
if "${PY}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  bash "${_SCRIPT_DIR}/install_flash_attn_omini_eval.sh"
else
  echo "  skip flash-attn: no CUDA visible"
fi

echo "[4] Deploy TRACE modeling patches"
"${PY}" "${_SCRIPT_DIR}/restore_q3vl_geopact_patches.py"
if "${PY}" -c "from transformers import Qwen2_5_VLForConditionalGeneration" 2>/dev/null; then
  "${PY}" "${_SCRIPT_DIR}/restore_q25vl_geopact_patches.py" || true
fi

echo "[5] Verify imports"
"${PY}" -c "
import torch, transformers
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('transformers', transformers.__version__)
import lmms_eval
print('lmms_eval', getattr(lmms_eval, '__file__', 'ok'))
from qwen_vl_utils import process_vision_info
print('qwen_vl_utils: OK')
print('=== setup_eval_env: ALL CHECKS PASSED ===')
"
