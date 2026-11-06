#!/usr/bin/env bash
# Fail fast before lmms_eval when CUDA / flash_attn are unavailable.
#
#   bash scripts/check_eval_cuda.sh
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=eval_env_defaults.sh
source "${_SCRIPT_DIR}/eval_env_defaults.sh"

EVAL="${EVAL_PYTHON}"

if "${EVAL}" -c "
import sys
import warnings
import torch

driver_issue = None
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    cuda_ok = torch.cuda.is_available()
    if not cuda_ok:
        try:
            _ = torch.cuda.device_count()
        except Exception:
            pass

for w in caught:
    msg = str(w.message)
    if 'NVIDIA driver' in msg or 'CUDA driver' in msg:
        driver_issue = msg
        break

if not cuda_ok:
    print('CUDA unavailable (torch.cuda.is_available() == False)', file=sys.stderr)
    print('hostname:', __import__('socket').gethostname(), file=sys.stderr)
    if driver_issue:
        print('driver_mismatch:', driver_issue, file=sys.stderr)
        print('GPU driver too old for torch cu121 — switch DSW GPU instance.', file=sys.stderr)
    else:
        print('Likely CPU-only node — reconnect to a GPU DSW instance.', file=sys.stderr)
    sys.exit(1)

try:
    import flash_attn  # noqa: F401
except Exception as exc:
    print(f'flash_attn import failed: {exc}', file=sys.stderr)
    sys.exit(2)

print('cuda_ok', torch.cuda.get_device_name(0))
print('flash_attn_ok')
"; then
  exit 0
fi

rc=$?
echo "" >&2
echo "ERROR: GUI-Owl eval requires GPU + flash_attn (model_args use attn_implementation=flash_attention_2)." >&2
echo "  cluster=${EVAL_CLUSTER} host=$(hostname 2>/dev/null || echo unknown)" >&2
echo "  python=${EVAL_PYTHON}" >&2
echo "" >&2
echo "On DSW: SSH to a GPU instance where this passes:" >&2
echo "  ${EVAL_PYTHON} scripts/_check_torch_flash.py" >&2
echo "  bash scripts/check_eval_cuda.sh" >&2
echo "" >&2
echo "If cuda_available is False, switch DSW port / notebook (CPU-only nodes cannot run this eval)." >&2
exit "${rc}"
