"""Quick CUDA + flash_attn probe before GUI-Owl eval."""
from __future__ import annotations

import sys
import warnings

import torch

print("torch", torch.__version__)
print("cuda", torch.version.cuda)

driver_issue: str | None = None
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    cuda_ok = torch.cuda.is_available()
    if not cuda_ok:
        try:
            _ = torch.cuda.device_count()
        except Exception:
            pass

for w in caught:
    msg = str(w.message)
    if "NVIDIA driver" in msg or "CUDA driver" in msg:
        driver_issue = msg
        break

print("cuda_available", cuda_ok)

if not cuda_ok:
    if driver_issue:
        print("cuda_driver_issue", driver_issue, file=sys.stderr)
        print(
            "hint: PyTorch cu121 needs a newer NVIDIA driver than this node provides; "
            "switch to a GPU DSW instance with a matching driver.",
            file=sys.stderr,
        )
    else:
        print(
            "hint: no CUDA device visible — connect to a GPU DSW instance, not a CPU-only node.",
            file=sys.stderr,
        )

try:
    import flash_attn

    print("flash_attn", flash_attn.__version__)
except Exception as exc:
    print("flash_attn MISSING:", exc)
    sys.exit(1 if cuda_ok else 2)

sys.exit(0 if cuda_ok else 2)
