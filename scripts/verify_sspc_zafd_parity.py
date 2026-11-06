#!/usr/bin/env python3
"""Golden parity smoke for SSPC-ZAFD in-budget path (wrapper around verify_sspc)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    cmd = [sys.executable, str(REPO / "scripts" / "verify_sspc.py")]
    cmd.extend(sys.argv[1:])
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
