#!/usr/bin/env python3
"""Generate cross_dataset_abc_grid.tsv for 4B sliding-rewrite fleet supervisors."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "scripts" / "cross_dataset_abc_grid.tsv"

ABC_ARM = "18_prod_drop_scale_a02_sspc_zafd"
RATIOS = (
    ("80", "0.80"),
    ("40", "0.40"),
    ("20", "0.20"),
    ("10", "0.10"),
    ("05", "0.05"),
)

# Multi-step: hist9 + prune_target=history (sliding rewrite on 19144 / 14687).
OMNIGUI_PORT = 19144
ANDROID_PORT = 14687
SINGLE_PORT = 14075

SINGLE = (
    ("screenspot_v2", "task_screenspot_v2_guiowl"),
    ("screenspot_pro", "task_screenspot_pro_guiowl"),
    ("mmbench_l1", "task_mmbench_gui_l1_guiowl"),
    ("mmbench_l2", "task_mmbench_gui_l2_guiowl"),
)


def main() -> int:
    lines = ["tier\tbenchmark\ttask\ttrack\texp_arm\tratio_pct\tratio_val\twave\tport\tgpu_slot"]
    wave = 1

    # OmniGUI main + ratio sweep on 19144 (no baseline_full).
    for pct, val in RATIOS:
        track = "abc_r40_sliding" if pct == "40" else f"abc_r{pct}"
        lines.append(
            f"1\tomnigui\ttask_v2_qwen3_omni_novideo\t{track}\t{ABC_ARM}\t{pct}\t{val}\t{wave}\t{OMNIGUI_PORT}\t-"
        )
    wave += 1

    # AndroidControl ABC r40 on 14687.
    lines.append(
        f"1\tandroid_control\ttask_android_control_qwen3_omni\tabc_r40\t{ABC_ARM}\t40\t0.40\t{wave}\t{ANDROID_PORT}\t-"
    )
    wave += 1

    # OmniGUI baseline_kv GP anchor on 14075 only.
    lines.append(
        f"1\tomnigui\ttask_v2_qwen3_omni_novideo\tbaseline_kv\t01_dialogue_kv_noprune\t-\t-\t{wave}\t{SINGLE_PORT}\t-"
    )
    wave += 1

    # Single-step benchmarks: abc_r40 on 14075.
    for benchmark, task in SINGLE:
        lines.append(
            f"2\t{benchmark}\t{task}\tabc_r40\t{ABC_ARM}\t40\t0.40\t{wave}\t{SINGLE_PORT}\t-"
        )

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(lines) - 1} jobs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
