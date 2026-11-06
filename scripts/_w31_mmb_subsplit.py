#!/usr/bin/env python3
"""W31: offline Basic/Adv. split for every landed MMBench-GUI L2 arm.

Walks logs/ for *samples_task_mmbench_gui_l2*.jsonl, keeps the newest file
per arm directory, and recomputes overall / per-grounding_type accuracy from
the mmbench_gui_l2_acc payload (hit, platform, grounding_type). Emits a TSV:
    arm_dir  n  overall  <type1>=acc1  <type2>=acc2 ...
so the local table generator can map Basic/Adv. columns; grounding_type
values are reported verbatim (no assumption on naming).
"""
import glob
import json
import os
import sys
from collections import defaultdict

ROOT = "logs"


def newest_per_arm():
    by_arm = {}
    pattern = os.path.join(ROOT, "**", "*samples_task_mmbench_gui_l2*.jsonl")
    for path in glob.iglob(pattern, recursive=True):
        # arm dir = path up to (and excluding) the trailing results/... tail
        arm = path.split("/results/")[0]
        mtime = os.path.getmtime(path)
        if arm not in by_arm or mtime > by_arm[arm][0]:
            by_arm[arm] = (mtime, path)
    return {arm: p for arm, (_, p) in by_arm.items()}


def score(path):
    tot, hit = 0, 0
    by_type = defaultdict(lambda: [0, 0])
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = rec.get("mmbench_gui_l2_acc")
            if payload is None:
                continue
            h = 1 if payload.get("hit") else 0
            gt = str(payload.get("grounding_type", "unknown")) or "unknown"
            tot += 1
            hit += h
            by_type[gt][0] += 1
            by_type[gt][1] += h
    return tot, hit, by_type


def main():
    arms = newest_per_arm()
    for arm in sorted(arms):
        tot, hit, by_type = score(arms[arm])
        if tot == 0:
            continue
        parts = [arm, str(tot), f"{100.0 * hit / tot:.2f}"]
        for gt in sorted(by_type):
            n, h = by_type[gt]
            parts.append(f"{gt}={100.0 * h / n:.2f}({n})")
        print("\t".join(parts))


if __name__ == "__main__":
    main()
