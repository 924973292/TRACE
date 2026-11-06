#!/usr/bin/env python3
"""W31b: pooled (micro) Text/Icon accuracies for every landed ScreenSpot arm.

Walks logs/ for ScreenSpot-v2 and ScreenSpot-Pro samples files, keeps the
newest per arm dir, and computes micro accuracy overall and per ui/data type,
so the headline table's Text/Icon sub-columns share the official micro
convention (each cell = fraction of official docs correct on that subset,
sub-columns recombine exactly into the official Avg). Output TSV:
    bench  arm_dir  n  overall  text  icon
"""
import glob
import json
import os
from collections import defaultdict

BENCHES = {
    "SSV2": ("*samples_task_screenspot_v2_guiowl*.jsonl", "screenspot_v2_center_acc"),
    "SSPRO": ("*samples_task_screenspot_pro_guiowl*.jsonl", "screenspot_pro_center_acc"),
}


def newest_per_arm(pattern):
    by_arm = {}
    for path in glob.iglob(os.path.join("logs", "**", pattern), recursive=True):
        arm = path.split("/results/")[0]
        mtime = os.path.getmtime(path)
        if arm not in by_arm or mtime > by_arm[arm][0]:
            by_arm[arm] = (mtime, path)
    return {arm: p for arm, (_, p) in by_arm.items()}


def score(path, key):
    tot = hit = 0
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
            payload = rec.get(key)
            if not isinstance(payload, dict):
                continue
            h = 1 if payload.get("hit") else 0
            t = str(payload.get("ui_type") or payload.get("data_type") or "?").lower()
            t = "text" if "text" in t else ("icon" if "icon" in t else t)
            tot += 1
            hit += h
            by_type[t][0] += 1
            by_type[t][1] += h
    return tot, hit, by_type


def main():
    for bench, (pattern, key) in BENCHES.items():
        for arm, path in sorted(newest_per_arm(pattern).items()):
            tot, hit, by_type = score(path, key)
            if tot == 0:
                continue
            def acc(t):
                n, h = by_type.get(t, [0, 0])
                return f"{100.0 * h / n:.2f}" if n else "-"
            print("\t".join([bench, arm, str(tot), f"{100.0 * hit / tot:.2f}",
                             acc("text"), acc("icon")]))


if __name__ == "__main__":
    main()
