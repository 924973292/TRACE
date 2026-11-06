#!/usr/bin/env python3
"""Verify Tier1 cross-benchmark smoke outputs (limit=100 per dataset)."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BENCHMARKS: dict[str, dict] = {
    "omnigui": {
        "dir": "omnigui",
        "metrics": ("gp", "tm", "em"),
        "log_overall": True,
        "min_samples": 1,
    },
    "screenspot_v2": {
        "dir": "screenspot_v2",
        "result_keys": ("screenspot_v2_center_acc",),
        "min_samples": 1,
    },
    "screenspot_pro": {
        "dir": "screenspot_pro",
        "result_keys": ("screenspot_pro_center_acc",),
        "min_samples": 1,
    },
    "mmbench_l1": {
        "dir": "mmbench_l1",
        "result_keys": ("mmbench_gui_l1_acc",),
        "min_samples": 1,
    },
    "mmbench_l2": {
        "dir": "mmbench_l2",
        "result_keys": ("mmbench_gui_l2_acc",),
        "min_samples": 1,
    },
}

FAIL_PATTERNS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "RuntimeError:",
    "ModuleNotFoundError:",
)


def _check(name: str, ok: bool, detail: str = "") -> tuple[str, bool, str]:
    mark = "OK" if ok else "FAIL"
    msg = f"[{mark}] {name}" + (f" — {detail}" if detail else "")
    print(msg)
    return name, ok, detail


def _read_log(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _count_samples(out_dir: Path) -> int:
    n = 0
    for p in out_dir.rglob("samples_*.jsonl"):
        try:
            n += sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
        except OSError:
            continue
    return n


def _find_results_json(out_dir: Path) -> Path | None:
    candidates = sorted(out_dir.rglob("*results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _metric_finite(val: object, *, pct_scale: bool = False) -> bool:
    if val is None:
        return False
    if isinstance(val, (int, float)):
        f = float(val)
        if not math.isfinite(f):
            return False
        if pct_scale:
            return 0.0 <= f <= 100.0
        if f > 1.0:
            f /= 100.0
        return 0.0 <= f <= 1.0
    if isinstance(val, str):
        try:
            s = val.strip()
            if s.endswith("%"):
                f = float(s[:-1])
                return math.isfinite(f) and 0.0 <= f <= 100.0
            f = float(s)
            if pct_scale:
                return math.isfinite(f) and 0.0 <= f <= 100.0
            if f > 1.0:
                f /= 100.0
            return math.isfinite(f) and 0.0 <= f <= 1.0
        except ValueError:
            return False
    return False


def _parse_overall(log_text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in log_text.splitlines():
        if not line.startswith("OVERALL ALL"):
            continue
        for m in re.finditer(r"(\w+)=([\d.]+)%", line):
            out[m.group(1).lower()] = float(m.group(2))
    return out


def verify_benchmark(root: Path, spec: dict, *, expect_limit: int) -> list[tuple[str, bool, str]]:
    out_dir = root / spec["dir"]
    checks: list[tuple[str, bool, str]] = []
    name = spec["dir"]
    log_path = out_dir / "run.log"
    log_text = _read_log(log_path)

    checks.append(_check(f"{name}/run.log", log_path.is_file(), str(log_path)))

    err_hits = [p for p in FAIL_PATTERNS if p in log_text]
    checks.append(_check(f"{name}/no_crash", not err_hits, ", ".join(err_hits) or "clean"))

    n_samples = _count_samples(out_dir)
    checks.append(
        _check(
            f"{name}/samples",
            n_samples >= spec.get("min_samples", 1),
            f"count={n_samples} (expect ~{expect_limit})",
        )
    )

    missing_img = len(re.findall(r"missing image", log_text, re.I))
    checks.append(
        _check(
            f"{name}/images",
            missing_img == 0 or missing_img < max(1, expect_limit // 20),
            f"missing_image_mentions={missing_img}",
        )
    )

    if spec.get("log_overall"):
        overall = _parse_overall(log_text)
        for key in spec.get("metrics", ()):
            checks.append(
                _check(
                    f"{name}/{key}",
                    key in overall and _metric_finite(overall[key], pct_scale=True),
                    f"{overall.get(key, 'n/a')}",
                )
            )

    results_path = _find_results_json(out_dir)
    if results_path:
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            checks.append(_check(f"{name}/results.json", False, str(exc)))
            data = {}
        if isinstance(data, dict) and "results" in data:
            data = data["results"]
        for key in spec.get("result_keys", ()):
            val = data.get(key) if isinstance(data, dict) else None
            if isinstance(val, dict):
                val = val.get(",none") or val.get("none") or next(iter(val.values()), None)
            checks.append(
                _check(
                    f"{name}/{key}",
                    _metric_finite(val),
                    str(val),
                )
            )
    elif not spec.get("log_overall"):
        checks.append(_check(f"{name}/results.json", False, "not found"))

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="outputs/tier1_smoke_limit100",
        help="smoke output root (contains omnigui/, screenspot_v2/, ...)",
    )
    parser.add_argument("--expect-limit", type=int, default=100)
    args = parser.parse_args()

    root = (REPO / args.root).resolve()
    if not root.is_dir():
        print(f"smoke root missing: {root}", file=sys.stderr)
        return 1

    all_checks: list[tuple[str, bool, str]] = []
    for _key, spec in BENCHMARKS.items():
        all_checks.extend(verify_benchmark(root, spec, expect_limit=args.expect_limit))

    failed = [c for c in all_checks if not c[1]]
    if failed:
        print(f"\nverify_tier1_smoke_outputs FAILED ({len(failed)} checks)", file=sys.stderr)
        return 1
    print("\nverify_tier1_smoke_outputs OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
