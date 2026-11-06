#!/usr/bin/env python3
"""Verify origpos MRoPE prune chain (Qwen2.5-Omni-style) before full ablation."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Broken MRoPE chain signature on hist_k5 full run
BROKEN_EM_FULL = 0.006
R99_MAX_DROP_PP = 8.0
R05_MIN_EM_ABS = 0.03
R05_MIN_EM_FRAC_OF_BASE = 0.08


def parse_metrics(log_path: Path) -> dict[str, float | None]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, float | None] = {"tm": None, "em": None}
    matches = re.findall(
        r"^OVERALL ALL TM=([\d.]+)% EM=([\d.]+)%",
        text,
        flags=re.MULTILINE,
    )
    if matches:
        tm_s, em_s = matches[-1]
        out["tm"] = float(tm_s) / 100.0
        out["em"] = float(em_s) / 100.0
    out["origpos_lines"] = text.count("GeoPACT origpos")
    out["source_compact_lines"] = text.count("GeoPACT source-compact:")
    out["text_only_rope_fail"] = "text-only RoPE" in text or "compact MRoPE must come from the hook" in text
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    tags = ("baseline", "random_r99", "random_r05")
    metrics: dict[str, dict] = {}
    for tag in tags:
        log_path = args.root / tag / "run.log"
        if not log_path.is_file():
            print(f"FAIL: missing {log_path}", file=sys.stderr)
            return 1
        metrics[tag] = parse_metrics(log_path)

    base = metrics["baseline"]
    r99 = metrics["random_r99"]
    r05 = metrics["random_r05"]

    if base["em"] is None or r99["em"] is None or r05["em"] is None:
        print("FAIL: could not parse OVERALL ALL from one or more runs", file=sys.stderr)
        return 1

    print(f"baseline  TM={base['tm']:.2%} EM={base['em']:.2%}")
    print(f"random99  TM={r99['tm']:.2%} EM={r99['em']:.2%}  delta={100*(r99['em']-base['em']):+.1f}pp")
    print(f"random05  TM={r05['tm']:.2%} EM={r05['em']:.2%}  delta={100*(r05['em']-base['em']):+.1f}pp")
    print(f"source-compact log lines: base={base['source_compact_lines']} r99={r99['source_compact_lines']} r05={r05['source_compact_lines']}")

    errors: list[str] = []

    for tag, m in metrics.items():
        if m.get("text_only_rope_fail"):
            errors.append(f"{tag}: text-only RoPE fallback detected in log")

    drop_pp = 100.0 * (base["em"] - r99["em"])
    if drop_pp > R99_MAX_DROP_PP:
        errors.append(
            f"random_r99 EM dropped {drop_pp:.1f}pp vs baseline (max {R99_MAX_DROP_PP}pp) — origpos MRoPE likely broken"
        )

    if r99["em"] <= BROKEN_EM_FULL * 2:
        errors.append(f"random_r99 EM={r99['em']:.2%} near broken-chain floor (~0.6%)")

    r05_min = max(R05_MIN_EM_ABS, base["em"] * R05_MIN_EM_FRAC_OF_BASE)
    if r05["em"] < r05_min:
        errors.append(
            f"random_r05 EM={r05['em']:.2%} < threshold {r05_min:.2%} "
            f"(abs>={R05_MIN_EM_ABS:.0%} or >={R05_MIN_EM_FRAC_OF_BASE:.0%} of baseline) — prune chain collapsed"
        )

    if r05["em"] <= BROKEN_EM_FULL * 2:
        errors.append(f"random_r05 EM={r05['em']:.2%} near broken-chain floor (~0.6%)")

    if errors:
        print("FAIL:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(
        f"PASS: origpos prune sanity (limit={args.limit}; "
        f"r99 within {R99_MAX_DROP_PP}pp; r05 EM>={r05_min:.2%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
