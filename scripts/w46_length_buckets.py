#!/usr/bin/env python3
"""Episode-length-stratified Step SR margins.

Question. The paper motivates a trajectory-scale problem, but OmniGUI episodes
average 3.6 steps and a history-window scan shows flat accuracy from H=5 to
H=20. The long-horizon framing is therefore hard to test
on this data. This script asks: within the
frozen suites, does our margin over the baselines GROW with episode length? If
it does, the trajectory framing gains support from our own data; if it does not,
the framing must be narrowed to the token-budget axis and said so plainly.

Design. Pure re-analysis of already-harvested per-step dumps -- no GPU. Episode
length is the step count per (subset, episode_id); Step SR is the per-step
`exact_match` the main tables use. Bootstrap resamples EPISODES, not steps,
because steps within an episode are not independent under teacher forcing.

Usage (remote):
    python3 scripts/w46_length_buckets.py [root]
Default root logs/w45_ret2 (four selectors x two budgets x OmniGUI/AndroidControl).
Writes docs/w46_length_buckets.json.
"""
import glob
import json
import os
import random
import sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO)

ROOT = sys.argv[1] if len(sys.argv) > 1 else "logs/w45_ret2"
N_BOOT = 2000
SEED = 20260731


def find_samples(run_dir):
    cands = sorted(glob.glob(os.path.join(run_dir, "results", "**", "*samples*.jsonl"),
                             recursive=True))
    return cands[-1] if cands else None


def per_step(run_dir):
    """(episode_key, step_id) -> 0/1 correctness."""
    path = find_samples(run_dir)
    if not path:
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            se = rec.get("step_eval") or {}
            if isinstance(se, str):
                try:
                    se = json.loads(se)
                except Exception:
                    continue
            if not isinstance(se, dict) or "episode_id" not in se:
                continue
            ep = f"{se.get('subset', '?')}-{se['episode_id']}"
            try:
                st = int(se.get("step_id", -1))
            except (TypeError, ValueError):
                continue
            out[(ep, st)] = 1.0 if se.get("exact_match") else 0.0
    return out


def episode_lengths(step_map):
    lens = defaultdict(int)
    for ep, _ in step_map:
        lens[ep] += 1
    return lens


def bucket_of(n, edges):
    for i, e in enumerate(edges):
        if n <= e:
            return i
    return len(edges)


def paired_delta(ours, base, lens, edges, n_buckets):
    """Per-bucket (n_episodes, n_steps, ours_sr, base_sr, delta, ci95)."""
    common = sorted(set(ours) & set(base))
    by_ep = defaultdict(list)
    for key in common:
        by_ep[key[0]].append(key)
    eps = sorted(by_ep)
    rng = random.Random(SEED)
    res = []
    for b in range(n_buckets):
        b_eps = [e for e in eps if bucket_of(lens.get(e, 0), edges) == b]
        keys = [k for e in b_eps for k in by_ep[e]]
        if not keys:
            res.append(None)
            continue
        o = sum(ours[k] for k in keys) / len(keys) * 100.0
        c = sum(base[k] for k in keys) / len(keys) * 100.0
        boots = []
        for _ in range(N_BOOT):
            samp = [b_eps[rng.randrange(len(b_eps))] for _ in b_eps]
            kk = [k for e in samp for k in by_ep[e]]
            if not kk:
                continue
            boots.append(
                (sum(ours[k] for k in kk) - sum(base[k] for k in kk)) / len(kk) * 100.0
            )
        boots.sort()
        lo = boots[int(0.025 * len(boots))] if boots else float("nan")
        hi = boots[int(0.975 * len(boots))] if boots else float("nan")
        res.append({
            "bucket": b, "n_episodes": len(b_eps), "n_steps": len(keys),
            "ours_sr": round(o, 2), "base_sr": round(c, 2),
            "delta": round(o - c, 2),
            "ci95": [round(lo, 2), round(hi, 2)],
        })
    return res


def main():
    suites = {"omni": "omni", "ac": "ac"}
    budgets = ["c50_h10", "c25_h05"]
    baselines = ["uniform", "divprune", "cdpruner"]
    report = {
        "question": ("Does the Step SR margin over baselines grow with episode "
                     "length inside the frozen suites?"),
        "protocol": ("Pure re-analysis of per-step step_eval dumps. Episode length "
                     "= step count per (subset, episode_id). Bootstrap resamples "
                     "episodes (not steps) because teacher-forced steps within an "
                     "episode are dependent. Buckets are terciles of the episode "
                     "length distribution of the paired step set."),
        "root": ROOT, "n_boot": N_BOOT, "cells": [],
        "missing": [],
    }

    for suite in suites:
        for budget in budgets:
            ours_dir = os.path.join(ROOT, f"{suite}_ours_{budget}")
            ours = per_step(ours_dir)
            if not ours:
                report["missing"].append(ours_dir)
                continue
            lens = episode_lengths(ours)
            vals = sorted(lens.values())
            if len(vals) < 6:
                report["missing"].append(f"{ours_dir} (too few episodes)")
                continue
            e1 = vals[len(vals) // 3]
            e2 = vals[2 * len(vals) // 3]
            edges = sorted({e1, e2})
            for base_name in baselines:
                base_dir = os.path.join(ROOT, f"{suite}_{base_name}_{budget}")
                base = per_step(base_dir)
                if not base:
                    report["missing"].append(base_dir)
                    continue
                buckets = paired_delta(ours, base, lens, edges, len(edges) + 1)
                report["cells"].append({
                    "suite": suite, "budget": budget, "baseline": base_name,
                    "length_edges": edges,
                    "episode_len_mean": round(sum(vals) / len(vals), 2),
                    "episode_len_max": vals[-1],
                    "buckets": [b for b in buckets if b],
                })

    os.makedirs("docs", exist_ok=True)
    json.dump(report, open("docs/w46_length_buckets.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    for c in report["cells"]:
        print(f"\n{c['suite']} {c['budget']} vs {c['baseline']}  "
              f"edges={c['length_edges']} mean_len={c['episode_len_mean']} "
              f"max={c['episode_len_max']}")
        print(f"  {'bucket':<8}{'eps':>6}{'steps':>7}{'ours':>8}{'base':>8}"
              f"{'delta':>8}  ci95")
        for b in c["buckets"]:
            print(f"  {b['bucket']:<8}{b['n_episodes']:>6}{b['n_steps']:>7}"
                  f"{b['ours_sr']:>8}{b['base_sr']:>8}{b['delta']:>+8}  "
                  f"[{b['ci95'][0]}, {b['ci95'][1]}]")
    if report["missing"]:
        print("\nmissing run dirs:")
        for m in report["missing"]:
            print("  " + m)
    print("\nwrote docs/w46_length_buckets.json")


if __name__ == "__main__":
    main()
