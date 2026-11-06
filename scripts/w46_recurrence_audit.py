#!/usr/bin/env python
"""W46 P0 data audit: episode length and within-goal screen-recurrence
distance distribution over the evaluated multi-step datasets.

Purpose (pre-registered before any long-horizon claim is written): the
lifecycle argument says evidence admitted at step t is consumed at a later
step t' of the SAME goal. The honest scope of that claim is bounded by how
far apart same-screen pairs actually are in the evaluated trajectories.
This audit measures, per dataset:
  - episode length histogram (steps per episode),
  - same-screen (t, t') pair counts by step distance delta = t' - t,
    using the same content-identity rule as the W43-A retention probe
    (exact file md5, or 64-bit dhash within 4 bits, size-matched),
  - the fraction of pairs with delta > 1 / > 3 / > 5.

Input: one prune-diag dump JSONL per dataset (any arm; frames and images
are arm-invariant). Only `is_current` records are used.

Usage (remote):
    "${EVAL_PYTHON}" scripts/w46_recurrence_audit.py \
        omnigui=logs/w45_ret2/omni_ours_c50_h10.jsonl \
        android_control=logs/w45_ret2/ac_ours_c50_h10.jsonl

Writes/merges docs/w46_recurrence_audit.json.
"""
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from w43_same_goal_retention import (  # noqa: E402
    _same_screen,
    episode_frames,
)


def audit(dump_path, max_delta_cap=40):
    episodes = episode_frames(dump_path)
    ep_lens = Counter()
    pair_delta = Counter()
    pair_kind = Counter()
    episodes_with_pairs = 0
    for ep, recs in episodes.items():
        ep_lens[len(recs)] += 1
        found = False
        for i in range(len(recs)):
            pi = recs[i].get("image_path") or ""
            if not pi:
                continue
            for j in range(i + 1, len(recs)):
                pj = recs[j].get("image_path") or ""
                if not pj:
                    continue
                kind = _same_screen(pi, pj)
                if kind is None:
                    continue
                delta = j - i
                pair_delta[min(delta, max_delta_cap)] += 1
                pair_kind[kind] += 1
                found = True
        if found:
            episodes_with_pairs += 1

    n_pairs = sum(pair_delta.values())
    n_eps = sum(ep_lens.values())

    def frac_gt(th):
        return (
            sum(v for d, v in pair_delta.items() if d > th) / n_pairs
            if n_pairs
            else 0.0
        )

    return {
        "episodes": n_eps,
        "steps_total": sum(k * v for k, v in ep_lens.items()),
        "episode_len_hist": {str(k): v for k, v in sorted(ep_lens.items())},
        "episode_len_mean": (
            sum(k * v for k, v in ep_lens.items()) / n_eps if n_eps else 0.0
        ),
        "episode_len_max": max(ep_lens) if ep_lens else 0,
        "same_screen_pairs": n_pairs,
        "pairs_exact": pair_kind.get("exact", 0),
        "pairs_near": pair_kind.get("near", 0),
        "episodes_with_pairs": episodes_with_pairs,
        "pair_delta_hist": {str(k): v for k, v in sorted(pair_delta.items())},
        "frac_delta_gt1": frac_gt(1),
        "frac_delta_gt3": frac_gt(3),
        "frac_delta_gt5": frac_gt(5),
        "dump": dump_path,
    }


def main():
    out_path = os.path.join(HERE, "..", "docs", "w46_recurrence_audit.json")
    out_path = os.path.abspath(out_path)
    merged = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            merged = json.load(f)
    for spec in sys.argv[1:]:
        name, _, dump = spec.partition("=")
        if not dump:
            raise SystemExit(f"bad arg (want name=dump.jsonl): {spec}")
        print(f"[audit] {name}: {dump}", flush=True)
        merged[name] = audit(dump)
        print(json.dumps({k: v for k, v in merged[name].items()
                          if k != "episode_len_hist" and k != "pair_delta_hist"},
                         indent=2), flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2, sort_keys=True)
    os.replace(tmp, out_path)
    print(f"[audit] wrote {out_path}")


if __name__ == "__main__":
    main()
