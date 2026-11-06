#!/usr/bin/env python
"""W43-A: same-goal future-target retention probe (protocol implementation).

Direct test of the goal--trajectory asymmetry that the instruction-swap
stress test only upper-bounds: within ONE episode (one global goal), does
the write-time keep frozen at step t still witness the target that a LATER
step t' > t of the same goal acts on, when that later target is registerable
on the step-t screen?

Registration rule (natural pairs, no synthetic forks): a future step t' is
mapped back to frame t iff both steps show the same screen. OmniGUI and
AndroidControl save one uniquely named screenshot per step, so identity is
established by CONTENT, not filename: exact duplicates (equal file md5) or
near duplicates (64-bit difference hash within 4 bits, which tolerates
clock/status-bar pixels but not scrolls or layout changes). Pairs whose two
images differ in pixel size are rejected, because the t' ground-truth rect
must be transplanted onto frame t's grid. The probe then asks whether the
*history* keep committed at t (the k_h prefix, which is what a reusable
cache actually serves at t') contains a patch overlapping the ground-truth
rect of the t' action. Exact and near pair counts are reported separately;
retention uses both.

Arms are compared on the SAME (t, t') pair set; the difference is purely
which evidence each selector committed at frame entry. Episode-cluster
percentile bootstrap gives the interval.

This script produces NO paper claim by itself; results enter the paper only
after a real run, with the pair count and screen-repeat coverage reported.

Usage (remote, after sourcing scripts/eval_env_defaults.sh):
    "${EVAL_PYTHON}" scripts/w43_same_goal_retention.py \
        <tag> <arm1>=<dump1.jsonl>:<run_dir1> [<arm2>=... ...] [--H 5]

Example:
    "${EVAL_PYTHON}" scripts/w43_same_goal_retention.py omni_c50_h10 \
        ours=logs/w40/omni_ours.jsonl:logs/w40/omni_ours_run \
        uniform=logs/w40/omni_unif.jsonl:logs/w40/omni_unif_run

Writes/merges docs/w43_same_goal_retention.json:
    {tag: {"pairs": int, "episodes": int, "screen_repeat_rate": float,
           arms: {name: {"retention": pct, "ci95_episode": [lo, hi],
                          "n_pairs": int}},
           "paired_diffs": {"a_minus_b": {"diff": pct, "ci95": [lo, hi]}}}}
"""
import json
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _w41_commit_conditional import (analyse_dump, find_samples,  # noqa: E402
                                     gt_rect_from_target, img_size,
                                     load_jsonl, rect_to_patches)

N_BOOT = 10000
SEED = 20260729


def episode_frames(dump_path):
    """episode_id -> ordered list of is_current dump records."""
    episodes = defaultdict(list)
    for rec in load_jsonl(dump_path):
        if rec.get("is_current"):
            episodes[rec["episode_key"].split("-", 1)[-1]].append(rec)
    return episodes


_MD5_CACHE = {}
_DHASH_CACHE = {}


def _file_md5(path):
    if path not in _MD5_CACHE:
        try:
            import hashlib
            _MD5_CACHE[path] = hashlib.md5(open(path, "rb").read()).hexdigest()
        except Exception:
            _MD5_CACHE[path] = None
    return _MD5_CACHE[path]


def _file_dhash(path):
    if path not in _DHASH_CACHE:
        try:
            from PIL import Image
            im = Image.open(path).convert("L").resize((9, 8))
            px = list(im.getdata())
            bits = 0
            for r in range(8):
                for c in range(8):
                    bits = (bits << 1) | (
                        1 if px[r * 9 + c] > px[r * 9 + c + 1] else 0
                    )
            _DHASH_CACHE[path] = bits
        except Exception:
            _DHASH_CACHE[path] = None
    return _DHASH_CACHE[path]


def _same_screen(path_a, path_b):
    """('exact'|'near'|None) content identity of two screenshots."""
    ma, mb = _file_md5(path_a), _file_md5(path_b)
    if ma is not None and ma == mb:
        return "exact"
    ha, hb = _file_dhash(path_a), _file_dhash(path_b)
    if ha is not None and hb is not None and bin(ha ^ hb).count("1") <= 4:
        return "near"
    return None


def gt_rects(run_dir):
    """(episode_id, step_id) -> pixel-or-normalized rect of the GT action."""
    gt = {}
    for rec in load_jsonl(find_samples(run_dir)):
        se = rec.get("step_eval") or {}
        if isinstance(se, str):
            try:
                se = json.loads(se)
            except Exception:
                se = {}
        rect = gt_rect_from_target(rec.get("target", ""))
        if rect is not None:
            gt[(str(se.get("episode_id")), int(se.get("step_id", -1)))] = rect
    return gt


def hist_keep(rec):
    """The committed history subset for this frame (what reuse serves later)."""
    order = rec.get("nap_order") or []
    if order and rec.get("ratio_hist"):
        k_h = max(1, int(round(rec["n_frame"] * float(rec["ratio_hist"]))))
        return set(order[:k_h])
    # selectors without a nested order commit their (only) keep
    return set(rec.get("local_keep") or [])


def collect_pairs(episodes, gt):
    """All same-goal registerable (t, t') pairs with the t-frame record.

    Returns (pairs, repeats, frames, n_exact, n_near); pairs are content
    matches with equal pixel sizes (see module docstring)."""
    pairs = []
    repeats, frames = 0, 0
    n_exact = n_near = 0
    for ep, recs in episodes.items():
        paths = [r.get("image_path", "") for r in recs]
        sizes = [img_size(p) for p in paths]
        frames += len(recs)
        for t, rec in enumerate(recs):
            for tp in range(t + 1, len(recs)):
                if sizes[t] is None or sizes[t] != sizes[tp]:
                    continue
                mode = _same_screen(paths[t], paths[tp])
                if mode is None:
                    continue
                repeats += 1
                rect = gt.get((ep, tp))
                if rect is None:
                    continue
                if mode == "exact":
                    n_exact += 1
                else:
                    n_near += 1
                pairs.append((ep, t, tp, rec, rect))
    return pairs, repeats, frames, n_exact, n_near


def retention_outcomes(pairs):
    """episode -> list of 0/1: does the frozen t-keep witness the t' target?"""
    per_ep = defaultdict(list)
    for ep, t, tp, rec, rect in pairs:
        size = img_size(rec.get("image_path", ""))
        if size is None:
            continue
        if max(rect) <= 1.5:
            rect = (rect[0] * size[0], rect[1] * size[1],
                    rect[2] * size[0], rect[3] * size[1])
        g = rect_to_patches(rect, size[0], size[1],
                            rec["grid_h"], rec["grid_w"])
        per_ep[ep].append(1.0 if g & hist_keep(rec) else 0.0)
    return per_ep


def ep_boot_rate(per_ep, seed_offset=0):
    eps = sorted(per_ep)
    rng = random.Random(SEED + seed_offset)
    rates = []
    for _ in range(N_BOOT):
        pick = [eps[rng.randrange(len(eps))] for _ in eps]
        vals = [v for e in pick for v in per_ep[e]]
        if vals:
            rates.append(100.0 * sum(vals) / len(vals))
    rates.sort()
    return [round(rates[int(0.025 * len(rates))], 2),
            round(rates[int(0.975 * len(rates)) - 1], 2)]


def ep_boot_diff(per_a, per_b, seed_offset=0):
    eps = sorted(set(per_a) | set(per_b))
    rng = random.Random(SEED + seed_offset)
    diffs = []
    for _ in range(N_BOOT):
        pick = [eps[rng.randrange(len(eps))] for _ in eps]
        va = [v for e in pick for v in per_a.get(e, [])]
        vb = [v for e in pick for v in per_b.get(e, [])]
        if va and vb:
            diffs.append(100.0 * (sum(va) / len(va) - sum(vb) / len(vb)))
    diffs.sort()
    return [round(diffs[int(0.025 * len(diffs))], 2),
            round(diffs[int(0.975 * len(diffs)) - 1], 2)]


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    tag = sys.argv[1]
    arms = {}
    for a in sys.argv[2:]:
        if a.startswith("--"):
            continue
        name, rest = a.split("=", 1)
        dump, run_dir = rest.split(":", 1)
        arms[name] = (dump, run_dir)
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    out = {"tag": tag, "arms": {}}
    per_arm_ep = {}
    out["registration"] = "content_md5_exact_or_dhash64_le4_same_size"
    for k, (name, (dump, run_dir)) in enumerate(sorted(arms.items())):
        episodes = episode_frames(dump)
        gt = gt_rects(run_dir)
        pairs, repeats, frames, n_exact, n_near = collect_pairs(episodes, gt)
        per_ep = retention_outcomes(pairs)
        flat = [v for vs in per_ep.values() for v in vs]
        per_arm_ep[name] = per_ep
        out["arms"][name] = {
            "n_pairs": len(flat),
            "n_pairs_exact": n_exact,
            "n_pairs_near": n_near,
            "episodes": len(per_ep),
            "screen_repeat_rate": round(repeats / max(1, frames), 4),
            "retention": (round(100.0 * sum(flat) / len(flat), 2)
                          if flat else None),
            "ci95_episode": ep_boot_rate(per_ep, k) if flat else None,
        }
        print(name, json.dumps(out["arms"][name]))
    names = sorted(per_arm_ep)
    out["paired_diffs"] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra = out["arms"][a]["retention"]
            rb = out["arms"][b]["retention"]
            if ra is None or rb is None:
                continue
            out["paired_diffs"][f"{a}_minus_{b}"] = {
                "diff": round(ra - rb, 2),
                "ci95": ep_boot_diff(per_arm_ep[a], per_arm_ep[b],
                                     1000 + i * 50 + ord(b[0])),
            }
    n_any = max((v["n_pairs"] for v in out["arms"].values()), default=0)
    if n_any == 0:
        print("W43_SAME_GOAL: no registerable same-screen pairs found; "
              "fall back to the trajectory-fork protocol "
              "(notes/missing_experiments_protocols.md, protocol A2).")

    agg_path = "docs/w43_same_goal_retention.json"
    agg = {}
    if os.path.exists(agg_path):
        agg = json.load(open(agg_path, encoding="utf-8"))
    agg[tag] = out
    node = os.uname().nodename if hasattr(os, "uname") else "local"
    tmp = agg_path + f".tmp.{node}.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=1)
    os.replace(tmp, agg_path)
    print("W43_SAME_GOAL_DONE")


if __name__ == "__main__":
    main()
