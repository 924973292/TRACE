#!/usr/bin/env python
"""Harvest the W53 batch and check it against the published anchors.

Two things are reported per group:

  anchor gate   the base arm must reproduce the published operating point it was
                cloned from (SSPro 37.63 at r=10%, 20.30 at r=5%). If it does not,
                the queue's lineage differs from the main table and no delta from
                it can be compared against the SSv2 panel.

  deltas        each knockout minus its own base twin, so the two benchmarks are
                read the same way.

Mind2Web groups additionally pool the three official cross-splits by step count,
matching the convention used everywhere else in the paper.
"""
import glob
import json
import os
import sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.isdir(REPO):
    os.chdir(REPO)

SSPRO_ANCHOR = {"r10": 37.63, "r5": 20.30}


def metrics(run):
    f = glob.glob(f"{run}/results/models__*/*_results.json")
    if not f:
        return None
    try:
        d = json.load(open(f[0], encoding="utf-8"))
    except Exception:
        return None
    out = {}
    for _task, m in (d.get("results") or {}).items():
        for k, v in (m or {}).items():
            if isinstance(v, (int, float)):
                out[k] = v * 100 if 0 <= v <= 1 else v
    return out


def pick(m, *needles):
    if not m:
        return None
    for n in needles:
        for k, v in m.items():
            if n in k.lower():
                return v
    return None


def n_steps(run):
    f = glob.glob(f"{run}/results/models__*/*_samples_*.jsonl")
    if not f:
        return 0
    return sum(1 for line in open(f[0], encoding="utf-8") if line.strip())


print("=" * 74)
print("GROUP 1  ScreenSpot-Pro module ablations (tab:lip-energy / neo-form / 4b)")
print("=" * 74)
for tier, suffix in (("r10", "_r10"), ("r5", "_r5")):
    base = metrics(f"logs/w53_sspro/screenspot_pro/base{suffix}")
    b = pick(base, "center_acc", "acc")
    anchor = SSPRO_ANCHOR[tier]
    if b is None:
        print(f"\n-- {tier}: base not finished yet")
        continue
    gate = "OK" if abs(b - anchor) < 0.6 else f"DRIFT vs {anchor}"
    print(f"\n-- {tier}: base={b:.2f} (published anchor {anchor}) [{gate}]")
    rows = [
        ("-H (texture entropy)", f"noH{suffix}"),
        ("-C (boundary contrast)", f"noC{suffix}"),
        ("-G (containment)", f"noG{suffix}"),
        ("-R (alignment)", f"noR{suffix}"),
    ]
    if tier == "r10":
        rows += [
            ("NEO: unit coefficient", "neo_coef_r10"),
            ("NEO: log-mean-exp pooling", "neo_lse_r10"),
            # neo_meanrow_r10 is the abandoned inert-flag arm (MEANROW=1 without
            # TURN_QUERY_TOKENS=1 is a no-op); the published 32.01 cell is the
            # corrected meanonly arm from w53_sspro_fix.tsv.
            ("NEO: mean row only", "neo_meanonly_r10"),
            ("AWC: medoid representative", "rep_medoid_r10"),
        ]
    for label, arm in rows:
        m = metrics(f"logs/w53_sspro/screenspot_pro/{arm}")
        v = pick(m, "center_acc", "acc")
        if v is None:
            print(f"   {label:32s}     pending")
        else:
            print(f"   {label:32s} {v:6.2f}   delta {v - b:+6.2f}")

print()
print("=" * 74)
print("GROUP 2  Mind2Web pooled: transfer control (tab:transfer)")
print("=" * 74)
SPLITS = ("mind2web_task", "mind2web_website", "mind2web_domain")
for tier, c in (("mild", "c50_h10"), ("tight", "c25_h05")):
    print(f"\n-- {tier} budget, Step SR pooled by step count")
    for sel in ("divprune", "vispruner", "zooprune"):
        line = f"   {sel:12s}"
        for path, tree in (("native", "native"), ("ported", "pubsel")):
            num = den = 0.0
            miss = False
            for ds in SPLITS:
                # Each path has its own legacy home for the task split. Mixing
                # them would silently report one path twice, so the fallbacks are
                # kept disjoint per path.
                if tree == "pubsel":
                    cand = [
                        f"logs/w53_transfer/{ds}/pubsel_{sel}_{c}",
                        f"logs/xds8b_abl/{ds}/pubsel_{sel}_{c}",
                    ]
                else:
                    cand = [
                        f"logs/w53_transfer/{ds}/native_{sel}_{c}",
                        f"logs/xds8b_pub/{ds}/{sel}_{c}",
                        f"logs/xds8b_pub2/{ds}/{sel}_{c}",
                        f"logs/xds8b_pubattn/{ds}/{sel}_{c}",
                        # zooprune's native task-split home (transfer.tex header
                        # names xds8b_w20 as a native source tree)
                        f"logs/xds8b_w20/{ds}/{sel}_{c}",
                    ]
                hit = None
                for p in cand:
                    m = metrics(p)
                    if m:
                        hit = (p, m)
                        break
                if not hit:
                    miss = True
                    break
                p, m = hit
                sr = pick(m, "step_sr", "step_success")
                w = n_steps(p) or 1
                if sr is None:
                    miss = True
                    break
                num += sr * w
                den += w
            line += f"  {path}=" + ("pending" if miss else f"{num / den:.2f}")
        print(line)

print()
print("=" * 74)
print("GROUP 3  Mind2Web pooled: NEO factor removal (tab:ablation-modules a)")
print("=" * 74)
for label, arm, legacy in (
    ("base", "base_a6nocov_c25_h05", "logs/xds8b_tune/{ds}/a6nocov_c25_h05"),
    # noA flattens the layout-prior mass m_j (BAM_NO_MASS=1). The task split ran
    # in W23-era xds8b_abl2; website/domain completed via w53_abl4a_mass.tsv.
    ("- mass", "noA_c25_h05", "logs/xds8b_abl2/{ds}/noA_c25_h05"),
    ("- diversity", "nodiv_a6nocov_c25_h05", "logs/xds8b_abl2/{ds}/nodiv_a6nocov_c25_h05"),
    # UNIFORM_A flattens the relevance likelihood a_j (geopact_qd_selector
    # L1072-1075), which is the instruction-removal knockout, not mass removal.
    # The table's own audit note pairs unifa with the -instr row (p=0.142).
    ("- instruction", "unifa_a6nocov_c25_h05", "logs/xds8b_abl2/{ds}/unifa_a6nocov_c25_h05"),
):
    num = den = 0.0
    miss = []
    for ds in SPLITS:
        m = metrics(f"logs/w53_abl4a/{ds}/{arm}") or metrics(legacy.format(ds=ds))
        if not m:
            miss.append(ds.replace("mind2web_", ""))
            continue
        sr = pick(m, "step_sr", "step_success")
        p = f"logs/w53_abl4a/{ds}/{arm}"
        w = n_steps(p) or n_steps(legacy.format(ds=ds)) or 1
        if sr is None:
            miss.append(ds.replace("mind2web_", ""))
            continue
        num += sr * w
        den += w
    val = f"{num / den:.2f}" if den else "--"
    print(f"   {label:14s} pooled Step SR = {val:>6s}"
          + (f"   pending: {','.join(miss)}" if miss else ""))
