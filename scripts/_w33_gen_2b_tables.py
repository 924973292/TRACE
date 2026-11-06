#!/usr/bin/env python3
"""W33: 2B-carrier row provider for the combined 8B+2B paper tables.

Parses the collected dump docs/xds2b_tabledata.txt (2B replication corpus
under logs/xds2b_rep) and exposes row builders consumed by
gen_tables.py, which renders the combined main tables (Tables 1-2,
8B+2B bands) and the supplementary 50%/25% completion table:
  - HARVEST     native results.json metrics per arm (OmniGUI/M2W/overalls)
  - AC_HIST     HistPrune-protocol recompute (Type/Ground/StepSR)
  - SS_TI       pooled micro Text/Icon splits (ScreenSpot v2/Pro)
  - MMB_SUB     Basic/Adv difficulty split (MMBench-GUI L2)
Arms not landed yet render as \\tbd{}. Re-run scripts/_w33_2b_tabledata.sh,
pull the dump, then re-run gen_tables.py to refresh.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_tables as G

_ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(_ROOT, "artifacts", "metrics", "xds2b_tabledata.txt")
if not os.path.isfile(SRC):
    SRC = os.path.join(_ROOT, "docs", "xds2b_tabledata.txt")
OUT = G.OUT

TBD, DASH = G.TBD, G.DASH

# method display name -> arm leaf prefix (same baseline set as Tables 1-2)
METHODS = [
    ("Random", "random"),
    ("Uniform", "uniform"),
    (G.VEN["DivPrune"], "divprune"),
    (G.VEN["CDPruner"], "cdpruner"),
    (G.VEN["VisPruner"], "vispruner"),
    (G.VEN["PruMerge+"], "prumergeplus"),
    (G.VEN["TRIM"], "trim"),
    (G.VEN["FastV"], "fastv"),
    # DART dropped from the paper (W34 user directive)
    (G.VEN["VisionTrim"], "visiontrim"),
    (G.VEN["ZOO-Prune"], "zooprune"),
    (G.VEN["PruneSID"], "prunesid"),
]

# adopted \method{} arms on the 2B carrier (same rungs as the 8B adoption)
OURS_SINGLE = {
    "ssv2": {"50": "a2fix_r50", "25": "a2fix_r25", "10": "covdose50_r10", "5": "a2fix_r5"},
    "sspro": {"50": "a4fix_r50", "25": "a4cov50_r25", "10": "a4fix_r10", "5": "covdose30_r05"},
    "mmb": {"50": "a4fix_r50", "25": "a6fix_r25", "10": "a4fix_r10", "5": "a4fix_r5"},
}
OURS_MULTI = {
    "omnigui": {"I": "nested_rho05_c50_h10", "II": "a2cap_c25_h05"},
    "mind2web_task": {"I": "a6fix_c50_h10", "II": "a6nocov_c25_h05"},
    # The landed 2B twin follows the adopted 8B domain-I rung exactly.
    "mind2web_domain": {"I": "a6cov30_c50_h10", "II": "a6nocov_c25_h05"},
    "mind2web_website": {"I": "a4cov30_c50_h10", "II": "a4nocov_c25_h05"},
    "android_control": {"I": "covdose_rho50_c50_h10", "II": "rho10cov50_c25_h05"},
}
TIER_SUFFIX = {"I": "c50_h10", "II": "c25_h05"}
M2W_COUNTS = G.M2W_COUNTS  # (task, domain, website) official step counts


def read_lines(path):
    for enc in ("utf-8", "utf-16"):
        try:
            with open(path, encoding=enc) as f:
                return f.read().splitlines()
        except UnicodeError:
            continue
    raise SystemExit(f"cannot decode {path}")


def parse_dump():
    harvest = {}       # (bench, arm) -> {metric: float}
    done = set()       # (bench, arm) with landed results
    ac_hist = {}       # arm -> (type, ground, stepsr) as strings
    ss_ti = {}         # (bench, arm) -> (text, icon) strings
    mmb_sub = {}       # arm -> (basic, adv) strings
    section = None
    for line in read_lines(SRC):
        if line.startswith("==="):
            section = line.strip("= ").strip()
            continue
        if not line.strip():
            continue
        if section == "HARVEST":
            parts = line.split("\t")
            if len(parts) >= 3 and parts[0] == "DONE" and "/" in parts[1]:
                bench, arm = parts[1].split("/", 1)
                d = {}
                for tok in parts[2].split():
                    if "=" not in tok:
                        continue
                    k, v = tok.rsplit("=", 1)
                    try:
                        d[k.split(":", 1)[-1]] = float(v)
                    except ValueError:
                        pass
                harvest[(bench, arm)] = d
                done.add((bench, arm))
        elif section and section.startswith("AC_HIST"):
            m = re.match(r"^(\S+)\s+(\d+)\s+\|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\|", line)
            if m and m.group(1) not in ac_hist:
                # glob order lists logs/xds2b_rep before logs/xds8b_w25, so
                # first occurrence per arm name is the 2B run; arms without a
                # landed 2B twin are cross-checked against `done` below.
                ac_hist[m.group(1)] = (m.group(3), m.group(4), m.group(6))
        elif section == "SS_TI":
            parts = line.split("\t")
            if len(parts) == 6 and parts[1].startswith("logs/xds2b_rep/"):
                bench = "ssv2" if parts[0] == "SSV2" else "sspro"
                arm = parts[1].rstrip("/").split("/")[-1]
                ss_ti[(bench, arm)] = (parts[3], parts[4], parts[5])  # overall, text, icon
        elif section == "MMB_SUB":
            parts = line.split("\t")
            if len(parts) >= 4 and parts[0].startswith("logs/xds2b_rep/"):
                arm = parts[0].rstrip("/").split("/")[-1]
                kv = {}
                for tok in parts[3:]:
                    mm = re.match(r"(\w+)=([\d.]+)\(", tok)
                    if mm:
                        kv[mm.group(1)] = mm.group(2)
                mmb_sub[arm] = (parts[2], kv.get("basic", DASH), kv.get("advanced", DASH))
    # AC protocol numbers only count when the 2B native run has landed
    # (drops the 8B w25 prunesid rows sharing the same arm names).
    ac_hist = {a: v for a, v in ac_hist.items()
               if ("android_control", a) in done or a == "baseline_full"}
    return harvest, ac_hist, ss_ti, mmb_sub


HARVEST, AC_HIST, SS_TI, MMB_SUB = parse_dump()


def arm_variants(prefix, r_pct):
    tag = {"50": ["r50"], "25": ["r25"], "10": ["r10"], "5": ["r05", "r5"]}[r_pct]
    return [f"{prefix}_{t}" for t in tag]


def ss_cells(bench, prefix, r_pct, ours=False):
    arms = [OURS_SINGLE[bench][r_pct]] if ours else arm_variants(prefix, r_pct)
    for arm in arms:
        v = SS_TI.get((bench, arm))
        if v:
            return v[1], v[2], v[0]  # text, icon, overall
    return TBD, TBD, TBD


def mmb_cells(prefix, r_pct, ours=False):
    arms = [OURS_SINGLE["mmb"][r_pct]] if ours else arm_variants(prefix, r_pct)
    for arm in arms:
        v = MMB_SUB.get(arm)
        if v:
            return v[1], v[2], v[0]  # basic, adv, overall
    return TBD, TBD, TBD


# 2B dense upper bounds (noprune_ref / baseline_full on the 2B carrier)
UB_SSV2 = SS_TI[("ssv2", "noprune_ref")]    # overall, text, icon
UB_SSPRO = SS_TI[("sspro", "noprune_ref")]
UB_MMB = MMB_SUB["noprune_ref"]             # overall, basic, adv
SS2B_MAIN_UB = [float(UB_SSV2[0]), float(UB_SSPRO[0]), float(UB_MMB[0])]


def single_rows(r_pct):
    rows = []
    for name, prefix in METHODS:
        sv_t, sv_i, sv = ss_cells("ssv2", prefix, r_pct)
        sp_t, sp_i, sp = ss_cells("sspro", prefix, r_pct)
        mb, ma, mo = mmb_cells(prefix, r_pct)
        ret = G._ret_avg([(sv, SS2B_MAIN_UB[0]), (sp, SS2B_MAIN_UB[1]),
                          (mo, SS2B_MAIN_UB[2])])
        rows.append((name, "n", [sv_t, sv_i, sv, sp_t, sp_i, sp, mb, ma, mo,
                                 G._fmt_ret(ret)]))
    sv_t, sv_i, sv = ss_cells("ssv2", None, r_pct, ours=True)
    sp_t, sp_i, sp = ss_cells("sspro", None, r_pct, ours=True)
    mb, ma, mo = mmb_cells(None, r_pct, ours=True)
    ret = G._ret_avg([(sv, SS2B_MAIN_UB[0]), (sp, SS2B_MAIN_UB[1]),
                      (mo, SS2B_MAIN_UB[2])])
    rows.append((r"\method{}", "ours",
                 [sv_t, sv_i, sv, sp_t, sp_i, sp, mb, ma, mo, G._fmt_ret(ret)]))
    return rows


def omni_cells(prefix, tier, ours=False):
    arm = OURS_MULTI["omnigui"][tier] if ours else f"{prefix}_{TIER_SUFFIX[tier]}"
    d = HARVEST.get(("omnigui", arm))
    if not d or "type_match" not in d:
        return TBD, TBD
    return f"{d['type_match'] * 100:.2f}", f"{d['exact_match'] * 100:.2f}"


def m2w_pooled(prefix, tier, ours=False):
    acc = [0.0, 0.0, 0.0]
    tot = sum(M2W_COUNTS)
    for bench, n_s in zip(("mind2web_task", "mind2web_domain", "mind2web_website"), M2W_COUNTS):
        arm = OURS_MULTI[bench][tier] if ours else f"{prefix}_{TIER_SUFFIX[tier]}"
        d = HARVEST.get((bench, arm))
        if not d or "m2w_ele_acc" not in d:
            return None
        for j, k in enumerate(("m2w_ele_acc", "m2w_op_f1", "m2w_step_sr")):
            acc[j] += d[k] * 100 * n_s
    return [f"{a / tot:.2f}" for a in acc]


def m2w_pooled_ub():
    acc = [0.0, 0.0, 0.0]
    tot = sum(M2W_COUNTS)
    for bench, anchor, n_s in zip(
            ("mind2web_task", "mind2web_domain", "mind2web_website"),
            ("baseline_full", "noprune_kv_ref", "noprune_kv_ref"), M2W_COUNTS):
        d = HARVEST[(bench, anchor)]
        for j, k in enumerate(("m2w_ele_acc", "m2w_op_f1", "m2w_step_sr")):
            acc[j] += d[k] * 100 * n_s
    return [f"{a / tot:.2f}" for a in acc]


def ac_cells(prefix, tier, ours=False):
    arm = OURS_MULTI["android_control"][tier] if ours else f"{prefix}_{TIER_SUFFIX[tier]}"
    return AC_HIST.get(arm, (TBD, TBD, TBD))


UB_OMNI = (
    f"{HARVEST[('omnigui', 'baseline_full')]['type_match'] * 100:.2f}",
    f"{HARVEST[('omnigui', 'baseline_full')]['exact_match'] * 100:.2f}")
UB_M2W = m2w_pooled_ub()
UB_AC = AC_HIST["baseline_full"]


def multi_rows(tier):
    ub_omni_sr, ub_m2w_sr, ub_ac_sr = (
        float(UB_OMNI[1]), float(UB_M2W[2]), float(UB_AC[2]))
    rows = []
    entries = [(name, prefix, False) for name, prefix in METHODS]
    entries.append((r"\method{}", None, True))
    for name, prefix, ours in entries:
        o_type, o_sr = omni_cells(prefix, tier, ours)
        m2w = m2w_pooled(prefix, tier, ours) or [TBD] * 3
        a_type, a_grd, a_sr = ac_cells(prefix, tier, ours)
        ret = G._ret_avg([(o_sr, ub_omni_sr), (m2w[2], ub_m2w_sr),
                          (a_sr, ub_ac_sr)])
        rows.append((name, "ours" if ours else "n",
                     [o_type, o_sr] + m2w + [a_type, a_grd, a_sr,
                                             G._fmt_ret(ret)]))
    return rows


if __name__ == "__main__":
    # smoke: report landed-cell coverage per band
    for key in ("50", "25", "10", "5"):
        rows = single_rows(key)
        n_tbd = sum(v.count(TBD) for _, _, v in rows)
        print(f"single r{key}: {len(rows)} rows, {n_tbd} TBD cells")
    for tier in ("I", "II"):
        rows = multi_rows(tier)
        n_tbd = sum(v.count(TBD) for _, _, v in rows)
        print(f"multi tier {tier}: {len(rows)} rows, {n_tbd} TBD cells")
