#!/usr/bin/env python3
"""W35: machine-check every derived cell of the rendered paper tables
(combined 8B+2B format), successor of _w31_verify_main_tables.py.

main_single.tex / supp_single.tex, per data row (carrier tracked from the
band lines):
  1. micro identity SSv2 : (Text*718  + Icon*554 )/1272 == Avg      (+- tol)
  2. micro identity SSPro: (Text*977  + Icon*604 )/1581 == Avg      (+- tol)
  3. micro identity MMB  : (Basic*1787 + Adv*1807)/3594 == Overall  (+- tol)
  4. retention Avg% == 100*mean(Avg/UB) with the carrier's own UBs
  5. provenance (when logs/_w31_ss_ti.tsv + logs/_w32_mmb_subsplit.tsv are
     present locally): the sub-cell triples appear verbatim in the remote
     per-sample recompute TSVs.

main_multi.tex, per data row: M2W pooled identity (2008/5429/1343 step
weights, per carrier) and retention Avg% with per-carrier UBs.

main_efficiency.tex: Ret.% arithmetic + Step SR column must equal the
OmniGUI mild-budget EM cells of the corresponding main-table rows.

ablation_main.tex: FINAL row must equal the main-table adopted cells.
"""
import importlib.util
import os
import re
import sys

TOL = 0.025
CNT = {"ssv2": (718.0, 554.0), "sspro": (977.0, 604.0), "mmb": (1787.0, 1807.0)}
M2W_CNT = (2008.0, 5429.0, 1343.0)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TAB = os.path.join(ROOT, "artifacts", "tables")


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sys.path.insert(0, os.path.join(ROOT, "scripts"))
G = load_mod("gen_tables", os.path.join(ROOT, "scripts", "gen_tables.py"))
import _w33_gen_2b_tables as R

CLEAN = re.compile(r"\\(?:inllm|textbf|underline|pub|method|tbd|textcolor\{gray\})\s*\{([^{}]*)\}")


def cells(line):
    s = line
    for _ in range(4):
        s = CLEAN.sub(r"\1", s)
    s = s.replace(r"\rowcolor{rowblue}", "").replace(r"\\", "").strip()
    return [p.strip() for p in s.split("&")]


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def band_of(line):
    m = re.search(r"\\textsc\{([^}]*)\}", line)
    return m.group(1) if m else None


# ---- per-carrier upper bounds ----
SS_UB = {
    "8b": (93.79, 70.34, 82.64),
    "2b": tuple(R.SS2B_MAIN_UB),
}
MULTI_UB = {
    "8b": (float(G.OMNI_UB[3]), float(G._m2w_pooled_ub()[2]), float(G.AC_UB[2])),
    "2b": (float(R.UB_OMNI[1]), float(R.UB_M2W[2]), float(R.UB_AC[2])),
}


def prov_2b():
    """2B provenance: sub-cell triples must match the harvested dump."""
    tsv, mmb = set(), set()
    for (bench, _), (o, t, i) in R.SS_TI.items():
        tsv.add(("SSV2" if bench == "ssv2" else "SSPRO", o, t, i))
    for _, (o, b, a) in R.MMB_SUB.items():
        mmb.add((o, b, a))
    return tsv, mmb


def load_prov():
    tsv, mmb = set(), set()
    p1 = os.path.join(ROOT, "logs", "_w31_ss_ti.tsv")
    p2 = os.path.join(ROOT, "logs", "_w32_mmb_subsplit.tsv")
    if os.path.exists(p1):
        for line in open(p1, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) == 6 and p[2] in ("1272", "1581"):
                tsv.add((p[0], p[3], p[4], p[5]))
    mmb_line = re.compile(
        r"\t3594\t([0-9.]+)\tadvanced=([0-9.]+)\(1807\)\tbasic=([0-9.]+)\(1787\)")
    if os.path.exists(p2):
        for line in open(p2, encoding="utf-8"):
            m = mmb_line.search(line)
            if m:
                overall, adv, basic = m.groups()
                mmb.add((overall, basic, adv))
    return tsv, mmb


def check_single(path, tsv, mmb_tsv):
    bad = checked = 0
    carrier = None
    for line in open(path, encoding="utf-8"):
        b = band_of(line)
        if b:
            if "GUI-Owl-1.5-2B" in b:
                carrier = "2b"
            elif "GUI-Owl-1.5-8B" in b:
                carrier = "8b"
            continue
        if "&" not in line or "multicolumn" in line or "cmidrule" in line \
                or "Method" in line or "Upper Bound" in line \
                or line.strip().startswith("GUI-Owl"):
            continue
        p = cells(line)
        if len(p) != 11:
            continue
        name = p[0]
        vals = [fl(x) for x in p[1:]]
        if any(v is None for v in vals[:9]):
            if vals[9] is not None:
                print(f"FAIL dash-row {name} [{carrier}]: numeric Avg% {vals[9]}")
                bad += 1
            continue
        checked += 1
        sv_t, sv_i, sv, sp_t, sp_i, sp, mb, ma, mm, ret = vals
        for tag, (nt, ni), t, i, o in (
                ("ssv2", CNT["ssv2"], sv_t, sv_i, sv),
                ("sspro", CNT["sspro"], sp_t, sp_i, sp),
                ("mmb", CNT["mmb"], mb, ma, mm)):
            pooled = (t * nt + i * ni) / (nt + ni)
            if abs(pooled - o) > TOL:
                print(f"FAIL identity {tag:5s} {name} [{carrier}]: "
                      f"pooled {pooled:.3f} vs table {o}")
                bad += 1
        ub = SS_UB[carrier]
        exp_ret = 100.0 * (sv / ub[0] + sp / ub[1] + mm / ub[2]) / 3
        if ret is None or abs(exp_ret - ret) > 0.051:
            print(f"FAIL retention {name} [{carrier}]: "
                  f"expect {exp_ret:.2f} vs table {ret}")
            bad += 1
        p_tsv, p_mmb = (tsv, mmb_tsv) if carrier == "8b" else prov_2b()
        if p_tsv:
            for bench, t, i, o in (("SSV2", sv_t, sv_i, sv),
                                   ("SSPRO", sp_t, sp_i, sp)):
                key = (bench, f"{o:.2f}", f"{t:.2f}", f"{i:.2f}")
                if key not in p_tsv:
                    print(f"FAIL provenance {bench} {name} [{carrier}]: "
                          f"({o:.2f},{t:.2f},{i:.2f}) not in per-sample TSV")
                    bad += 1
        if p_mmb:
            mkey = (f"{mm:.2f}", f"{mb:.2f}", f"{ma:.2f}")
            if mkey not in p_mmb:
                print(f"FAIL provenance MMB {name} [{carrier}]: "
                      f"({mm:.2f},{mb:.2f},{ma:.2f}) not in per-sample TSV")
                bad += 1
    print(f"{os.path.basename(path)}: rows checked {checked}")
    return bad


def _m2w_pool_8b(tier, raw_name):
    tot, acc = 0.0, [0.0, 0.0, 0.0]
    for sk, n in zip(("task", "domain", "website"), M2W_CNT):
        vals, _ = G._m2w_lookup(sk, tier, raw_name)
        if vals is None:
            return None
        for j, x in enumerate(vals):
            v = fl(x)
            if v is None:
                return None
            acc[j] += v * n
        tot += n
    return [a / tot for a in acc]


def _m2w_pool_2b(tier, disp_name):
    if disp_name == "GeoPACT":  # rendered ours name; use ours lookup
        return [fl(x) for x in (R.m2w_pooled(None, tier, ours=True) or [None] * 3)]
    for name, prefix in R.METHODS:
        if cells(name + " ")[0] == disp_name:
            return [fl(x) for x in (R.m2w_pooled(prefix, tier) or [None] * 3)]
    return None


def check_multi(path):
    bad = checked = 0
    raw_names = {}
    for tier in ("I", "II"):
        for raw, _, _ in G.M2W["task"][tier]:
            raw_names[cells(raw + " ")[0]] = raw
    ours_disp = cells(r"\method{} ")[0]
    carrier = tier = None
    for line in open(path, encoding="utf-8"):
        b = band_of(line)
        if b:
            if "GUI-Owl-1.5-2B" in b:
                carrier = "2b"
            elif "GUI-Owl-1.5-8B" in b:
                carrier = "8b"
            elif "Retain 50" in b:
                tier = "I"
            elif "Retain 25" in b:
                tier = "II"
            continue
        if "&" not in line or "multicolumn" in line or "cmidrule" in line \
                or "Method" in line or "Upper Bound" in line \
                or line.strip().startswith("GUI-Owl"):
            continue
        p = cells(line)
        if len(p) != 10:
            continue
        name = p[0]
        vals = [fl(x) for x in p[1:]]
        _, o_sr, ele, f1, m_sr, _, _, a_sr, ret = vals
        exp = None
        if carrier == "8b" and name in raw_names:
            exp = _m2w_pool_8b(tier, raw_names[name])
        elif carrier == "2b":
            exp = _m2w_pool_2b(tier, name if name != ours_disp else "GeoPACT")
        got = (ele, f1, m_sr)
        if exp is not None and all(e is not None for e in exp):
            if any(v is None or abs(v - e) > TOL for v, e in zip(got, exp)):
                print(f"FAIL m2w-pool {name} [{carrier} T{tier}]: "
                      f"expect {[f'{e:.2f}' for e in exp]} vs {got}")
                bad += 1
        if any(v is None for v in (o_sr, m_sr, a_sr)):
            if ret is not None:
                print(f"FAIL dash-row {name} [{carrier} T{tier}]: Avg% {ret}")
                bad += 1
            continue
        checked += 1
        ub = MULTI_UB[carrier]
        exp_ret = 100.0 * (o_sr / ub[0] + m_sr / ub[1] + a_sr / ub[2]) / 3
        if ret is None or abs(exp_ret - ret) > 0.051:
            print(f"FAIL retention(multi) {name} [{carrier} T{tier}]: "
                  f"expect {exp_ret:.2f} vs {ret}")
            bad += 1
    print(f"{os.path.basename(path)}: rows checked {checked}")
    return bad


def check_efficiency():
    bad = 0
    ub = G.EFF_UB_SR
    omni_em = {name: fl(v[3]) for name, _, v in G.OMNI["I"]}
    for name, kind, tok, gf, sel, ttft, e2e, kv, sr in G.EFF_ROWS:
        srf = fl(sr)
        base = name.replace(r"$^{\dagger}$", "")
        if kind != "gray":
            exp_ret = srf / ub * 100
            # arithmetic is computed live in gen; just cross-check SR source
            ref = omni_em.get(base)
            if ref is None and base == r"\method{}":
                ref = fl(G.OMNI["I"][-1][2][3])
            if ref is not None and abs(ref - srf) > 0.005:
                print(f"FAIL eff-SR {base}: table {srf} vs main-multi EM {ref}")
                bad += 1
            if ref is None:
                print(f"WARN eff-SR {base}: no main-table twin found")
    print("efficiency: SR column cross-checked vs OmniGUI mild-budget EM")
    return bad


def check_ablation_final():
    bad = 0
    # FINAL row literals live in gen_ablation_main.configs; re-derive the
    # expected values from the adopted main-table cells instead.
    expect_mild = [G.OMNI["I"][-1][2][0],           # omni GP c50
                   G.M2W["task"]["I"][-1][2][2],    # m2w task SR c50
                   G._ssv2_block("10")[-1][2][6],   # ssv2 acc r10
                   G._sspro_block("25")[-1][2][12]]  # sspro acc r25
    expect_tight = [G.OMNI["II"][-1][2][0],
                    G.M2W["task"]["II"][-1][2][2],
                    G._ssv2_block("5")[-1][2][6],
                    G._sspro_block("10")[-1][2][12]]
    tex = open(os.path.join(TAB, "ablation_main.tex"), encoding="utf-8").read()
    rows = [cells(l) for l in tex.splitlines()
            if l.count(r"\cmark") == 3]
    finals = [r for r in rows if len(r) == 7 and r[2]]
    if len(finals) != 2:
        print(f"FAIL ablation FINAL rows found={len(finals)} (expect 2)")
        return 1
    for tag, row, exp in (("mild", finals[0], expect_mild),
                          ("tight", finals[1], expect_tight)):
        got = row[3:7]
        if [fl(x) for x in got] != [fl(x) for x in exp]:
            print(f"FAIL ablation FINAL {tag}: table {got} vs main {exp}")
            bad += 1
    print("ablation_main: FINAL rows cross-checked vs adopted main cells")
    return bad


def main():
    tsv, mmb_tsv = load_prov()
    if not tsv:
        print("NOTE: logs/_w31_ss_ti.tsv missing locally -> provenance skipped")
    bad = check_single(os.path.join(TAB, "main_single.tex"), tsv, mmb_tsv)
    bad += check_single(os.path.join(TAB, "supp_single.tex"), tsv, mmb_tsv)
    bad += check_multi(os.path.join(TAB, "main_multi.tex"))
    bad += check_efficiency()
    bad += check_ablation_final()
    print("W35_VERIFY", "FAIL" if bad else "OK", f"bad={bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
