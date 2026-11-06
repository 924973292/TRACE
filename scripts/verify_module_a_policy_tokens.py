#!/usr/bin/env python3
"""Unit checks for Module A scoring policy tokens (E, rho, denominators)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lmms_eval.models.miu_info_density import (
    attach_module_a_info_density_proxy,
    estimate_miu_grid_token_count,
    estimate_miu_pad_count,
    info_density_raw,
    miu_interaction_energy,
)


def _mock_miu() -> dict:
    return {
        "box": (0.1, 0.1, 0.5, 0.5),
        "det_score": 0.8,
        "pentagonal_scale": 0.6,
        "pentagonal_entropy": 0.4,
        "pentagonal_contrast": 0.5,
        "pentagonal_containment": 0.7,
        "pentagonal_resonance": 0.3,
        "pentagonal_confidence_attr": 0.75,
    }


def main() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if ok:
            print(f"[OK] {name}" + (f" — {detail}" if detail else ""))
        else:
            print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""), file=sys.stderr)
            failures.append(name)

    miu = _mock_miu()
    pol_base = "oas_ickh_cdct_no_coord_yolo"
    e_base = miu_interaction_energy(miu, pol_base)
    check("default E > 0", e_base > 0.0, f"E={e_base:.4f}")

    e_no_r = miu_interaction_energy(miu, f"{pol_base}_drop_pent_resonance")
    check("drop_pent_resonance lowers E", e_no_r < e_base, f"{e_no_r:.4f} < {e_base:.4f}")

    e_add = miu_interaction_energy(miu, f"{pol_base}_no_conf_gate")
    e_mult = miu_interaction_energy(miu, pol_base)
    check("no_conf_gate changes E vs mult", abs(e_add - e_mult) > 1e-6)

    e_conf_only = miu_interaction_energy(miu, f"{pol_base}_conf_attr_only")
    check("conf_attr_only positive", e_conf_only > 0.0)

    e_no_conf = miu_interaction_energy(miu, f"{pol_base}_no_conf")
    pent_sum = sum(
        float(miu[k])
        for k in (
            "pentagonal_scale",
            "pentagonal_entropy",
            "pentagonal_contrast",
            "pentagonal_containment",
            "pentagonal_resonance",
        )
    )
    check("no_conf structural only", abs(e_no_conf - pent_sum) < 1e-6, f"E={e_no_conf:.4f} sum={pent_sum:.4f}")

    n_grid = estimate_miu_grid_token_count(miu, 24, 32)
    n_area = estimate_miu_pad_count(miu)
    check("grid vs area token count differ", n_grid != n_area, f"grid={n_grid} area={n_area}")

    rho_div = info_density_raw(e_base, n_grid, "image", pol_base)
    rho_raw = info_density_raw(e_base, n_grid, "image", f"{pol_base}_raw_energy")
    check("raw_energy skips division", abs(rho_raw - e_base) < 1e-6, f"rho={rho_raw:.4f}")
    check("default divides by n", abs(rho_div - e_base / n_grid) < 1e-6)

    attached_grid, = attach_module_a_info_density_proxy(
        [dict(miu)], pol_base, grid_hw=(24, 32)
    )
    check("grid attach source", attached_grid.get("est_token_count_source") == "grid")
    check("grid attach density", float(attached_grid.get("info_density", 0)) > 0)

    attached_area, = attach_module_a_info_density_proxy(
        [dict(miu)], f"{pol_base}_div_area_heuristic", grid_hw=(24, 32)
    )
    check(
        "div_area_heuristic forces area source",
        attached_area.get("est_token_count_source") == "area_heuristic",
    )
    check(
        "div_area_heuristic uses area n",
        int(attached_area.get("est_token_count", 0)) == n_area,
    )

    attached_raw, = attach_module_a_info_density_proxy(
        [dict(miu)], f"{pol_base}_raw_energy", grid_hw=(24, 32)
    )
    check(
        "raw_energy density equals E",
        abs(float(attached_raw.get("info_density", 0)) - e_base) < 1e-5,
    )

    null_list = attach_module_a_info_density_proxy(
        [dict(miu)], f"{pol_base}_no_info_density", grid_hw=(24, 32)
    )
    check("no_info_density zeros rho", float(null_list[0].get("info_density", -1)) == 0.0)

    pol_v2 = "oas_ickh_cdct_no_coord_yolo_v2"
    e_v2 = miu_interaction_energy(miu, pol_v2)
    check("v2 structural E equals pent sum", abs(e_v2 - pent_sum) < 1e-6, f"E={e_v2:.4f}")
    check("v2 differs from legacy det×struct", abs(e_v2 - e_mult) > 1e-6)
    e_v2_drop_c = miu_interaction_energy(miu, f"{pol_v2}_drop_pent_contrast")
    check(
        "v2 drop_contrast lowers E",
        e_v2_drop_c < e_v2,
        f"{e_v2_drop_c:.4f} < {e_v2:.4f}",
    )
    attached_v2, = attach_module_a_info_density_proxy([dict(miu)], pol_v2, grid_hw=(24, 32))
    check("v2 uses grid denominator", attached_v2.get("est_token_count_source") == "grid")
    check(
        "v2 density equals E/n_grid",
        abs(float(attached_v2.get("info_density", 0)) - e_v2 / n_grid) < 1e-6,
    )

    if failures:
        print(f"\n{len(failures)} check(s) failed", file=sys.stderr)
        return 1
    print("\nAll Module A policy token checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
