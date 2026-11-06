#!/usr/bin/env python3

"""Unit checks for ephemeral Module C on dialogue-KV sticky prune path."""

from __future__ import annotations



import os

import sys

from pathlib import Path



REPO = Path(__file__).resolve().parents[1]

if str(REPO) not in sys.path:

    sys.path.insert(0, str(REPO))





def _check(name: str, ok: bool, detail: str = "") -> bool:

    status = "OK" if ok else "FAIL"

    msg = f"[{status}] {name}"

    if detail:

        msg += f" — {detail}"

    print(msg)

    return ok





def main() -> int:

    failures = 0



    from lmms_eval.models.geopact_coord_appendix import (

        build_sdb_appendix,

        estimate_token_budget,

        miu_center_qwen1000,

        resolve_coord_format,

        to_qwen1000,

    )

    from lmms_eval.models.geopact_vl_hook import (

        HESResult,

        _build_coord_text,

        _coord_metadata_only,

        _coord_miu_indices_from_pick_order,

        _coord_policy_active,

    )



    mius = [

        {"box": (0.1, 0.1, 0.2, 0.2), "type": "btn_a", "info_density": 0.9},

        {"box": (0.5, 0.5, 0.6, 0.6), "type": "btn_b", "info_density": 0.5},

        {"box": (0.7, 0.1, 0.8, 0.2), "type": "btn_c", "info_density": 0.7},

    ]

    miu_token_idxs = [[0, 1], [2, 3], [4]]

    pick_order = [0, 2, 3, 1, 4]



    ordered = _coord_miu_indices_from_pick_order(

        pick_order, miu_token_idxs, mius, topk=5

    )

    if not _check(

        "pick-order MIU map — respects pick order",

        ordered == [0, 1, 2],

        f"got {ordered}",

    ):

        failures += 1



    top2 = _coord_miu_indices_from_pick_order(

        pick_order, miu_token_idxs, mius, topk=2

    )

    if not _check("pick-order MIU map — topk=2", top2 == [0, 1], f"got {top2}"):

        failures += 1



    hes = HESResult([0, 2, 4], selected_miu_indices=[0, 1], picked_token_order=[0, 2, 4])

    if not _check(

        "HESResult stores picked_token_order",

        getattr(hes, "picked_token_order", None) == [0, 2, 4],

        "",

    ):

        failures += 1



    if not _check("to_qwen1000 round bias", to_qwen1000(0.4999) == 500, ""):

        failures += 1



    cx, cy = miu_center_qwen1000((0.1, 0.1, 0.2, 0.2))

    if not _check(

        "miu_center_qwen1000",

        cx == 150 and cy == 150,

        f"got ({cx},{cy})",

    ):

        failures += 1



    if not _check(

        "resolve_coord_format sdb policy",

        resolve_coord_format("oas_ickh_cdct_coord_sdb_yolo") == "sdb",

        "",

    ):

        failures += 1



    sdb = build_sdb_appendix([(412, 583), (720, 112)])

    if not _check(

        "SDB appendix structure",

        "[Appendix" in sdb

        and "Anchors do NOT imply action_type" in sdb

        and "A1<|box_start|>(412,583)<|box_end|>" in sdb,

        "",

    ):

        failures += 1



    prev_topk = os.environ.get("GEOPACT_COORD_TOPK")

    prev_fmt = os.environ.get("GEOPACT_COORD_FORMAT")

    try:

        os.environ["GEOPACT_COORD_TOPK"] = "3"

        os.environ["GEOPACT_COORD_FORMAT"] = "sdb"

        coord_sdb = _build_coord_text(

            selected_miu_indices=[0, 1, 2],

            keep_token_set={0, 1, 2, 3, 4},

            miu_token_idxs=miu_token_idxs,

            mius=mius,

            policy="oas_ickh_cdct_coord_sdb_yolo",

            miu_order=[0, 1, 2],

        )

        if not _check(

            "SDB coord text via _build_coord_text",

            bool(coord_sdb and "A1<|box_start|>" in coord_sdb and "miu_0" not in coord_sdb),

            "",

        ):

            failures += 1

        if not _check(

            "SDB token budget heuristic",

            estimate_token_budget(coord_sdb or "") <= 70,

            f"est={estimate_token_budget(coord_sdb or '')}",

        ):

            failures += 1



        os.environ["GEOPACT_COORD_FORMAT"] = "legacy"

        os.environ["GEOPACT_COORD_TOPK"] = "5"

        coord_legacy = _build_coord_text(

            selected_miu_indices=[0, 1],

            keep_token_set={0, 1, 2, 3},

            miu_token_idxs=miu_token_idxs,

            mius=mius,

            policy="oas_ickh_cdct_coord_ephemeral_yolo",

            miu_order=[0, 1],

        )

        lines = [ln for ln in str(coord_legacy or "").splitlines() if ln.strip()]

        if not _check(

            "legacy coord text format",

            bool(coord_legacy and len(lines) == 2 and "miu_0" in coord_legacy),

            f"lines={len(lines)}",

        ):

            failures += 1

    finally:

        if prev_topk is None:

            os.environ.pop("GEOPACT_COORD_TOPK", None)

        else:

            os.environ["GEOPACT_COORD_TOPK"] = prev_topk

        if prev_fmt is None:

            os.environ.pop("GEOPACT_COORD_FORMAT", None)

        else:

            os.environ["GEOPACT_COORD_FORMAT"] = prev_fmt



    prev_ephemeral = os.environ.pop("GEOPACT_COORD_EPHEMERAL", None)

    prev_meta = os.environ.get("GEOPACT_COORD_METADATA_ONLY")

    try:

        os.environ["GEOPACT_COORD_EPHEMERAL"] = "1"

        os.environ["GEOPACT_COORD_METADATA_ONLY"] = "1"

        if not _check(

            "ephemeral forces splice (metadata_only off)",

            not _coord_metadata_only(),

            "",

        ):

            failures += 1

        if not _check(

            "coord_sdb policy active",

            _coord_policy_active("oas_ickh_cdct_coord_sdb_yolo"),

            "",

        ):

            failures += 1

        if not _check(

            "coord_ephemeral policy active",

            _coord_policy_active("oas_ickh_cdct_coord_ephemeral_yolo"),

            "",

        ):

            failures += 1

    finally:

        if prev_ephemeral is None:

            os.environ.pop("GEOPACT_COORD_EPHEMERAL", None)

        else:

            os.environ["GEOPACT_COORD_EPHEMERAL"] = prev_ephemeral

        if prev_meta is None:

            os.environ.pop("GEOPACT_COORD_METADATA_ONLY", None)

        else:

            os.environ["GEOPACT_COORD_METADATA_ONLY"] = prev_meta



    compact = list(range(100))
    tail = 5
    coord_n = 12
    ins = len(compact) - tail - coord_n
    end = len(compact) - tail - coord_n
    boundary = compact[:end]
    if not _check(
        "boundary strip arithmetic (legacy tail)",
        len(boundary) == 83 and boundary[-1] == 82,
        f"len={len(boundary)}",
    ):
        failures += 1

    from lmms_eval.models.geopact_coord_injection.lifecycle import strip_for_kv_commit
    from lmms_eval.models.geopact_coord_injection.record import (
        CoordInjectionMode,
        CoordInjectionRecord,
    )
    from lmms_eval.models.geopact_coord_injection.renderer import (
        render_spatial_plain,
        render_spatial_ranked,
    )

    plain = render_spatial_plain([(832, 73), (500, 670)])
    if not _check(
        "spatial_plain has no schema words",
        "action_type" not in plain
        and "<|box_start|>" not in plain
        and "(832,73)" in plain,
        plain[:80],
    ):
        failures += 1

    ranked = render_spatial_ranked([(100, 200)])
    if not _check(
        "spatial_ranked format",
        ranked.startswith("[Ref]") and "P1=(100,200)" in ranked,
        ranked,
    ):
        failures += 1

    from lmms_eval.models.geopact_coord_injection.renderer import (
        SIDECAR_HEADER,
        render_sidecar,
    )

    sample_entries = [
        {
            "rank": 0,
            "type": "icon_button",
            "box_xyxy": (128, 214, 346, 501),
            "text": "支付",
            "conf": 0.93,
        },
        {
            "rank": 1,
            "type": "text",
            "box_xyxy": (480, 180, 820, 270),
            "text": "",
            "conf": 0.88,
        },
    ]
    sidecar_l0 = render_sidecar(sample_entries, format_id="sidecar_bbox")
    if not _check(
        "GATS sidecar_bbox schema",
        SIDECAR_HEADER in sidecar_l0
        and "[KEEP] R1" in sidecar_l0
        and "bbox=[128,214,346,501]" in sidecar_l0
        and "role=" not in sidecar_l0
        and "action_type" not in sidecar_l0,
        sidecar_l0[:120],
    ):
        failures += 1

    sidecar_l3 = render_sidecar(sample_entries, format_id="sidecar_keep")
    if not _check(
        "GATS sidecar_keep full fields",
        "role=icon_button" in sidecar_l3
        and 'text="支付"' in sidecar_l3
        and "conf=0.9" in sidecar_l3
        and "text=-" in sidecar_l3,
        sidecar_l3[:160],
    ):
        failures += 1

    prev_fmt_sc = os.environ.get("GEOPACT_COORD_FORMAT")
    try:
        os.environ["GEOPACT_COORD_TOPK"] = "2"
        os.environ["GEOPACT_COORD_FORMAT"] = "sidecar_keep"
        coord_sidecar = _build_coord_text(
            selected_miu_indices=[0, 1],
            keep_token_set={0, 1, 2, 3, 4},
            miu_token_idxs=miu_token_idxs,
            mius=mius,
            policy="oas_ickh_cdct_coord_sdb_yolo",
            miu_order=[0, 1],
        )
        if not _check(
            "sidecar_keep via _build_coord_text",
            bool(coord_sidecar and "[KEEP] R1" in coord_sidecar and "bbox=[" in coord_sidecar),
            (coord_sidecar or "")[:100],
        ):
            failures += 1
        if not _check(
            "sidecar token budget heuristic",
            estimate_token_budget(coord_sidecar or "") <= 80,
            f"est={estimate_token_budget(coord_sidecar or '')}",
        ):
            failures += 1
    finally:
        if prev_fmt_sc is None:
            os.environ.pop("GEOPACT_COORD_FORMAT", None)
        else:
            os.environ["GEOPACT_COORD_FORMAT"] = prev_fmt_sc

    inj_rec = CoordInjectionRecord(
        text="ref",
        token_count=coord_n,
        insert_pos=ins,
        mode=CoordInjectionMode.EPHEMERAL,
        ephemeral=True,
        spliced=True,
        gen_tail_len=tail,
    )
    precise_boundary = strip_for_kv_commit(compact, record=inj_rec, gen_tail_len=tail)
    if not _check(
        "strip_for_kv_commit insert_pos",
        len(precise_boundary) == 83 and precise_boundary[-1] == 82,
        f"len={len(precise_boundary)}",
    ):
        failures += 1

    stale_rec = CoordInjectionRecord(
        text="ref",
        token_count=coord_n,
        insert_pos=999,
        mode=CoordInjectionMode.EPHEMERAL,
        ephemeral=True,
        spliced=True,
        gen_tail_len=tail,
    )
    stale_boundary = strip_for_kv_commit(compact, record=stale_rec, gen_tail_len=tail)
    if not _check(
        "strip_for_kv_commit stale insert_pos falls back to tail",
        len(stale_boundary) == 83 and stale_boundary[-1] == 82,
        f"len={len(stale_boundary)}",
    ):
        failures += 1

    from lmms_eval.models.geopact_vl_hook import _should_build_coord_text

    prev_eph2 = os.environ.pop("GEOPACT_COORD_EPHEMERAL", None)
    prev_meta2 = os.environ.get("GEOPACT_COORD_METADATA_ONLY")
    try:
        os.environ.pop("GEOPACT_COORD_EPHEMERAL", None)
        os.environ["GEOPACT_COORD_METADATA_ONLY"] = "1"
        if not _check(
            "should_build skips metadata-only without ephemeral",
            not _should_build_coord_text("oas_ickh_cdct_coord_sdb_yolo"),
            "",
        ):
            failures += 1
    finally:
        if prev_eph2 is None:
            os.environ.pop("GEOPACT_COORD_EPHEMERAL", None)
        else:
            os.environ["GEOPACT_COORD_EPHEMERAL"] = prev_eph2
        if prev_meta2 is None:
            os.environ.pop("GEOPACT_COORD_METADATA_ONLY", None)
        else:
            os.environ["GEOPACT_COORD_METADATA_ONLY"] = prev_meta2

    from lmms_eval.models.geopact_session_kv import _validate_suffix_prune_mrope_metadata



    pruned_prefill = {
        "geopact_compact_stats": {
            "orig_seq_len": 713,
            "compact_seq_len": 415,
            "n_visual_total": 646,
            "n_visual_kept": 258,
            "coord_ephemeral_tokens": 90,
            "coord_spliced_tokens": 90,
            "coord_ephemeral": True,
            "coord_injection": {
                "token_count": 90,
                "ephemeral": True,
                "spliced": True,
                "insert_pos": 325,
            },
        },
        "mrope_keep_mask": [True] * 325 + [False] * (713 - 325),
        "mrope_source_input_ids": list(range(713)),
    }

    try:

        audit = _validate_suffix_prune_mrope_metadata(

            pruned_prefill, suffix_len=415, where="verify"

        )

        if not _check(
            "MRoPE audit accepts coord_spliced tokens",
            audit.get("coord_spliced_tokens") == 90
            and audit.get("keep_mask_kept") == 325,
            f"audit={audit}",
        ):

            failures += 1

    except RuntimeError as exc:

        if not _check(
            "MRoPE audit accepts coord_spliced tokens",
            False,
            str(exc),
        ):

            failures += 1



    if failures:

        print(f"\nverify_dialogue_coord_ephemeral FAILED ({failures} checks)", file=sys.stderr)

        return 1

    print("\nverify_dialogue_coord_ephemeral PASSED")

    return 0





if __name__ == "__main__":

    raise SystemExit(main())

