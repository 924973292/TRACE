#!/usr/bin/env python
"""Rescore AndroidControl arms under the HistPrune official protocol
(https://github.com/ldq20040602/HistPrune-GUI AndroidControl_sum.py) and
compare with our stored protocol, offline from existing samples.jsonl.

HistPrune protocol:
  Type   = pred action_type == gt action_type, over all steps
  Ground = among type-matched click/long_press steps, normalized Euclidean
           distance <= 0.14 (their "distance <= 140" on 0-1000 coords)
  Text   = among type-matched type/open_app steps, containment match
  Step   = type match AND all fields match (dist<=0.14, text containment)

Ours (stored): tap threshold 0.04, text containment OR similarity>=0.8.

samples.jsonl rows: doc is None; gt lives in `target` (python-dict repr),
pred text in filtered_resps. Screen size defaults to 1080x2400 (validated
against stored exact_match agreement, reported as `agree`).

Usage: python scripts/_ac_hist_metrics.py [logs_root ...]
"""
import ast
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmms_eval.tasks.gui_owl_cross.android_control_eval import (
    _normalize_ac_type,
    _norm_coord,
    ac_actions_match,
    guiowl_pred_to_ac_action,
)

HIST_TAP_THRESHOLD = 0.14
DEFAULT_SCREEN = (1080, 2400)


def _resp_text(row):
    for key in ("filtered_resps", "resps"):
        v = row.get(key)
        while isinstance(v, list) and v:
            v = v[0]
        if isinstance(v, str) and v:
            return v
    return ""


def _parse_target(row):
    t = row.get("target")
    if isinstance(t, dict):
        return t
    if isinstance(t, str) and t.strip():
        try:
            v = ast.literal_eval(t)
            return v if isinstance(v, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    return {}


def _text_contain(p, g):
    p = str(p or "").strip().lower()
    g = str(g or "").strip().lower()
    if not p or not g:
        return p == g
    return p == g or p in g or g in p


def hist_step(pred, gt, screen):
    """Return (type_ok, ground_tot, ground_ok, text_tot, text_ok, step_ok)."""
    g_type = _normalize_ac_type(gt.get("action_type", ""))
    p_type = _normalize_ac_type(pred.get("action_type", ""))
    type_ok = bool(p_type) and p_type == g_type
    ground_tot = ground_ok = text_tot = text_ok = 0
    step_ok = False
    if not type_ok:
        return type_ok, ground_tot, ground_ok, text_tot, text_ok, step_ok
    if g_type in ("click", "long_press") and gt.get("x") is not None:
        ground_tot = 1
        if pred.get("x") is not None and pred.get("y") is not None:
            w, h = screen
            px, py = _norm_coord(float(pred["x"]), w), _norm_coord(float(pred["y"]), h)
            gx, gy = _norm_coord(float(gt["x"]), w), _norm_coord(float(gt["y"]), h)
            if math.hypot(px - gx, py - gy) <= HIST_TAP_THRESHOLD:
                ground_ok = 1
        step_ok = bool(ground_ok)
    elif g_type in ("type",):
        text_tot = 1
        text_ok = int(_text_contain(pred.get("text"), gt.get("text")))
        step_ok = bool(text_ok)
    elif g_type == "open_app":
        text_tot = 1
        text_ok = int(_text_contain(pred.get("app_name", pred.get("text")), gt.get("app_name")))
        step_ok = bool(text_ok)
    elif g_type == "scroll":
        step_ok = str(pred.get("direction", "")).lower() == str(gt.get("direction", "")).lower()
    else:  # navigate_back / navigate_home / wait
        step_ok = True
    return type_ok, ground_tot, ground_ok, text_tot, text_ok, step_ok


def score_file(path):
    n = 0
    agree = 0
    h = dict(type=0, g_tot=0, g_ok=0, t_tot=0, t_ok=0, step=0)
    ours_em = 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        gt = _parse_target(row)
        if not gt:
            continue
        pred = guiowl_pred_to_ac_action(_resp_text(row))
        n += 1
        t, g_tot, g_ok, t_tot, t_ok, s = hist_step(pred, gt, DEFAULT_SCREEN)
        h["type"] += t
        h["g_tot"] += g_tot
        h["g_ok"] += g_ok
        h["t_tot"] += t_tot
        h["t_ok"] += t_ok
        h["step"] += s
        # validation: recompute our protocol and compare to stored flag
        ours = ac_actions_match(pred, gt, screen_size=DEFAULT_SCREEN)
        ours_em += ours
        stored = row.get("exact_match")
        if isinstance(stored, dict):
            stored = stored.get("exact_match")
        if stored is not None and bool(stored) == bool(ours):
            agree += 1
    return n, h, ours_em, agree


def main():
    roots = sys.argv[1:] or ["logs/xds8b_full"]
    files = sorted(
        {
            f
            for r in roots
            for f in glob.glob(os.path.join(r, "android_control", "*", "results", "*", "*samples*.jsonl"))
        }
    )
    hdr = f"{'arm':<22} {'n':>5} | {'Type':>6} {'Ground':>7} {'Text':>6} {'StepSR':>7} | {'ourEM':>6} {'agree':>6}"
    print(hdr)
    print("-" * len(hdr))
    for f in files:
        parts = f.replace("\\", "/").split("/")
        arm = parts[parts.index("android_control") + 1]
        n, h, ours_em, agree = score_file(f)
        if not n:
            continue
        print(
            f"{arm:<22} {n:>5} | {100*h['type']/n:>6.2f} "
            f"{(100*h['g_ok']/h['g_tot'] if h['g_tot'] else float('nan')):>7.2f} "
            f"{(100*h['t_ok']/h['t_tot'] if h['t_tot'] else float('nan')):>6.2f} "
            f"{100*h['step']/n:>7.2f} | {100*ours_em/n:>6.2f} {100*agree/n:>6.2f}"
        )


if __name__ == "__main__":
    main()
