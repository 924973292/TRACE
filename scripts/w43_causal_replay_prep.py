#!/usr/bin/env python
"""W43-B: causal replay preparation (interventions only; no replay here).

The commitment-error audit is observational: it reports whether the
acted-on patch survives retirement, and the conditional Step SR split is
directional because retention correlates with frame content. The causal
version intervenes on the SAME frames at matched budgets:

  restore-arm  for steps whose in-window history frame committed a keep
               that misses the future-demand patch, force that patch into
               S_hist by evicting the lowest-ranked kept token
               (budget unchanged);
  delete-arm   for steps whose history frame retained the patch, evict it
               and admit the highest-ranked excluded token
               (budget unchanged).

Teacher-forced replay of exactly these steps under the two overrides, versus
the unmodified run, then attributes any Step SR change to the presence of
the single future-demand patch. Both arms are needed: restore tests
sufficiency of the retained patch, delete tests necessity.

This script only PREPARES the intervention files from an existing dump; the
replay itself must run through the eval harness with the keep-override hook
(GEOPACT_KEEP_OVERRIDE=<overrides.jsonl>), which loads per-(episode, step,
frame) token sets in place of the dumped nested keep. If the hook is not yet
implemented in the harness, implement it before running; do not simulate.

Usage (remote):
    "${EVAL_PYTHON}" scripts/w43_causal_replay_prep.py \
        <dump.jsonl> <run_dir> <out_prefix> [H]

Emits:
    <out_prefix>_restore.jsonl   keep overrides for the restore arm
    <out_prefix>_delete.jsonl    keep overrides for the delete arm
    <out_prefix>_manifest.json   step lists, counts, and the result schema
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _w41_commit_conditional import (find_samples, gt_rect_from_target,  # noqa: E402
                                     img_size, load_jsonl, rect_to_patches)


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    dump_path, run_dir, out_prefix = sys.argv[1:4]
    h_window = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

    episodes = {}
    for rec in load_jsonl(dump_path):
        if rec.get("is_current"):
            episodes.setdefault(rec["episode_key"].split("-", 1)[-1],
                                []).append(rec)

    restore, delete = [], []
    for ep, recs in episodes.items():
        for t, rec in enumerate(recs):
            if t == len(recs) - 1:
                continue
            rect = gt.get((ep, t))
            if rect is None:
                continue
            size = img_size(rec.get("image_path", ""))
            if size is None:
                continue
            if max(rect) <= 1.5:
                rect = (rect[0] * size[0], rect[1] * size[1],
                        rect[2] * size[0], rect[3] * size[1])
            g = rect_to_patches(rect, size[0], size[1],
                                rec["grid_h"], rec["grid_w"])
            order = rec.get("nap_order") or []
            if not order or not rec.get("ratio_hist"):
                continue
            k_h = max(1, int(round(rec["n_frame"] * float(rec["ratio_hist"]))))
            s_hist = list(order[:k_h])
            hit = sorted(g & set(s_hist))
            if not hit:
                target_tok = sorted(g & set(rec.get("local_keep") or []))
                if not target_tok:
                    continue  # entry miss: nothing committable to restore
                # evict the lowest-ranked history token, admit the patch
                new_keep = s_hist[:-1] + [target_tok[0]]
                restore.append({"episode_id": ep, "frame_step": t,
                                "image_path": rec.get("image_path", ""),
                                "arm": "restore", "k_h": k_h,
                                "keep": sorted(new_keep)})
            else:
                # evict the patch token(s); admit next-ranked excluded tokens
                removed = [j for j in s_hist if j in g]
                kept = [j for j in s_hist if j not in g]
                refill = [j for j in order[k_h:] if j not in g]
                new_keep = kept + refill[:len(removed)]
                if len(new_keep) != k_h:
                    continue  # cannot match budget; skip rather than distort
                delete.append({"episode_id": ep, "frame_step": t,
                               "image_path": rec.get("image_path", ""),
                               "arm": "delete", "k_h": k_h,
                               "keep": sorted(new_keep)})

    for arm, rows in (("restore", restore), ("delete", delete)):
        p = f"{out_prefix}_{arm}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{arm}: {len(rows)} frame interventions -> {p}")

    manifest = {
        "dump": dump_path, "run_dir": run_dir, "h_window": h_window,
        "n_restore": len(restore), "n_delete": len(delete),
        "replay_hook": "GEOPACT_KEEP_OVERRIDE=<overrides.jsonl>",
        "replay_scope": ("teacher-forced replay of only the steps whose "
                         "in-window history contains an intervened frame; "
                         "all other frames use the original dumped keeps"),
        "result_schema": {
            "per_arm": {"steps": "int", "step_sr": "pct",
                        "step_sr_baseline_same_steps": "pct",
                        "diff": "pct", "ci95_episode": "[lo, hi]"},
            "claim_rule": ("a causal statement enters the paper only if the "
                           "episode-bootstrap interval excludes zero; "
                           "otherwise report as directional")},
    }
    with open(f"{out_prefix}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print("W43_CAUSAL_PREP_DONE", json.dumps({k: manifest[k] for k in
                                              ("n_restore", "n_delete")}))


if __name__ == "__main__":
    main()
