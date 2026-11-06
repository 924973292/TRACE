"""Unit smoke for the self-conditioning store: no model, no GPU.

Checks the properties that make or break the protocol:
  1. step 0 must be byte-identical to teacher forcing (no history to differ on),
  2. at step t>0 the history must carry the PREDICTED action, not the GT one,
  3. screenshots must still be the ground-truth frames,
  4. a missing prefix must degrade to GT rather than fabricate history,
  5. an unparseable generation must be recorded explicitly, not silently GT,
  6. the injection file must override the prediction at the targeted step,
  7. the tail invariant must fire when the override does not take effect.
"""

import json
import os
import sys

import importlib.util

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# Load by path: `lmms_eval.tasks.__init__` imports `datasets` and the whole task
# registry, which this unit check neither needs nor should depend on.
_spec = importlib.util.spec_from_file_location(
    "_selfcond_under_test",
    os.path.join(_ROOT, "lmms_eval", "tasks", "self_conditioned.py"),
)
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)


def make_doc(step, n_hist=3):
    return {
        "subset": "omnigui",
        "episode_id": "ep777",
        "step_index": step,
        "action_history": [
            {
                "step_index": i,
                "action_text": "click(start_box='(100,200)')",
                "image_path": f"/frames/ep777/{i}.png",
            }
            for i in range(n_hist)
        ],
    }


def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


def t1_disabled_by_default():
    sc.reset()
    os.environ.pop("GEOPACT_SELFCOND", None)
    if sc.enabled():
        fail("selfcond active without the env flag")
    print("ok  disabled by default")


def t2_step0_is_empty_history():
    sc.reset()
    os.environ["GEOPACT_SELFCOND"] = "1"
    if sc.history_for(make_doc(0)) != []:
        fail("step 0 must yield an empty history, identical to teacher forcing")
    print("ok  step 0 identical to teacher forcing")


def t3_history_is_prediction_frames_are_gt():
    sc.reset()
    os.environ["GEOPACT_SELFCOND"] = "1"
    for i in range(3):
        sc.record(make_doc(i), "raw", {"action_type": 0, "coordinate": [i, i]})
    hist = sc.history_for(make_doc(3))
    if hist is None or len(hist) != 3:
        fail(f"expected 3 history entries, got {hist}")
    for i, h in enumerate(hist):
        act = json.loads(h["action_json"])
        if act.get("coordinate") != [i, i]:
            fail(f"history step {i} does not carry the prediction: {act}")
        if h.get("image_path") != f"/frames/ep777/{i}.png":
            fail(f"history step {i} lost the ground-truth frame: {h}")
    print("ok  history carries predictions, frames stay ground truth")


def t4_missing_prefix_degrades():
    sc.reset()
    os.environ["GEOPACT_SELFCOND"] = "1"
    sc.record(make_doc(0), "raw", {"action_type": 0})
    # Steps 1 and 2 were never produced, so the fallback template covers steps
    # the store does not know about and the whole history must degrade to GT.
    if sc.history_for(make_doc(3)) is not None:
        fail("incomplete prefix must return None so the caller falls back to GT")
    print("ok  incomplete prefix degrades to GT instead of fabricating")


def t5_unparseable_is_explicit():
    sc.reset()
    os.environ["GEOPACT_SELFCOND"] = "1"
    rec = sc.record(make_doc(0), "total garbage", None, gt_action_json='{"action_type": 4}')
    if not rec["unparseable"]:
        fail("unparseable generation was not flagged")
    if rec["action"].get("action_type") == 4:
        fail("unparseable generation silently fell back to GT -> teacher forcing")
    hist = sc.history_for(make_doc(1))
    if json.loads(hist[0]["action_json"]).get("action_type") == 4:
        fail("GT leaked into history through the unparseable path")
    print("ok  unparseable recorded explicitly, no GT leak")


def t6_injection_overrides():
    sc.reset()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_w60_unit_inject.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "episode_id": "ep777", "step_index": 1,
            "action": {"action_type": 8}, "arm": "inject",
        }) + "\n")
    os.environ["GEOPACT_SELFCOND"] = "1"
    os.environ["GEOPACT_SELFCOND_INJECT"] = path
    try:
        r0 = sc.record(make_doc(0), "raw", {"action_type": 0, "coordinate": [5, 5]})
        r1 = sc.record(make_doc(1), "raw", {"action_type": 0, "coordinate": [7, 7]})
        if r0["injected"] or r0["action"].get("coordinate") != [5, 5]:
            fail("step 0 must not be injected")
        if not r1["injected"] or r1["action"].get("action_type") != 8:
            fail(f"step 1 injection did not take effect: {r1['action']}")
        if r1["predicted_action"].get("coordinate") != [7, 7]:
            fail("the true prediction must still be auditable after injection")
        if json.loads(sc.history_for(make_doc(2))[1]["action_json"])["action_type"] != 8:
            fail("injected action did not reach the history")
    finally:
        os.environ.pop("GEOPACT_SELFCOND_INJECT", None)
        os.remove(path)
    print("ok  injection overrides history at the targeted step only")


def t8_template_preserves_gt_fields():
    """The GT history is a template: only action_json may change.

    This is the path the live hook takes, and it is what guarantees the only
    difference between the two protocols is the action text.
    """
    sc.reset()
    os.environ["GEOPACT_SELFCOND"] = "1"
    gt_hist = [
        {"step_index": 0, "action_json": '{"action_type": 4}',
         "image_path": "/frames/ep777/0.png", "step_instruction": "open the app"},
        {"step_index": 2, "action_json": '{"action_type": 7, "text": "hi"}',
         "image_path": "/frames/ep777/2.png", "step_instruction": "type it"},
    ]
    sc.record(make_doc(0), "raw", {"action_type": 0, "coordinate": [3, 3]})
    doc2 = make_doc(2)
    doc2["step_index"] = 2
    sc.record(doc2, "raw", {"action_type": 8})
    doc = make_doc(5)
    out = sc.history_for(doc, gt_history=gt_hist)
    if out is None or len(out) != 2:
        fail(f"template path lost entries: {out}")
    for orig, new in zip(gt_hist, out):
        for k in ("step_index", "image_path", "step_instruction"):
            if new.get(k) != orig.get(k):
                fail(f"template field {k} was altered: {orig} -> {new}")
        if new["action_json"] == orig["action_json"]:
            fail("action_json was NOT replaced, so history is still GT")
    if json.loads(out[0]["action_json"]).get("coordinate") != [3, 3]:
        fail("non-contiguous step ids broke the action lookup")
    if json.loads(out[1]["action_json"]).get("action_type") != 8:
        fail("step 2 action did not reach the template entry")
    print("ok  GT history used as template, only action_json rewritten")


def t9_missing_template_step_degrades():
    sc.reset()
    os.environ["GEOPACT_SELFCOND"] = "1"
    gt_hist = [{"step_index": 0, "action_json": "{}"},
               {"step_index": 1, "action_json": "{}"}]
    sc.record(make_doc(0), "raw", {"action_type": 0})
    if sc.history_for(make_doc(2), gt_history=gt_hist) is not None:
        fail("a template step with no stored action must degrade to GT")
    print("ok  template step without a prediction degrades to GT")


def t7_tail_invariant_fires():
    sc.reset()
    os.environ["GEOPACT_SELFCOND"] = "1"
    os.environ["GEOPACT_SELFCOND_STRICT"] = "1"
    sc.record(make_doc(0), "raw", {"action_type": 0, "coordinate": [1, 1]})
    render = lambda step, act: f"step{step}:{json.dumps(act, sort_keys=True)}"
    good = [
        {"role": "assistant", "content": [{"type": "text", "text": render(0, {"action_type": 0, "coordinate": [1, 1]})}]},
        {"role": "user", "content": [{"type": "text", "text": "now"}]},
    ]
    sc.check_history_tail(make_doc(1), good, render)
    bad = [
        {"role": "assistant", "content": [{"type": "text", "text": render(0, {"action_type": 4})}]},
        {"role": "user", "content": [{"type": "text", "text": "now"}]},
    ]
    try:
        sc.check_history_tail(make_doc(1), bad, render)
    except RuntimeError:
        print("ok  tail invariant rejects a GT-carrying history")
        return
    fail("tail invariant did NOT fire on a GT-carrying history")


def t10_missing_inject_file_is_loud():
    sc.reset()
    os.environ["GEOPACT_SELFCOND"] = "1"
    os.environ["GEOPACT_SELFCOND_INJECT"] = "/no/such/inject.jsonl"
    try:
        sc.injected_for(make_doc(0))
    except FileNotFoundError:
        os.environ.pop("GEOPACT_SELFCOND_INJECT", None)
        print("ok  missing inject file raises instead of silently going clean")
        return
    os.environ.pop("GEOPACT_SELFCOND_INJECT", None)
    fail("missing inject file did not raise")


if __name__ == "__main__":
    t1_disabled_by_default()
    t2_step0_is_empty_history()
    t3_history_is_prediction_frames_are_gt()
    t4_missing_prefix_degrades()
    t5_unparseable_is_explicit()
    t6_injection_overrides()
    t7_tail_invariant_fires()
    t8_template_preserves_gt_fields()
    t9_missing_template_step_degrades()
    t10_missing_inject_file_is_loud()
    print("\nall self-conditioning unit checks passed")
