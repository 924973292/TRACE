#!/usr/bin/env bash
# Phased smoke tests: GeoPACT + GUI-Owl on hist_k5 (FIFO K=5 history screenshots).
# Run from repo root on remote GPU machine.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
EVAL="${EVAL_PYTHON:-python3}"
TASK="task_v2_guiowl_imageonly_a_hist_k5"
MODEL="gui_owl"
PRETRAINED="models/GUI-Owl-1.5-2B-Instruct"
LIMIT="${LIMIT:-10}"
OUT_ROOT="${OUT_ROOT:-outputs/_smoke_geopact_hist_k5}"

cd "$REPO"
export MIU_DISABLE_OCR=1
export LMMS_LIGHT_LOG=1

echo "=== Phase 0: site-packages patches + pyc (Qwen3-VL / GUI-Owl) ==="
$EVAL scripts/fix_q3vl_rope.py || true
$EVAL scripts/fix_q3vl_deepstack.py || true
$EVAL scripts/fix_reset_position_ids.py || true
PYC_DIR="$(
  $EVAL - <<'PY'
import transformers.models.qwen3_vl.modeling_qwen3_vl as m
import os
print(os.path.dirname(m.__file__))
PY
)"
rm -f "${PYC_DIR}/__pycache__/modeling_qwen3_vl"*.pyc

$EVAL - <<'PY'
import inspect
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
src = inspect.getsource(Qwen3VLModel.forward)
ok = "GeoPACT" in src and "reset position_ids" in src
print("qwen3_vl patch:", "OK" if ok else "MISSING")
assert ok
PY

echo "=== Phase 1: hist_k5 message image counts (data preprocessing) ==="
$EVAL - <<'PY'
import json
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "utils", Path("lmms_eval/tasks/agentcpm_gui/utils.py")
)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

docs = json.loads(Path("data/merged_sorted.json").read_text(encoding="utf-8"))
kwargs = {
    "media_dir": "data/benchmark",
    "coordinate_type": "normalized",
    "history_image_mode": "fifo",
    "max_history_images": 5,
}
for i in range(min(12, len(docs))):
    doc = docs[i]
    msgs = utils.ablation_v2_no_av_hist_k5_doc_to_messages(doc, kwargs)
    nimg = sum(
        1
        for m in msgs
        for p in (m.get("content") or [])
        if isinstance(p, dict) and p.get("type") == "image"
    )
    print(
        f"doc_id={i} episode={doc.get('episode_id')} step={doc.get('step_index')} "
        f"hist={len(doc.get('action_history') or [])} n_images={nimg}"
    )
PY

echo "=== Phase 1b: import + sticky token ranges (spatial_merge_size=2) ==="
$EVAL - <<'PY'
from lmms_eval.models.geopact_vl_hook import make_geopact_hook, _flat_image_token_ranges, _llm_tokens_per_image, _spatial_merge_size
from lmms_eval.models.simple.qwen_vl_omnigui_common import cache_geopact_image
import torch

hook = make_geopact_hook(0.1, "random", visual_prune_scope="image_sticky")
assert callable(hook)

merge = _spatial_merge_size()
raw_h, raw_w = 14, 18
n_per = _llm_tokens_per_image(1, raw_h, raw_w, merge)
grid = torch.tensor([[1, raw_h, raw_w], [1, raw_h, raw_w], [1, raw_h, raw_w]])
ranges = _flat_image_token_ranges(grid, n_per * 3)
assert len(ranges) == 3, ranges
assert all(e - s == n_per for s, e in ranges), ranges
print("hook + token_ranges (merged): OK", "n_per=", n_per, "ranges=", ranges)
PY

run_eval() {
  local tag="$1"
  local extra_args="${2:-}"
  local out="${OUT_ROOT}/${tag}"
  mkdir -p "$out"
  echo "=== Phase run: ${tag} ==="
  GUIOWL_DEBUG_MESSAGES=1 GUIOWL_DEBUG_VISION=0 \
  $EVAL -m lmms_eval \
    --model "$MODEL" \
    --model_args "pretrained=${PRETRAINED},attn_implementation=flash_attention_2,device_map=auto${extra_args}" \
    --tasks "$TASK" \
    --batch_size 1 \
    --limit "$LIMIT" \
    --log_samples \
    --output_path "$out" \
    2>&1 | tee "${out}/nohup.log"
  grep -E "^(OVERALL|LATENCY_STEP|LATENCY_EPISODE|DIM_|FAIL|Error|Traceback)" "${out}/nohup.log" | tail -25 || true
}

echo "=== Phase 2: baseline (no pruning) ==="
run_eval "baseline" ""

echo "=== Phase 3: random prune r=0.10 scope=image_sticky ==="
run_eval "random_r10_sticky" ",visual_prune_ratio=0.10,visual_prune_policy=random,visual_prune_scope=image_sticky,miu_detector=canny_morph"

echo "=== Phase 4: GeoPACT A+B (Canny) r=0.10 image_sticky ==="
run_eval "ab_canny_r10_sticky" ",visual_prune_ratio=0.10,visual_prune_policy=oas_ickh_no_coord,visual_prune_scope=image_sticky,miu_detector=canny_morph"

echo "=== Phase 5: GeoPACT A+B+C (Canny) r=0.10 image_sticky ==="
run_eval "abc_canny_r10_sticky" ",visual_prune_ratio=0.10,visual_prune_policy=oas_ickh_coord_center,visual_prune_scope=image_sticky,miu_detector=canny_morph"

echo "ALL SMOKE PHASES DISPATCHED — check each ${OUT_ROOT}/*/nohup.log for errors."
