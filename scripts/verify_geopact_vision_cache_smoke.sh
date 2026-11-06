#!/usr/bin/env bash
# Compare hist_k5 with vision embed cache off vs on (limit small).
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
EVAL="${EVAL_PYTHON:-python3}"
cd "$REPO"
source scripts/geopact_yolo_eval_env.sh
export GEOPACT_COMPACT_BACKEND=source
$EVAL scripts/restore_q3vl_geopact_patches.py

LIMIT="${LIMIT:-20}"
TASK="task_v2_guiowl_imageonly_a_hist_k5"
MODEL_ARGS="pretrained=models/GUI-Owl-1.5-2B-Instruct,attn_implementation=flash_attention_2,device_map=auto,visual_prune_ratio=0.05,visual_prune_policy=random,visual_prune_scope=image_sticky"

run_one() {
  local tag="$1"
  local cache_flag="$2"
  local out="outputs/_vision_cache_smoke_${tag}_limit${LIMIT}"
  mkdir -p "$out"
  echo "=== RUN ${tag} (GEOPACT_VISION_EMBED_CACHE=${cache_flag}) ==="
  GEOPACT_VISION_EMBED_CACHE="$cache_flag" \
  $EVAL -m lmms_eval \
    --model gui_owl \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASK" \
    --batch_size 1 \
    --limit "$LIMIT" \
    --log_samples \
    --output_path "$out" \
    2>&1 | tee "${out}/run.log"
  echo "--- summary ${tag} ---"
  grep -E 'GeoPACT vision-cache:|OVERALL ALL|LATENCY_STEP|InputTokens=' "${out}/run.log" | tail -20 || true
}

echo "=== unit: vision cache module ==="
$EVAL - <<'PY'
from lmms_eval.models.geopact_vision_cache import (
    _per_image_pixel_lengths,
    _split_pixel_values,
    vision_embed_cache_enabled,
    clear_vision_embed_cache,
)
import torch

assert not vision_embed_cache_enabled()
grid = torch.tensor([[1, 4, 6], [1, 4, 6]])
assert _per_image_pixel_lengths(grid) == [24, 24]
pv = torch.randn(48, 16)
slices = _split_pixel_values(pv, grid)
assert len(slices) == 2 and slices[0].shape[0] == 24
print("vision_cache unit: OK")
PY

run_one cache_off 0
run_one cache_on 1

echo "DONE — compare vision_encode_ms / LATENCY_STEP between cache_off and cache_on logs."
