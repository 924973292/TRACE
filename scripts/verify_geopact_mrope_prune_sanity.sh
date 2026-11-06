#!/usr/bin/env bash
# MRoPE sanity: r=0.99 should match baseline; r=0.20 uniform should not collapse EM.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
EVAL="${EVAL_PYTHON:-python3}"
cd "$REPO"

export MIU_DISABLE_OCR=1
export LMMS_LIGHT_LOG=1
LIMIT="${LIMIT:-200}"

echo "=== Apply GeoPACT MRoPE v3 patches ==="
$EVAL scripts/restore_q3vl_geopact_patches.py
PYC_DIR="$($EVAL -c 'import transformers.models.qwen3_vl.modeling_qwen3_vl as m, os; print(os.path.dirname(m.__file__))')"
rm -f "${PYC_DIR}/__pycache__/modeling_qwen3_vl"*.pyc

echo "=== Patch verification ==="
$EVAL scripts/check_q3vl_import.py

run_one() {
  local tag="$1"
  local extra="$2"
  local out="outputs/_sanity_mrope_${tag}"
  mkdir -p "$out"
  echo "=== RUN $tag limit=$LIMIT ==="
  $EVAL -m lmms_eval \
    --model gui_owl \
    --model_args "pretrained=models/GUI-Owl-1.5-2B-Instruct,attn_implementation=flash_attention_2,device_map=auto${extra}" \
    --tasks task_v2_guiowl_imageonly_a_hist_k5 \
    --batch_size 1 \
    --limit "$LIMIT" \
    --log_samples \
    --output_path "$out" \
    2>&1 | tee "${out}/run.log"
  grep -E '^(OVERALL ALL|LATENCY_EPISODE|GeoPACT compact)' "${out}/run.log" | tail -5 || true
}

run_one "baseline" ""
run_one "random_r99" ",visual_prune_ratio=0.99,visual_prune_policy=random,visual_prune_scope=image,miu_detector=canny_morph"
run_one "uniform_r20" ",visual_prune_ratio=0.20,visual_prune_policy=uniform,visual_prune_scope=image"

echo "=== Compare EM from logs ==="
$EVAL - <<'PY'
import re
from pathlib import Path

def em(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^OVERALL ALL .* EM=([\d.]+)%", text, re.M)
    return float(m.group(1)) if m else None

base = em("outputs/_sanity_mrope_baseline/run.log")
r99 = em("outputs/_sanity_mrope_random_r99/run.log")
r20 = em("outputs/_sanity_mrope_uniform_r20/run.log")
print(f"baseline EM={base}%  random_r99 EM={r99}%  uniform_r20 EM={r20}%")
if base is None or r99 is None or r20 is None:
    raise SystemExit("FAIL: missing EM in logs")
if r99 < base - 8.0:
    raise SystemExit(f"FAIL: r=0.99 EM dropped {base - r99:.1f}pp vs baseline")
if r20 < 5.0:
    raise SystemExit(f"FAIL: r=0.20 uniform EM={r20}% still collapsed (<5%)")
if r20 < base * 0.15:
    print(f"WARN: r=0.20 EM={r20}% is <15% of baseline — check MIU selection next")
print("PASS: MRoPE prune sanity (r99 within 8pp; r20 EM >= 5%)")
PY
