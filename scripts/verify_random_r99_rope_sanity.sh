#!/usr/bin/env bash
# GeoPACT MRoPE v3 sanity check — origpos must survive heavy prune.
# See scripts/verify_geopact_mrope_prune_sanity.sh for r=0.99 + r=0.20 matrix.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
EVAL="${EVAL_PYTHON:-python3}"
cd "$REPO"

export MIU_DISABLE_OCR=1
export LMMS_LIGHT_LOG=1
LIMIT="${LIMIT:-200}"

echo "=== Apply site-packages patches (MRoPE v3) ==="
$EVAL scripts/restore_q3vl_geopact_patches.py
PYC_DIR="$($EVAL -c 'import transformers.models.qwen3_vl.modeling_qwen3_vl as m, os; print(os.path.dirname(m.__file__))')"
rm -f "${PYC_DIR}/__pycache__/modeling_qwen3_vl"*.pyc

echo "=== Patch verification ==="
$EVAL scripts/check_q3vl_import.py

run_one() {
  local tag="$1"
  local extra="$2"
  local out="outputs/_sanity_rope_${tag}"
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
  grep -E '^(OVERALL ALL|LATENCY_EPISODE)' "${out}/run.log" | tail -3
}

run_one "baseline" ""
run_one "random_r99_scope_image" ",visual_prune_ratio=0.99,visual_prune_policy=random,visual_prune_scope=image,miu_detector=canny_morph"

echo "=== Compare EM from logs ==="
$EVAL - <<'PY'
import re
from pathlib import Path

def em(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^OVERALL ALL .* EM=([\d.]+)%", text, re.M)
    return float(m.group(1)) if m else None

base = em("outputs/_sanity_rope_baseline/run.log")
r99 = em("outputs/_sanity_rope_random_r99_scope_image/run.log")
print(f"baseline EM={base}%  random_r99 EM={r99}%  delta={None if base is None or r99 is None else r99-base:.2f}pp")
if base is not None and r99 is not None:
    if r99 < base - 5.0:
        print("FAIL: r=0.99 random EM dropped >5pp — MRoPE chain still broken")
        raise SystemExit(1)
    print("PASS: r=0.99 random within 5pp of baseline")
PY
