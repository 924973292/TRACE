# Transformers modeling patches

TRACE installs modified `modeling_*.py` files into the eval venv. Do not edit site-packages by hand.

| File | Carrier |
|------|---------|
| `qwen3_vl/modeling_qwen3_vl.py` | GUI-Owl-1.5 8B / 2B |
| `qwen2_5_vl/modeling_qwen2_5_vl.py` | UI-TARS-1.5-7B |
| `qwen2_vl/modeling_qwen2_vl.py` | Qwen2-VL path (2B VL) |

```bash
export EVAL_PYTHON=/path/to/venv/bin/python
python scripts/restore_q3vl_geopact_patches.py
python scripts/restore_q25vl_geopact_patches.py   # UI-TARS only
```

The restore scripts copy the files above into `site-packages/transformers/models/...`, then reload the module. `EVAL_PYTHON` must be the interpreter that will run `lmms_eval`.
