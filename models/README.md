# models/

Put checkpoints here. Weights are not in git.

```bash
python scripts/download_omini_models.py --models-dir models trace
```

Expected directories:

```text
models/GUI-Owl-1.5-8B-Instruct/
models/GUI-Owl-1.5-2B-Instruct/
models/UI-TARS-1.5-7B/
models/omniparser/icon_detect/model.pt
```

Set `MODEL_DIR` and `OMNIPARSER_ICON_WEIGHTS` accordingly. Missing OmniParser weights fail the LIP detector (no silent `yolov8n.pt` fallback unless `GEOPACT_ALLOW_FALLBACK_YOLO=1`).
