# checkpoints/

TRACE does not ship training checkpoints. Inference uses public Hugging Face weights placed under [`../models/`](../models/README.md).

There is no extra TRACE-specific finetune to download. GUI-Owl-1.5 and UI-TARS-1.5 are used as released; TRACE is a training-free visual-token method plus a Transformers modeling patch (`patches/`).
