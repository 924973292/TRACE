# GUI-Owl native_mt4 Correctness Standard

## Prompt And History

- Task: `task_v2_guiowl_native_mt4`.
- System prompt and `mobile_use` output schema: GUI-Owl / Mobile-Agent-v3.5 source at commit `8cf3966e`, with a required OmniGUI coordinate wording patch.
- Raw official system prompt SHA256: `275599396258056288eb58e66697f178f8b38f0ada9955f1181940c3f3ce5fcf`.
- Native resized-pixel system prompt SHA256: `802248033effdbf895fe50ca9b8b79ed71d34e845db5f730f0f670249a1326bf`.
- Native `task_v2_guiowl_native_mt4` prompt must state that coordinates use the Qwen-resized screenshot pixel space shown in the current user message.
- Coordinate protocol: `[x,y]`, `x` left-to-right and `y` top-to-bottom, in pixels of the screenshot after Qwen vision preprocessing.
- Locked evaluation history: `history_n=6`, rolling latest-history window.
- Prompt image budget: at most 6 historical screenshots plus the current screenshot.

For step 0, the user turn contains the instruction text and current screenshot.
For steps 1-6, the first retained history user turn contains the instruction
text and oldest retained history screenshot; later history user turns are
image-only; each retained history action is a teacher-forced assistant
`Action: ... <tool_call>` turn. The current user turn is image-only.

For steps greater than 6, only the latest 6 historical screenshots/actions are
retained. Older actions are summarized in the `Previous actions:` text.

## Coordinate Semantics

OmniGUI native_mt4 scoring uses `coordinate_type: qwen_resized_pixel`. Model
outputs, teacher-forced retained history actions, Module C coordinate text, and
GT comparison must use the resized pixel coordinates of the corresponding Qwen
vision input.

The original GUI-Owl/Mobile-Agent prompt says "The screen's resolution is
1000x1000." In this benchmark that sentence is misleading because input
screenshots can have arbitrary real pixel sizes and Qwen can smart-resize them
before vision encoding. The actual native_mt4 prompt keeps the official
`mobile_use` schema but rewrites that line to describe the resized-pixel
coordinate space. GT `result_touch_xy` is treated as a raw-image pixel bbox,
clamped to the raw image bounds, scaled to the Qwen-resized image size, and a
TAP/LONG_PRESS is exact when the predicted resized-pixel point falls inside the
scaled bbox.

## Vision Cache Semantics

When pruning is enabled and vision cache is enabled, the cache stores only
post-prune compact tensors for a screenshot after it has appeared as the current
frame:

- `compact_embeds`
- `compact_deepstack`
- `local_keep_seq`
- `orig_grid_row`

The cache does not store MRoPE `position_ids` or `rope_deltas`. Those must be
recomputed for each prompt because text before the image block can change across
steps.

When the same screenshot appears later as a history frame, the compact path must
reuse the cached `local_keep_seq` before sequence compaction. The compact
embeddings, every deepstack layer, and visual placeholder count must all have
the same row count.

## MRoPE Semantics

The default pruning path is original-position MRoPE:

1. Compute full-prompt `position_ids` from the original `input_ids`,
   original `image_grid_thw`, and attention mask.
2. Build the compact sequence by dropping unkept visual tokens.
3. Subset the full-prompt `position_ids` by the same compact keep mask.
4. Pass pinned compact `position_ids` into generation prefill.

On this path, forward should keep the original `image_grid_thw` metadata with
the pinned original-position `position_ids`. Compact grid metadata is reserved
for explicit sequence-position recomputation paths.

## Validation Gates

Before accepting new full metrics as locked baselines, run:

```bash
python scripts/audit_guiowl_baseline_settings.py
python scripts/audit_guiowl_compact_invariants.py
```

Then run small remote smokes for no-prune, random no-cache, random vision-cache,
and sticky vision-cache. If any invariant fails or MRoPE/cache handling changes,
previous full metrics should be treated as historical and rerun under the fixed
standard.
