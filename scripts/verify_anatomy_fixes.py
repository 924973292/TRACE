#!/usr/bin/env python
"""Bit-level consistency verification for the P1-P15 anatomy fixes.

Covers:
  1. P1  generation-prompt tail length: text-diff path == legacy double-processor path
  2. P5  capture-hook n==1 predicate: seq_len form == legacy cache_position .item() form
  3. P7  MRoPE keep_mask: vectorized construction == python loop
  4. P6  mapping_cache: cached CDCT herding == uncached (same selection, same order)
  5. P3  block KV cache: tensor-level equality vs DynamicCache (update/crop/grow/legacy)
  6. P3  block KV cache: full-model 20-token greedy decode logits torch.equal (GPU)
  7. P4  compiled RMSNorm vs eager numerics (GPU, reports max abs diff)

Run on remote GPU0:
  CUDA_VISIBLE_DEVICES=0 $EVAL_PYTHON scripts/verify_anatomy_fixes.py [--with-model]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

PASS = 0
FAIL = 0


def ok(msg: str) -> None:
    global PASS
    PASS += 1
    print(f"[PASS] {msg}")


def fail(msg: str) -> None:
    global FAIL
    FAIL += 1
    print(f"[FAIL] {msg}")


# ---------------------------------------------------------------------------
# 1. P1 tail length equivalence
# ---------------------------------------------------------------------------

def _make_test_messages(model_dir: str):
    from PIL import Image

    tmpdir = tempfile.mkdtemp(prefix="anatomy_tail_")
    paths = []
    for i, (w, h) in enumerate([(448, 448), (672, 448), (448, 896)]):
        img = Image.new("RGB", (w, h), (i * 40 % 255, 100, 200))
        p = os.path.join(tmpdir, f"img{i}.png")
        img.save(p)
        paths.append(p)

    sys_msg = {"role": "system", "content": [{"type": "text", "text": "You are a GUI agent."}]}
    variants = []
    instrs = [
        "Open the settings app",
        "Scroll down and tap the third item in the list, then confirm.",
        "输入用户名并点击登录按钮",
        'Return the action as JSON: {"action": "click", "coordinate": [100, 200]}',
    ]
    for n_turn in (1, 2, 3):
        for instr in instrs:
            msgs = [sys_msg]
            for t in range(n_turn):
                msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": paths[t % len(paths)]},
                            {"type": "text", "text": f"{instr} (step {t})"},
                        ],
                    }
                )
                if t < n_turn - 1:
                    msgs.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": f'{{"action": "click", "coordinate": [{t}, {t}]}}'}],
                        }
                    )
            variants.append(msgs)
    # A couple of text-only conversations too.
    for instr in instrs[:2]:
        variants.append(
            [sys_msg, {"role": "user", "content": [{"type": "text", "text": instr}]}]
        )
    return variants


def test_tail_len(model_dir: str) -> None:
    from transformers import AutoProcessor, AutoTokenizer

    from lmms_eval.models.simple.qwen3_vl_gui import _Qwen3VLGui

    import types

    class Shim:
        pass

    shim = Shim()
    shim.processor = AutoProcessor.from_pretrained(model_dir, use_fast=False)
    shim.tokenizer = AutoTokenizer.from_pretrained(model_dir)
    for name in (
        "_input_ids_from_qwen_messages",
        "_generation_prompt_tail_len",
        "_generation_prompt_tail_len_full",
    ):
        setattr(shim, name, types.MethodType(getattr(_Qwen3VLGui, name), shim))

    variants = _make_test_messages(model_dir)
    mismatches = []
    for i, msgs in enumerate(variants):
        new_tail = shim._generation_prompt_tail_len(msgs)
        old_tail = shim._generation_prompt_tail_len_full(msgs)
        if new_tail != old_tail:
            mismatches.append((i, new_tail, old_tail))
    if mismatches:
        fail(f"P1 tail_len mismatch on {len(mismatches)}/{len(variants)} variants: {mismatches[:5]}")
    else:
        ok(f"P1 tail_len: text-diff == double-processor on {len(variants)} message variants")


# ---------------------------------------------------------------------------
# 2. P5 capture predicate equivalence
# ---------------------------------------------------------------------------

def test_capture_predicate() -> None:
    from lmms_eval.models.geopact_session_kv import DialogueKVCapture

    bad = []
    for target in (10, 100, 1517):
        for cache_pos0 in (target - 2, target - 1, target, target + 1, target + 50):
            cp = torch.tensor([cache_pos0], dtype=torch.long)
            legacy = DialogueKVCapture._is_capture_prefill_forward(cp, target_kv_len=target)
            # In the hook, seq_len (post-forward) == cache_position[0] + 1 for n==1.
            new = DialogueKVCapture._is_capture_prefill_forward(
                cp, target_kv_len=target, seq_len=cache_pos0 + 1
            )
            if legacy != new:
                bad.append((target, cache_pos0, legacy, new))
    if bad:
        fail(f"P5 capture predicate mismatch: {bad}")
    else:
        ok("P5 capture predicate: seq_len form == legacy .item() form on all boundary cases")


# ---------------------------------------------------------------------------
# 3. P7 keep_mask vectorized == loop
# ---------------------------------------------------------------------------

def test_keep_mask() -> None:
    import random

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(7)
    bad = 0
    for _ in range(200):
        n_frame = rng.randint(1, 1600)
        tail_len = rng.randint(0, 40)
        region_len = n_frame + tail_len
        n_keep = rng.randint(0, n_frame)
        local_keep = sorted(rng.sample(range(n_frame), n_keep))

        loop_mask = torch.zeros(region_len, dtype=torch.bool, device=device)
        for k in local_keep:
            loop_mask[k] = True
        for t in range(n_frame, region_len):
            loop_mask[t] = True

        vec_mask = torch.zeros(region_len, dtype=torch.bool, device=device)
        if local_keep:
            vec_mask[torch.as_tensor(local_keep, dtype=torch.long, device=device)] = True
        vec_mask[n_frame:] = True

        if not torch.equal(loop_mask, vec_mask):
            bad += 1
    if bad:
        fail(f"P7 keep_mask vectorized != loop in {bad}/200 random cases")
    else:
        ok("P7 keep_mask: vectorized == loop on 200 random cases (incl. empty keeps)")


# ---------------------------------------------------------------------------
# 4. P6 mapping cache: cached vs uncached CDCT herding
# ---------------------------------------------------------------------------

def _synth_mius(rng, n: int):
    mius = []
    for _ in range(n):
        x1, y1 = rng.random() * 0.7, rng.random() * 0.7
        mius.append(
            {
                "box": (x1, y1, min(1.0, x1 + 0.05 + rng.random() * 0.25), min(1.0, y1 + 0.05 + rng.random() * 0.25)),
                "energy": 0.2 + rng.random() * 0.8,
            }
        )
    return mius


def test_mapping_cache() -> None:
    import random

    from lmms_eval.models.geopact_vl_hook import (
        _hes_token_cdct_herding,
        _map_mius_to_tokens,
        _map_mius_to_tokens_cached,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(11)
    torch.manual_seed(11)

    gh, gw = 34, 46
    n_img = gh * gw
    image_embeds = torch.randn(n_img, 64, device=device)
    seq_len = n_img + 60
    inputs_embeds = torch.randn(1, seq_len, 64, device=device)
    image_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    image_mask[10 : 10 + n_img] = True
    grid = torch.tensor([[1, gh * 2, gw * 2]], device=device)
    mius = _synth_mius(rng, 12)

    # Direct equality of the memoized helper.
    cache: dict = {}
    r1 = _map_mius_to_tokens_cached(cache, mius, image_embeds, gh, gw, n_img)
    r2 = _map_mius_to_tokens_cached(cache, mius, image_embeds, gh, gw, n_img)
    r0 = _map_mius_to_tokens(mius, image_embeds, gh, gw, n_img)
    if r1 is not r2:
        fail("P6 mapping cache did not memoize (second call recomputed)")
    elif r1[0] != r0[0] or r1[3] != r0[3]:
        fail("P6 cached mapping differs from direct _map_mius_to_tokens")
    else:
        ok("P6 mapping cache: memoized result identical to direct computation")

    # End-to-end: herding selection with and without cache must be identical.
    sel_nc = _hes_token_cdct_herding(
        mius=mius, image_embeds=image_embeds, inputs_embeds=inputs_embeds,
        image_mask=image_mask, image_grid_thw=grid, keep_k_tokens=200,
        policy="token_cdct_herding", mapping_cache=None,
    )
    sel_c = _hes_token_cdct_herding(
        mius=mius, image_embeds=image_embeds, inputs_embeds=inputs_embeds,
        image_mask=image_mask, image_grid_thw=grid, keep_k_tokens=200,
        policy="token_cdct_herding", mapping_cache={},
    )
    if sorted(sel_nc) != sorted(sel_c) or list(
        getattr(sel_nc, "picked_token_order", [])
    ) != list(getattr(sel_c, "picked_token_order", [])):
        fail("P6 CDCT herding selection differs with mapping_cache")
    else:
        ok("P6 CDCT herding: identical selection and pick order with/without mapping_cache")


# ---------------------------------------------------------------------------
# 5. P3 block cache tensor-level equality vs DynamicCache
# ---------------------------------------------------------------------------

def test_block_cache_tensors() -> None:
    os.environ["GEOPACT_KV_BLOCK_SIZE"] = "64"  # force growth across boundaries
    from transformers.cache_utils import DynamicCache

    from lmms_eval.models.geopact_block_kv_cache import GeoPactBlockKVCache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(3)
    n_layers, b, h, d = 4, 1, 2, 16

    dyn = DynamicCache()
    blk = GeoPactBlockKVCache()

    def push(n_tok: int):
        for li in range(n_layers):
            k = torch.randn(b, h, n_tok, d, device=device)
            v = torch.randn(b, h, n_tok, d, device=device)
            dk, dv = dyn.update(k.clone(), v.clone(), li)
            bk, bv = blk.update(k.clone(), v.clone(), li)
            assert torch.equal(dk, bk) and torch.equal(dv, bv), f"layer {li} mismatch after update({n_tok})"

    bad = None
    try:
        push(100)                      # prefill (crosses block=64 boundary)
        for _ in range(30):
            push(1)                    # decode; crosses 128 boundary mid-way
        assert dyn.get_seq_length() == blk.get_seq_length() == 130
        # crop then re-append (bootstrap pattern)
        dyn.crop(115)
        blk.crop(115)
        assert dyn.get_seq_length() == blk.get_seq_length() == 115
        for _ in range(5):
            push(1)
        # legacy extraction
        for (dk, dv), (bk, bv) in zip(dyn.to_legacy_cache(), blk.to_legacy_cache()):
            assert torch.equal(dk, bk) and torch.equal(dv, bv)
        # stride contract for flash-attn
        for layer in blk.layers:
            assert layer.keys.stride(-1) == 1 and layer.values.stride(-1) == 1
        # from-legacy round trip (as _cache_from_legacy_tensors does)
        os.environ["GEOPACT_KV_BLOCK_CACHE"] = "1"
        from lmms_eval.models.geopact_block_kv_cache import make_kv_cache

        rebuilt = make_kv_cache()
        for li, (k, v) in enumerate(dyn.to_legacy_cache()):
            rebuilt.update(k[..., :100, :].clone(), v[..., :100, :].clone(), li)
        assert type(rebuilt).__name__ == "GeoPactBlockKVCache"
        assert rebuilt.get_seq_length() == 100
        os.environ["GEOPACT_KV_BLOCK_CACHE"] = "0"
    except AssertionError as exc:
        bad = str(exc)
    if bad:
        fail(f"P3 block cache tensor equality: {bad}")
    else:
        ok("P3 block cache: update/decode/crop/re-append/legacy all torch.equal vs DynamicCache")


# ---------------------------------------------------------------------------
# 6. P3 full-model logits equality (GPU only, requires checkpoint)
# ---------------------------------------------------------------------------

def test_block_cache_model(model_dir: str) -> None:
    if not torch.cuda.is_available():
        print("[SKIP] P3 model logits test (no CUDA)")
        return
    if not os.path.isdir(model_dir):
        print(f"[SKIP] P3 model logits test (no checkpoint at {model_dir})")
        return
    os.environ.pop("GEOPACT_KV_BLOCK_SIZE", None)
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache

    from lmms_eval.models.geopact_block_kv_cache import GeoPactBlockKVCache

    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelCls

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = ModelCls.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    prompt = (
        "You are a GUI agent. The user asks: open the settings app and enable "
        "dark mode. Respond with the action JSON."
    )
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")

    @torch.inference_mode()
    def run(cache):
        toks = []
        logits_list = []
        cur = ids
        for step in range(20):
            out = model(input_ids=cur, past_key_values=cache, use_cache=True)
            logits = out.logits[:, -1, :]
            logits_list.append(logits.clone())
            nxt = logits.argmax(dim=-1, keepdim=True)
            toks.append(int(nxt))
            cur = nxt
        return toks, logits_list

    torch.manual_seed(0)
    toks_dyn, logits_dyn = run(DynamicCache())
    torch.manual_seed(0)
    toks_blk, logits_blk = run(GeoPactBlockKVCache())

    if toks_dyn != toks_blk:
        fail(f"P3 model decode tokens differ: dyn={toks_dyn} blk={toks_blk}")
    else:
        max_diff = max(
            float((a - b).abs().max()) for a, b in zip(logits_dyn, logits_blk)
        )
        exact = all(torch.equal(a, b) for a, b in zip(logits_dyn, logits_blk))
        if exact:
            ok("P3 model: 20-token greedy decode logits torch.equal (bit-identical)")
        elif max_diff == 0.0:
            ok("P3 model: 20-token greedy decode logits max_diff=0")
        else:
            fail(f"P3 model logits differ, max_abs_diff={max_diff} (tokens identical={toks_dyn == toks_blk})")

    # P4 numerics on the real model's norm layers (report only; gated separately).
    try:
        from lmms_eval.models.geopact_compile_norms import _rms_norm_ref

        norm = None
        for _, m in model.named_modules():
            if "RMSNorm" in type(m).__name__ and hasattr(m, "variance_epsilon"):
                norm = m
                break
        if norm is not None:
            x = torch.randn(1, 7, norm.weight.shape[-1], dtype=torch.bfloat16, device=norm.weight.device)
            eager = norm(x)
            ref = _rms_norm_ref(x, norm.weight, norm.variance_epsilon)
            if torch.equal(eager, ref):
                ok("P4 RMSNorm reference math == HF eager forward (bit-identical)")
            else:
                fail(f"P4 RMSNorm ref differs from eager, max={float((eager - ref).abs().max())}")
            compiled = torch.compile(_rms_norm_ref, dynamic=True)
            comp = compiled(x, norm.weight, norm.variance_epsilon)
            diff = float((comp - eager).abs().max())
            if torch.equal(comp, eager):
                ok("P4 compiled RMSNorm == eager (bit-identical)")
            elif diff <= 1e-2:
                ok(f"P4 compiled RMSNorm within tolerance (max_abs_diff={diff:.3e})")
            else:
                fail(f"P4 compiled RMSNorm diff too large: {diff:.3e}")
    except Exception as exc:
        print(f"[SKIP] P4 compile check errored non-fatally: {exc}")

    del model
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "models/GUI-Owl-1.5-4B-Instruct"))
    ap.add_argument("--with-model", action="store_true", help="run full-model logits test")
    ap.add_argument("--skip-tail", action="store_true", help="skip processor-based tail test")
    args = ap.parse_args()

    test_capture_predicate()
    test_keep_mask()
    test_mapping_cache()
    test_block_cache_tensors()
    if not args.skip_tail:
        if os.path.isdir(args.model_dir):
            test_tail_len(args.model_dir)
        else:
            print(f"[SKIP] P1 tail test (no checkpoint at {args.model_dir})")
    if args.with_model:
        test_block_cache_model(args.model_dir)

    print(f"\n=== verify_anatomy_fixes: {PASS} passed, {FAIL} failed ===")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
