"""Qwen3-Omni local-inference adapter for OmniGUI evaluation.

This module reuses the full OmniGUI compression stack (OME-MIU build, C-MMR
selection, post-projector token compaction, MRoPE-preserving virtual grids
and audio mask) by inheriting from :class:`Qwen2_5_Omni_Chat` and only
overriding the four backbone-specific extension points exposed by the parent:

* ``_MODEL_CLASS`` / ``_PROCESSOR_CLASS`` --- the HF model + processor;
* ``_extract_visual_features`` / ``_extract_audio_features`` --- pull the per
  token tensor out of Qwen3's richer encoder outputs (``BaseModelOutputWith
  DeepstackFeatures`` / ``BaseModelOutputWithPooling``);
* ``_thinker_spatial_merge_size`` --- look up the merge factor under
  ``vision_config`` (Qwen3) instead of the top-level thinker config (Qwen2.5);
* ``_compute_audio_feature_length`` --- Qwen3's audio tower has 3 stride-2
  convolutions and a 13-token deepstack contribution per 100-frame block,
  which produces different MRoPE alignment math than Qwen2.5;
* ``_call_model_generate`` --- Qwen3-Omni's top-level ``generate`` re-prefixes
  thinker-specific kwargs (``thinker_max_new_tokens`` / ``thinker_eos_token_id``)
  and uses a different default speaker (``Ethan``).

No MIU / MMR / pruning code is touched, so any pruning ablation that runs on
Qwen2.5-Omni is reproducible on Qwen3-Omni with the same CLI flags.
"""
from __future__ import annotations

from typing import Optional, Union

import torch
from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen2_5_omni import Qwen2_5_Omni_Chat


try:
    from transformers import (
        Qwen3OmniMoeForConditionalGeneration,
        Qwen3OmniMoeProcessor,
    )
except ImportError as exc:
    Qwen3OmniMoeForConditionalGeneration = None
    Qwen3OmniMoeProcessor = None
    eval_logger.warning(
        f"Qwen3-Omni-Moe classes not available in this transformers build "
        f"({exc!r}); upgrade to transformers>=4.57 to use qwen3_omni_chat."
    )


# transformers >= 4.55 ships fast image processors that call
# ``torch.compiler.is_compiling()`` during resize.  That helper was only added in
# torch >= 2.5; the cluster image still pins torch 2.2.1+cu121, so we polyfill it
# here as a no-op (we never invoke torch.compile during inference).  Doing it
# inside the adapter module keeps the patch scoped: it is only applied when this
# file is imported, so the Qwen2.5-Omni pipeline (older transformers, slow
# processor) is unaffected.
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False  # type: ignore[attr-defined]


# Translations from Qwen2.5-Omni generate kwargs -> Qwen3-Omni generate kwargs.
# Keys with these names override the Qwen3 top-level defaults (which are pre-set
# inside Qwen3OmniMoeForConditionalGeneration.generate before our shared_kwargs
# merge runs and would otherwise be locked at e.g. max_new_tokens=1024).
_THINKER_PREFIX_MAP = {
    "max_new_tokens": "thinker_max_new_tokens",
    "eos_token_id": "thinker_eos_token_id",
}


@register_model("qwen3_omni_chat")
class Qwen3_Omni_Chat(Qwen2_5_Omni_Chat):
    """Local Qwen3-Omni-Moe inference with OmniGUI OME-MIU + C-MMR pruning."""

    _MODEL_CLASS = Qwen3OmniMoeForConditionalGeneration
    _PROCESSOR_CLASS = Qwen3OmniMoeProcessor

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        skip_speaker_loading: bool = True,  # default off: we never decode audio
        system_prompt: str = (
            "You are Qwen3-Omni, a multimodal assistant developed by the Qwen "
            "Team, Alibaba Group, capable of perceiving auditory and visual "
            "inputs as well as generating text responses."
        ),
        attn_implementation: Optional[str] = "flash_attention_2",
        torch_dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        **kwargs,
    ) -> None:
        if Qwen3OmniMoeForConditionalGeneration is None:
            raise ImportError(
                "transformers>=4.57 is required for Qwen3-Omni-Moe; install it "
                "into a dedicated env (e.g. OminiEval3) and rerun."
            )
        super().__init__(
            pretrained=pretrained,
            skip_speaker_loading=skip_speaker_loading,
            system_prompt=system_prompt,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            **kwargs,
        )

    # ---- backbone-specific feature extraction ----
    @staticmethod
    def _extract_visual_features(out):
        # Qwen3-Omni's vision tower returns BaseModelOutputWithDeepstackFeatures;
        # ``pooler_output`` is the per-token projector output that gets scattered
        # into ``inputs_embeds`` for the LLM prefill.  We deliberately ignore the
        # ``deepstack_features`` (multi-scale residuals injected mid-decoder) so
        # the compaction path stays symmetric with the Qwen2.5-Omni recipe; this
        # is the smallest change that avoids re-implementing the deepstack
        # scatter against pruned token positions.
        if out is None:
            return None
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state
        return out

    @staticmethod
    def _extract_audio_features(out):
        # Qwen3-Omni's audio tower returns BaseModelOutputWithPooling; the
        # per-token features live in ``last_hidden_state``.
        if out is None:
            return None
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out

    @staticmethod
    def _thinker_spatial_merge_size(thinker) -> int:
        msz = getattr(getattr(thinker.config, "vision_config", None), "spatial_merge_size", None)
        if msz is None:
            msz = getattr(thinker.config, "spatial_merge_size", 2)
        return int(msz or 2)

    # ---- audio-tower output length (3 stride-2 + 13/block deepstack) ----
    def _compute_audio_feature_length(self, raw_len: int) -> int:
        if raw_len <= 0:
            return 0
        leave = raw_len % 100
        feat = (leave - 1) // 2 + 1
        return max(0, ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (raw_len // 100) * 13)

    # ---- generate-kwarg translation ----
    def _call_model_generate(self, **gen_kwargs):
        # Qwen3-Omni's top-level generate pre-seeds thinker_kwargs with its own
        # defaults BEFORE merging shared kwargs, which means plain
        # ``max_new_tokens`` / ``eos_token_id`` silently get capped at the
        # 1024/151645 defaults.  Translate them to the thinker_-prefixed names
        # so our values actually take effect.
        for src, dst in _THINKER_PREFIX_MAP.items():
            if src in gen_kwargs and dst not in gen_kwargs:
                gen_kwargs[dst] = gen_kwargs.pop(src)
        # Qwen3-Omni dropped the Chelsie speaker; fall back to Ethan if the
        # talker is still loaded (no effect when return_audio=False).
        gen_kwargs.setdefault("speaker", "Ethan")
        return self.model.generate(**gen_kwargs)
