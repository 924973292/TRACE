import base64
import copy
import hashlib
import os
import re
import time
import types
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import audioread
import av
import decord
import librosa
import numpy as np
import soundfile as sf
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
try:
    from moviepy import VideoFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        VideoFileClip = None
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, StoppingCriteria, StoppingCriteriaList

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.gui_miu_detectors import build_raw_mius, filter_redundant_image_mius
from lmms_eval.models.cross_modal_miu_links import (
    attach_cross_modal_resonance,
    cross_modal_summary_meta,
)
from lmms_eval.models.miu_info_density import (
    attach_module_a_info_density_proxy,
    mmr_importance_from_miu,
    module_a_summary_meta,
)
from lmms_eval.models.oas_miu import (
    attach_oas_attributes,
    format_miu_binding_descriptor,
    format_miu_type_instance_label,
    oas_summary_meta,
    parse_oas_drop_set,
)
from lmms_eval.models.ome_miu import build_ome_mius
from lmms_eval.models.model_utils.audio_processing import split_audio
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    eval_logger.warning("Failed to import qwen_omni_utils; Please install it via `pip install qwen-omni-utils[decord]`")


@register_model("qwen2_5_omni")
class Qwen2_5_Omni(lmms):
    """
    Qwen2.5-Omni-7B
    "https://huggingface.co/Qwen/Qwen2.5-Omni-7B"

    For better performance, please visit the Qwen-Omni repo to get the latest system prompt based on your running tasks.
    https://github.com/QwenLM/Qwen2.5-Omni/tree/main/cookbooks
    """

    # ---- minimal extension points for sibling Omni backbones (e.g. Qwen3) ----
    # Subclasses only need to override these to swap the underlying HF classes;
    # all MIU / MMR / compaction logic below is model-agnostic.
    _MODEL_CLASS = Qwen2_5OmniForConditionalGeneration
    _PROCESSOR_CLASS = Qwen2_5OmniProcessor

    @staticmethod
    def _resolve_torch_dtype(torch_dtype):
        if isinstance(torch_dtype, str):
            return {
                "auto": "auto",
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "fp16": torch.float16,
                "float16": torch.float16,
                "fp32": torch.float32,
                "float32": torch.float32,
            }.get(torch_dtype.lower(), torch_dtype)
        return torch_dtype

    @staticmethod
    def _log_thinker_attention_backend(model, attn_implementation: str) -> None:
        """Confirm GUI eval runs thinker with FA2; Token2Wav is not on the hot path."""
        thinker = getattr(model, "thinker", None)
        if thinker is None:
            return
        text_cfg = getattr(thinker, "config", None)
        text_cfg = getattr(text_cfg, "text_config", text_cfg)
        impl = (
            getattr(text_cfg, "_attn_implementation", None)
            or getattr(text_cfg, "attn_implementation", None)
            or attn_implementation
        )
        has_talker = bool(getattr(model, "has_talker", False))
        eval_logger.info(
            "Qwen Omni load: thinker_attn={} requested_attn={} has_talker={} "
            "(Token2Wav/TTS skipped when enable_audio_output=False)",
            impl,
            attn_implementation,
            has_talker,
        )

    @staticmethod
    def _timing_log_enabled() -> bool:
        return str(os.environ.get("LMMS_LOG_PER_SAMPLE", "0")).strip().lower() in {"1", "true", "yes", "on"}

    def _media_token_ids(self) -> dict[str, Optional[int]]:
        thinker = getattr(self.model, "thinker", None)
        config = getattr(thinker, "config", None)
        if config is None:
            return {"image": None, "video": None, "audio": None}
        return {
            "image": getattr(config, "image_token_id", getattr(config, "image_token_index", None)),
            "video": getattr(config, "video_token_id", getattr(config, "video_token_index", None)),
            "audio": getattr(config, "audio_token_id", getattr(config, "audio_token_index", None)),
        }

    @staticmethod
    def _model_input_ids(model_inputs):
        if isinstance(model_inputs, dict):
            return model_inputs.get("input_ids")
        return getattr(model_inputs, "input_ids", None)

    def _sequence_token_breakdown(self, input_ids) -> dict[str, float]:
        if input_ids is None:
            return {}
        try:
            ids = input_ids[0] if getattr(input_ids, "dim", lambda: 1)() > 1 else input_ids
            total = int(ids.shape[-1])
            out: dict[str, float] = {"sequence_tokens": float(total)}
            media_total = 0
            for name, token_id in self._media_token_ids().items():
                count = int((ids == int(token_id)).sum().item()) if token_id is not None else 0
                out[f"{name}_tokens"] = float(count)
                media_total += count
            out["media_tokens"] = float(media_total)
            out["text_tokens"] = float(max(total - media_total, 0))
            return out
        except Exception:
            return {}

    @staticmethod
    def _generated_token_count(generated_ids_trimmed) -> int:
        total = 0
        for item in generated_ids_trimmed or []:
            try:
                total += int(item.shape[-1])
            except Exception:
                try:
                    total += len(item)
                except Exception:
                    pass
        return total

    def _log_generation_timing(
        self,
        *,
        step_t0: float,
        ttft_ms: float,
        prefill_ms: float,
        decode_ms: float,
        generate_ms: float,
        peak_mem_mb: float,
        prompt_len: int,
        generated_ids_trimmed,
        active_inputs,
        forward_calls: float = 0.0,
    ) -> None:
        if not self._timing_log_enabled():
            return
        stats = self._last_visual_prune_stats or {}
        seq = self._sequence_token_breakdown(self._model_input_ids(active_inputs))
        generated_tokens = self._generated_token_count(generated_ids_trimmed)
        decode_forward_ms = decode_ms
        e2e_ms = (time.perf_counter() - step_t0) * 1000.0
        decode_ms = max(e2e_ms - ttft_ms, 0.0)
        llm_prefill_forward_ms = float(stats.get("llm_prefill_forward_ms", stats.get("prefill_forward_ms", prefill_ms)))
        model_prefill_total_ms = float(stats.get("model_prefill_total_ms", prefill_ms))
        compact_ms = float(stats.get("compact_total_ms", 0.0))
        projector_feature_ms = float(stats.get("projector_feature_ms", stats.get("compact_feature_ms", 0.0)))
        projector_scatter_ms = float(stats.get("projector_scatter_ms", 0.0))
        compact_non_feature_ms = float(stats.get("compact_non_feature_ms", 0.0))
        decode_for_token_ms = decode_forward_ms if decode_forward_ms > 0 else max(generate_ms - prefill_ms, 0.0)
        if decode_for_token_ms <= 0:
            decode_for_token_ms = generate_ms
        generated_token_ms = decode_for_token_ms / generated_tokens if generated_tokens > 0 else 0.0
        retained_media_tokens = float(stats.get("retained_media_tokens", seq.get("media_tokens", 0.0)))
        retained_text_tokens = float(stats.get("retained_text_tokens", seq.get("text_tokens", 0.0)))
        media_before = float(stats.get("media_tokens_before", retained_media_tokens))
        media_after = float(stats.get("media_tokens_after", retained_media_tokens))
        text_before = float(stats.get("text_tokens_before", retained_text_tokens))
        text_after = float(stats.get("text_tokens_after", retained_text_tokens))
        message = (
            "Qwen generation timing "
            f"policy={self.visual_prune_policy} ratio={self.visual_prune_ratio} "
            f"ttft_ms={ttft_ms:.2f} prefill_ms={prefill_ms:.2f} decode_ms={decode_ms:.2f} "
            f"model_prefill_total_ms={model_prefill_total_ms:.2f} "
            f"prefill_forward_ms={llm_prefill_forward_ms:.2f} "
            f"llm_prefill_forward_ms={llm_prefill_forward_ms:.2f} "
            f"source_timing_impl={float(stats.get('source_timing_impl', 0.0)):.0f} "
            f"source_forward_prefill_ms={float(stats.get('source_forward_prefill_ms', 0.0)):.2f} "
            f"source_llm_model_prefill_ms={float(stats.get('source_llm_model_prefill_ms', 0.0)):.2f} "
            f"source_lm_head_prefill_ms={float(stats.get('source_lm_head_prefill_ms', 0.0)):.2f} "
            f"source_rope_prefill_ms={float(stats.get('source_rope_prefill_ms', 0.0)):.2f} "
            f"source_input_embed_prefill_ms={float(stats.get('source_input_embed_prefill_ms', 0.0)):.2f} "
            f"source_image_feature_prefill_ms={float(stats.get('source_image_feature_prefill_ms', 0.0)):.2f} "
            f"source_video_feature_prefill_ms={float(stats.get('source_video_feature_prefill_ms', 0.0)):.2f} "
            f"source_audio_feature_prefill_ms={float(stats.get('source_audio_feature_prefill_ms', 0.0)):.2f} "
            f"llm_decode_forward_ms={float(stats.get('llm_decode_forward_ms', 0.0)):.2f} "
            f"source_forward_decode_ms={float(stats.get('source_forward_decode_ms', 0.0)):.2f} "
            f"source_model_attention_mask_all_ones={float(stats.get('source_model_attention_mask_all_ones', -1.0)):.0f} "
            f"source_model_attention_mask_zero_count={float(stats.get('source_model_attention_mask_zero_count', -1.0)):.0f} "
            f"source_model_attention_mask_length={float(stats.get('source_model_attention_mask_length', -1.0)):.0f} "
            f"source_flash_attention_mask_passthrough={float(stats.get('source_flash_attention_mask_passthrough', -1.0)):.0f} "
            f"source_flash_attention_mask_zero_count={float(stats.get('source_flash_attention_mask_zero_count', -1.0)):.0f} "
            f"source_flash_attention_mask_length={float(stats.get('source_flash_attention_mask_length', -1.0)):.0f} "
            f"decode_forward_ms={decode_forward_ms:.2f} generate_ms={generate_ms:.2f} e2e_ms={e2e_ms:.2f} "
            f"forward_calls={float(forward_calls):.0f} "
            f"peak_mem_mb={peak_mem_mb:.1f} prompt_tokens={int(prompt_len)} "
            f"generated_tokens={int(generated_tokens)} generated_token_ms={generated_token_ms:.4f} "
            f"retained_media_tokens={retained_media_tokens:.0f} retained_text_tokens={retained_text_tokens:.0f} "
            f"media_tokens_before={media_before:.0f} media_tokens_after={media_after:.0f} "
            f"text_tokens_before={text_before:.0f} text_tokens_after={text_after:.0f} "
            f"detector_seconds={float(stats.get('detector_seconds', getattr(self, '_last_detect_seconds', 0.0))):.4f} "
            f"miu_detect_ms={float(stats.get('miu_detect_ms', 0.0)):.2f} "
            f"miu_detect_cache_hit={float(stats.get('miu_detect_cache_hit', 0.0)):.0f} "
            f"miu_cache_hit_rate={float(stats.get('miu_cache_hit_rate', 0.0)):.4f} "
            f"detector_backend={str(stats.get('miu_detect_backend', 'na'))} "
            f"position_mode={float(stats.get('position_mode_label', -1.0)):.0f} "
            f"exact_pos={float(stats.get('source_exact_compact_positions', -1.0)):.0f} "
            f"exact_tail_pos={float(stats.get('source_exact_compact_tail_positions', 0.0)):.0f} "
            f"coord_mode={float(stats.get('coord_text_mode', -1.0)):.0f} "
            f"coord_added_tokens={float(stats.get('coord_text_inserted_tokens', 0.0)):.0f} "
            f"module_c_added_tokens={float(stats.get('module_c_added_tokens', 0.0)):.0f} "
            f"compact_ms={compact_ms:.2f} "
            f"compact_non_feature_ms={compact_non_feature_ms:.2f} "
            f"compact_backend_source={float(stats.get('compact_backend_source', 0.0)):.0f} "
            f"source_compact_impl={float(stats.get('source_compact_impl', 0.0)):.0f} "
            f"compact_projector_prepare_ms={float(stats.get('compact_projector_prepare_ms', 0.0)):.2f} "
            f"compact_input_embed_ms={float(stats.get('compact_input_embed_ms', 0.0)):.2f} "
            f"projector_feature_ms={projector_feature_ms:.2f} "
            f"projector_scatter_ms={projector_scatter_ms:.2f} "
            f"compact_feature_ms={projector_feature_ms:.2f} "
            f"compact_image_feature_ms={float(stats.get('compact_image_feature_ms', 0.0)):.2f} "
            f"compact_video_feature_ms={float(stats.get('compact_video_feature_ms', 0.0)):.2f} "
            f"compact_audio_feature_ms={float(stats.get('compact_audio_feature_ms', 0.0)):.2f} "
            f"image_feature_cache_hit={float(stats.get('image_feature_cache_hit', 0.0)):.0f} "
            f"feature_cache_key_ms={float(stats.get('feature_cache_key_ms', 0.0)):.2f} "
            f"image_selection_ms={float(stats.get('image_selection_ms', 0.0)):.2f} "
            f"image_scatter_ms={float(stats.get('image_scatter_ms', 0.0)):.2f} "
            f"video_scatter_ms={float(stats.get('video_scatter_ms', 0.0)):.2f} "
            f"audio_scatter_ms={float(stats.get('audio_scatter_ms', 0.0)):.2f} "
            f"compact_rebuild_ms={float(stats.get('compact_rebuild_ms', 0.0)):.2f} "
            f"instruction_embed_ms={float(stats.get('instruction_embed_ms', 0.0)):.2f} "
            f"instruction_embed_cache_hit={float(stats.get('instruction_embed_cache_hit', 0.0)):.0f} "
            f"coord_preembed_ms={float(stats.get('coord_preembed_ms', 0.0)):.2f} "
            f"coord_preembed_candidates={float(stats.get('coord_preembed_candidates', 0.0)):.0f} "
            f"coord_preembed_tokens={float(stats.get('coord_preembed_tokens', 0.0)):.0f} "
            f"coord_reuse_preembedded={float(stats.get('coord_reuse_preembedded', 0.0)):.0f} "
            f"coord_system_prompt_note={float(stats.get('coord_system_prompt_note', 0.0)):.0f} "
            f"image_miu_hard_crop={float(stats.get('image_miu_hard_crop', 0.0)):.0f} "
            f"coord_ms={float(stats.get('module_c_coord_ms', 0.0)):.2f} "
            f"coord_insert_ms={float(stats.get('coord_insert_ms', 0.0)):.2f}"
        )
        eval_logger.info(message)
        print(message, flush=True)

    def _call_model_generate_with_step_timing(self, **gen_kwargs):
        thinker = getattr(self.model, "thinker", None)
        timing: dict[str, float] = {"forward_calls": 0.0}
        if thinker is None:
            return self._call_model_generate(**gen_kwargs), timing

        prompt_ids = gen_kwargs.get("input_ids")
        try:
            prompt_len = int(prompt_ids.shape[-1]) if prompt_ids is not None else 0
        except Exception:
            prompt_len = 0
        generate_wall_t0 = time.perf_counter()
        ttft_start_event = None
        ttft_end_event = None

        class _FirstTokenTimingCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                nonlocal ttft_end_event
                if "_ttft_seen" not in timing:
                    try:
                        current_len = int(input_ids.shape[-1])
                    except Exception:
                        current_len = 0
                    if current_len > prompt_len:
                        timing["_ttft_seen"] = 1.0
                        if torch.cuda.is_available():
                            ttft_end_event = torch.cuda.Event(enable_timing=True)
                            ttft_end_event.record()
                        else:
                            timing["ttft_ms"] = float((time.perf_counter() - generate_wall_t0) * 1000.0)
                return False

        first_token_criteria = _FirstTokenTimingCriteria()
        existing_criteria = gen_kwargs.get("stopping_criteria")
        if existing_criteria is None:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([first_token_criteria])
        else:
            try:
                criteria_items = list(existing_criteria)
            except TypeError:
                criteria_items = [existing_criteria]
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(criteria_items + [first_token_criteria])

        reset_timing = getattr(thinker, "_omnigui_reset_generation_timing", None)
        if callable(reset_timing):
            try:
                clear_projector = not bool((self._last_visual_prune_stats or {}).get("compact_backend_source", 0.0))
                reset_timing(clear_projector=clear_projector)
            except TypeError:
                reset_timing()

        if torch.cuda.is_available():
            ttft_start_event = torch.cuda.Event(enable_timing=True)
            ttft_start_event.record()
        generate_wall_t0 = time.perf_counter()
        cont = self._call_model_generate(**gen_kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            if ttft_start_event is not None and ttft_end_event is not None:
                timing["ttft_ms"] = float(ttft_start_event.elapsed_time(ttft_end_event))
        else:
            timing.setdefault("ttft_ms", float((time.perf_counter() - generate_wall_t0) * 1000.0))
        source_timing = getattr(thinker, "_omnigui_last_generation_timing", None)
        if isinstance(source_timing, dict):
            for key, value in source_timing.items():
                if isinstance(value, (int, float)):
                    timing[key] = float(value)
        timing.setdefault("prefill_ms", float(timing.get("model_prefill_total_ms", timing.get("llm_prefill_forward_ms", 0.0))))
        timing.setdefault("prefill_forward_ms", float(timing.get("llm_prefill_forward_ms", 0.0)))
        timing.setdefault("decode_ms", float(timing.get("decode_forward_ms", 0.0)))
        return cont, timing

    def _call_model_generate(self, **gen_kwargs):
        """Thin wrapper around ``model.generate`` so subclasses can rename kwargs.

        Default implementation matches Qwen2.5-Omni's ``generate`` signature.
        """
        mrope_deltas = gen_kwargs.pop("_geopact_rope_deltas", None)
        if mrope_deltas is None and "rope_deltas" in gen_kwargs:
            mrope_deltas = gen_kwargs.get("rope_deltas", None)
        position_ids = gen_kwargs.get("position_ids", None)
        thinker = getattr(self.model, "thinker", None)
        if mrope_deltas is None or position_ids is None or thinker is None:
            return self.model.generate(**gen_kwargs)

        original_prepare = thinker.prepare_inputs_for_generation
        original_rope_deltas = getattr(thinker, "rope_deltas", None)

        def _prepare_with_prefill_position_ids(
            _self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            use_cache=True,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
            input_features=None,
            feature_attention_mask=None,
            use_audio_in_video=False,
            video_second_per_grid=None,
            **kwargs,
        ):
            model_inputs = original_prepare(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_ids=position_ids,
                use_cache=use_cache,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                use_audio_in_video=use_audio_in_video,
                video_second_per_grid=video_second_per_grid,
                **kwargs,
            )
            is_prefill = cache_position is None or (
                hasattr(cache_position, "numel")
                and cache_position.numel() > 0
                and int(cache_position[0].item()) == 0
            )
            if is_prefill and position_ids is not None:
                model_inputs["position_ids"] = position_ids
            return model_inputs

        try:
            thinker.rope_deltas = mrope_deltas.to(position_ids.device)
            thinker.prepare_inputs_for_generation = types.MethodType(
                _prepare_with_prefill_position_ids,
                thinker,
            )
            return self.model.generate(**gen_kwargs)
        finally:
            thinker.prepare_inputs_for_generation = original_prepare
            thinker.rope_deltas = original_rope_deltas

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-Omni-7B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[bool] = "eager",
        torch_dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_num_frames: int = 768,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        visual_prune_ratio: Optional[float] = None,
        visual_prune_scope: str = "all",
        visual_prune_policy: str = "none",
        visual_prune_anchor_limit: int = 24,
        visual_prune_mmr_lambda: float = 0.65,
        miu_detector: str = "heuristic",
        token_prune_ratio: Optional[float] = None,
        token_prune_scope: Optional[str] = None,
        token_prune_policy: Optional[str] = None,
        skip_speaker_loading: bool = False,
        system_prompt: str = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        ModelCls = type(self)._MODEL_CLASS
        ProcessorCls = type(self)._PROCESSOR_CLASS

        if getattr(ModelCls, "_tp_plan", None) is None:
            ModelCls._tp_plan = []
        original_load_speakers = None
        original_from_pretrained = None
        original_model_init = None
        if skip_speaker_loading and hasattr(ModelCls, "load_speakers"):
            original_load_speakers = ModelCls.load_speakers

            def _skip_load_speakers(model, path):
                model.speaker_map = {"Chelsie": {}}

            ModelCls.load_speakers = _skip_load_speakers
        if skip_speaker_loading:

            def _config_without_audio_output(config):
                """Force thinker-only init (works even if HF lacks enable_audio_output kwarg)."""
                if config is None:
                    return config
                if hasattr(config, "enable_audio_output"):
                    config.enable_audio_output = False
                elif isinstance(config, dict):
                    config["enable_audio_output"] = False
                return config

            if hasattr(ModelCls, "__init__"):
                original_model_init = ModelCls.__init__

                def _init_skip_talker(self, config, *args, **kwargs):
                    return original_model_init(self, _config_without_audio_output(config), *args, **kwargs)

                ModelCls.__init__ = _init_skip_talker

            if hasattr(ModelCls, "from_pretrained"):
                original_from_pretrained = ModelCls.from_pretrained.__func__

                @classmethod
                def _from_pretrained_gui_eval(cls, pretrained_model_name_or_path, *model_args, **kwargs):
                    kwargs.setdefault("enable_audio_output", False)
                    if kwargs.get("config") is not None:
                        _config_without_audio_output(kwargs["config"])
                    try:
                        return original_from_pretrained(
                            cls, pretrained_model_name_or_path, *model_args, **kwargs
                        )
                    except ValueError as err:
                        if "spk_dict" not in str(err):
                            raise
                        eval_logger.warning(
                            "Speaker dict missing (%s); continuing with enable_audio_output=False.",
                            err,
                        )
                        kwargs["enable_audio_output"] = False
                        return original_from_pretrained(
                            cls, pretrained_model_name_or_path, *model_args, **kwargs
                        )

                ModelCls.from_pretrained = _from_pretrained_gui_eval

        torch_dtype = self._resolve_torch_dtype(torch_dtype)
        fp_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": self.device_map,
            "attn_implementation": attn_implementation,
        }
        if skip_speaker_loading:
            fp_kwargs["enable_audio_output"] = False

        try:
            self._model = ModelCls.from_pretrained(pretrained, **fp_kwargs).eval()
        finally:
            if original_load_speakers is not None:
                ModelCls.load_speakers = original_load_speakers
            if original_from_pretrained is not None:
                ModelCls.from_pretrained = classmethod(original_from_pretrained)
            if original_model_init is not None:
                ModelCls.__init__ = original_model_init

        self._log_thinker_attention_backend(self._model, str(attn_implementation or ""))
        if getattr(self._model, "has_talker", False) and hasattr(self._model, "disable_talker"):
            try:
                self._model.disable_talker()
            except Exception as exc:
                eval_logger.warning(f"disable_talker failed: {exc!r}; continuing with talker loaded")
        self.processor = ProcessorCls.from_pretrained(pretrained)
        self.max_num_frames = max_num_frames
        self._tokenizer = self.processor.tokenizer

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        if visual_prune_ratio is None and token_prune_ratio is not None:
            visual_prune_ratio = token_prune_ratio
            visual_prune_scope = token_prune_scope or visual_prune_scope
            visual_prune_policy = token_prune_policy or visual_prune_policy
        self.visual_prune_ratio = float(visual_prune_ratio) if visual_prune_ratio is not None else None
        self.visual_prune_scope = str(visual_prune_scope or "all").lower()
        self.visual_prune_policy = str(visual_prune_policy or "none").lower()
        self.visual_prune_anchor_limit = int(visual_prune_anchor_limit)
        self.visual_prune_mmr_lambda = float(visual_prune_mmr_lambda)
        self.miu_detector = str(miu_detector or "heuristic").lower()
        self._last_visual_prune_stats = {}
        self._last_miu_detect_meta: dict[str, float] = {}
        self._ocr_empty_warned = False
        self._last_ome_miu_meta: dict[str, float] = {}
        # Optional case-study buffer for surviving image-modality MIUs.
        # Token pruning / MIU selection now lives entirely in the Transformer
        # source backend (``thinker.prepare_omnigui_projector_inputs``), so the
        # frontend no longer populates this list; it stays empty and is only
        # read by the optional geopact case-study dump.
        self._last_selected_image_mius: list[dict] = []
        if self._should_preload_miu_detector():
            try:
                from lmms_eval.models.gui_miu_detectors import warm_omniparser_icon_yolo

                warm_meta = warm_omniparser_icon_yolo()
                eval_logger.info("MIU detector warmup complete: %s", warm_meta)
                print(
                    "MIU_YOLO_MODEL_PRELOAD_OK "
                    f"first_ms={warm_meta.get('miu_yolo_warmup_first_ms', 0.0):.2f} "
                    f"warm_ms={warm_meta.get('miu_yolo_warmup_second_ms', 0.0):.2f} "
                    f"mius={int(warm_meta.get('miu_yolo_warmup_mius', 0.0))}",
                    flush=True,
                )
            except Exception as exc:
                if os.environ.get("MIU_REQUIRE_YOLO_GPU", "0").strip().lower() in {"1", "true", "yes"}:
                    raise
                eval_logger.warning("MIU detector warmup failed: %r", exc)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    def _should_preload_miu_detector(self) -> bool:
        if os.environ.get("MIU_YOLO_MODEL_PRELOAD", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            return False
        policy = str(self.visual_prune_policy or "").lower()
        if not self._use_visual_pruning() or not any(key in policy for key in ("miu", "ome", "oas")):
            return False
        det = str(self.miu_detector or "").lower()
        # Canny+Morph metadata extractor has no YOLO weights to preload.
        if det in {"omniparser_lean", "omniparser_icon_attr", "canny", "canny_morph", "cannymorph"}:
            return False
        return det.startswith(("omniparser", "yolo"))

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_Omni")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def resample_audio(self, audio: np.ndarray, current_sample_rate: int):
        """
        Resample the audio to the target sample rate.
        """
        if current_sample_rate != 16000:  # The sample rate for Qwen2.5-Omni is 16kHz
            if isinstance(audio, np.ndarray):
                audio = librosa.resample(audio, orig_sr=current_sample_rate, target_sr=16000).astype(np.float32)
        return audio

    def _check_if_video_has_audio(self, video_path):
        if VideoFileClip is None:
            return False
        clip = VideoFileClip(video_path)
        return clip.audio is not None

    def _use_visual_pruning(self) -> bool:
        return (
            self.visual_prune_ratio is not None
            and 0.0 < self.visual_prune_ratio < 0.999
            and self.visual_prune_policy not in {"", "none", "off", "false"}
        )

    def _use_coord_system_prompt_note(self) -> bool:
        policy = str(self.visual_prune_policy or "").lower()
        return self._use_visual_pruning() and "coord" in policy and "no_coord" not in policy

    @staticmethod
    def _coord_system_prompt_note() -> str:
        return (
            "When current screenshot candidate coordinates are provided from compacted visual evidence, "
            "treat them as spatial anchors for the retained key GUI elements. These coordinates are "
            "normalized to the original screenshot frame after accounting for detector and model resizing; "
            "choose the candidate coordinate that best matches the requested action."
        )

    def _append_coord_system_prompt_note_to_text(self, text: str) -> str:
        prompt = str(text or "")
        note = self._coord_system_prompt_note()
        if note in prompt:
            return prompt
        sep = " " if prompt.rstrip().endswith((".", "!", "?")) else "\n"
        return f"{prompt.rstrip()}{sep}{note}"

    def _system_prompt_for_current_policy(self) -> str:
        if not self._use_coord_system_prompt_note():
            return str(self.system_prompt or "")
        self._last_visual_prune_stats["coord_system_prompt_note"] = 1.0
        return self._append_coord_system_prompt_note_to_text(str(self.system_prompt or ""))

    def _append_coord_system_prompt_note_to_messages(self, qwen_messages: List[Dict]) -> List[Dict]:
        if not self._use_coord_system_prompt_note():
            return qwen_messages
        qwen_messages = copy.deepcopy(qwen_messages)
        self._last_visual_prune_stats["coord_system_prompt_note"] = 1.0
        for message in qwen_messages:
            if message.get("role") != "system":
                continue
            content = message.get("content", [])
            if isinstance(content, str):
                message["content"] = [{"type": "text", "text": self._append_coord_system_prompt_note_to_text(content)}]
                return qwen_messages
            if isinstance(content, list):
                for item in reversed(content):
                    if isinstance(item, dict) and item.get("type") == "text":
                        item["text"] = self._append_coord_system_prompt_note_to_text(str(item.get("text", "")))
                        return qwen_messages
                content.append({"type": "text", "text": self._coord_system_prompt_note()})
                message["content"] = content
                return qwen_messages
        qwen_messages.insert(0, {"role": "system", "content": [{"type": "text", "text": self._system_prompt_for_current_policy()}]})
        return qwen_messages

    @staticmethod
    def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return inter / max(area_a + area_b - inter, 1e-6)

    @staticmethod
    def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def _detect_ocr_mius(self, image: Image.Image) -> list[dict]:
        try:
            import pytesseract  # type: ignore
        except Exception:
            return []
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        except Exception:
            return []

        width, height = image.size
        mius: list[dict] = []
        for idx, text in enumerate(data.get("text", [])):
            text = str(text or "").strip()
            try:
                conf = float(data.get("conf", [0])[idx])
            except Exception:
                conf = 0.0
            if not text or conf < 25:
                continue
            x = float(data["left"][idx]) / max(width, 1)
            y = float(data["top"][idx]) / max(height, 1)
            w = float(data["width"][idx]) / max(width, 1)
            h = float(data["height"][idx]) / max(height, 1)
            if w * h < 0.00002:
                continue
            mius.append(
                {
                    "box": (x, y, min(1.0, x + w), min(1.0, y + h)),
                    "type": "text",
                    "text": text[:48],
                    "source": "ocr",
                    "det_score": min(max(conf / 100.0, 0.0), 1.0),
                }
            )
        return mius

    def _detect_visual_component_mius(self, image: Image.Image, max_edge: int = 384) -> list[dict]:
        width, height = image.size
        scale = min(max_edge / max(width, height), 1.0)
        small = image.convert("RGB")
        if scale < 1.0:
            small = small.resize((max(1, int(width * scale)), max(1, int(height * scale))))
        arr = np.asarray(small).astype(np.float32)
        gray = arr.mean(axis=2)
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        grad = gx + gy
        color_var = np.max(arr, axis=2) - np.min(arr, axis=2)
        threshold = max(float(np.percentile(grad, 82)), 10.0)
        mask = (grad > threshold) | (color_var > 38)
        # Light dilation connects icon/text strokes into control-like regions.
        for _ in range(2):
            padded = np.pad(mask, 1, mode="constant")
            mask = (
                padded[1:-1, 1:-1]
                | padded[:-2, 1:-1]
                | padded[2:, 1:-1]
                | padded[1:-1, :-2]
                | padded[1:-1, 2:]
            )

        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        mius: list[dict] = []
        min_area = max(8, int(h * w * 0.00015))
        max_area = int(h * w * 0.35)
        for yy in range(h):
            xs = np.flatnonzero(mask[yy] & ~visited[yy])
            for start_x in xs:
                if visited[yy, start_x] or not mask[yy, start_x]:
                    continue
                stack = [(yy, int(start_x))]
                visited[yy, start_x] = True
                min_x = max_x = int(start_x)
                min_y = max_y = yy
                area = 0
                while stack:
                    cy, cx = stack.pop()
                    area += 1
                    min_x, max_x = min(min_x, cx), max(max_x, cx)
                    min_y, max_y = min(min_y, cy), max(max_y, cy)
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                bw = max_x - min_x + 1
                bh = max_y - min_y + 1
                if area < min_area or area > max_area or bw < 3 or bh < 3:
                    continue
                nx1, ny1 = min_x / w, min_y / h
                nx2, ny2 = (max_x + 1) / w, (max_y + 1) / h
                aspect = bw / max(bh, 1)
                comp_type = "text_like" if aspect > 3.0 else "icon_button"
                if bw * bh > h * w * 0.04:
                    comp_type = "panel"
                mius.append(
                    {
                        "box": (float(nx1), float(ny1), float(nx2), float(ny2)),
                        "type": comp_type,
                        "text": "",
                        "source": "vision",
                        "det_score": min(1.0, area / max(bw * bh, 1)),
                    }
                )
        return mius

    def _layout_skeleton_mius(self) -> list[dict]:
        boxes = [
            ((0.0, 0.0, 1.0, 0.12), "top_bar"),
            ((0.0, 0.88, 1.0, 1.0), "bottom_bar"),
            ((0.0, 0.12, 0.18, 0.88), "left_rail"),
            ((0.82, 0.12, 1.0, 0.88), "right_rail"),
            ((0.12, 0.18, 0.88, 0.82), "content_panel"),
        ]
        return [
            {"box": box, "type": role, "text": "", "source": "skeleton", "det_score": 0.35}
            for box, role in boxes
        ]

    def _merge_gui_mius(self, mius: list[dict], max_mius: int = 96) -> list[dict]:
        ordered = sorted(
            mius,
            key=lambda m: (
                -float(m.get("det_score", 0.0)),
                m["box"][1],
                m["box"][0],
            ),
        )
        kept: list[dict] = []
        for miu in ordered:
            box = miu["box"]
            if any(self._box_iou(box, other["box"]) > 0.72 for other in kept):
                continue
            kept.append(miu)
            if len(kept) >= max_mius:
                break
        return sorted(kept, key=lambda m: (m["box"][1], m["box"][0]))

    def _build_gui_mius_from_image(
        self,
        image: Optional[Image.Image],
        include_skeleton: bool = True,
    ) -> list[dict]:
        if image is None:
            return []
        mius, detect_meta = build_raw_mius(
            image,
            detector=self.miu_detector,
            include_skeleton=include_skeleton,
        )
        # Rule-based redundancy filter: drop OmniParser container / noise
        # boxes (panels, full-screen backgrounds, 1-pixel detector noise,
        # parents-of-many) before they consume budget in the downstream
        # MMR.  This is a UI-domain prior with NO learned weights so the
        # method-side hyperparameter count stays at zero.  The
        # ``no_prefilter`` policy flag disables it for ablation studies
        # (see §6 of communication.md / R8 in the ablation matrix).
        if "no_prefilter" in str(self.visual_prune_policy or "").lower():
            prefilter_meta = {
                "miu_prefilter_in": float(len(mius)),
                "miu_prefilter_kept": float(len(mius)),
                "miu_prefilter_dropped": 0.0,
                "miu_prefilter_drop_area": 0.0,
                "miu_prefilter_drop_cover": 0.0,
                "miu_prefilter_drop_tiny": 0.0,
                "miu_prefilter_drop_aspect": 0.0,
                "miu_prefilter_drop_parent": 0.0,
            }
        else:
            mius, prefilter_meta = filter_redundant_image_mius(mius)
        detect_meta.update(prefilter_meta)
        self._last_miu_detect_meta = detect_meta
        import os

        ocr_disabled = os.environ.get("MIU_DISABLE_OCR", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if (
            not self._ocr_empty_warned
            and not ocr_disabled
            and float(detect_meta.get("miu_ocr_count", 0)) <= 0
            and self.miu_detector
            in {
                "omniparser",
                "omniparser_icon",
                "omniparser_v2",
                "omniparser_full",
                "yolo_ocr",
            }
            and "oas" in str(self.visual_prune_policy or "")
        ):
            eval_logger.warning(
                "OAS path has 0 OCR text MIUs (miu_ocr_backend=%s); Module A remains text-free. "
                "Set MIU_DISABLE_OCR=1 to skip OCR (recommended for speed).",
                detect_meta.get("miu_ocr_backend", "?"),
            )
            self._ocr_empty_warned = True
        merged = self._merge_gui_mius(mius)
        for idx, miu in enumerate(merged):
            x1, y1, x2, y2 = miu["box"]
            cx, cy = self._box_center(miu["box"])
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            boundary = max(cx, 1.0 - cx, cy, 1.0 - cy)
            center = max(0.0, 1.0 - (((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5) / 0.7072)
            interactable = 1.0 if miu["type"] in {"icon_button", "text_like", "text"} else 0.45
            action_prior = 0.45 * center + 0.35 * cy + 0.20 * interactable
            topology = 0.45 * boundary + 0.35 * (1.0 - min(area * 8.0, 1.0)) + 0.20 * interactable
            miu.update(
                {
                    "miu_id": f"M{idx:02d}",
                    "area": area,
                    "center": (cx, cy),
                    "instruction_alignment": 0.0,
                    "interactable": interactable,
                    "action_prior": action_prior,
                    "topology_prior": topology,
                }
            )
        return merged

    def _format_miu_anchor_text(self, mius: list[dict]) -> str:
        if not mius:
            return ""
        ranked = sorted(
            mius,
            key=lambda m: (
                -(0.55 * float(m.get("action_prior", 0.0)) + 0.45 * float(m.get("topology_prior", 0.0))),
                m["box"][1],
                m["box"][0],
            ),
        )[: max(0, self.visual_prune_anchor_limit)]
        lines = [
            "\n[GUI component anchors for spatial grounding; coordinates are normalized x1,y1,x2,y2.]"
        ]
        for miu in ranked:
            x1, y1, x2, y2 = miu["box"]
            text = f' text="{miu.get("text")}"' if miu.get("text") else ""
            lines.append(
                f"{miu['miu_id']} {miu.get('type','component')} "
                f"({x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}){text}"
            )
        return "\n".join(lines)

    def _build_ome_mius(
        self,
        image: Optional[Image.Image],
        audios=None,
        videos=None,
        include_skeleton: bool = True,
    ) -> list[dict]:
        image_mius = self._build_gui_mius_from_image(
            image,
            include_skeleton=include_skeleton,
        )
        ome_mius, ome_meta = build_ome_mius(image_mius, videos=videos, audios=audios)
        self._last_ome_miu_meta = ome_meta
        filtered = self._filter_ome_mius_by_policy(ome_mius, self.visual_prune_policy)
        if "oas" in self.visual_prune_policy:
            # Legacy OAS diagnostics / ablations. Module A itself is now the
            # text-free interaction energy density attached below.
            drops = parse_oas_drop_set(self.visual_prune_policy)
            filtered = attach_oas_attributes(filtered, drop=drops)
            attach_module_a_info_density_proxy(filtered, self.visual_prune_policy)
            attach_cross_modal_resonance(filtered, self.visual_prune_policy)
            self._last_ome_miu_meta.update(oas_summary_meta(filtered))
            self._last_ome_miu_meta.update(module_a_summary_meta(filtered))
            self._last_ome_miu_meta.update(cross_modal_summary_meta(filtered))
        return filtered

    def _filter_ome_mius_by_policy(self, mius: list[dict], policy: str) -> list[dict]:
        policy = str(policy or "").lower()
        filtered: list[dict] = []
        for miu in mius:
            modality = str(miu.get("modality", "image"))
            source = str(miu.get("source", ""))
            if "no_audio" in policy and modality == "audio":
                continue
            if "no_video" in policy and modality == "video":
                continue
            if "no_skeleton" in policy and source == "skeleton":
                continue
            kept = dict(miu)
            if "no_sync" in policy:
                kept["sync_group"] = ""
                kept["sync_score"] = 0.0
            filtered.append(kept)
        return filtered

    def _dump_geopact_case_study_if_requested(
        self,
        *,
        doc: dict,
        doc_id: int,
        answer: str,
        gui_mius: Optional[list[dict]],
        image: Optional[Image.Image],
    ) -> None:
        """Append one JSON case record when ``GEOPACT_CASE_STUDY_DUMP`` is set."""
        import json
        import os
        from pathlib import Path

        dump_dir = os.environ.get("GEOPACT_CASE_STUDY_DUMP", "").strip()
        if not dump_dir:
            return

        def _ser(m: dict) -> dict:
            box = m.get("box")
            center = m.get("center")
            return {
                "miu_id": m.get("miu_id", ""),
                "modality": m.get("modality", "image"),
                "type": m.get("type", m.get("role", "")),
                "source": m.get("source", ""),
                "text": str(m.get("text") or "")[:120],
                "box": list(box) if box else None,
                "center": list(center) if center else None,
                "time": list(m.get("time")) if isinstance(m.get("time"), (list, tuple)) else None,
                "time_center": float(m.get("time_center", 0) or 0),
                "det_score": float(m.get("det_score", 0) or 0),
                "oas_imp": float(m.get("oas_imp", 0) or 0),
                "oas_exec": int(m.get("oas_exec", 0) or 0),
                "oas_ref": int(m.get("oas_ref", 0) or 0),
                "oas_state": int(m.get("oas_state", 0) or 0),
                "oas_cross": int(m.get("oas_cross", 0) or 0),
                "info_density": float(m.get("info_density", 0) or 0),
                "info_density_proxy": float(m.get("info_density_proxy", 0) or 0),
                "score": float(m.get("score", 0) or 0),
                "interaction_energy": float(m.get("interaction_energy", 0) or 0),
                "pentagonal_energy": float(m.get("pentagonal_energy", 0) or 0),
                "pentagonal_energy_sum": float(m.get("pentagonal_energy_sum", 0) or 0),
                "pentagonal_confidence_attr": float(m.get("pentagonal_confidence_attr", 0) or 0),
                "rho_image": float(m.get("rho_image", 0) or 0),
                "rho_video_support": float(m.get("rho_video_support", 0) or 0),
                "rho_audio_support": float(m.get("rho_audio_support", 0) or 0),
                "rho_multimodal": float(m.get("rho_multimodal", 0) or 0),
                "cross_graph_video_degree": float(m.get("cross_graph_video_degree", 0) or 0),
                "cross_graph_audio_path_count": float(m.get("cross_graph_audio_path_count", 0) or 0),
            }

        def _jsonable(value):
            if isinstance(value, dict):
                return {str(k): _jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_jsonable(v) for v in value]
            if hasattr(value, "item"):
                try:
                    return value.item()
                except Exception:
                    pass
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(value)

        def _metric_value(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return _jsonable(value)

        raw_mius: list[dict] = []
        if image is not None:
            try:
                raw, _ = build_raw_mius(image, detector=self.miu_detector, include_skeleton=False)
                raw_mius = [_ser(m) for m in raw[:80]]
            except Exception as exc:
                eval_logger.warning(f"case-study raw MIU dump failed: {exc!r}")

        image_mius = [m for m in (gui_mius or []) if str(m.get("modality", "image")) == "image"]
        video_mius = [m for m in (gui_mius or []) if str(m.get("modality", "")) == "video"]
        audio_mius = [m for m in (gui_mius or []) if str(m.get("modality", "")) == "audio"]
        last_selected_image_mius = getattr(self, "_last_selected_image_mius", []) or []
        selected = [_ser(m) for m in last_selected_image_mius]
        coord_text = self._format_inline_coord_text(last_selected_image_mius, self.visual_prune_policy)
        stats = {str(k): _metric_value(v) for k, v in (getattr(self, "_last_visual_prune_stats", {}) or {}).items()}
        detect_meta = {str(k): _metric_value(v) for k, v in (getattr(self, "_last_miu_detect_meta", {}) or {}).items()}

        gt_info: dict = {}
        eval_out: dict = {}
        try:
            from lmms_eval.tasks.agentcpm_gui import utils as ag_utils

            gt_info = ag_utils.parse_ground_truth_coordinate(doc)
            eval_out = ag_utils.evaluate_omni_step(
                doc,
                answer,
                screen_width=int(doc.get("image_width") or (image.size[0] if image else 1080)),
                screen_height=int(doc.get("image_height") or (image.size[1] if image else 2400)),
                coordinate_type="normalized",
            )
        except Exception as exc:
            eval_logger.warning(f"case-study eval dump failed: {exc!r}")

        record = {
            "doc_index": int(doc_id),
            "episode_id": doc.get("episode_id"),
            "step_index": doc.get("step_index", doc.get("step_id")),
            "instruction": (doc.get("instruction_en") or doc.get("instruction") or "")[:400],
            "image_path": doc.get("image_path", ""),
            "video_path": doc.get("video_path", ""),
            "audio_path": doc.get("audio_path", ""),
            "app": doc.get("app", ""),
            "task_category": doc.get("task_category", ""),
            "task_category_full": doc.get("task_category_full", ""),
            "action_history": doc.get("action_history", []),
            "target_action": doc.get("target_action", {}),
            "result_action_json": doc.get("result_action_json", ""),
            "policy": self.visual_prune_policy,
            "prune_ratio": self.visual_prune_ratio,
            "detect_meta": detect_meta,
            "prune_stats": stats,
            "raw_mius": raw_mius,
            "filtered_mius": [_ser(m) for m in image_mius],
            "video_mius": [_ser(m) for m in video_mius],
            "audio_mius": [_ser(m) for m in audio_mius],
            "selected_mius": selected,
            "coord_text": coord_text,
            "model_answer": answer,
            "gt_coord": gt_info,
            "eval": {
                "hit_format": bool(eval_out.get("hit_format")),
                "type_match": bool(eval_out.get("type_match")),
                "exact_match": bool(eval_out.get("exact_match")),
                "coordinate_distance": eval_out.get("coordinate_distance"),
            },
        }
        out_dir = Path(dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"case_doc{doc_id:05d}.json"
        record = _jsonable(record)
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        eval_logger.info(f"GeoPACT case-study dump -> {out_path}")

    def _message_has_modality(self, messages: List[Dict], modality: str) -> bool:
        for message in messages:
            for item in message.get("content", []) or []:
                if isinstance(item, dict) and item.get("type") == modality:
                    return True
        return False

    def _sanitize_one_media_item(self, item):
        if isinstance(item, np.ndarray):
            if item.dtype == np.uint8:
                if item.ndim == 3:
                    return Image.fromarray(item)
                if item.ndim == 4:
                    return [Image.fromarray(frame) for frame in item]
            return item
        if isinstance(item, (list, tuple)):
            return [self._sanitize_one_media_item(x) for x in item]
        return item

    def _sanitize_media_inputs(self, media):
        if media is None:
            return None
        if isinstance(media, list):
            cleaned = [self._sanitize_one_media_item(x) for x in media if x is not None]
            return cleaned or None
        return self._sanitize_one_media_item(media)

    def _prepare_processor_media(
        self,
        audios,
        images,
        videos,
        use_audio_in_video: bool,
        qwen_messages: Optional[List[Dict]] = None,
    ):
        has_audio = self._message_has_modality(qwen_messages or [], "audio")
        has_video = self._message_has_modality(qwen_messages or [], "video")

        if not has_audio:
            audios = None
        elif isinstance(audios, list):
            audios = [a for a in audios if a is not None]
            if not audios:
                audios = None
            elif len(audios) == 1:
                audios = audios[0]

        if audios is None:
            use_audio_in_video = bool(has_video and use_audio_in_video)

        images = self._sanitize_media_inputs(images)
        videos = self._sanitize_media_inputs(videos)
        return audios, images, videos, use_audio_in_video

    def _call_omni_processor(
        self,
        text: str,
        audios,
        images,
        videos,
        use_audio_in_video: bool,
        qwen_messages: Optional[List[Dict]] = None,
    ):
        audios, images, videos, use_audio_in_video = self._prepare_processor_media(
            audios, images, videos, use_audio_in_video, qwen_messages
        )
        kwargs = dict(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )
        try:
            return self.processor(**kwargs)
        except ValueError as error:
            if audios is None:
                raise
            eval_logger.warning(f"Omni processor audio fallback ({error!r}); retrying without audio.")
            kwargs["audio"] = None
            kwargs["use_audio_in_video"] = False
            return self.processor(**kwargs)

    @staticmethod
    def _drop_audio_items(messages: List[Dict]) -> List[Dict]:
        cleaned: List[Dict] = []
        for message in messages:
            content = []
            for item in message.get("content", []) or []:
                if isinstance(item, dict) and item.get("type") == "audio":
                    continue
                content.append(item)
            if content:
                cleaned.append({**message, "content": content})
        return cleaned

    def _process_mm_info_with_audio_fallback(
        self,
        qwen_messages: List[Dict],
        use_audio_in_video: bool,
    ) -> tuple[List[Dict], object, object, object, bool]:
        try:
            audios, images, videos = process_mm_info(qwen_messages, use_audio_in_video=use_audio_in_video)
            return qwen_messages, audios, images, videos, use_audio_in_video
        except Exception as error:
            eval_logger.warning(
                f"process_mm_info failed on official audio/video path ({error!r}); retrying without video-audio extraction."
            )

        try:
            audios, images, videos = process_mm_info(qwen_messages, use_audio_in_video=False)
            return qwen_messages, audios, images, videos, False
        except Exception as error:
            eval_logger.warning(
                f"process_mm_info still failed with explicit audio ({error!r}); retrying after dropping audio items."
            )

        audio_free_messages = self._drop_audio_items(qwen_messages)
        audios, images, videos = process_mm_info(audio_free_messages, use_audio_in_video=False)
        return audio_free_messages, None, images, videos, False

    @staticmethod
    def _format_inline_coord_text(mius: list[dict], policy: str) -> str:
        """Render kept current-screenshot MIU boxes as a Qwen-native grounding block.

        Coordinate space is **integer in [0, 1000]** in ``[x, y]`` order, to be
        bit-identical with:

        * Qwen2.5-VL / Qwen2.5-Omni pretraining grounding format
          (``<|box_start|>(x1,y1),(x2,y2)<|box_end|>`` with int [0,1000]),
        * the AgentCPM-GUI task system prompt's "## Coordinate System"
          section, which tells the model "All coordinates are normalized
          values in [0, 1000] range in [x, y] format" and shows
          ``{"coordinate": [500, 100]}`` as the click example, and
        * the touch_xy evaluation path in
          ``lmms_eval/tasks/agentcpm_gui/utils.py::auto_normalize_coordinate``
          which compares pred against GT in the same [0, 1000] space.

        Using float [0, 1] here was a real bug: it mismatched both Qwen's
        grounding pretraining distribution and the system-prompt convention
        the model is currently being asked to follow, leaving the model to
        guess which of two coordinate spaces the inline boxes live in.

        ``policy`` controls only the wrapper, not the numeric scale:
        - default                       : ``<|box_start|>(x1,y1),(x2,y2)<|box_end|>``
                                          (Qwen-VL/Omni pretraining-native)
        - ``coord_plain`` in policy     : ``(x1,y1,x2,y2)`` bare tuples
                                          (ablation: did the special tokens help?)
        - ``coord_named`` in policy     : ``<|object_ref_start|>{label}<|object_ref_end|>``
                                          prefix (ablation: does the type label help?)
        - ``coord_center`` in policy    : ``<|box_start|>(cx,cy)<|box_end|>`` -- v5 default
                                          (matches Qwen's single-point click output;
                                          ~3x fewer tokens than the box form).
        - ``coord_center_size``         : ``<|box_start|>(cx,cy,w,h)<|box_end|>``
                          compact size-aware variant for C2.
        - ``coord_shuffle/offset/random/dummy``: causal controls that keep
                                          insertion locality and marker format
                                          but corrupt coordinate semantics.
        """
        plain = "coord_plain" in policy
        named = "coord_named" in policy
        # v5: center-only mode.  The agent's final answer is a single
        # ``{"coordinate": [x, y]}`` (see AgentCPM-GUI utils.auto_normalize_coordinate),
        # so injecting box corners is redundant; the center point alone
        # already grounds the click.  Saves ~3x on coord text tokens and
        # leaves more compression budget for the LLM forward.
        center_size_mode = "coord_center_size" in policy or "compact_center_size" in policy
        center_mode = "coord_center" in policy and not center_size_mode
        do_shuffle = "coord_shuffle" in policy
        do_offset = "coord_offset" in policy
        do_random = "coord_random" in policy
        do_dummy = "coord_dummy" in policy
        tagged_policy = (
            ("coord_tagged" in policy or "coord_center" in policy or center_size_mode)
            and "coord_untagged" not in policy
            and "coord_plain" not in policy
            and not plain
        )

        def _candidate_header() -> str:
            return (
                "\n[Current screenshot candidates. Coordinates are normalized to the original screenshot frame in [0,1000].]\n"
            )

        if not mius:
            return ""
        ordered = sorted(mius, key=lambda m: (m["box"][1], m["box"][0]))
        chunks: list[str] = []

        def _to_q(v: float) -> int:
            # Quantize a [0, 1] normalized coord to the Qwen [0, 1000]
            # integer grid used by both the model's pretraining and the
            # task's coordinate-system spec.  ``round`` (not ``int``) is
            # important: ``int(0.4999*1000)`` would silently bias the kept
            # boxes toward the top-left corner.
            return int(round(max(0.0, min(1.0, float(v))) * 1000))

        def _shift_box(box: tuple[float, float, float, float], dx: float, dy: float) -> tuple[float, float, float, float]:
            x1, y1, x2, y2 = box
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            nx1 = max(0.0, min(1.0 - w, x1 + dx))
            ny1 = max(0.0, min(1.0 - h, y1 + dy))
            return (nx1, ny1, min(1.0, nx1 + w), min(1.0, ny1 + h))

        coord_boxes = [tuple(m["box"]) for m in ordered]
        if do_shuffle and len(coord_boxes) > 1:
            coord_boxes = coord_boxes[1:] + coord_boxes[:1]
        elif do_shuffle and coord_boxes:
            coord_boxes = [(1.0 - b[2], 1.0 - b[3], 1.0 - b[0], 1.0 - b[1]) for b in coord_boxes]
        if do_offset:
            coord_boxes = [_shift_box(b, 0.17, 0.13) for b in coord_boxes]
        if do_random:
            generated = []
            for i, _box in enumerate(coord_boxes):
                x1 = ((i * 37 + 11) % 83) / 100.0
                y1 = ((i * 53 + 7) % 83) / 100.0
                w = 0.06 + ((i * 17) % 19) / 200.0
                h = 0.06 + ((i * 29) % 19) / 200.0
                generated.append((x1, y1, min(0.99, x1 + w), min(0.99, y1 + h)))
            coord_boxes = generated
        if do_dummy:
            coord_boxes = [(0.0, 0.0, 0.0, 0.0) for _ in coord_boxes]

        def _position_words(box: tuple[float, float, float, float]) -> str:
            x1, y1, x2, y2 = box
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            if cy < 0.18:
                vertical = "top"
            elif cy > 0.78:
                vertical = "bottom"
            else:
                vertical = "middle"
            if cx < 0.33:
                horizontal = "left"
            elif cx > 0.67:
                horizontal = "right"
            else:
                horizontal = "center"
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if w * h > 0.04:
                size = "large"
            elif w > 0.25:
                size = "wide"
            elif h > 0.12:
                size = "tall"
            else:
                size = "small"
            return f"{size} {vertical}-{horizontal} UI region"

        def _candidate_label(miu: dict, box: tuple[float, float, float, float]) -> str:
            text = str(miu.get("text") or miu.get("content") or "").strip().replace("\n", " ")[:48]
            role = str(miu.get("type") or miu.get("role") or "ui").replace("_", " ").strip()
            if text:
                return f'{role} labeled "{text}"'
            return f"{_position_words(box)} ({role})"

        def _reason_text(miu: dict) -> str:
            parts = []
            rho_image = float(miu.get("rho_image", miu.get("info_density", miu.get("info_density_proxy", 0.0))) or 0.0)
            rho_video = float(miu.get("rho_video_support", 0.0) or 0.0)
            rho_audio = float(miu.get("rho_audio_support", 0.0) or 0.0)
            if rho_image > 0.0:
                parts.append(f"image={rho_image:.2f}")
            if rho_video > 0.05:
                parts.append(f"video={rho_video:.2f}")
            if rho_audio > 0.05:
                parts.append(f"audio={rho_audio:.2f}")
            return ", ".join(parts[:3]) if parts else "selected visual candidate"

        # BICA-T: tagged bundled block — keep C2 as a simple coordinate list.
        # Action-output rules live only in the task system prompt.
        tagged_bundle = tagged_policy
        if tagged_bundle and ordered:
            lines: list[str] = [_candidate_header()]
            for cand_idx, (miu, box) in enumerate(zip(ordered, coord_boxes), start=1):
                x1, y1, x2, y2 = box
                if center_mode:
                    qcx = _to_q(0.5 * (x1 + x2))
                    qcy = _to_q(0.5 * (y1 + y2))
                    block = f"<|box_start|>({qcx},{qcy})<|box_end|>"
                elif center_size_mode:
                    qcx = _to_q(0.5 * (x1 + x2))
                    qcy = _to_q(0.5 * (y1 + y2))
                    qw = _to_q(x2 - x1)
                    qh = _to_q(y2 - y1)
                    block = f"<|box_start|>({qcx},{qcy},{qw},{qh})<|box_end|>"
                else:
                    qx1, qy1, qx2, qy2 = _to_q(x1), _to_q(y1), _to_q(x2), _to_q(y2)
                    block = f"<|box_start|>({qx1},{qy1}),({qx2},{qy2})<|box_end|>"
                    qcx = _to_q(0.5 * (x1 + x2))
                    qcy = _to_q(0.5 * (y1 + y2))
                if named:
                    label = str(miu.get("type", miu.get("role", "ui")))[:24]
                    block = f"<|object_ref_start|>{label}<|object_ref_end|>{block}"
                lines.append(f"Candidate {cand_idx}: center={block}\n")
            return "".join(lines)

        for miu, box in zip(ordered, coord_boxes):
            x1, y1, x2, y2 = box
            if center_mode or center_size_mode:
                qcx = _to_q(0.5 * (x1 + x2))
                qcy = _to_q(0.5 * (y1 + y2))
                if plain:
                    if center_size_mode:
                        chunks.append(f"({qcx},{qcy},{_to_q(x2 - x1)},{_to_q(y2 - y1)})")
                    else:
                        chunks.append(f"({qcx},{qcy})")
                else:
                    if center_size_mode:
                        block = f"<|box_start|>({qcx},{qcy},{_to_q(x2 - x1)},{_to_q(y2 - y1)})<|box_end|>"
                    else:
                        block = f"<|box_start|>({qcx},{qcy})<|box_end|>"
                    if named:
                        label = str(miu.get("type", miu.get("role", "ui")))[:24]
                        chunks.append(f"<|object_ref_start|>{label}<|object_ref_end|>{block}")
                    else:
                        chunks.append(block)
                continue
            qx1, qy1, qx2, qy2 = _to_q(x1), _to_q(y1), _to_q(x2), _to_q(y2)
            if plain:
                chunks.append(f"({qx1},{qy1},{qx2},{qy2})")
            else:
                bbox = f"<|box_start|>({qx1},{qy1}),({qx2},{qy2})<|box_end|>"
                if named:
                    label = str(miu.get("type", miu.get("role", "ui")))[:24]
                    chunks.append(f"<|object_ref_start|>{label}<|object_ref_end|>{bbox}")
                else:
                    chunks.append(bbox)
        return "".join(chunks)

    def _use_source_projector_compact_backend(self) -> bool:
        backend = os.environ.get("MIU_COMPACT_BACKEND", "source").strip().lower()
        return backend not in {"0", "false", "off", "manual", "legacy", "wrapper"}

    def _source_projector_backend_strict(self) -> bool:
        backend = os.environ.get("MIU_COMPACT_BACKEND", "source").strip().lower()
        return backend in {"source_strict", "internal_strict", "qwen_strict"}

    def _prepare_source_projector_payload(
        self,
        inputs,
        use_audio_in_video: bool,
        external_av_support_only: bool,
        gui_mius: Optional[list[dict]] = None,
        instruction_text: str = "",
    ) -> Optional[dict]:
        thinker = self.model.thinker
        prepare = getattr(thinker, "prepare_omnigui_projector_inputs", None)
        if prepare is None:
            return None

        def _return_payload(payload):
            return payload

        try:
            payload = prepare(
                input_ids=inputs.get("input_ids", None),
                input_features=inputs.get("input_features", None),
                pixel_values=inputs.get("pixel_values", None),
                pixel_values_videos=inputs.get("pixel_values_videos", None),
                image_grid_thw=inputs.get("image_grid_thw", None),
                video_grid_thw=inputs.get("video_grid_thw", None),
                attention_mask=inputs.get("attention_mask", None),
                feature_attention_mask=inputs.get("feature_attention_mask", None),
                audio_feature_lengths=inputs.get("audio_feature_lengths", None),
                use_audio_in_video=use_audio_in_video,
                video_second_per_grid=inputs.get("video_second_per_grid", None),
                prune_hook=None,
                omnigui_prune_config={
                    "ratio": float(self.visual_prune_ratio),
                    "scope": str(self.visual_prune_scope or "all"),
                    "policy": str(self.visual_prune_policy or ""),
                    "gui_mius": gui_mius or [],
                    "instruction_text": instruction_text or "",
                    "tokenizer": self.tokenizer,
                    "external_av_support_only": bool(external_av_support_only),
                },
                skip_audio_features=external_av_support_only,
                skip_video_features=external_av_support_only,
            )
        except TypeError as exc:
            if "omnigui_prune_config" not in str(exc):
                raise
            if self._source_projector_backend_strict():
                raise
            eval_logger.warning("Qwen source compact API is old; falling back to wrapper backend: %r", exc)
            payload = prepare(
                input_ids=inputs.get("input_ids", None),
                input_features=inputs.get("input_features", None),
                pixel_values=inputs.get("pixel_values", None),
                pixel_values_videos=inputs.get("pixel_values_videos", None),
                image_grid_thw=inputs.get("image_grid_thw", None),
                video_grid_thw=inputs.get("video_grid_thw", None),
                attention_mask=inputs.get("attention_mask", None),
                feature_attention_mask=inputs.get("feature_attention_mask", None),
                audio_feature_lengths=inputs.get("audio_feature_lengths", None),
                use_audio_in_video=use_audio_in_video,
                video_second_per_grid=inputs.get("video_second_per_grid", None),
                prune_hook=_return_payload,
                skip_audio_features=external_av_support_only,
                skip_video_features=external_av_support_only,
            )
        except Exception as exc:
            if self._source_projector_backend_strict():
                raise
            eval_logger.warning("Qwen source projector compact backend failed; falling back to wrapper backend: %r", exc)
            return None
        if not payload:
            return None
        payload = dict(payload)
        payload["_omnigui_projector_payload"] = True
        return payload

    @torch.inference_mode()
    def _compact_qwen_inputs_after_projector(
        self,
        inputs,
        use_audio_in_video: bool,
        gui_mius: Optional[list[dict]] = None,
        instruction_text: str = "",
    ):
        input_ids_probe = inputs.get("input_ids", None) if hasattr(inputs, "get") else None
        if input_ids_probe is None:
            input_ids_probe = inputs.input_ids
        if not self._use_visual_pruning() or input_ids_probe.shape[0] != 1:
            return None

        precompact_stats = dict(self._last_visual_prune_stats or {})
        policy_lower = str(self.visual_prune_policy or "").lower()
        external_av_support_only = (
            ("oas" in policy_lower or "ome" in policy_lower)
            and "keep_av_tokens" not in policy_lower
            and "retain_av_tokens" not in policy_lower
        )
        projector_payload = bool(hasattr(inputs, "get") and inputs.get("_omnigui_projector_payload", False))
        if (not projector_payload) and self._use_source_projector_compact_backend():
            source_prepare_t0 = time.perf_counter()
            source_payload = self._prepare_source_projector_payload(
                inputs,
                use_audio_in_video=use_audio_in_video,
                external_av_support_only=external_av_support_only,
                gui_mius=gui_mius,
                instruction_text=instruction_text,
            )
            if source_payload is not None:
                source_stats = dict(source_payload.get("_omnigui_projector_stats", {}))
                source_stats["compact_projector_prepare_ms"] = float(
                    (time.perf_counter() - source_prepare_t0) * 1000.0
                )
                source_payload["_omnigui_projector_stats"] = source_stats
                if source_payload.get("_omnigui_source_compact", False):
                    source_stats["compact_backend_source"] = 1.0
                    source_stats["source_compact_impl"] = 1.0
                    if float(precompact_stats.get("coord_system_prompt_note", 0.0)) > 0.0:
                        source_stats["coord_system_prompt_note"] = float(precompact_stats["coord_system_prompt_note"])
                    source_stats["detector_seconds"] = float(getattr(self, "_last_detect_seconds", 0.0))
                    source_stats.update(self._last_miu_detect_meta)
                    source_stats.update(self._last_ome_miu_meta)
                    try:
                        from lmms_eval.models.gui_miu_detectors import detector_cache_stats

                        source_stats.update(detector_cache_stats())
                    except Exception:
                        pass
                    self._last_visual_prune_stats = source_stats
                    return {
                        key: value
                        for key, value in source_payload.items()
                        if key in {
                            "input_ids",
                            "inputs_embeds",
                            "attention_mask",
                            "image_grid_thw",
                            "video_grid_thw",
                            "video_second_per_grid",
                            "feature_attention_mask",
                            "position_ids",
                            "rope_deltas",
                        }
                        and value is not None
                    }
                # Source backend returned a payload but did not perform a full
                # compaction. Under strict module decoupling the frontend must
                # not run any pruning itself, so decline to compact this step.
                return None
        # Strict module decoupling: the evaluation frontend performs NO token
        # pruning / compression / feature integration. All heavy lifting lives
        # in the Transformer source (``thinker.prepare_omnigui_projector_inputs``).
        # When the source backend is disabled or cannot produce a compacted
        # payload, fall back to the unpruned inputs; the caller treats ``None``
        # as "generate on the original sequence".
        return None

    @staticmethod
    def _clean_pure_instruction_text(text: str) -> str:
        """Keep task instruction text while dropping prompt/action scaffolding."""
        if not text:
            return ""
        kept: list[str] = []
        for raw_line in str(text).replace("\r", "\n").split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            low = line.lower()
            if line.startswith("[") and (
                "history screenshot" in low or "gui component anchors" in low
            ):
                continue
            prefixed = re.match(r"^(goal|task|instruction)\s*[:：]\s*(.+)$", line, re.IGNORECASE)
            if prefixed:
                value = prefixed.group(2).strip()
                if value:
                    kept.append(value)
                continue
            if re.match(r"^(action history|history|previous action|step)\s*[:：]", line, re.IGNORECASE):
                continue
            if re.match(r"^step\s*\d+\s*[:：]", line, re.IGNORECASE):
                continue
            if re.match(r"^(answer|response|output|format|system)\s*[:：]", line, re.IGNORECASE):
                continue
            if "<|box_start|>" in line or "<|object_ref_start|>" in line:
                continue
            if "output the next action" in low or "next action as json" in low:
                continue
            if "analyze the current screenshot" in low:
                continue
            if (
                "candidate coordinate" in low
                or "candidate coordinates" in low
                or "gui component anchor" in low
                or "gui component anchors" in low
                or "bounding box" in low
                or "bounding boxes" in low
            ):
                continue
            if re.search(r"\b(?:x1|y1|x2|y2|center_x|center_y|left|top|right|bottom)\s*=\s*[-+]?\d*\.?\d+", low):
                continue
            if re.search(r"\([-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+(?:\s*,\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+)?\)", line):
                continue
            kept.append(line)
        return " ".join(kept).strip()

    def _pure_instruction_text_from_doc(self, doc: Dict, qwen_messages: Optional[List[Dict]] = None) -> str:
        """Extract the task instruction only; exclude system/action/history text."""
        if isinstance(doc, dict):
            for key in ("instruction", "task_description", "instruction_en"):
                value = doc.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        pieces: list[str] = []
        for message in qwen_messages or []:
            if message.get("role") == "system":
                continue
            for item in message.get("content", []):
                if isinstance(item, dict) and item.get("type") == "text":
                    cleaned = self._clean_pure_instruction_text(str(item.get("text", "")))
                    if cleaned:
                        pieces.append(cleaned)
        return " ".join(pieces).strip()

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        current_use_audio = False  # Flag to check whether we are using video or not

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            step_t0 = time.perf_counter()
            self._last_visual_prune_stats = {}
            self._last_detect_seconds = 0.0
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            # For better performance, please visit the Qwen-Omni repo to get the latest system prompt based on tasks.
            # https://github.com/QwenLM/Qwen2.5-Omni/tree/main/cookbooks
            message = [{"role": "system", "content": [{"type": "text", "text": self._system_prompt_for_current_policy()}]}]
            for i, context in enumerate(contexts):
                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        current_use_audio = self._check_if_video_has_audio(visual)
                        if self.use_custom_video_loader:
                            visual = read_video_pyav_base64(visual, num_frm=self.max_num_frames, fps=self.fps, img_format="JPEG", max_image_size=self.max_image_size)
                            image_contents = list(map(lambda x: f"data:image/jpeg;base64,{x}", visual))
                            message.append({"role": "user", "content": [{"type": "video", "video": image_contents}, {"type": "text", "text": context}]})
                        else:  # Model video loader
                            message.append({"role": "user", "content": [{"type": "video", "video": visual}, {"type": "text", "text": context}]})

                    elif isinstance(visual, Image.Image):  # Single image
                        message.append({"role": "user", "content": [{"type": "image", "image": visual}, {"type": "text", "text": context}]})

                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                        single_message = {"role": "user", "content": []}
                        for v in visual:
                            single_message["content"].append({"type": "image", "image": v})
                        single_message["content"].append({"type": "text", "text": context})
                        message.append(single_message)

                    # Fixed code for audio messages
                    elif isinstance(visual, dict):  # Single audio
                        current_use_audio = True
                        audio = self.resample_audio(visual["array"], visual["sampling_rate"])
                        audio_splits = split_audio(audio, 4800000)  # Split the audio to 5 min chunks
                        single_message = {"role": "user", "content": []}
                        for i in range(len(audio_splits)):
                            single_message["content"].append({"type": "audio", "audio": audio_splits[i]})
                        single_message["content"].append({"type": "text", "text": context})
                        message.append(single_message)

                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, dict) for v in visual):  # Multiple audios
                        current_use_audio = True
                        for i, v in enumerate(visual):
                            audio = self.resample_audio(v["array"], v["sampling_rate"])
                            audio_splits = split_audio(audio, 4800000)  # Split the audio to 5 min chunks
                            single_message = {"role": "user", "content": []}
                            for j in range(len(audio_splits)):
                                single_message["content"].append({"type": "audio", "audio": audio_splits[j]})
                            single_message["content"].append({"type": "text", "text": context})
                            message.append(single_message)

                    else:
                        raise ValueError(f"Unknown visual type: {type(visual)}")

            text = self.processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(message, use_audio_in_video=current_use_audio)
            inputs = self._call_omni_processor(text, audios, images, videos, current_use_audio, message)

            if self.device_map == "auto":
                inputs = inputs.to("cuda").to(self.model.dtype)
            else:
                inputs = inputs.to(self.model.device).to(self.model.dtype)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id

            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                generate_t0 = time.perf_counter()
                cont, gen_timing = self._call_model_generate_with_step_timing(
                    **inputs,
                    return_audio=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    use_audio_in_video=current_use_audio,
                    thinker_do_sample=False,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                generate_ms = (time.perf_counter() - generate_t0) * 1000.0
                peak_mem_mb = (
                    torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                    if torch.cuda.is_available()
                    else 0.0
                )
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                answer = ""
                res.append(answer)
                pbar.update(1)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                continue

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            self._last_visual_prune_stats.update({k: v for k, v in gen_timing.items() if isinstance(v, (int, float))})
            prefill_forward_ms = float(gen_timing.get("llm_prefill_forward_ms", gen_timing.get("prefill_forward_ms", 0.0)))
            model_prefill_total_ms = float(gen_timing.get("model_prefill_total_ms", gen_timing.get("prefill_ms", prefill_forward_ms)))
            step_ttft_ms = (generate_t0 - step_t0) * 1000.0 + float(gen_timing.get("ttft_ms", gen_timing.get("prefill_ms", 0.0)))
            self._log_generation_timing(
                step_t0=step_t0,
                ttft_ms=step_ttft_ms,
                prefill_ms=model_prefill_total_ms,
                decode_ms=float(gen_timing.get("decode_ms", 0.0)),
                generate_ms=generate_ms,
                peak_mem_mb=peak_mem_mb,
                prompt_len=int(inputs.input_ids.shape[1]),
                generated_ids_trimmed=generated_ids_trimmed,
                active_inputs=inputs,
                forward_calls=float(gen_timing.get("forward_calls", 0.0)),
            )
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                answers[i] = ans
            content = []
            for ans, context in zip(answers, contexts):
                res.append(ans)
                content.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res


@register_model("qwen2_5_omni_chat")
class Qwen2_5_Omni_Chat(Qwen2_5_Omni):
    is_simple = False

    def _convert_messages_to_qwen(self, messages: List[Dict]) -> Tuple[List[Dict], bool]:
        qwen_messages = []
        use_audio_in_video = False

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]

            qwen_content = []
            for item in content:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")
                if item_type == "text":
                    text = item.get("text", "")
                    if text:
                        qwen_content.append({"type": "text", "text": text})
                elif item_type == "image":
                    image = item.get("image", item.get("url"))
                    if image:
                        qwen_content.append({"type": "image", "image": image})
                elif item_type == "video":
                    video = item.get("video", item.get("url"))
                    if video:
                        video_item = {"type": "video", "video": video}
                        if self.use_custom_video_loader and isinstance(video, str):
                            try:
                                frames = read_video_pyav_base64(
                                    video,
                                    num_frm=self.max_num_frames,
                                    fps=self.fps,
                                    img_format="JPEG",
                                    max_image_size=self.max_image_size,
                                )
                                video = [f"data:image/jpeg;base64,{frame}" for frame in frames]
                                video_item["video"] = video
                            except Exception as error:
                                eval_logger.warning(f"Custom video loader failed for {video!r}: {error!r}; using processor loader.")
                        elif self.max_num_frames:
                            video_item["nframes"] = int(self.max_num_frames)
                        qwen_content.append(video_item)
                elif item_type == "audio":
                    audio = item.get("audio", item.get("url"))
                    if isinstance(audio, dict):
                        audio = audio.get("path") or audio.get("url")
                    if audio:
                        qwen_content.append({"type": "audio", "audio": audio})

            if qwen_content:
                qwen_messages.append({"role": role, "content": qwen_content})

        return qwen_messages, use_audio_in_video

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Qwen2.5-Omni Responding")

        for request in requests:
            step_t0 = time.perf_counter()
            self._last_visual_prune_stats = {}
            self._last_detect_seconds = 0.0
            ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            messages = doc_to_messages(doc)
            qwen_messages, use_audio_in_video = self._convert_messages_to_qwen(messages)
            qwen_messages = self._append_coord_system_prompt_note_to_messages(qwen_messages)

            try:
                qwen_messages, audios, images, videos, use_audio_in_video = self._process_mm_info_with_audio_fallback(
                    qwen_messages,
                    use_audio_in_video,
                )
                gui_mius = None
                pure_instruction_text = self._pure_instruction_text_from_doc(doc, qwen_messages)
                if self._use_visual_pruning() and (
                    "miu" in self.visual_prune_policy
                    or "ome" in self.visual_prune_policy
                    or "oas" in self.visual_prune_policy
                ) and images:
                    # v4-min skeleton policy: OAS always drops skeleton (per
                    # user direction; the 3B ablation showed skeleton anchors
                    # provided no measurable lift), while legacy OME policies
                    # remain controllable via ``no_skeleton`` in the name.
                    if "oas" in self.visual_prune_policy:
                        include_skeleton = False
                    else:
                        include_skeleton = "no_skeleton" not in self.visual_prune_policy
                    # Module 3 option 1: time the detector (YOLO+caption+OCR)
                    # separately so the per-step log can attribute latency to
                    # detection vs. LLM forward (the detector is a constant
                    # overhead independent of the pruned sequence length).
                    detect_t0 = time.perf_counter()
                    if "ome" in self.visual_prune_policy or "oas" in self.visual_prune_policy:
                        gui_mius = self._build_ome_mius(
                            images[-1],
                            audios=audios,
                            videos=videos,
                            include_skeleton=include_skeleton,
                        )
                    else:
                        gui_mius = self._build_gui_mius_from_image(
                            images[-1],
                            include_skeleton=include_skeleton,
                        )
                    self._last_detect_seconds = time.perf_counter() - detect_t0
                    # OAS path performs spatial anchoring AFTER the projector
                    # via inline grounding text (module 3); the legacy v3
                    # ``_format_miu_anchor_text`` would crash on audio MIUs
                    # (no box) and is also redundant with inline coord. Skip
                    # it for OAS/OME/no_rebind policies.
                    anchor_text = "" if (
                        "no_rebind" in self.visual_prune_policy
                        or "ome" in self.visual_prune_policy
                        or "oas" in self.visual_prune_policy
                    ) else self._format_miu_anchor_text(gui_mius)
                    if anchor_text and qwen_messages:
                        qwen_messages[-1].setdefault("content", []).append({"type": "text", "text": anchor_text})
                text = self.processor.apply_chat_template(qwen_messages, add_generation_prompt=True, tokenize=False)
                inputs = self._call_omni_processor(text, audios, images, videos, use_audio_in_video, qwen_messages)

                if self.device_map == "auto":
                    inputs = inputs.to("cuda").to(self.model.dtype)
                else:
                    inputs = inputs.to(self.model.device).to(self.model.dtype)

                current_gen_kwargs = dict(gen_kwargs)
                max_new_tokens = current_gen_kwargs.get("max_new_tokens", 4096)
                temperature = current_gen_kwargs.get("temperature", 0)
                top_p = current_gen_kwargs.get("top_p", None)
                num_beams = current_gen_kwargs.get("num_beams", 1)
                do_sample = bool(temperature and temperature > 0)
                if not do_sample:
                    temperature = None
                    top_p = None

                compact_inputs = self._compact_qwen_inputs_after_projector(
                    inputs,
                    use_audio_in_video,
                    gui_mius,
                    instruction_text=pure_instruction_text,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                generate_t0 = time.perf_counter()
                if compact_inputs is not None:
                    prompt_len = compact_inputs["input_ids"].shape[1]
                    cont, gen_timing = self._call_model_generate_with_step_timing(
                        **compact_inputs,
                        return_audio=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        num_beams=num_beams,
                        max_new_tokens=max_new_tokens,
                        use_cache=self.use_cache,
                        use_audio_in_video=use_audio_in_video,
                        thinker_do_sample=False,
                    )
                else:
                    prompt_len = inputs.input_ids.shape[1]
                    cont, gen_timing = self._call_model_generate_with_step_timing(
                        **inputs,
                        return_audio=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        num_beams=num_beams,
                        max_new_tokens=max_new_tokens,
                        use_cache=self.use_cache,
                        use_audio_in_video=use_audio_in_video,
                        thinker_do_sample=False,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                generate_ms = (time.perf_counter() - generate_t0) * 1000.0
                peak_mem_mb = (
                    torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                    if torch.cuda.is_available()
                    else 0.0
                )
                generated_ids_trimmed = [
                    out_ids[prompt_len:] if out_ids.shape[-1] > prompt_len else out_ids
                    for out_ids in cont
                ]
                self._last_visual_prune_stats.update({k: v for k, v in gen_timing.items() if isinstance(v, (int, float))})
                prefill_forward_ms = float(gen_timing.get("llm_prefill_forward_ms", gen_timing.get("prefill_forward_ms", 0.0)))
                model_prefill_total_ms = float(gen_timing.get("model_prefill_total_ms", gen_timing.get("prefill_ms", prefill_forward_ms)))
                step_ttft_ms = (generate_t0 - step_t0) * 1000.0 + float(gen_timing.get("ttft_ms", gen_timing.get("prefill_ms", 0.0)))
                self._log_generation_timing(
                    step_t0=step_t0,
                    ttft_ms=step_ttft_ms,
                    prefill_ms=model_prefill_total_ms,
                    decode_ms=float(gen_timing.get("decode_ms", 0.0)),
                    generate_ms=generate_ms,
                    peak_mem_mb=peak_mem_mb,
                    prompt_len=int(prompt_len),
                    generated_ids_trimmed=generated_ids_trimmed,
                    active_inputs=compact_inputs if compact_inputs is not None else inputs,
                    forward_calls=float(gen_timing.get("forward_calls", 0.0)),
                )
                answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                answer = answers[0] if answers else ""
                if not answer and generated_ids_trimmed:
                    debug_ids = generated_ids_trimmed[0].detach().cpu().tolist()
                    eval_logger.warning(
                        "Empty decoded answer after generation: "
                        f"prompt_len={prompt_len} cont_shape={tuple(cont[0].shape)} "
                        f"trimmed_len={len(debug_ids)} first_ids={debug_ids[:12]}"
                    )
            except Exception as error:
                eval_logger.exception(f"Error {error!r} in generating")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                answer = ""

            try:
                doc = self.task_dict[task][split][doc_id]
                dump_images = locals().get("images")
                dump_image = dump_images[-1] if dump_images else None
                self._dump_geopact_case_study_if_requested(
                    doc=doc,
                    doc_id=int(doc_id),
                    answer=answer,
                    gui_mius=gui_mius,
                    image=dump_image,
                )
            except Exception as dump_exc:
                import sys

                print(f"case-study dump skipped: {dump_exc!r}", file=sys.stderr, flush=True)
                eval_logger.warning(f"case-study dump skipped: {dump_exc!r}")

            res.append(answer)
            self.cache_hook.add_partial("generate_until", (ctx, gen_kwargs), answer)
            pbar.update(1)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
