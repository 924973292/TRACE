"""Repository overlay re-exports (not part of omni-gui/OmniGUI upstream utils.py).

Keeps upstream ``utils.py`` byte-identical to official except for a single import
of this module at file end. GeoPACT / GUI-Owl / timing metrics live here.
"""

from lmms_eval.tasks.agentcpm_gui.utils_eval_extensions import (  # noqa: F401
    gui_owl_native_current_doc_to_messages,
    gui_owl_native_multiturn_doc_to_messages,
    omni_aggregate_decode_per_token_ms,
    omni_aggregate_decode_total_ms,
    omni_aggregate_e2e_latency_ms,
    omni_aggregate_episode_max_gpu_memory_gb,
    omni_aggregate_episode_mean_step_e2e_latency_ms,
    omni_aggregate_episode_sum_e2e_latency_ms,
    omni_aggregate_episode_sum_generation_ms,
    omni_aggregate_episode_sum_input_tokens,
    omni_aggregate_exact_match,
    omni_aggregate_generation_ms,
    omni_aggregate_goal_progress,
    omni_aggregate_input_tokens,
    omni_aggregate_llm_prefill_forward_ms,
    omni_aggregate_output_tokens,
    omni_aggregate_prefill_ms,
    omni_aggregate_preprocess_ms,
    omni_aggregate_success_rate,
    omni_aggregate_throughput_tokens_per_s,
    omni_aggregate_ttft_ms,
    omni_aggregate_type_match,
    omni_aggregate_vlm_prefill_ms,
    omni_process_results,
    qwen_official_merged_doc_to_messages,
)

from lmms_eval.tasks.agentcpm_gui.utils_pre_official_replace import (  # noqa: F401
    _session_append_screens_to_content,
    _session_tail_prompt_parts,
    _session_task_description,
    ablation_v2_no_av_current_image_only_doc_to_messages,
    ablation_v2_no_av_hist_k5_doc_to_messages,
    build_session_prefix_messages,
)
