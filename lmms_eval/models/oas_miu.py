"""Legacy OAS diagnostics for omni-modal GUI MIUs.

Design goals (Method Upgrade v8-min, communication.md §2 module 1):
* **Module A** is strict **InfoDensity** = interaction energy E / n_tokens.
    The OAS fields below are retained as diagnostics / legacy ablations only;
    they do not consume instruction text.
* Importance (InfoDensity) and diversity are **strictly decoupled**: diversity
    is handled only in Module B.
* The production Module A energy is the pentagonal detector-native energy in
    ``miu_info_density.py``.  The four OAS binary fields below are kept so older
    ablations and score-only diagnostic rows remain auditable.
* Audio cue keywords are kept as a fixed, English+Chinese list so the dictionary
    is auditable for legacy diagnostics, not a tuned hyperparameter.

The four attributes are computed in a single pass over the full MIU list
because two of them ("executable" and "state") use step-level top-50 %
thresholds; doing it once and attaching the binary flags to each MIU dict
keeps the downstream selector branch-free.
"""

from __future__ import annotations

from typing import Iterable, Sequence

# Fixed, deterministic audio-cue dictionary used only for ``A_ref`` on audio
# MIUs.  Keep it short and frozen so the method has zero tunable knobs.
_AUDIO_CUE_KEYWORDS: tuple[str, ...] = (
    "sound", "ring", "alert", "click", "voice", "notification", "tone", "beep",
    "声", "响", "音", "提示", "通知", "铃",
)

# OAS attribute keys we attach to every MIU dict.
_OAS_KEYS: tuple[str, ...] = (
    "oas_exec",   # A_exec  — executable / actionable
    "oas_ref",    # A_ref   — legacy referential diagnostic
    "oas_state",  # A_state — step-event aligned state evidence
    "oas_cross",  # A_cross — cross-modal supported
)

_DEFAULT_OAS_DROP: frozenset[str] = frozenset()

# Fast GUI A_ref substrate (no OCR): frozen instruction-token → detector-role hints.
# Auditable dictionary, not a tuned weight vector.
_INSTRUCTION_ROLE_HINTS: tuple[tuple[str, str], ...] = (
    ("search", "input"),
    ("search", "text_field"),
    ("search", "icon_button"),
    ("find", "icon_button"),
    ("click", "icon_button"),
    ("click", "button"),
    ("tap", "icon_button"),
    ("press", "icon_button"),
    ("button", "icon_button"),
    ("button", "button"),
    ("login", "input"),
    ("login", "text_field"),
    ("sign", "input"),
    ("type", "input"),
    ("input", "input"),
    ("enter", "input"),
    ("menu", "icon_button"),
    ("settings", "icon_button"),
    ("setting", "icon_button"),
    ("home", "icon_button"),
    ("back", "icon_button"),
    ("close", "icon_button"),
    ("播放", "icon_button"),
    ("点击", "icon_button"),
    ("搜索", "input"),
    ("输入", "input"),
    ("设置", "icon_button"),
    ("登录", "input"),
    ("返回", "icon_button"),
    ("关闭", "icon_button"),
)


def miu_referential_text(miu: dict) -> str:
    """Lowercased blob for A_ref: optional OCR text + detector role/type tokens."""
    parts: list[str] = []
    for key in ("text", "caption", "content"):
        val = str(miu.get(key) or "").strip()
        if val:
            parts.append(val.lower())
    role = str(miu.get("type") or miu.get("role") or "").strip().lower()
    if role:
        parts.append(role.replace(" ", "_"))
        parts.append(role.replace("_", " "))
    return " ".join(parts)


def instruction_role_referential(miu: dict, keywords: Sequence[str]) -> bool:
    """Fast A_ref: instruction ↔ OmniParser role/type (O(1) per MIU, no OCR)."""
    if not keywords:
        return False
    role = str(miu.get("type") or miu.get("role") or "").lower()
    if not role:
        return False
    blob = miu_referential_text(miu)
    for kw in keywords:
        if len(kw) < 2:
            continue
        if kw in blob or kw in role or role in kw:
            return True
        for hint, expected in _INSTRUCTION_ROLE_HINTS:
            if hint in kw or kw in hint:
                if expected in role:
                    return True
    return False


def format_miu_type_instance_label(miu: dict) -> str:
    """Minimal BICA identity: instance id + type (disambiguates duplicate types).

    Example: ``M02 icon_button``
    """
    tokens: list[str] = []
    miu_id = str(miu.get("miu_id") or "").strip()
    if miu_id:
        tokens.append(miu_id)
    role = str(miu.get("type") or miu.get("role") or "component").strip()
    if role:
        tokens.append(role.replace(" ", "_")[:24])
    return " ".join(tokens) if tokens else "component"


def format_miu_binding_descriptor(miu: dict, *, max_text_len: int = 48) -> str:
    """One-line MIU identity for BICA-T (tagged bundled coordinates).

    Example: ``M02 icon_button text="Settings"``
    """
    tokens: list[str] = []
    miu_id = str(miu.get("miu_id") or "").strip()
    if miu_id:
        tokens.append(miu_id)
    role = str(miu.get("type") or miu.get("role") or "component").strip()
    if role:
        tokens.append(role.replace(" ", "_")[:24])
    blob = miu_referential_text(miu)
    if blob:
        snippet = blob[:max_text_len].replace('"', "'")
        tokens.append(f'text="{snippet}"')
    return " ".join(tokens) if tokens else "component"


def _instruction_keywords(instruction_text: str) -> tuple[str, ...]:
    if not instruction_text:
        return ()
    cleaned = instruction_text.replace("，", " ").replace("。", " ").replace(",", " ").replace(".", " ")
    tokens = [tok.lower() for tok in cleaned.split() if len(tok) > 1]
    return tuple(tokens)


def _instruction_has_audio_cue(instruction_text: str) -> bool:
    if not instruction_text:
        return False
    lower = instruction_text.lower()
    return any(cue in lower for cue in _AUDIO_CUE_KEYWORDS)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _median(values: Sequence[float]) -> float:
    """Top-50 % threshold proxy: every value >= median is in the top half.

    Using the median (not a percentile rank) keeps the threshold deterministic
    and ties-friendly: identical scores stay together in the same OAS bucket.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return 0.5 * (float(ordered[mid - 1]) + float(ordered[mid]))


def attach_oas_attributes(
    mius: list[dict],
    instruction_text: str = "",
    drop: Iterable[str] = _DEFAULT_OAS_DROP,
) -> list[dict]:
    """Attach legacy ``oas_exec/ref/state/cross`` diagnostics to every MIU.

    ``instruction_text`` is accepted for backward compatibility but ignored:
    Module A construction is text-free; pure instruction enters only in Module B.

    ``drop`` may include any subset of {"exec", "func", "ref", "state", "change", "cross"};
    dropped attributes are forced to 0 so the mean shrinks but stays in [0, 1].
    This lets §6.1 run #3..#6 ablate one attribute at a time without changing
    the selector code path.
    """
    if not mius:
        return mius

    del instruction_text
    drop_set = {d.strip().lower() for d in drop if d}
    keywords: tuple[str, ...] = ()
    has_audio_cue = False

    # ---- step-level thresholds for A_exec / A_state ----------------------
    image_action_scores = [
        _safe_float(m.get("action_prior", m.get("det_score", 0.0)))
        for m in mius if str(m.get("modality")) == "image"
    ]
    video_change_scores = [
        _safe_float(m.get("modality_score", m.get("det_score", 0.0)))
        for m in mius if str(m.get("modality")) == "video"
    ]
    audio_onset_scores = [
        _safe_float(m.get("modality_score", m.get("det_score", 0.0)))
        for m in mius if str(m.get("modality")) == "audio"
    ]
    image_action_thr = _median(image_action_scores)
    video_change_thr = _median(video_change_scores)
    audio_onset_thr = _median(audio_onset_scores)

    # video MIUs that overlap an image MIU box -> source of A_change for that image MIU
    image_change_ids: set[str] = set()
    for m in mius:
        if str(m.get("modality")) != "video":
            continue
        linked = str(m.get("linked_component") or "")
        if linked:
            image_change_ids.add(linked)

    # v6 orthogonality fix: A_cross for image MIUs should NOT be a literal
    # alias of A_change.  v4-v5 defined both as ``miu_id in image_change_ids``,
    # which made the two attributes perfectly correlated for image MIUs
    # (rate-cross == rate-change in oas_summary_meta) and produced bit-identical
    # ``drop_cross`` / ``drop_change`` ablation results.  v6 redirects A_cross
    # to genuinely cross-modal evidence: an image MIU is cross-supported when
    # it shares a sync_group with at least one AUDIO MIU.  A_change keeps its
    # video <-> image semantics.  No new hyperparameter is introduced.
    image_audio_cross_ids: set[str] = set()
    audio_sync_groups: set[str] = set()
    for m in mius:
        if str(m.get("modality")) == "audio":
            sg = str(m.get("sync_group") or "")
            if sg:
                audio_sync_groups.add(sg)
    for m in mius:
        if str(m.get("modality")) != "image":
            continue
        sg = str(m.get("sync_group") or "")
        if sg and sg in audio_sync_groups:
            image_audio_cross_ids.add(str(m.get("miu_id") or ""))

    # video <-> image linked pairs (used so video MIUs can borrow A_ref from
    # their anchor image MIU when that image MIU is itself referential).
    referential_image_ids: set[str] = set()
    for m in mius:
        if str(m.get("modality")) != "image":
            continue
        text = str(m.get("text") or "").lower()
        if text and keywords:
            if any(kw in text or text in kw for kw in keywords):
                referential_image_ids.add(str(m.get("miu_id") or ""))

    def _bin(flag: bool, attr: str) -> int:
        if attr in drop_set:
            return 0
        return 1 if flag else 0

    for miu in mius:
        modality = str(miu.get("modality") or "image").lower()
        miu_id = str(miu.get("miu_id") or "")

        # --- A_exec: "does it have executable / actionable substance?" ----
        # For images, avoid the saturated legacy ``A_func`` behavior
        # (detector fired => 1 for almost everything).  A widget must have an
        # actionable class/source signal and be in the step's top half by
        # action_prior.  The top-half threshold is deterministic and adds no
        # method-side hyperparameter.
        func_flag = False
        if modality == "image":
            role = str(miu.get("type") or miu.get("role") or "").lower()
            source = str(miu.get("source") or "").lower()
            actionable_role = role in {
                "icon_button", "button", "icon", "link", "input", "input_field", "text_field", "search_button",
            }
            # Text can be executable when it comes from OCR/text-like UI, but
            # plain panel/container boxes are intentionally excluded.
            text_actionable = role in {"text_like", "text"} and source != "skeleton"
            action_score = _safe_float(miu.get("action_prior", miu.get("det_score", 0.0)))
            func_flag = (actionable_role or text_actionable) and action_score >= image_action_thr
        elif modality == "video":
            func_flag = (
                _safe_float(miu.get("modality_score", miu.get("det_score", 0.0))) >= video_change_thr
            )
        elif modality == "audio":
            func_flag = (
                _safe_float(miu.get("modality_score", miu.get("det_score", 0.0))) >= audio_onset_thr
            )

        # --- A_ref: instruction-referential (fast role/type; OCR text optional) ---
        ref_flag = False
        if modality == "image":
            ref_flag = instruction_role_referential(miu, keywords)
        elif modality == "video":
            linked = str(miu.get("linked_component") or "")
            if linked and linked in referential_image_ids:
                ref_flag = True
        elif modality == "audio":
            ref_flag = bool(has_audio_cue)

        # --- A_cross: "is it cross-modally corroborated?" -----------------
        # v6: orthogonal to A_change by construction.
        #   image MIU -> looks for an AUDIO sync_group match (image<->audio).
        #   video MIU -> sync_group OR linked_component (video<->{img,aud}).
        #   audio MIU -> sync_group (audio<->{img,vid}).
        cross_flag = False
        sync_group = str(miu.get("sync_group") or "")
        if modality == "image":
            cross_flag = miu_id in image_audio_cross_ids
        elif modality == "video":
            cross_flag = bool(sync_group) or bool(str(miu.get("linked_component") or ""))
        elif modality == "audio":
            cross_flag = bool(sync_group)

        # --- A_change: "does it correspond to a step event?" --------------
        change_flag = False
        if modality == "image":
            # Image MIUs are step-event-aligned when an overlapping video MIU
            # peaks here.  (Pure-static screen MIUs without any frame_diff
            # support stay at 0; this is the differentiator.)
            change_flag = miu_id in image_change_ids
        elif modality == "video":
            change_flag = True  # constructive: video MIUs ARE the change events
        elif modality == "audio":
            change_flag = True  # constructive: audio MIUs ARE onset/event windows

        a_func = _bin(func_flag, "func") if "exec" not in drop_set else 0
        a_ref = _bin(ref_flag, "ref")
        a_cross = _bin(cross_flag, "cross")
        a_change = _bin(change_flag, "change") if "state" not in drop_set else 0

        miu["oas_exec"] = a_func
        miu["oas_func"] = a_func
        miu["oas_ref"] = a_ref
        miu["oas_state"] = a_change
        miu["oas_cross"] = a_cross
        miu["oas_change"] = a_change
        # 4-attribute mean — zero weights, exactly the design from
        # communication.md §2 module 1.
        miu["oas_imp"] = 0.25 * float(a_func + a_ref + a_cross + a_change)

    return mius


def oas_summary_meta(mius: list[dict]) -> dict[str, float]:
    """Per-step statistics, used in logs / debug tables (not in selection)."""
    if not mius:
        return {
            "oas_count": 0.0,
            "oas_imp_mean": 0.0,
            "oas_exec_rate": 0.0,
            "oas_func_rate": 0.0,
            "oas_ref_rate": 0.0,
            "oas_state_rate": 0.0,
            "oas_cross_rate": 0.0,
            "oas_change_rate": 0.0,
        }
    n = float(len(mius))
    return {
        "oas_count": n,
        "oas_imp_mean": float(sum(_safe_float(m.get("oas_imp")) for m in mius) / n),
        "oas_exec_rate": float(sum(int(m.get("oas_exec", m.get("oas_func", 0))) for m in mius) / n),
        "oas_func_rate": float(sum(int(m.get("oas_func", m.get("oas_exec", 0))) for m in mius) / n),
        "oas_ref_rate": float(sum(int(m.get("oas_ref", 0)) for m in mius) / n),
        "oas_state_rate": float(sum(int(m.get("oas_state", m.get("oas_change", 0))) for m in mius) / n),
        "oas_cross_rate": float(sum(int(m.get("oas_cross", 0)) for m in mius) / n),
        "oas_change_rate": float(sum(int(m.get("oas_change", m.get("oas_state", 0))) for m in mius) / n),
    }


def parse_oas_drop_set(policy: str) -> frozenset[str]:
    """Read ``policy`` for OAS drop-one ablation tokens."""
    policy = (policy or "").lower()
    drops: set[str] = set()
    aliases = {
        "exec": ("exec", "func"),
        "func": ("func", "exec"),
        "ref": ("ref",),
        "state": ("state", "change"),
        "change": ("change", "state"),
        "cross": ("cross",),
    }
    for attr, names in aliases.items():
        if any(f"drop_{name}" in policy or f"no_{name}" in policy for name in names):
            drops.add(attr)
    return frozenset(drops)
