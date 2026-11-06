#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文基线「Uniform / Random」剪枝 + GUI-Owl-1.5 prompt + 各数据集测评核心。

本文件 **不接入 eval harness**，也不 import ``geopact_vl_hook`` / ``geopact_nap``。
用途是把散落在 hook / NAP / task YAML / process_results 里的内容收成一份可对照的参考：

1. **Uniform pruning** —— 论文表里的 Uniform 行
   （``visual_prune_policy=uniform`` / ``uniform_stride``）。
2. **Random pruning** —— 论文表里的 Random 行
   （``visual_prune_policy=random``，纯 token 随机，不是 MIU 随机）。
3. **GUI-Owl-1.5 测评 prompt** —— OmniGUI 多步 / ScreenSpot·MMBench-L2 单步
   grounding / Mind2Web 三套 system + user 模板。
4. **各数据集测评代码** —— task 入口、消息函数、打分规则（第 5 节）。
   完整容错解析仍在 ``agentcpm_gui/utils.py``（约四千行）；这里只保留
   GUI-Owl 实际走的 ``<tool_call>`` 主路径 + 官方 grounding 坐标抽取。

源文件：

* ``lmms_eval/models/geopact_vl_hook.py`` / ``geopact_nap.py``
* ``lmms_eval/tasks/agentcpm_gui/gui_owl_native_messages.py``
* ``lmms_eval/tasks/gui_owl_cross/{utils,process_results,android_control_eval,mind2web_eval,aitw_action_match}.py``
* ``lmms_eval/tasks/{screenspot_v2,screenspot_pro,mmbench_gui}/utils.py``
* 各 ``task_*_guiowl.yaml`` / ``task_*_qwen3_omni*.yaml``

跑一遍自检::

    python scripts/ref_uniform_random_prune_and_guiowl_prompts.py
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# =============================================================================
# 0. 先把几个容易混的名字钉死
# =============================================================================
#
# visual_prune_ratio
#     是 **keep 比例**，不是 drop 比例。hook 工厂注释写明
#     "Fraction of image tokens to keep (0 < r < 1)"。
#     keep_k = max(1, round(n_img * ratio))。
#     所以 ratio=0.40 表示每帧保留约 40% 视觉 token。
#
# visual_prune_policy=uniform / uniform_stride
#     论文 Uniform 行。栅格顺序等间距抽样。跳过 Module A/B/C（GCC/HES/SBR）。
#     走 dialogue-sticky token 路径（``_is_dialogue_sticky_token_prune``）。
#
# visual_prune_policy=random
#     论文 Random 行。对 [0, n_img) 做一次 shuffle，取前 keep_k。
#     同样跳过 Module A/B/C。sticky 下用 SHA256 派生的确定性 RNG，
#     同一 (episode, image) 复用同一 mask。
#
# oas_random_*
#     **不是** 论文 Random 行。那是 MIU 级随机（先打乱检测框再并 token）。
#     只有 0 个 MIU 时才退回 token-random。不要把两条路径当成同一基线。
#
# GEOPACT_BAM_UNIFORM_A=1
#     **不是** Uniform 行。那是 BAM 选择器里把指令似然 a_j 抹平的消融，
#     仍然走 GeoPACT 选择器，只是去掉 instruction-likelihood 通道。
#
# image_sticky
#     多步对话里，同一截图的 keep mask 按 (episode_key, image_path) 缓存；
#     该帧从 current 滑进 history 时不再重选，只可能走 NAP 的嵌套收缩。
#
# NAP（Nested Anticipative Pruning, GEOPACT_TWO_STAGE=1）
#     当前帧以较大预算 k_cur 进 LLM，滑进 history 时在 **S_cur 内部** 收到
#     k_hist。Uniform 在 S_cur 上等间距再抽样（stride）；Random 取该帧
#     置换的前缀（honest nested-random）。约束：S_hist ⊂ S_cur。


# =============================================================================
# 1. Uniform pruning —— 栅格等间距
# =============================================================================
#
# 视觉 token 按 Qwen 的 raster 顺序编号：先行后列，index ∈ [0, n_img)。
# Uniform 要在这条 1D 链上取 keep_k 个点，并且：
#
#   * 确定性（无 RNG）；
#   * 尽量覆盖整幅图（含两端）；
#   * keep_k == 1 时取正中，避免总是钉在左上角。
#
# 公式：step = (n_img - 1) / (keep_k - 1)，第 i 个点 round(i * step)。
# 用 set 去重：round 碰撞时实际保留数可能略少于 keep_k（n 很小、k 很大时）。
# 这与 hook 行为一致，论文表也按这个实现出数。
#
# ``_hes_select`` 还有一条 candidate 子集路径：若只允许在 cand_list 里选
# （例如 NAP/历史帧已经收缩过），则对 **已排序的 cand_list** 做
# ``cand_list[::step][:keep_k]``，step = len(cand) // keep_k。
# 这比下面的 round-linspace 更粗，只在「候选已经是子集」时触发。
# 主表单帧/首帧走的是 ``_uniform_keep_indices`` 这条。


def keep_k_from_ratio(n_img: int, ratio: float) -> int:
    """``visual_prune_ratio`` → 本帧保留 token 数。

    与 ``geopact_vl_hook`` 里 sticky / rectify 路径一致：
    ``keep_k = max(1, int(round(n_img * ratio)))``，再夹到 ``[1, n_img]``。
    """
    if n_img <= 0:
        return 0
    return min(max(1, int(round(n_img * float(ratio)))), n_img)


def uniform_keep_indices(n_img: int, keep_k: int) -> Set[int]:
    """论文 Uniform：raster 上等间距下标（含两端）。

    源：``geopact_vl_hook._uniform_keep_indices``。

    例子（n=20, k=5）::

        step = 19/4 = 4.75
        i=0..4 → round(0, 4.75, 9.5, 14.25, 19) = {0, 5, 10, 14, 19}

    keep_k == 1 → ``{n_img // 2}``（正中一枚，不是 0）。
    keep_k >= n_img → 全留。
    """
    if n_img <= 0:
        return set()
    keep_k = min(max(1, keep_k), n_img)
    if keep_k >= n_img:
        return set(range(n_img))
    if keep_k == 1:
        return {n_img // 2}
    step = (n_img - 1) / (keep_k - 1)
    return {int(round(i * step)) for i in range(keep_k)}


def uniform_keep_from_candidates(cand_list: Sequence[int], keep_k: int) -> List[int]:
    """Uniform 在已排序候选子集上的粗 stride（``_hes_select`` cand 分支）。

    ``step = max(len(cand) // max(1, keep_k), 1)``，然后 ``cand[::step][:keep_k]``。
    注意这 **不是** round-linspace，NAP stage-2 也不走这条，走下面的
    ``stride_subset``。
    """
    cand = list(cand_list)
    if not cand:
        return []
    keep_k = min(max(1, int(keep_k)), len(cand))
    step = max(len(cand) // max(1, keep_k), 1)
    return cand[::step][:keep_k]


def stride_subset(sorted_items: Sequence[int], k: int) -> List[int]:
    """NAP 嵌套-Uniform：在 **S_cur 内部** 再做一次含端点的等间距。

    源：``geopact_nap._stride_subset``。

    为什么不直接对 S_cur 取 prefix？
        Uniform 的「顺序」就是 raster 坐标，没有 greedy 优先级；
        prefix(S_cur) 会把保留集挤到图的一侧，不是 honest nested-uniform。

    碰撞处理：``round(i * step)`` 撞到已用下标时先向右 bump，再向左 bump，
    保证输出恰好 k 个（在 k <= n 时）。
    """
    items = list(sorted_items)
    n = len(items)
    if k >= n:
        return list(items)
    if k <= 1:
        return [items[n // 2]]
    step = (n - 1) / (k - 1)
    picked: List[int] = []
    used: Set[int] = set()
    for i in range(k):
        j = int(round(i * step))
        while j in used and j + 1 < n:
            j += 1
        while j in used and j - 1 >= 0:
            j -= 1
        used.add(j)
        picked.append(items[j])
    return picked


# =============================================================================
# 2. Random pruning —— token 级无放回抽样
# =============================================================================
#
# 论文 Random 行 = 把全部视觉 token 下标打乱，取前 keep_k。
# **不看** 检测框、指令、注意力。sticky 下必须可复现：
# 同一 episode 里同一张图在 step t 和 step t+1（已变成 history）必须留下
# **同一组** token，否则 RoPE / KV 对不齐。
#
# 种子：
#     key = "\\0".join((episode_key, norm_image_path))
#     seed = int.from_bytes(sha256(key)[:8], "big")
#     rng = random.Random(seed)
#
# （若打开 sticky instruction key，还会把该帧指令的 sha256[:12] 拼进 key；
#  默认评测关着，本文件按默认路径实现。）
#
# NAP 嵌套-Random：
#     stage-1 得到 S_cur 后，用 **同一 sticky rng** 把 S_cur 再 shuffle 一次，
#     存成 order；stage-2 取 order[:k_hist]。
#     这是 S_cur 的均匀随机子集，满足 S_hist ⊂ S_cur。
#     若每步重新对全图 random，history 帧会换一套 token，那就不是 nested。


def sticky_rng(episode_key: str, image_path: str) -> random.Random:
    """Sticky 确定性 RNG。源：``geopact_vl_hook._sticky_rng``。

    只用 SHA256 的前 8 字节当 seed（64-bit），与 hook 一致。
    ``image_path`` 在 hook 里会先做 ``_norm_image_path``；这里假定调用方
    已经传入规范化路径（正斜杠、去掉多余 ``./`` 等）。
    """
    key = "\0".join((episode_key, image_path))
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return random.Random(seed)


def random_keep_indices(
    n_img: int,
    keep_k: int,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """论文 Random：shuffle ``[0, n_img)``，取前 keep_k。

    源：``geopact_vl_hook._hes_token_random``。
    返回 **list**（hook 里再包进 HESResult / set）；顺序是 shuffle 后的
    前缀，NAP 可以把这份顺序直接当 nested-random 的 order。
    """
    keep_k = min(max(1, int(keep_k)), max(0, n_img))
    idxs = list(range(n_img))
    r = rng if rng is not None else random
    r.shuffle(idxs)
    return idxs[:keep_k]


def random_keep_from_candidates(
    cand_list: Sequence[int],
    keep_k: int,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Random 在候选子集上的同款：shuffle cand，取前 keep_k。

    ``_hes_select`` 在 ``candidate_local`` 非空时走这条
    （历史帧 / NAP 收缩后再选）。
    """
    shuffled = list(cand_list)
    if not shuffled:
        return []
    keep_k = min(max(1, int(keep_k)), len(shuffled))
    r = rng if rng is not None else random
    r.shuffle(shuffled)
    return shuffled[:keep_k]


def miu_random_keep_indices(
    miu_token_groups: Sequence[Sequence[int]],
    n_img: int,
    keep_k: int,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """MIU 级随机（``oas_random_*``）—— **不是** 论文 Random 行。

    源：``geopact_vl_hook._hes_miu_random``。
    打乱检测组，按组并 token，直到凑满预算；不够再用全图 token-random 补。
    放在这里只为对照：主表 Random 从未走这条。
    """
    r = rng if rng is not None else random
    groups = list(range(len(miu_token_groups)))
    r.shuffle(groups)
    keep: Set[int] = set()
    for gi in groups:
        keep.update(int(t) for t in miu_token_groups[gi])
        if len(keep) >= keep_k:
            break
    keep_list = list(keep)[:keep_k]
    if len(keep_list) < keep_k:
        remaining = [i for i in range(n_img) if i not in keep]
        r.shuffle(remaining)
        keep_list.extend(remaining[: keep_k - len(keep_list)])
    return keep_list


# =============================================================================
# 3. NAP 嵌套：Uniform / Random 各自怎么从 S_cur 收到 S_hist
# =============================================================================
#
# kind 由 policy 字符串决定（``nap_order_kind``）：
#     "uniform" in policy → "stride"
#     "random"  in policy → "random"
#     其余（BAM / ICKH / …）→ "greedy"（本文件不展开 greedy/strata）
#
# stage-1 ``build_nap_order``：
#     stride → sorted(S_cur)          # raster
#     random → shuffle(sorted(S_cur)) # 用 sticky rng
#
# stage-2 ``contract_nap_order``：
#     stride → stride_subset(sorted(order), k_hist)
#     random → order[:k_hist]


def nap_order_kind(policy: str) -> str:
    """源：``geopact_nap.nap_order_kind``。"""
    p = str(policy or "").lower()
    if "uniform" in p:
        return "stride"
    if "random" in p:
        return "random"
    return "greedy"


def build_nap_order(
    keep_local: Sequence[int],
    kind: str = "greedy",
    rng: Optional[random.Random] = None,
) -> List[int]:
    """源：``geopact_nap.build_nap_order`` 的 stride / random 分支。

    greedy 分支需要 selector 的 pick_order，这里省略；调用 greedy 请去 nap.py。
    """
    keep_set = {int(i) for i in keep_local}
    if kind == "stride":
        return sorted(keep_set)
    if kind == "random":
        out = sorted(keep_set)
        if rng is not None:
            rng.shuffle(out)
        return out
    raise ValueError(f"this extract only implements stride/random, got {kind!r}")


def contract_nap_order_uniform_or_random(order: Sequence[int], kind: str, k_hist: int) -> List[int]:
    """源：``geopact_nap.contract_nap_order`` 的 stride / random 分支。"""
    order_list = [int(x) for x in order]
    if not order_list:
        return []
    k = min(max(1, int(k_hist)), len(order_list))
    if kind == "stride":
        return stride_subset(sorted(order_list), k)
    if kind == "random":
        return order_list[:k]
    raise ValueError(f"this extract only implements stride/random, got {kind!r}")


# =============================================================================
# 4. GUI-Owl-1.5 测评 prompt
# =============================================================================
#
# 模型：mPLUG / 本地 ``GUI-Owl-1.5-{2,4,8}B-Instruct``，包装在
# ``lmms_eval/models/simple/qwen3_vl_gui.py``。
#
# 官方源（X-PLUG/MobileAgent, commit 8cf3966e）：
#   Mobile-Agent-v3.5/mobile_use/utils.py          → 多步 mobile_use
#   Mobile-Agent-v3.5/grounding_and_kb/
#       eval_grounding_benchmarks.py               → 单步 computer_use
#       (`only_two_action_system_prompt`)
#
# 官方默认 ``build_messages(..., history_n=4)``；
# **本仓库评测锁死 history_n=6**（``LOCKED_EVAL_HISTORY_N``）。
# 在线 GUI-Owl 把模型自己的输出追加进 history；OmniGUI 是 step-level
# 离线基准，history 的 assistant 轮由 GT 动作教师强制写成同一
# ``Action: ... <tool_call>`` 格式。
#
# 三套 prompt 对照：
#
# ┌─────────────────────────────┬──────────────────┬─────────────────────────┐
# │ 任务                         │ system 工具       │ user 怎么写              │
# ├─────────────────────────────┼──────────────────┼─────────────────────────┤
# │ OmniGUI / AC / AITW 多步     │ mobile_use 全集   │ 见 instruction 模板；    │
# │                             │ 坐标已改成归一化   │ 滚动窗口 6 张历史截图    │
# │ ScreenSpot / MMBench-GUI L2 │ computer_use      │ [image, 原始 instruction]│
# │ 单步 grounding               │ 仅 left_click /   │ **没有任何包装句**       │
# │                             │ mouse_move        │                         │
# │ Mind2Web                    │ mobile_use 子集   │ 与多步同一套 native      │
# │                             │ click/type/select │ system，动作空间收窄     │
# └─────────────────────────────┴──────────────────┴─────────────────────────┘
#
# 坐标：官方原文写 "The screen's resolution is 1000x1000"（看起来像像素）。
# OmniGUI 计分用归一化 [0, 1000]，所以 native system 只改了三处措辞
# （``_patch_prompt_to_normalized_1000``）。Grounding 官方 prompt **故意
# 保留** "1000x1000" 这句话，这是官方用来引出归一化坐标的 trick，评测
# 未改。另有可选 resized-pixel 变体，默认评测不用。


OFFICIAL_SOURCE_COMMIT = "8cf3966e"
OFFICIAL_DEFAULT_HISTORY_N = 4
LOCKED_EVAL_HISTORY_N = 6  # 评测锁死；build_messages 会忽略传入的其它值

# ----- 4.1 官方原文 SYSTEM_PROMPT（Mobile-Agent-v3.5/mobile_use/utils.py）-----
# 保持不可变，方便和上游 diff。真正喂给 OmniGUI 多步评测的是打过补丁的
# GUI_OWL_NATIVE_SYSTEM_PROMPT。

OFFICIAL_SYSTEM_PROMPT = '''# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is 1000x1000.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: \\"volume_up\\", \\"volume_down\\", \\"power\\", \\"camera\\", \\"clear\\".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `answer`: Terminate the current task and output the answer.
* `interact`: Resolve the blocking window by interacting with the user.
* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "system_button", "open", "wait", "answer", "interact", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=key`, `action=type`, `action=open`, `action=answer`,and `action=interact`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one for Action.
- Do not output anything else outside those two parts.
- If finishing, use action=terminate in the tool call.'''


def patch_prompt_to_normalized_1000(prompt: str) -> str:
    """OmniGUI 多步评测对官方 system 的 **唯一** 改动：像素措辞 → 归一化 [0,1000]。

    三处替换与 ``gui_owl_native_messages._patch_prompt_to_normalized_1000`` 逐字相同。
    工具列表、Action / <tool_call> 格式、terminate 规则一律不动。
    """
    return (
        prompt.replace(
            "* The screen's resolution is 1000x1000.",
            "* Coordinates use a normalized 1000x1000 coordinate space, regardless of the screenshot's real pixel resolution: [0,0] is top-left and [1000,1000] is bottom-right.",
        )
        .replace(
            "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
            "(x, y): normalized coordinates in the 0-1000 screen space; x increases left-to-right and y increases top-to-bottom. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
        )
        .replace(
            "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.",
            "(x, y): normalized end coordinates in the 0-1000 screen space. Required only by `action=swipe`.",
        )
    )


# OmniGUI / AndroidControl / AITW 多步实际使用的 system prompt。
GUI_OWL_NATIVE_SYSTEM_PROMPT = patch_prompt_to_normalized_1000(OFFICIAL_SYSTEM_PROMPT)


# ----- 4.2 多步 user 模板（官方 build_messages 的 instruction 块）-----
#
# 只有 **滚动窗口里的第一张图** 带这段文字；后续历史帧只有 image；
# 当前帧也只有 image。older-than-window 的动作压成 "Step i: …" 摘要，
# 窗口内的动作则以完整 assistant 轮（Action + tool_call）出现。
#
# Instruction 行把日期和任务拼在一起，中间 **没有** 空格/换行：
#     "Instruction: Today's date is: 2026-09-01 Tuesday.{task}"
# 这是官方原文如此，评测未改。
#
# 英文 app（amazon / instagram / …）用 ``instruction_en``，其余用中文
# ``instruction``。见 ``ENGLISH_APPS``。
#
# Previous actions:
#     窗口之外的更早步 → 从 GT assistant 文本抽出 Action 摘要；
#     若当前步 < history_n（还没有溢出窗口）→ 字面量 "None"。

ENGLISH_APPS = {
    "amazon", "instagram", "snapchat", "spotify", "x", "youtube", "tiktok",
    "duolingo", "googletranslate", "imdb", "ted", "vimeo", "redbulltv", "tasty",
}


def official_instruction_prompt(
    instruction: str,
    history_output: List[Dict[str, str]],
    *,
    history_n: int = LOCKED_EVAL_HISTORY_N,
    today: Optional[datetime] = None,
) -> str:
    """源：``gui_owl_native_messages._official_instruction_prompt``。

    ``history_output[i]["output"]`` 是该步教师强制的
    ``Action: ...\\n<tool_call>...</tool_call>``。
    窗口外的步只保留 ``Action:`` 与 ``<tool_call>`` 之间的短句。
    """
    current_step = len(history_output)
    history_start_idx = max(0, current_step - history_n)

    previous_actions = []
    for i in range(history_start_idx):
        text = history_output[i]["output"]
        if "Action:" in text and "<tool_call>" in text:
            text = text.split("Action:")[1].split("<tool_call>")[0].strip()
        previous_actions.append(f"Step {i + 1}: {text}")
    previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

    now = today or datetime.today()
    weekday_names = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]
    date_info = (
        f"Today's date is: {now.strftime('%Y-%m-%d')} {weekday_names[now.weekday()]}."
    )
    return (
        "Please generate the next move according to the UI screenshot, "
        "instruction and previous actions.\n\n"
        f"Instruction: {date_info}{instruction}\n\n"
        f"Previous actions:\n{previous_actions_str}"
    )


def format_native_assistant_turn(*, action_summary: str, tool_call: Dict[str, Any]) -> str:
    """GT → 教师强制 assistant 轮。源：``format_native_assistant_turn``。

    模型必须按这个骨架解码：一行 Action，然后一个 ``<tool_call>`` JSON。
    ``tool_call`` 形如 ``{"name": "mobile_use", "arguments": {"action": "click", ...}}``。
    """
    return (
        f"Action: {action_summary}\n"
        f"<tool_call>\n{json.dumps(tool_call, ensure_ascii=False)}\n</tool_call>"
    )


def sketch_omnigui_messages(
    *,
    instruction: str,
    current_image: str,
    history: List[Dict[str, str]],
    history_n: int = LOCKED_EVAL_HISTORY_N,
    system_prompt: str = GUI_OWL_NATIVE_SYSTEM_PROMPT,
) -> List[Dict[str, Any]]:
    """多步对话布局草图（与 ``build_messages_compatible`` 同构）。

    ``history`` 每项 ``{"image": path, "output": assistant_text}``，按时间升序。
    实际评测会把 ``history_n`` 强制改回 6，这里同样锁死。
    """
    history_n = LOCKED_EVAL_HISTORY_N
    instr = official_instruction_prompt(instruction, history, history_n=history_n)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
    ]
    window = history[-history_n:] if history else []
    if window:
        for idx, item in enumerate(window):
            if idx == 0:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instr},
                        {"type": "image", "url": item["image"]},
                    ],
                })
            else:
                messages.append({
                    "role": "user",
                    "content": [{"type": "image", "url": item["image"]}],
                })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": item["output"]}],
            })
        messages.append({
            "role": "user",
            "content": [{"type": "image", "url": current_image}],
        })
    else:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": instr},
                {"type": "image", "url": current_image},
            ],
        })
    return messages


# ----- 4.3 单步 grounding（ScreenSpot / MMBench-GUI L2）-----
#
# L2 不是选择题：截图 + 指令 → 点击坐标，落在 GT bbox 内算命中。
# user 内容 = [image, raw instruction]，**不要** 再包
# "Please generate the next move..."。
# UITARS_NATIVE_GROUNDING=1 时改走 UI-TARS 模板，GUI-Owl 臂不要开这个开关。

GUI_OWL_OFFICIAL_GROUNDING_SYSTEM_PROMPT = '''# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.
* The screen's resolution is 1000x1000.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
* don't use any other computer use tool like type, key, scroll, left_click_drag and so on.
* you can only use the left_click and mouse_move action to interact with the computer. if you can't find the element, you should terminate the task and report the failure.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button with coordinate (x, y) pixel coordinate on the screen.", "enum": ["mouse_move", "left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click`.", "type": "array"}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>'''


def sketch_grounding_messages(image_path: str, instruction: str) -> List[Dict[str, Any]]:
    """源：``gui_owl_grounding_doc_to_messages``（非 UI-TARS 分支）。"""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": GUI_OWL_OFFICIAL_GROUNDING_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": instruction},
            ],
        },
    ]


# ----- 4.4 Mind2Web：同一 mobile_use 表面，动作空间收到 click/type/select -----

GUI_OWL_MIND2WEB_SYSTEM_PROMPT = '''# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a mouse and keyboard to interact with a web page, and take screenshots.
* This is an interface to a web page shown in the screenshot. You can click elements, type into inputs, and select options from dropdowns.
* Coordinates use a normalized 1000x1000 coordinate space, regardless of the screenshot's real pixel resolution: [0,0] is top-left and [1000,1000] is bottom-right.
* Make sure to click elements at the center of the target element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `click`: Click the element at coordinate (x, y).
* `type`: Type the specified text into the input element at coordinate (x, y).
* `select`: Select the specified option text from the dropdown element at coordinate (x, y).", "enum": ["click", "type", "select"], "type": "string"}, "coordinate": {"description": "(x, y): normalized coordinates in the 0-1000 screen space; x increases left-to-right and y increases top-to-bottom. Required by every action.", "type": "array"}, "text": {"description": "The text to type or the option to select. Required only by `action=type` and `action=select`.", "type": "string"}}, "required": ["action", "coordinate"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one for Action.
- Do not output anything else outside those two parts.'''


# MMBench-GUI L1 才是选择题；L2 用上面的 grounding prompt。
# 放在这里避免和 L2 搞混。
MMBENCH_L1_QA_SYSTEM_PROMPT = "You are a helpful assistant."
MMBENCH_L1_USER_SUFFIX = "Answer with the option letter only (A/B/C/D/E)."


# =============================================================================
# 5. 各数据集测评代码（GUI-Owl-1.5 / lmms_eval）
# =============================================================================
#
# 论文主表分两摊：
#   单步 grounding : ScreenSpot-v2 / ScreenSpot-Pro / MMBench-GUI L2
#   多步 agent     : OmniGUI / Mind2Web 三 split / AndroidControl
# 另外仓库里还有 MMBench L1（选择题）、AITW、ScreenSpot-v2-mobile。
#
# 共同约定：
#   * generate_until，temperature=0
#   * GUI-Owl 输出 ``Action: ...`` + ``<tool_call>{json}</tool_call>``
#   * 坐标默认归一化 [0, 1000]；grounding 官方用「1000x1000 分辨率」trick
#   * 多步 history 由 GT 教师强制，不是模型自己的上一步输出
#
# 下面 DATASETS 是路由表；再往后是真正可跑的解析 / 打分函数。


DATASETS: List[Dict[str, Any]] = [
    {
        "name": "OmniGUI",
        "family": "multi-step",
        "task": "task_v2_guiowl_native_mt4  /  task_guiowl_omnigui_native_mt4_8b",
        "data": "data/merged_sorted.json",
        "messages": "gui_owl_native_multiturn_doc_to_messages  (history_n=6)",
        "prompt": "GUI_OWL_NATIVE_SYSTEM_PROMPT  (mobile_use, 坐标已补丁)",
        "score": "TM / EM / SR / GP   EM=逐步 exact_match；SR=整 episode 全中；GP=失败前前缀",
        "yaml": "lmms_eval/tasks/agentcpm_gui/task_v2_guiowl_native_mt4.yaml",
        "paper": "主表多步；W58 起 OmniGUI 列报的是 Step SR = EM，不是 GP",
    },
    {
        "name": "Mind2Web",
        "family": "multi-step",
        "task": "task_mind2web_qwen3_omni_{domain,website,task}",
        "data": "data/mind2web/merged_sorted_{domain,website,task}.json",
        "messages": "gui_owl_mind2web_dialogue_doc_to_messages  (history_n=9)",
        "prompt": "GUI_OWL_MIND2WEB_SYSTEM_PROMPT  (click/type/select)",
        "score": "Ele.Acc + Op.F1 + Step SR(=Ele.Acc 且 Op.F1==1)；主表 pooled Step SR",
        "yaml": "lmms_eval/tasks/gui_owl_cross/task_mind2web_qwen3_omni_*.yaml",
        "paper": "三官方 split 按步数加权池化",
    },
    {
        "name": "AndroidControl",
        "family": "multi-step",
        "task": "task_android_control_qwen3_omni  (对话) / task_android_control_high_guiowl (单步)",
        "data": "data/android_control/merged_sorted.json  或  eval_steps.json",
        "messages": "gui_owl_android_control_dialogue_doc_to_messages  (history_n=6)",
        "prompt": "GUI_OWL_NATIVE_SYSTEM_PROMPT",
        "score": "type 对齐后再看坐标 L2<=0.04（归一化屏）/ 文本包含 / 方向相等",
        "yaml": "lmms_eval/tasks/gui_owl_cross/task_android_control_*.yaml",
        "paper": "主表多步；sticky 覆盖实验是 supporting evidence",
    },
    {
        "name": "ScreenSpot-v2",
        "family": "grounding",
        "task": "task_screenspot_v2_guiowl",
        "data": "data/screenspot_v2/eval_index.json",
        "messages": "gui_owl_grounding_doc_to_messages  (user=[image, raw instruction])",
        "prompt": "GUI_OWL_OFFICIAL_GROUNDING_SYSTEM_PROMPT  (computer_use)",
        "score": "center acc：预测点落在 GT bbox；bbox 永远是 [x,y,w,h]",
        "yaml": "lmms_eval/tasks/screenspot_v2/task_screenspot_v2_guiowl.yaml",
        "paper": "主表单步；另报 mobile/web/desktop macro",
    },
    {
        "name": "ScreenSpot-Pro",
        "family": "grounding",
        "task": "task_screenspot_pro_guiowl",
        "data": "data/screenspot_pro/eval_index.json",
        "messages": "同上 grounding",
        "prompt": "同上 computer_use",
        "score": "center acc；bbox 默认 xyxy，若 x2<=x1 则当 [x,y,w,h]",
        "yaml": "lmms_eval/tasks/screenspot_pro/task_screenspot_pro_guiowl.yaml",
        "paper": "主表单步，高分辨率/图标密界面",
    },
    {
        "name": "MMBench-GUI L2",
        "family": "grounding",
        "task": "task_mmbench_gui_l2_guiowl",
        "data": "data/mmbench_gui/l2_eval_index.json",
        "messages": "gui_owl_grounding_doc_to_messages",
        "prompt": "同上 computer_use",
        "score": "point-in-bbox；标注是 xyxy，<=1.5 时先乘 image_size",
        "yaml": "lmms_eval/tasks/mmbench_gui/task_mmbench_gui_l2_guiowl.yaml",
        "paper": "主表单步。L2 不是选择题，是 click grounding",
    },
    {
        "name": "MMBench-GUI L1",
        "family": "mcq",
        "task": "task_mmbench_gui_l1_guiowl",
        "data": "data/mmbench_gui/l1_eval_index.json",
        "messages": "mmbench_gui_l1_doc_to_messages",
        "prompt": "You are a helpful assistant. + Answer with the option letter only",
        "score": "抽 A-E；overall = 按 1/n_options 加权（论文 Eq.4）",
        "yaml": "lmms_eval/tasks/mmbench_gui/task_mmbench_gui_l1_guiowl.yaml",
        "paper": "内容理解 MCQ，不要和 L2 数字混用",
    },
    {
        "name": "AITW",
        "family": "multi-step-offline",
        "task": "task_aitw_guiowl_step",
        "data": "data/aitw/steps/standard_test_steps.json",
        "messages": "gui_owl_aitw_doc_to_messages  (无历史，单步 BC)",
        "prompt": "GUI_OWL_NATIVE_SYSTEM_PROMPT",
        "score": "Google AITW action-matching：tap 14% 屏、swipe 比主轴",
        "yaml": "lmms_eval/tasks/gui_owl_cross/task_aitw_guiowl_step.yaml",
        "paper": "交叉基准，不进主表",
    },
    {
        "name": "ScreenSpot-v2-mobile",
        "family": "grounding",
        "task": "task_screenspot_v2_mobile_guiowl",
        "data": "data/screenspot_v2_mobile/eval_index.json",
        "messages": "grounding 同款",
        "prompt": "computer_use",
        "score": "与 SSv2 相同，仅 mobile 子集",
        "yaml": "lmms_eval/tasks/screenspot_v2_mobile/task_screenspot_v2_mobile_guiowl.yaml",
        "paper": "子集消融，不是主表 SSv2 全量",
    },
]


# ----- 5.1 共享：从 GUI-Owl 输出里抽出 tool_call / 点击坐标 -----
#
# 官方 grounding 抽取（eval_grounding_benchmarks.py）：
#   先找所有 ``(x, y)``，没有再找 ``[x, y]``，取 **最后一处**；
#   什么都没有则 (0, 0)。返回值已经在 [0, 1000] 空间。
# 漏解析会变成点 (0,0)，几乎必 miss —— 不要改成「解析失败就跳过」。

_OFFICIAL_PAREN_RE = re.compile(r"\((\d+),\s*(\d+)\)")
_OFFICIAL_BRACKET_RE = re.compile(r"\[(\d+),\s*(\d+)\]")

# OmniGUI 动作编号（``OmniActionType``）。GUI-Owl mobile_use 动词映射见
# ``tool_call_to_omni``。
OMNI_TAP, OMNI_LONG, OMNI_TYPE = 0, 2, 7
OMNI_SWIPE_UP, OMNI_SWIPE_DOWN, OMNI_SWIPE_LEFT, OMNI_SWIPE_RIGHT = 3, 4, 5, 6
OMNI_BACK, OMNI_HOME, OMNI_DONE, OMNI_FAIL, OMNI_WAIT = 8, 9, 10, 11, -1


def parse_click_point_official(output: str) -> Tuple[float, float]:
    """源：``gui_owl_cross.utils.parse_click_point_official``。"""
    text = output or ""
    matches = _OFFICIAL_PAREN_RE.findall(text)
    if not matches:
        matches = _OFFICIAL_BRACKET_RE.findall(text)
    if not matches:
        return 0.0, 0.0
    x, y = matches[-1]
    return float(x), float(y)


def point_in_bbox_1000(
    point_1000: Tuple[float, float],
    bbox_pixels: Sequence[float],
    width: int,
    height: int,
) -> bool:
    """官方 grounding 计分：GT 像素框先映到 1000×1000（左上 floor、右下 ceil）。

    源：``gui_owl_cross.utils.point_in_bbox_1000``。
    若传入的是 ``x2<=x1`` 的 [x, y, w, h]，这里同样转成角点。
    """
    if not bbox_pixels or len(bbox_pixels) < 4 or width <= 0 or height <= 0:
        return False
    x1, y1, x2, y2 = (float(v) for v in bbox_pixels[:4])
    if x2 <= x1 or y2 <= y1:
        x2, y2 = x1 + float(bbox_pixels[2]), y1 + float(bbox_pixels[3])
    gx1 = math.floor(x1 * 1000.0 / width)
    gy1 = math.floor(y1 * 1000.0 / height)
    gx2 = math.ceil(x2 * 1000.0 / width)
    gy2 = math.ceil(y2 * 1000.0 / height)
    px, py = point_1000
    return gx1 <= px <= gx2 and gy1 <= py <= gy2


def extract_tool_call_args(pred_text: str) -> Dict[str, Any]:
    """抽出 ``<tool_call>...</tool_call>`` 里的 ``arguments`` 字典。

    GUI-Owl 主路径。完整 ``parse_json_output`` 还会吃 markdown / 裸 JSON /
    DSL；漏掉那些格式时请回 ``agentcpm_gui/utils.py``。
    """
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", pred_text or "", re.S)
    if not m:
        return {}
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}
    args = data.get("arguments") or data.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return args if isinstance(args, dict) else {}


def tool_call_to_omni(pred_text: str) -> Dict[str, Any]:
    """``mobile_use`` / ``computer_use`` → Omni 动作字典。

    源：``agentcpm_gui.utils._convert_tool_call_to_action`` 的主分支
    （click / long_press / type / wait / swipe / system_button / terminate /
    left_click / mouse_move）。grounding 的 left_click 也映射成 TAP=0。
    """
    args = extract_tool_call_args(pred_text)
    action = str(args.get("action") or "").lower().strip()
    if not action:
        return {}

    simple = {
        "click": OMNI_TAP,
        "left_click": OMNI_TAP,
        "mouse_move": OMNI_TAP,
        "double_click": 1,
        "long_press": OMNI_LONG,
        "type": OMNI_TYPE,
        "wait": OMNI_WAIT,
    }
    out: Dict[str, Any] = {}
    if action in simple:
        out["action_type"] = simple[action]
        coord = args.get("coordinate")
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            out["coordinate"] = [int(float(coord[0])), int(float(coord[1]))]
        if action == "type" and args.get("text"):
            out["text"] = str(args["text"])
        return out

    if action == "swipe":
        c1 = args.get("coordinate") or args.get("start_coordinate")
        c2 = args.get("coordinate2") or args.get("end_coordinate")
        if not (isinstance(c1, (list, tuple)) and isinstance(c2, (list, tuple))):
            return {}
        x1, y1 = int(float(c1[0])), int(float(c1[1]))
        x2, y2 = int(float(c2[0])), int(float(c2[1]))
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) > abs(dy):
            at = OMNI_SWIPE_RIGHT if dx > 0 else OMNI_SWIPE_LEFT
        else:
            at = OMNI_SWIPE_DOWN if dy > 0 else OMNI_SWIPE_UP
        return {
            "action_type": at,
            "coordinate": [x1, y1],
            "start_coordinate": [x1, y1],
            "end_coordinate": [x2, y2],
        }

    if action == "system_button":
        button = str(args.get("button") or "").lower()
        if button in ("home", "home_screen"):
            return {"action_type": OMNI_HOME}
        return {"action_type": OMNI_BACK}

    if action == "terminate":
        status = str(args.get("status") or "success").lower()
        return {"action_type": OMNI_DONE if status in ("success", "complete", "done") else OMNI_FAIL}

    if action in ("open",):
        return {"action_type": "open_app", "app_name": str(args.get("text") or "")}

    return {}


def wrap_tool_call(arguments: Dict[str, Any], *, name: str = "mobile_use") -> str:
    """构造一条评测用的模型输出，方便下面的打分自检。"""
    payload = {"name": name, "arguments": arguments}
    return (
        "Action: demo\n"
        f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"
    )


# ----- 5.2 ScreenSpot / MMBench-L2：point-in-bbox -----


def screenspot_v2_pixel_bbox(bbox: Sequence[float], width: int, height: int) -> List[float]:
    """SSv2 标注 **永远** 是 [x, y, w, h]（OS-Copilot）。不要做 xyxy 启发式。"""
    x, y, w, h = (float(v) for v in bbox[:4])
    if max(x, y, w, h) <= 1.5:
        return [x * width, y * height, (x + w) * width, (y + h) * height]
    return [x, y, x + w, y + h]


def score_screenspot_v2(pred_text: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    """源：``screenspot_v2.utils.screenspot_v2_process_result``。"""
    width = int((doc.get("img_size") or [0, 0])[0] or doc.get("width") or 0)
    height = int((doc.get("img_size") or [0, 0])[1] or doc.get("height") or 0)
    point = parse_click_point_official(pred_text)
    bbox = screenspot_v2_pixel_bbox(doc.get("bbox") or doc.get("gt_bbox") or [0, 0, 0, 0], width, height)
    hit = point_in_bbox_1000(point, bbox, width, height)
    return {"hit": hit, "point": point, "bbox_px": bbox, "domain": doc.get("domain", "")}


def score_screenspot_pro(pred_text: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    """源：``screenspot_pro.utils.screenspot_pro_process_result``。bbox 默认 xyxy。"""
    width = int((doc.get("img_size") or [0, 0])[0] or doc.get("width") or 1920)
    height = int((doc.get("img_size") or [0, 0])[1] or doc.get("height") or 1080)
    raw = list(doc.get("bbox") or doc.get("gt_bbox") or [0, 0, 0, 0])
    x1, y1, x2, y2 = (float(v) for v in raw[:4])
    if x2 <= x1 or y2 <= y1:
        x2, y2 = x1 + float(raw[2]), y1 + float(raw[3])
    point = parse_click_point_official(pred_text)
    hit = point_in_bbox_1000(point, [x1, y1, x2, y2], width, height)
    return {"hit": hit, "point": point, "bbox_px": [x1, y1, x2, y2]}


def score_mmbench_l2(pred_text: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    """源：``mmbench_gui.utils.mmbench_gui_l2_process_result``。xyxy，可归一化。"""
    size = doc.get("image_size") or [1920, 1080]
    width, height = int(size[0] or 1920), int(size[1] or 1080)
    bbox = list(doc.get("bbox") or [0, 0, 0, 0])
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    if max(x1, y1, x2, y2) <= 1.5:
        pixel = [x1 * width, y1 * height, x2 * width, y2 * height]
    else:
        pixel = [x1, y1, x2, y2]
    point = parse_click_point_official(pred_text)
    hit = point_in_bbox_1000(point, pixel, width, height)
    return {"hit": hit, "point": point, "grounding_type": doc.get("grounding_type", "")}


def aggregate_hit_rate(rows: Sequence[Dict[str, Any]], key: str = "hit") -> float:
    if not rows:
        return 0.0
    return sum(1.0 if r.get(key) else 0.0 for r in rows) / len(rows)


# ----- 5.3 MMBench-GUI L1：选项字母 + 1/n_options 加权 -----


def parse_option_letter(text: str) -> str:
    """源：``mmbench_gui.utils._parse_option_letter``。"""
    if not text:
        return ""
    m = re.search(r"\b([A-E])\b", text.upper())
    if m:
        return m.group(1)
    m2 = re.search(r"answer[\"']?\s*[:=]\s*[\"']?([A-E])", text, re.I)
    return m2.group(1).upper() if m2 else ""


def score_mmbench_l1(pred_text: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    pred = parse_option_letter(pred_text)
    gt = str(doc.get("answer", "")).strip().upper()
    n_opts = max(int(len(doc.get("options") or {}) or 4), 1)
    return {
        "correct": bool(pred and gt and pred == gt),
        "pred": pred,
        "gt": gt,
        "n_options": n_opts,
        "platform": doc.get("platform", ""),
    }


def aggregate_mmbench_l1_weighted(rows: Sequence[Dict[str, Any]]) -> float:
    """论文 Eq.4：每题权重 α = 1/num_options。"""
    total_w = total_s = 0.0
    for r in rows:
        w = 1.0 / max(int(r.get("n_options", 4)), 1)
        total_w += w
        total_s += w * (1.0 if r.get("correct") else 0.0)
    return total_s / total_w if total_w else 0.0


# ----- 5.4 OmniGUI：TM / EM / SR / GP -----
#
# 逐步：
#   type_match  = 预测 action_type == GT
#   exact_match = 类型对 **并且**
#       TAP/LONG : 点在 GT 框内（有框）或到中心距离 <= 140（[0,1000] 空间）
#       SWIPE    : 起终点都落在对应框 / 点阈值内（简化版：类型对即可若无坐标）
#       TYPE     : 文本互相包含（大小写不敏感）
#       BACK/HOME/DONE/FAIL/WAIT : 类型对即 EM
# episode：
#   SR = 该 episode 每一步都 EM
#   GP = 从第一步起连续 EM 的步数 / 总步数，再对 episode 取平均
#
# 论文 W58 之后主表 OmniGUI 列报的是 **Step SR = 逐步 EM**，不是 GP。


OMNI_CLICK_RADIUS_1000 = 140.0  # evaluate_coordinate_match 的点阈值


def _to_1000(xy: Sequence[float], width: int, height: int) -> Tuple[float, float]:
    x, y = float(xy[0]), float(xy[1])
    if max(abs(x), abs(y)) <= 1.5:
        return x * 1000.0, y * 1000.0
    if max(abs(x), abs(y)) > 1000.0 and width > 0 and height > 0:
        return x * 1000.0 / width, y * 1000.0 / height
    return x, y


def _point_in_rect_1000(pred_xy: Sequence[float], rect_px: Sequence[float], width: int, height: int) -> bool:
    px, py = _to_1000(pred_xy, width, height)
    x1, y1, x2, y2 = (float(v) for v in rect_px[:4])
    a = _to_1000([x1, y1], width, height)
    b = _to_1000([x2, y2], width, height)
    lo_x, hi_x = min(a[0], b[0]), max(a[0], b[0])
    lo_y, hi_y = min(a[1], b[1]), max(a[1], b[1])
    return lo_x <= px <= hi_x and lo_y <= py <= hi_y


def score_omnigui_step(pred_text: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    """逐步 OmniGUI 打分（GUI-Owl ``<tool_call>`` 主路径）。

    完整版 ``evaluate_omni_step`` 还处理像素/归一化自动探测、swipe 双框、
    以及非 tool_call 的容错解析。源：``agentcpm_gui.utils.evaluate_omni_step``。
    """
    pred = tool_call_to_omni(pred_text)
    gt_type = doc.get("result_action_type", doc.get("gt_action_type"))
    width = int(doc.get("image_width") or doc.get("width") or 1080)
    height = int(doc.get("image_height") or doc.get("height") or 2400)
    hit_format = bool(pred)
    type_match = bool(pred) and gt_type is not None and pred.get("action_type") == gt_type
    exact = False
    if type_match:
        at = pred["action_type"]
        if at in (OMNI_TAP, 1, OMNI_LONG):
            coord = pred.get("coordinate")
            rect = doc.get("gt_bbox") or doc.get("bbox")
            center = doc.get("gt_coord") or doc.get("result_touch_yx")
            if coord and rect and len(rect) >= 4:
                exact = _point_in_rect_1000(coord, rect, width, height)
            elif coord and center and len(center) >= 2:
                p = _to_1000(coord, width, height)
                c = _to_1000(center, width, height)
                dist = math.hypot(p[0] - c[0], p[1] - c[1])
                exact = dist <= OMNI_CLICK_RADIUS_1000
        elif at in (OMNI_SWIPE_UP, OMNI_SWIPE_DOWN, OMNI_SWIPE_LEFT, OMNI_SWIPE_RIGHT):
            # 完整实现比两端坐标；无框时类型对即过（与「无参动作」同类）。
            start_box = doc.get("gt_start_bbox")
            if start_box and pred.get("start_coordinate"):
                exact = _point_in_rect_1000(pred["start_coordinate"], start_box, width, height)
            else:
                exact = True
        elif at == OMNI_TYPE:
            pt = str(pred.get("text") or "").lower().strip()
            gt = str(doc.get("result_action_text") or doc.get("gt_text") or "").lower().strip()
            exact = bool(pt and gt and (pt in gt or gt in pt))
        elif at in (OMNI_BACK, OMNI_HOME, OMNI_DONE, OMNI_FAIL, OMNI_WAIT):
            exact = True
    return {
        "subset": doc.get("subset", "unknown"),
        "episode_id": doc.get("episode_id", "unknown"),
        "step_id": doc.get("step_index", doc.get("step_id", 0)),
        "type_match": type_match,
        "exact_match": exact,
        "hit_format": hit_format,
        "pred_action_type": pred.get("action_type") if pred else None,
        "gt_action_type": gt_type,
    }


def aggregate_type_match(rows: Sequence[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1.0 if r.get("type_match") else 0.0 for r in rows) / len(rows)


def aggregate_exact_match(rows: Sequence[Dict[str, Any]]) -> float:
    """逐步 EM = 论文 OmniGUI 列的 Step SR。"""
    if not rows:
        return 0.0
    return sum(1.0 if r.get("exact_match") else 0.0 for r in rows) / len(rows)


def _episode_key(row: Dict[str, Any]) -> str:
    return f"{row.get('subset', 'unknown')}-{row.get('episode_id', 'unknown')}"


def aggregate_success_rate(rows: Sequence[Dict[str, Any]]) -> float:
    """整条 episode 每步都 EM 才算成功。"""
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[_episode_key(r)].append(dict(r))
    if not by:
        return 0.0
    wins = 0
    for ep in by.values():
        ep.sort(key=lambda x: int(x.get("step_id", 0)))
        wins += int(all(x.get("exact_match") for x in ep))
    return wins / len(by)


def aggregate_goal_progress(rows: Sequence[Dict[str, Any]]) -> float:
    """失败前连续 EM 前缀 / 步数，再对 episode 平均。"""
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[_episode_key(r)].append(dict(r))
    if not by:
        return 0.0
    acc = 0.0
    for ep in by.values():
        ep.sort(key=lambda x: int(x.get("step_id", 0)))
        n = 0
        for x in ep:
            if x.get("exact_match"):
                n += 1
            else:
                break
        acc += n / len(ep)
    return acc / len(by)


# ----- 5.5 AndroidControl：归一化 L2 ≤ 0.04 -----
#
# 点击阈值 0.04 是 **归一化屏坐标** 的欧氏距离（不是 1000 空间、也不是 AITW 的 0.14）。
# scroll 只比方向；type / open_app 用包含或相似度 ≥ 0.8。

TAP_DISTANCE_THRESHOLD_AC = 0.04


def _norm_01(val: float, axis: int) -> float:
    v = float(val)
    return v / max(axis, 1) if v > 1.5 else v


def _text_ok(pred: str, gt: str) -> bool:
    p, g = pred.lower().strip(), gt.lower().strip()
    if not p or not g:
        return False
    if p in g or g in p:
        return True
    # stdlib 近似 Levenshtein.ratio
    from difflib import SequenceMatcher
    return SequenceMatcher(None, p, g).ratio() >= 0.8


def guiowl_pred_to_ac(pred_text: str) -> Dict[str, Any]:
    """源：``android_control_eval.guiowl_pred_to_ac_action``。"""
    pred = tool_call_to_omni(pred_text)
    if pred.get("action_type") == "open_app":
        return {"action_type": "open_app", "app_name": pred.get("app_name", "")}
    at = pred.get("action_type")
    swipe_dir = {
        OMNI_SWIPE_UP: "up",
        OMNI_SWIPE_DOWN: "down",
        OMNI_SWIPE_LEFT: "left",
        OMNI_SWIPE_RIGHT: "right",
    }
    if at in swipe_dir:
        return {"action_type": "scroll", "direction": swipe_dir[at]}
    mapping = {
        OMNI_TAP: "click",
        OMNI_LONG: "long_press",
        OMNI_TYPE: "type",
        OMNI_BACK: "navigate_back",
        OMNI_HOME: "navigate_home",
        OMNI_WAIT: "wait",
    }
    if at not in mapping:
        return {}
    out: Dict[str, Any] = {"action_type": mapping[at]}
    coord = pred.get("coordinate")
    if coord:
        scale = 1000.0 if max(float(coord[0]), float(coord[1])) > 1.5 else 1.0
        out["x"] = float(coord[0]) / scale
        out["y"] = float(coord[1]) / scale
    if at == OMNI_TYPE:
        out["text"] = str(pred.get("text") or "")
    return out


def score_android_control_step(pred_text: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    """源：``android_control_eval.score_android_control_step`` + ``ac_actions_match``。"""
    gt = doc.get("action") or doc.get("gt_action") or {}
    if isinstance(gt, str):
        gt = {}
    pred = guiowl_pred_to_ac(pred_text)
    size = doc.get("img_size") or [1080, 2400]
    w, h = int(size[0] or 1080), int(size[1] or 2400)
    p_type = str(pred.get("action_type") or "").lower()
    g_type = str(gt.get("action_type") or "").lower()
    aliases = {"tap": "click", "input_text": "type", "press_back": "navigate_back", "press_home": "navigate_home"}
    p_type, g_type = aliases.get(p_type, p_type), aliases.get(g_type, g_type)
    type_match = bool(p_type) and p_type == g_type
    exact = False
    if type_match:
        if g_type in ("click", "long_press"):
            if all(k in pred and k in gt for k in ("x", "y")):
                dist = math.hypot(
                    _norm_01(pred["x"], w) - _norm_01(gt["x"], w),
                    _norm_01(pred["y"], h) - _norm_01(gt["y"], h),
                )
                exact = dist <= TAP_DISTANCE_THRESHOLD_AC
        elif g_type == "scroll":
            exact = str(pred.get("direction", "")).lower() == str(gt.get("direction", "")).lower()
        elif g_type == "type":
            exact = _text_ok(str(pred.get("text", "")), str(gt.get("text", "")))
        elif g_type == "open_app":
            exact = _text_ok(str(pred.get("app_name", pred.get("text", ""))), str(gt.get("app_name", "")))
        elif g_type in ("navigate_back", "navigate_home", "wait"):
            exact = True
    return {"type_match": type_match, "exact_match": exact, "pred_action_type": p_type, "gt_action_type": g_type}


# ----- 5.6 Mind2Web：Ele.Acc / Op.F1 / Step SR -----
#
# Ele.Acc : 预测点（[0,1000] → 像素）落在 GT bbox_px 内
# Op.F1   : ``CLICK`` / ``TYPE value`` / ``SELECT value`` 的 token F1
# Step SR : Ele.Acc 且 Op.F1 == 1（主表报这个；三 split 按步数池化）


def _token_f1(pred: str, gt: str) -> float:
    pt, gtok = pred.lower().split(), gt.lower().split()
    if not pt or not gtok:
        return 1.0 if pt == gtok else 0.0
    bag: Dict[str, int] = {}
    for t in pt:
        bag[t] = bag.get(t, 0) + 1
    overlap = 0
    for t in gtok:
        if bag.get(t, 0) > 0:
            overlap += 1
            bag[t] -= 1
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(pt), overlap / len(gtok)
    return 2 * prec * rec / (prec + rec)


def score_mind2web_step(pred_text: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    """源：``mind2web_eval.score_mind2web_step``。"""
    gt = doc.get("gt_action") or doc.get("action") or {}
    args = extract_tool_call_args(pred_text)
    pred_type = str(args.get("action") or "").lower()
    if pred_type not in ("click", "type", "select"):
        pred_type = ""
    coord = args.get("coordinate") if isinstance(args.get("coordinate"), (list, tuple)) else None
    pred_text_val = str(args.get("text") or "")
    size = doc.get("img_size") or [1280, 720]
    w, h = int(size[0] or 1280), int(size[1] or 720)
    bbox = gt.get("bbox_px") or []
    ele_hit = False
    if coord and len(coord) >= 2 and len(bbox) >= 4:
        x = float(coord[0]) / 1000.0 * max(w, 1)
        y = float(coord[1]) / 1000.0 * max(h, 1)
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        ele_hit = x1 <= x <= x2 and y1 <= y <= y2
    gt_type = str(gt.get("action_type") or "").lower()
    pred_op = f"{pred_type.upper()} {pred_text_val}".strip()
    gt_op = f"{gt_type.upper()} {gt.get('text') or ''}".strip()
    op_f1 = _token_f1(pred_op, gt_op) if pred_type else 0.0
    step_sr = bool(ele_hit and op_f1 >= 1.0 - 1e-9)
    return {
        "ele_acc": ele_hit,
        "op_f1": op_f1,
        "step_sr": step_sr,
        "type_match": bool(pred_type) and pred_type == gt_type,
        "exact_match": step_sr,
    }


def aggregate_op_f1(rows: Sequence[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(float(r.get("op_f1") or 0.0) for r in rows) / len(rows)


# ----- 5.7 AITW：Google action-matching（tap 14% 屏） -----

AITW_TAP_FRAC = 0.14
AITW_DUAL_POINT_EQUAL = 0.04


def score_aitw_step(pred_text: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    """源：``aitw_action_match.aitw_actions_match``（坐标已是像素）。"""
    pred = tool_call_to_omni(pred_text)
    gt = doc.get("gt_action") or doc.get("action") or {}
    screen = (
        int(doc.get("screen_width") or 1080),
        int(doc.get("screen_height") or 2400),
    )
    p_touch = pred.get("start_coordinate") or pred.get("coordinate")
    p_lift = pred.get("end_coordinate") or pred.get("coordinate")
    g_touch = gt.get("touch") or gt.get("coordinate")
    g_lift = gt.get("lift") or gt.get("end_coordinate") or gt.get("coordinate")
    match = False
    if g_touch is None and g_lift is None:
        match = pred.get("action_type") == gt.get("action_type")
    elif p_touch and g_touch:
        def _as(v: Any) -> Tuple[float, float]:
            return float(v[0]), float(v[1])
        pt, gt_t = _as(p_touch), _as(g_touch)
        if p_lift is None or g_lift is None:
            dist = math.hypot(pt[0] - gt_t[0], pt[1] - gt_t[1])
            match = dist <= AITW_TAP_FRAC * max(screen)
        else:
            pl, gl = _as(p_lift), _as(g_lift)

            def _is_tap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
                return math.hypot(a[0] - b[0], a[1] - b[1]) <= AITW_DUAL_POINT_EQUAL

            if _is_tap(gt_t, gl) != _is_tap(pt, pl):
                match = False
            elif _is_tap(gt_t, gl):
                dist = math.hypot(pt[0] - gt_t[0], pt[1] - gt_t[1])
                match = dist <= AITW_TAP_FRAC * max(screen)
            else:
                g_axis = "horizontal" if abs(gl[0] - gt_t[0]) >= abs(gl[1] - gt_t[1]) else "vertical"
                p_axis = "horizontal" if abs(pl[0] - pt[0]) >= abs(pl[1] - pt[1]) else "vertical"
                match = g_axis == p_axis
    return {"match": match}


# ----- 5.8 统一入口：按数据集名字打分一步 -----

SCORERS = {
    "omnigui": score_omnigui_step,
    "screenspot_v2": score_screenspot_v2,
    "screenspot_pro": score_screenspot_pro,
    "mmbench_l2": score_mmbench_l2,
    "mmbench_l1": score_mmbench_l1,
    "android_control": score_android_control_step,
    "mind2web": score_mind2web_step,
    "aitw": score_aitw_step,
}


def score_step(dataset: str, pred_text: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    key = dataset.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ssv2": "screenspot_v2",
        "sspro": "screenspot_pro",
        "mmbench_gui_l2": "mmbench_l2",
        "mmbench_gui_l1": "mmbench_l1",
        "ac": "android_control",
        "m2w": "mind2web",
        "omni": "omnigui",
    }
    fn = SCORERS.get(aliases.get(key, key))
    if fn is None:
        raise KeyError(f"unknown dataset {dataset!r}; known={sorted(SCORERS)}")
    return fn(pred_text, doc)


# =============================================================================
# 6. 自检：keep 集合 + prompt 路由 + 各数据集打分
# =============================================================================


def _demo_keep_sets() -> None:
    n, k = 20, 5
    uni = sorted(uniform_keep_indices(n, k))
    rng = sticky_rng("ep_demo", "/abs/frame.png")
    rnd = sorted(random_keep_indices(n, k, rng))
    # sticky：同一 key 必须复现
    rnd2 = sorted(random_keep_indices(n, k, sticky_rng("ep_demo", "/abs/frame.png")))
    assert rnd == rnd2, "sticky random is not deterministic"

    # NAP：当前帧 keep 75%，滑进 history 收到 40%
    k_cur = keep_k_from_ratio(n, 0.75)
    k_hist = keep_k_from_ratio(n, 0.40)
    s_cur_u = sorted(uniform_keep_indices(n, k_cur))
    s_hist_u = contract_nap_order_uniform_or_random(
        build_nap_order(s_cur_u, kind="stride"), "stride", k_hist,
    )
    assert set(s_hist_u).issubset(set(s_cur_u))

    rng_n = sticky_rng("ep_demo", "/abs/frame.png")
    s_cur_r = random_keep_indices(n, k_cur, rng_n)
    order_r = build_nap_order(s_cur_r, kind="random", rng=sticky_rng("ep_demo", "/abs/frame.png"))
    s_hist_r = contract_nap_order_uniform_or_random(order_r, "random", k_hist)
    assert set(s_hist_r).issubset(set(s_cur_r))

    print("=== Uniform / Random keep-index demo (n=20, k=5, ratio keep) ===")
    print(f"  uniform raster:     {uni}")
    print(f"  random sticky:      {rnd}")
    print(f"  NAP uniform {k_cur}->{k_hist}: S_cur={s_cur_u}")
    print(f"                      S_hist={sorted(s_hist_u)}")
    print(
        f"  NAP random  {k_cur}->{k_hist}: "
        f"|S_cur|={len(s_cur_r)} |S_hist|={len(s_hist_r)} "
        f"nested={set(s_hist_r) <= set(s_cur_r)}"
    )


def _demo_prompts() -> None:
    sample = official_instruction_prompt(
        "打开支付宝并查看余额",
        history_output=[],
        today=datetime(2026, 9, 1),
    )
    print("\n=== GUI-Owl-1.5 prompt routing ===")
    print(f"  official history_n={OFFICIAL_DEFAULT_HISTORY_N}, eval locked history_n={LOCKED_EVAL_HISTORY_N}")
    print(f"  native system patched? {GUI_OWL_NATIVE_SYSTEM_PROMPT != OFFICIAL_SYSTEM_PROMPT}")
    print(f"  native contains 'normalized 1000x1000': {'normalized 1000x1000' in GUI_OWL_NATIVE_SYSTEM_PROMPT}")
    grounding_res_line = "* The screen's resolution is 1000x1000."
    print(
        f"  grounding keeps official 1000x1000 line: "
        f"{grounding_res_line in GUI_OWL_OFFICIAL_GROUNDING_SYSTEM_PROMPT}"
    )
    print("\n--- OmniGUI first-turn user text (no history) ---")
    print(sample)
    print("\n--- Grounding user is raw instruction, no wrapper ---")
    g = sketch_grounding_messages("shot.png", "click the search icon")
    print(f"  roles={[m['role'] for m in g]}")
    print(f"  user types={[c['type'] for c in g[1]['content']]}")


def _demo_datasets() -> None:
    print("\n=== Dataset eval routing ===")
    for d in DATASETS:
        print(f"  [{d['family']:20s}] {d['name']:22s}  task={d['task'].split()[0]}")

    click = wrap_tool_call({"action": "left_click", "coordinate": [500, 400]}, name="computer_use")
    ss = score_screenspot_v2(
        click,
        {"bbox": [400, 300, 200, 200], "img_size": [1000, 1000], "width": 1000, "height": 1000},
    )
    # bbox [400,300,200,200] xywh → pixel [400,300,600,500]; point (500,400) inside
    assert ss["hit"], ss

    l2 = score_mmbench_l2(
        click,
        {"bbox": [0.4, 0.3, 0.6, 0.5], "image_size": [1000, 1000]},
    )
    assert l2["hit"], l2

    l1 = score_mmbench_l1("The answer is B.", {"answer": "B", "options": {"A": "x", "B": "y", "C": "z"}})
    assert l1["correct"] and abs(aggregate_mmbench_l1_weighted([l1]) - 1.0) < 1e-9

    tap = wrap_tool_call({"action": "click", "coordinate": [500, 500]})
    omni = score_omnigui_step(
        tap,
        {
            "result_action_type": OMNI_TAP,
            "gt_bbox": [400, 400, 600, 600],
            "image_width": 1000,
            "image_height": 1000,
            "episode_id": "E1",
            "step_id": 0,
        },
    )
    miss = score_omnigui_step(
        tap,
        {
            "result_action_type": OMNI_TYPE,
            "result_action_text": "hello",
            "episode_id": "E1",
            "step_id": 1,
        },
    )
    assert omni["exact_match"] and not miss["exact_match"]
    assert abs(aggregate_exact_match([omni, miss]) - 0.5) < 1e-9
    assert aggregate_success_rate([omni, miss]) == 0.0
    assert abs(aggregate_goal_progress([omni, miss]) - 0.5) < 1e-9

    ac = score_android_control_step(
        tap,
        {"gt_action": {"action_type": "click", "x": 0.50, "y": 0.50}, "img_size": [1000, 1000]},
    )
    assert ac["exact_match"], ac

    m2w = score_mind2web_step(
        wrap_tool_call({"action": "click", "coordinate": [500, 500]}),
        {
            "gt_action": {"action_type": "click", "bbox_px": [400, 400, 600, 600]},
            "img_size": [1000, 1000],
        },
    )
    assert m2w["step_sr"] and m2w["ele_acc"] and m2w["op_f1"] == 1.0, m2w

    print("\n=== Dataset scorer smoke (synthetic) ===")
    print(f"  SSv2 hit={ss['hit']}  L2 hit={l2['hit']}  L1 correct={l1['correct']}")
    print(f"  OmniGUI EM={omni['exact_match']} miss_EM={miss['exact_match']} GP={aggregate_goal_progress([omni, miss]):.2f}")
    print(f"  AC EM={ac['exact_match']}  M2W step_sr={m2w['step_sr']}")


if __name__ == "__main__":
    _demo_keep_sets()
    _demo_prompts()
    _demo_datasets()
