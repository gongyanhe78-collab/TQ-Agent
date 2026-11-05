"""相似个例输出文本的业务化清洗工具。"""
from __future__ import annotations

import re
from typing import Any


COLLOQUIAL_REPLACEMENTS = {
    "咱们得": "需",
    "咱们要": "需",
    "得盯紧": "需重点关注",
    "盯紧": "重点关注",
    "很高": "较高",
    "这次": "当前过程",
}

CURRENT_ACTION_PATTERN = re.compile(
    r"(本次|此次|本轮|本过程|当前过程|新过程|建议|需要|需|应当|可考虑|重点关注|主观订正|预报上|服务上|"
    r"可作为.{0,12}(?:参考|依据)|提示当前|意味着当前|"
    r"对当前.{0,12}(?:参考|指示|研判)|"
    r"对.{0,12}(?:预报|研判|订正).{0,8}(?:参考价值|指示意义)|"
    r"(?:预报|研判|订正).{0,8}(?:可参考|提供参考|具有参考价值))"
)


ACTION_VERB_PATTERN = re.compile(
    r"(监测|跟踪|核查|核对|复核|订正|修正|调整|更新|明确提示|提示|发布|会商|"
    r"优先参考|参考|采用|加密观测|加密监测|对比|调阅)"
)

GENERIC_REASON_PATTERN = re.compile(
    r"(同属|均出现|均为|过程相似|相似过程|具有参考|参考意义|指示意义|有参照意义|相似性较高)"
)

GENERIC_REASON_SUFFIX_PATTERN = re.compile(
    r"[，,；;]?(?:对.{0,20})?(?:具有|有|可提供)?(?:参考意义|参考价值|指示意义|参照意义).*$"
)

METRIC_SIGNAL_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:hPa|毫米|mm|厘米|cm|米每秒|m/s|℃|小时|天|级)|"
    r"小时雨强|累计降水|降温幅度|温度距平|能见度|阵风|持续时间)"
)

TRANSFERABLE_PATTERN = re.compile(
    r"(在.{0,24}配置下|该类过程|类似过程|通常|往往|多表现为|一般表现为|易出现|可表现为|演变规律)"
)


def normalize_business_text(text: Any, limit: int = 240) -> str:
    """把模型输出统一为简洁、克制的业务书面语。"""
    value = re.sub(r"\s+", " ", str(text or "")).strip(" ；;。")
    for source, target in COLLOQUIAL_REPLACEMENTS.items():
        value = value.replace(source, target)
    value = value.replace("！", "。").replace("?", "？")
    # 词语替换可能增加字符数，必须在替换完成后再次按最终长度截断。
    if len(value) > limit:
        if limit <= 1:
            return value[:limit]
        value = value[:limit - 1].rstrip("，,；;。") + "。"
    return value


def normalize_complete_text(text: Any, limit: int = 120, hard_limit: int | None = None) -> str:
    """优先按完整句子收束摘要，避免在业务短语中间硬截断并补上假句号。"""
    raw_value = str(text or "").strip()
    # 模型在安全长度内自行加出的结尾省略号通常表示“按目标字数截断”，不能作为业务正文的一部分。
    if raw_value.endswith("…") and len(raw_value) <= (hard_limit or limit + 80):
        raw_value = raw_value[:-1].rstrip()
    had_terminal_mark = bool(raw_value and raw_value[-1] in "。！？!?；;")
    # 先做统一书面语清洗，但给内部步骤留出余量，真正的长度控制在句子边界处完成。
    value = normalize_business_text(raw_value, max(limit * 4, 512))
    # normalize_business_text 为兼容短字段会去掉末尾标点，这里恢复原文句尾，
    # 否则模型已经生成完整句时会被误判为无边界文本并改成省略号。
    if had_terminal_mark and value and value[-1] not in "。！？!?；;":
        value += "。"
    if len(value) <= limit:
        return value
    boundary_limit = hard_limit or limit + 80
    # 目标长度只是软提示；只要没有超过安全上限，就完整保留字段，不人为添加省略号。
    if len(value) <= boundary_limit:
        return value
    sentence_ends = [
        index + 1
        for index, char in enumerate(value)
        if char in "。！？!?；;" and index + 1 <= boundary_limit
    ]
    # 如果目标长度附近有完整句子，宁可略微超过软上限，也不切断句子。
    after_limit = [index for index in sentence_ends if index >= limit]
    if after_limit:
        return value[:after_limit[0]]
    if sentence_ends:
        return value[:sentence_ends[-1]]
    clause_ends = [
        index + 1
        for index, char in enumerate(value)
        if char in "，,、" and index + 1 <= boundary_limit
    ]
    if clause_ends:
        return value[:clause_ends[-1]].rstrip("，,、") + "…"
    # 极端情况下没有任何标点，明确用省略号表示未完整展示，不能伪造完整句号。
    return value[: max(1, boundary_limit - 1)].rstrip("，,；;") + "…"


def normalize_soft_text(text: Any, target_limit: int = 45, hard_limit: int | None = None) -> str:
    """按目标长度提示模型，但在安全上限内优先保留完整句子。

    目标长度是业务排版建议，不是对模型输出的硬切点；绝对上限只用于防止
    异常长文本挤占页面，正常略超目标长度的完整句不会被截成半句话。
    """
    target_limit = max(1, int(target_limit or 1))
    safe_limit = max(target_limit, int(hard_limit or target_limit + 55))
    return normalize_complete_text(text, target_limit, safe_limit)


def clean_reference_text(text: Any, limit: int = 180) -> str:
    """参考经验只保留历史个例事实和规律，过滤针对当前过程的行动建议。"""
    sentences = [
        normalize_business_text(sentence, limit)
        for sentence in re.split(r"(?<=[。；;])", str(text or ""))
        if sentence.strip()
    ]
    kept = [sentence for sentence in sentences if not CURRENT_ACTION_PATTERN.search(sentence)]
    if not kept and sentences:
        # 如果模型只给了一句但夹带了行动词，尽量删去明显的当前过程前缀后保留事实部分。
        value = CURRENT_ACTION_PATTERN.sub("", sentences[0])
        return normalize_business_text(value, limit)
    return normalize_business_text("".join(kept[:2]), limit)


def make_transferable_reference(text: Any, limit: int = 180) -> str:
    """将一次性历史描述收束为谨慎、可迁移的经验表达，不把单个个例夸大为普遍定律。"""
    value = clean_reference_text(text, limit)
    if not value or TRANSFERABLE_PATTERN.search(value):
        return value
    # 删除只说明来源的历史前缀，保留真正可核验的机制、强度和演变事实。
    fact = re.sub(r"^(?:历史个例中|历史过程中|该历史个例|该次过程|此次过程)[，,:：\s]*", "", value)
    fact = normalize_business_text(fact, limit)
    if not fact:
        return ""
    # “可表现为”只表达该类配置存在这种可能，不把一个样本强行推成“通常如此”。
    return normalize_business_text(f"该历史个例表明，在相近配置下可表现为{fact}", limit)


def filter_specific_match_reasons(
    reasons: Any,
    context: Any,
    signal_terms: list[str] | tuple[str, ...],
    limit: int = 2,
    text_limit: int = 80,
) -> list[str]:
    """只保留包含共同诊断信号或量级指标的理由，过滤仅靠月份、灾种标签的空泛表述。"""
    if not isinstance(reasons, list):
        return []
    context_text = str(context or "").lower()
    available_terms = [
        str(term).strip()
        for term in signal_terms
        if str(term).strip() and str(term).strip().lower() in context_text
    ]
    result = []
    for item in reasons:
        value = normalize_business_text(item, text_limit)
        # 前半句已有具体配置时只删除空泛价值判断，不连同有效物理依据一起丢弃。
        value = normalize_business_text(GENERIC_REASON_SUFFIX_PATTERN.sub("", value), text_limit)
        if not value:
            continue
        lowered = value.lower()
        matched_terms = [term for term in available_terms if term.lower() in lowered]
        metric_signal = bool(METRIC_SIGNAL_PATTERN.search(value))
        if not matched_terms and not metric_signal:
            continue
        # “同属某月、均有某灾种”即使带一个宽泛词也不足以区分个例，至少要有两个诊断信号或明确量级。
        if GENERIC_REASON_PATTERN.search(value) and len(set(matched_terms)) < 2 and not metric_signal:
            continue
        if value not in result:
            result.append(value)
        if len(result) >= limit:
            break
    return result


def build_evidence_bounded_reason(
    query_context: Any,
    case_context: Any,
    signal_terms: list[str] | tuple[str, ...],
    limit: int = 80,
) -> str:
    """在模型理由被过滤后，用双方正文共有的诊断信号生成不夸大的兜底说明。"""
    query_text = str(query_context or "").lower()
    case_text = str(case_context or "").lower()
    shared = [
        str(term).strip()
        for term in signal_terms
        if str(term).strip()
        and str(term).strip().lower() in query_text
        and str(term).strip().lower() in case_text
    ]
    shared = list(dict.fromkeys(shared))[:2]
    if shared:
        return normalize_business_text(f"当前与历史个例的共同诊断信号为{'、'.join(shared)}，相关配置具有可比性", limit)
    return normalize_business_text("灾种与基础条件相符，但历史正文缺少主导机制或强度量级的共同证据", limit)


def normalize_action_text(text: Any, limit: int = 45) -> str:
    """把建议整理成含明确动作动词的短句，并允许完整句略超目标长度。"""
    # 目标长度仅用于提示排版；安全上限只拦截异常长且缺少自然边界的输出。
    value = normalize_soft_text(text, limit, limit + 120)
    value = re.sub(r"^(?:建议|需要|需|应当|可考虑)[：:，,\s]*", "", value)
    if value and not ACTION_VERB_PATTERN.search(value):
        value = f"核查{value}"
    return normalize_soft_text(value, limit, limit + 120)


def build_action_tip(item: dict[str, Any], limit: int = 210) -> str:
    """优先按“关注对象 + 可能偏差 + 建议动作”组装预报提示。"""
    focus = normalize_soft_text(item.get("focus_object"), 30, 150)
    bias = normalize_soft_text(item.get("possible_bias"), 35, 170)
    action = normalize_action_text(item.get("suggested_action"), 45)
    parts = []
    if focus:
        parts.append(f"关注：{focus}。")
    if bias:
        parts.append(f"偏差：{bias}。")
    if action:
        parts.append(f"建议：{action}。")
    if parts:
        # 组合字段也采用软上限，避免在“关注/偏差/建议”之间再次硬截断。
        return normalize_complete_text("".join(parts), limit, limit + 80)
    # 兼容旧模型只返回 text 的情况，但仍限制长度和口语表达。
    return normalize_complete_text(item.get("text"), min(limit, 180), min(limit, 180) + 80)
