"""相似个例的输入标准化、结构化召回和候选内语义评分。"""
from __future__ import annotations

import logging
import math
import re
from datetime import date as date_type
from time import perf_counter
from typing import Any

from ..infrastructure.data_store import LocalCaseDataStore
from .dimension_profiles import resolve_dimension_profile
from .query_taxonomy import AREA_ALIASES, SHANXI_CITIES
logger = logging.getLogger("uvicorn.error")


# 所有可识别的灾害关键词，按从具体到宽泛的顺序排列（匹配时按顺序扫）
DISASTER_TERMS = (
    "雷暴大风", "短时强降水", "大暴雨", "强对流", "强降水", "暴雨", "暴雪", "雨雪",
    "冰雹", "寒潮", "低温", "高温", "沙尘", "霜冻", "降雪", "雷暴", "大风", "雾",
)

# 灾种家族：一个灾种对应的一组相关灾种。
# 用于灾种匹配时做"家族相似度"计算——不是精确匹配但同一家族的也给一定分数。
DISASTER_FAMILY = {
    "雷暴大风": ("雷暴大风", "雷暴", "大风"),
    "短时强降水": ("短时强降水", "强降水", "暴雨"),
    "大暴雨": ("大暴雨", "暴雨", "强降水"),
    "暴雨": ("暴雨", "大暴雨", "强降水", "短时强降水", "降水"),
    "强降水": ("强降水", "短时强降水", "暴雨", "大暴雨", "降水"),
    "强对流": ("强对流", "雷暴", "大风", "冰雹", "短时强降水"),
    "雨雪": ("雨雪", "降雪", "暴雪"),
    "降雪": ("降雪", "雨雪", "暴雪"),
    "暴雪": ("暴雪", "降雪", "雨雪"),
    "寒潮": ("寒潮", "低温", "霜冻"),
    "低温": ("低温", "寒潮", "霜冻"),
    "霜冻": ("霜冻", "低温", "寒潮"),
}

# 标题准入需要识别“降水”等灾种家族里的宽泛名称，但这些词不能进入用户灾种抽取枚举。
# 两套词表分离后，既能接纳“暴雨查询 + 大范围降水过程”，也不会把普通“降水”伪造成新灾种。
TITLE_DISASTER_TERMS = tuple(dict.fromkeys(
    [*DISASTER_TERMS, *(value for family in DISASTER_FAMILY.values() for value in family)]
))


def normalize_query(request: dict[str, Any]) -> dict[str, Any]:
    """把表单和自然语言补充合并成可计算的查询画像。

    输入可能来自结构化表单、自然语言描述，或两者混合。
    此函数做归一化：抽取灾种、区域、日期，生成统一的 query 结构，
    供后续结构化召回、语义检索等所有节点使用。
    """
    # 把所有文本类字段拼成一段原始描述，用于从中关键词抽取灾种和区域
    raw = " ".join(
        str(request.get(key) or "")
        for key in ("raw_query", "process_name", "date", "observation_description", "circulation_description", "intensity_description")
    ).strip()

    # 先取用户在表单里显式选择的灾种（去重、去空）
    disaster_types = _dedupe([str(item).strip() for item in request.get("disaster_types") or [] if str(item).strip()])
    # 先取用户在表单里显式选择的区域（去重、去空）
    areas = _dedupe([str(item).strip() for item in request.get("affected_areas") or [] if str(item).strip()])

    # 从原始文本中补充识别灾种关键词（没选过的才加）
    for term in DISASTER_TERMS:
        if term in raw and term not in disaster_types and not any(term in selected for selected in disaster_types):
            disaster_types.append(term)

    # 从原始文本中补充识别区域关键词（别名+全省+各地市，没选过的才加）
    for area in list(AREA_ALIASES) + ["全省", "山西省"] + list(SHANXI_CITIES):
        if area in raw and area not in areas:
            areas.append(area)

    # 日期文本：优先用起止日期字段，没有就用整段原文兜底
    date_text = " ".join(str(request.get(key) or "") for key in ("start_date", "end_date", "date"))
    if not date_text:
        date_text = raw

    # 解析开始/结束日期（解析失败返回 None）
    start = _parse_calendar_date(str(request.get("start_date") or "")) or _parse_calendar_date(date_text)
    end = _parse_calendar_date(str(request.get("end_date") or "")) or _parse_range_end(date_text, start)
    end = end or start

    # 从文本中提取涉及的月份列表
    months = _months_from_text(date_text or raw)
    # 如果解析出了精确日期，月份以日期的月份为准（更可靠）
    if start:
        months = [start.month]

    # 拼成一段"查询文本"，供向量检索和 LLM 使用
    # 只保留有内容的字段（以"："结尾说明值为空，跳过）
    query_text = "；".join(
        item for item in (
            "灾种：" + "、".join(disaster_types),
            "区域：" + "、".join(areas),
            "日期：" + date_text,
            "实况：" + str(request.get("observation_description") or ""),
            "环流：" + str(request.get("circulation_description") or ""),
            "强度：" + str(request.get("intensity_description") or ""),
            "指标：" + "；".join(
                f"{key}={value}"
                for key, value in (request.get("metric_descriptions") or {}).items()
            ),
            "补充：" + str(request.get("raw_query") or ""),
        ) if not item.endswith("：")
    )

    # 返回标准化后的查询字典
    # 灾种主次由原文中的提及频率、位置和强弱修饰共同决定，任何灾种都不享有固定优先级。
    importance_text = str(request.get("raw_query") or "").strip() or raw
    primary_disasters, secondary_disasters, disaster_importance = _split_disaster_types(
        disaster_types,
        importance_text,
        str(request.get("process_name") or ""),
    )
    return {
        "process_name": str(request.get("process_name") or "").strip(),   # 过程名称
        "raw_query": raw,                                                  # 合并后的原始文本
        "date_text": date_text.strip(),                                    # 日期描述文本
        "start_date": start.isoformat() if start else "",                  # 标准化开始日期
        "end_date": end.isoformat() if end else "",                        # 标准化结束日期
        "months": months,                                                  # 涉及的月份列表
        "disaster_types": disaster_types,                                  # 灾种列表
        # 主灾种用于准入和高权重评分，伴随灾种保留较低权重，适用于任意复合灾害而不偏向某一类别。
        "primary_disaster_types": primary_disasters,
        "secondary_disaster_types": secondary_disasters,
        "disaster_importance": disaster_importance,
        "affected_areas": areas,                                           # 影响区域列表
        "observation_description": str(request.get("observation_description") or "").strip(),  # 实况描述
        "circulation_description": str(request.get("circulation_description") or "").strip(),  # 环流描述
        "intensity_description": str(request.get("intensity_description") or "").strip(),      # 强度描述
        # 指标是自然语言解析的事实，不参与自由推断；后续候选比较按灾种配置读取。
        "metric_descriptions": {
            str(key): str(value).strip()[:300]
            for key, value in (request.get("metric_descriptions") or {}).items()
            if str(key).strip() and str(value).strip()
        },
        "query_text": query_text,                                          # 拼接好的完整查询文本
        "top_n": int(request.get("top_n") or 3),                           # 期望返回个数
        "diversity_mode": str(request.get("diversity_mode") or "moderate"),  # 多样性模式
        "include_images": bool(request.get("include_images", True)),       # 是否返回图片
    }


def structured_recall(query: dict[str, Any], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按地区、灾种、季节和过程时长完成第一阶段召回。

    对全部个例做三维度（区域/灾种/时间）打分，加权求和得到结构化分，
    按分数排序后取前 N 个作为候选，供后续语义精排使用。
    """
    scored = []
    dimension_profile = resolve_dimension_profile(query)
    structured_weights = dimension_profile.get("structured_weights") or {
        "area": 0.30,
        "disaster": 0.50,
        "temporal": 0.20,
    }
    for case in cases:
        disaster_gate_passed = not query.get("disaster_types") or _has_strong_disaster_match(query, case)
        # 分别计算三个维度的相似度得分
        breakdown = {
            "area": round(_area_score(query["affected_areas"], case.get("affected_areas") or []), 4),     # 区域匹配分
            "disaster": round(
                _disaster_score(
                    query["disaster_types"],
                    case,
                    query.get("primary_disaster_types"),
                    query.get("secondary_disaster_types"),
                    query.get("disaster_importance"),
                ),
                4,
            ),  # 灾种匹配分
            "temporal": round(_temporal_score(query, case), 4),                                          # 时间/季节匹配分
        }
        # 单独记录查询灾种被该历史个例覆盖的比例，便于复合灾害审计和人工复核。
        breakdown["disaster_coverage"] = round(_disaster_coverage_score(query, case), 4)

        # 只对"用户实际提出了要求"的维度做加权平均
        # （用户没说区域就不算区域分，避免 0 分拉低总分）
        active_weights = {key: value for key, value in breakdown.items() if _dimension_requested(query, key)}
        if active_weights:
            # 灾种是业务首要约束，避免全省落区或环流相似把明显不同灾害抬到前面。
            # 结构化召回也使用灾种配置，避免只有后置 LLM 才知道当前灾种最重要的指标。
            weight_sum = sum(float(structured_weights.get(key, 0.0)) for key in active_weights)
            score = sum(float(structured_weights.get(key, 0.0)) * value for key, value in active_weights.items()) / (weight_sum or 1.0)
        else:
            # 用户一个条件都没提 → 结构化分 0，完全靠后面的语义分排序
            score = 0.0

        # 查询明确给出灾种时，标题灾种冲突不能由语义相似度重新抬高。
        # （灾种差太远的直接打骨折，防止后面语义分把完全不对路的个例抬上来）
        if query.get("disaster_types") and breakdown["disaster"] < 0.45:
            score *= 0.35
        # 区域完全不匹配也做降权（但比灾种轻一些，因为语义可能拉回来）
        if query.get("affected_areas") and case.get("affected_areas") and breakdown["area"] == 0:
            score *= 0.5

        # 把结构化分、分数明细、匹配理由附加到个例上。
        # 标题未命中主灾种时保留为低分补充候选，避免候选为空；质量门槛会阻止它进入核心结果。
        if not disaster_gate_passed:
            score *= 0.35

        scored.append({
            **dict(case),
            "structured_score": round(score, 4),
            "score_breakdown": breakdown,
            # 把本次使用的配置透传到后续节点，保证动态维度的来源可追踪。
            "dimension_profile": dimension_profile,
            "disaster_gate_passed": disaster_gate_passed,
            "structured_reasons": _structured_reasons(query, case, breakdown),
        })

    # 按结构化分从高到低排序（同分按 case_id 倒序，保证稳定）
    scored.sort(key=lambda item: (item["structured_score"], item.get("case_id", "")), reverse=True)

    # 取候选数量：个例总数的 30%，但最少 15 个、最多 40 个。
    # 统一保留十五个候选，既给LLM足够比较空间，也保证每次审计能让业务人员复核前十五名。
    count = min(40, max(15, math.ceil(len(cases) * 0.3))) if cases else 0
    return scored[:count]


def add_semantic_scores(
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    data_store: LocalCaseDataStore,
    embedding_client: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    """只在结构化候选的 chunk 内计算语义分，避免语义召回冲掉地区和灾种。

    注意：不是全库做向量检索，而是"在结构化召回的候选范围内"算语义相似度。
    这样保证语义分只是锦上添花，不会把灾种/区域完全不对路的个例捞回来。
    返回 (带语义分的候选列表, 警告列表)。
    """
    warnings: list[str] = []

    # 没有候选或没有查询文本 → 直接返回中性语义分（0 分），不调用模型
    if not candidates or not query.get("query_text"):
        return _with_neutral_semantic(candidates), warnings

    # 获取 embedding 模型名（不同客户端属性名可能不一样，做兼容）
    model = str(getattr(embedding_client, "model", None) or getattr(embedding_client, "model_uid", "unknown"))
    started = perf_counter()

    try:
        # 先检查 embedding 客户端是否可用
        if hasattr(embedding_client, "is_available") and not embedding_client.is_available():
            logger.warning("[SmartCaseMatch][Embedding] 客户端不可用 model=%s", model)
            warnings.append("语义补充不可用，已退化为结构化排序：Embedding 客户端不可用")
            return _with_neutral_semantic(candidates), warnings

        logger.info("[SmartCaseMatch][Embedding] 开始调用 model=%s", model)
        # 调用 embedding 模型，把查询文本转成向量
        vector = embedding_client.embed_query(query["query_text"])
        expected_dim = len(vector)
        logger.info(
            "[SmartCaseMatch][Embedding] 调用成功 model=%s dimension=%d elapsed_ms=%.2f",
            model,
            len(vector),
            (perf_counter() - started) * 1000,
        )

        # 先收集结构化候选真正关联的 chunk，再计算余弦；库规模增长时开销由候选集决定。
        candidate_chunk_ids = {
            str(chunk_id)
            for candidate in candidates
            for chunk_id in candidate.get("source_chunk_ids") or []
        }
        valid_chunks = [
            data_store.get_chunk(chunk_id)
            for chunk_id in candidate_chunk_ids
        ]
        valid_chunks = [
            chunk for chunk in valid_chunks
            if chunk and len(chunk.get("embedding") or []) == expected_dim
        ]
        if not valid_chunks:
            raise RuntimeError(f"本地 chunk 向量维度与查询向量不一致：{expected_dim}")

        # 批量计算所有有效 chunk 与查询向量的余弦相似度
        chunk_scores = {
            str(chunk.get("chunk_id")): _cosine_score(vector, chunk.get("embedding") or [])
            for chunk in valid_chunks
        }

        # 逐个候选个例：取它关联的 chunk 中相似度最高的 2 个的平均分作为语义分
        updated = []
        for candidate in candidates:
            # 找出这个个例的所有 chunk 及其分数，按分数从高到低排序
            hits = sorted(
                ((chunk_id, chunk_scores[chunk_id]) for chunk_id in candidate.get("source_chunk_ids") or [] if chunk_id in chunk_scores),
                key=lambda item: item[1],
                reverse=True,
            )
            # 取前 2 个最高相似度的平均分作为个例的语义分
            top_scores = [score for _, score in hits[:2]]
            semantic = sum(top_scores) / len(top_scores) if top_scores else 0.0

            updated.append({
                **candidate,
                "semantic_score": round(semantic, 4),                    # 语义相似度分（前2平均）
                "semantic_chunk_ids": [chunk_id for chunk_id, _ in hits[:2]],  # 最相关的 2 个 chunk ID
                # 多保留少量向量命中供专用 Rerank 精排，不进入最终输出。
                "semantic_hits": [
                    {"chunk_id": chunk_id, "score": round(score, 6)}
                    for chunk_id, score in hits[:4]
                ],
                "semantic_available": bool(top_scores),                  # 是否有有效的语义分
            })
        return updated, warnings

    except Exception as exc:
        # embedding 调用失败 → 降级为纯结构化排序，把错误记为警告
        logger.exception("[SmartCaseMatch][Embedding] 调用失败 model=%s error=%s", model, exc)
        warnings.append(f"语义补充不可用，已退化为结构化排序：{exc}")
        return _with_neutral_semantic(candidates), warnings


def fuse_scores(candidates: list[dict[str, Any]], semantic_available: bool | None = None) -> list[dict[str, Any]]:
    """融合结构化分和语义分，语义不可用时不惩罚结构化候选。

    融合公式：retrieval_score = 0.65 * 结构化分 + 0.35 * 语义分
    如果某个个例没有语义分（semantic_available=False），就只用结构化分，
    避免"0 分语义分"把好的结构化候选拉低。
    """
    result = []
    for item in candidates:
        # 判断这个个例是否有有效的语义分（可外部强制指定，也可自动判断）
        semantic_ok = item.get("semantic_available", False) if semantic_available is None else semantic_available
        # 有语义分 → 加权融合；没语义分 → 纯结构化分
        fused = (
            0.65 * float(item.get("structured_score") or 0.0) + 0.35 * float(item.get("semantic_score") or 0.0)
            if semantic_ok else float(item.get("structured_score") or 0.0)
        )
        result.append({**item, "retrieval_score": round(fused, 4)})
    # 按融合后的检索分从高到低排序
    return sorted(result, key=lambda item: (item["retrieval_score"], item.get("case_id", "")), reverse=True)


def _with_neutral_semantic(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """给降级结果补齐语义字段，供后续审计展示。

    当语义检索不可用时（embedding 挂了或没数据），调用此函数给所有候选
    补上语义相关字段的默认空值，保证后续节点不用判空。
    """
    return [
        {
            **item,
            "semantic_score": 0.0,         # 语义分：0
            "semantic_chunk_ids": [],      # 相关 chunk ID：空
            "semantic_hits": [],           # 语义命中详情：空
            "semantic_available": False,   # 语义可用：否
        }
        for item in candidates
    ]


def _dimension_requested(query: dict[str, Any], dimension: str) -> bool:
    """判断用户是否在查询中明确指定了某个维度的条件。

    区域 → 看 affected_areas 是否有值
    灾种 → 看 disaster_types 是否有值
    时间 → 看 months 是否有值
    """
    return bool({"area": query.get("affected_areas"), "disaster": query.get("disaster_types"), "temporal": query.get("months")}.get(dimension))


def _area_score(query_areas: list[str], case_areas: list[str]) -> float:
    """计算地区覆盖率和 Jaccard，并对全省泛化个例做上限约束。

    打分逻辑：
    - coverage（覆盖率）：查询的地区中有多少被个例覆盖到了（查了 3 个市都中了 = 100%）
    - jaccard（杰卡德系数）：交集/并集，衡量两个集合整体的相似度
    - 两个指标加权：覆盖率更重要（0.65），jaccard 次之（0.35）
    特殊处理：
    - 个例是"全省"但用户只查局部 → 打上限（最高 0.68），避免全省个例永远排第一
    - 个例是具体地市且有分 → 小幅加分（+0.08），精确匹配优先
    """
    if not query_areas:
        return 0.0
    # 把查询区域和个例区域都展开为具体地市集合（"晋北"→"大同,朔州,忻州"）
    query_set = _expand_areas(query_areas, treat_shanxi_as_all=True)
    case_set = _expand_areas(case_areas)
    if not query_set or not case_set:
        return 0.0
    # 覆盖率：查询的地市中有多少被命中（用户视角：我关心的地方覆盖了多少）
    coverage = len(query_set & case_set) / len(query_set)
    # Jaccard 系数：交集/并集（整体视角：两个区域有多相似）
    jaccard = len(query_set & case_set) / len(query_set | case_set)
    # 加权求和
    score = 0.65 * coverage + 0.35 * jaccard
    # 个例是全省级但用户只查局部 → 上限 0.68，防止全省个例霸榜
    if _is_province_wide(case_areas) and not _is_province_wide(query_areas, treat_shanxi_as_all=True):
        score = min(score, 0.68)
    # 个例是具体地市且得分大于 0 → 小幅加 0.08，精确匹配比泛化匹配好
    elif score > 0 and not _is_province_wide(case_areas):
        score = min(1.0, score + 0.08)
    return score


def _disaster_score(
    query_types: list[str],
    case: dict[str, Any],
    primary_types: list[str] | None = None,
    secondary_types: list[str] | None = None,
    disaster_importance: dict[str, Any] | None = None,
) -> float:
    """计算灾种匹配得分。

    对每个查询灾种分别打分（取平均）：
    - 个例标题里直接出现这个灾种 → 1.0（最强信号）
    - 标题里出现同一家族的相关灾种 → 0.85
    - 个例灾种标签里有这个灾种 → 0.72
    - 灾种标签里有部分同家族的 → 按比例给 0.62 以下的分
    特殊处理：个例标题明确是其他灾种时，上限 0.35（防止正文标签误判）
    """
    if not query_types:
        return 0.0
    title = str(case.get("title") or "")
    case_types = [str(item) for item in case.get("disaster_types") or []]
    # 标题中出现的所有灾种关键词（用于判断标题级别的冲突）。
    title_terms = [term for term in TITLE_DISASTER_TERMS if term in title]

    importance = disaster_importance or {}
    default_weight = 1.0 / len(query_types)
    weighted_scores = []
    for term in query_types:
        family = DISASTER_FAMILY.get(term, (term,))
        if term in title:
            score = 1.0
        elif any(value in title for value in family):
            score = 0.85
        elif term in case_types:
            # 仅有标准标签属于弱证据，不能和标题命中等价。
            score = 0.48
        else:
            related_count = sum(1 for value in family if value in case_types)
            score = related_count / len(family) * 0.35
        item = importance.get(term) if isinstance(importance, dict) else None
        weight = float(item.get("weight") or default_weight) if isinstance(item, dict) else default_weight
        weighted_scores.append((score, weight))
    weight_sum = sum(weight for _, weight in weighted_scores) or 1.0
    result = sum(score * weight for score, weight in weighted_scores) / weight_sum

    # 把查询灾种展开成家族集合，用于判断标题是否完全跑题
    query_family = {value for term in query_types for value in DISASTER_FAMILY.get(term, (term,))}
    # 标题明确为其他灾种时，对正文误抽出的宽泛灾种标签做上限约束。
    # （标题说"暴雪"，正文标签有"大风"，不能算高度相关）
    if title_terms and not any(term in query_family for term in title_terms):
        result = min(result, 0.35)
    return result


def _split_disaster_types(
    values: list[str],
    source_text: str = "",
    process_name: str = "",
) -> tuple[list[str], list[str], dict[str, dict[str, Any]]]:
    """按原文证据强度划分主次灾种，不对雪、暴雨等任何类别设置固定偏好。"""
    values = _dedupe([str(value).strip() for value in values if str(value).strip()])
    if not values:
        return [], [], {}
    source = str(source_text or "")
    first_sentence = re.split(r"[。！？；]", source, maxsplit=1)[0]
    weak_modifiers = ("弱", "局地", "个别", "短暂", "伴有", "伴随")
    strong_modifiers = ("强", "严重", "大范围", "全省", "主要", "重点", "持续", "显著", "极端")
    raw_scores: dict[str, float] = {}
    mention_counts: dict[str, int] = {}
    for term in values:
        positions = [match.start() for match in re.finditer(re.escape(term), source)]
        # 模型从受控枚举识别出的灾种即使没有逐字出现，也保留一个较低基础权重。
        score = 0.75 if not positions else 1.0
        weighted_mentions = 0.0
        for position in positions:
            window = source[max(0, position - 8):position + len(term) + 8]
            factor = 1.0
            if any(value in window for value in weak_modifiers):
                factor *= 0.55
            if any(value in window for value in strong_modifiers):
                factor *= 1.25
            weighted_mentions += factor
        score += min(3.0, weighted_mentions) * 0.45
        if term in first_sentence:
            score += 0.55
        if term in process_name:
            score += 0.75
        raw_scores[term] = score
        mention_counts[term] = len(positions)
    total = sum(raw_scores.values()) or 1.0
    max_score = max(raw_scores.values())
    # 与最高证据强度接近的灾种共同作为主灾种，复合过程允许多个核心灾种并列。
    primary = [term for term in values if raw_scores[term] >= max_score * 0.72]
    secondary = [term for term in values if term not in primary]
    importance = {
        term: {
            "mention_count": mention_counts[term],
            "evidence_score": round(raw_scores[term], 4),
            "weight": round(raw_scores[term] / total, 6),
            "role": "primary" if term in primary else "secondary",
        }
        for term in values
    }
    return primary, secondary, importance


def _disaster_coverage_score(query: dict[str, Any], case: dict[str, Any]) -> float:
    """计算历史个例覆盖了多少查询灾种，权重来自用户原文而不是灾种类别常量。"""
    query_types = [str(value) for value in query.get("disaster_types") or []]
    if not query_types:
        return 0.0
    importance = query.get("disaster_importance") or {}
    title = str(case.get("title") or "")
    case_types = {str(value) for value in case.get("disaster_types") or []}
    default_weight = 1.0 / len(query_types)
    covered = 0.0
    total = 0.0
    for term in query_types:
        item = importance.get(term) if isinstance(importance, dict) else None
        weight = float(item.get("weight") or default_weight) if isinstance(item, dict) else default_weight
        family = DISASTER_FAMILY.get(term, (term,))
        if any(value in title for value in family) or term in case_types:
            covered += weight
        total += weight
    return covered / (total or 1.0)


def _has_strong_disaster_match(query: dict[str, Any], case: dict[str, Any]) -> bool:
    """判断个例标题是否明确属于用户请求的灾种家族。

    标准化标签可能来自正文中的次要描述，因此核心准入优先使用标题；
    这样任意灾种查询都不会仅因个例正文偶然提到一次相关词就误收无关过程。
    """
    query_types = query.get("disaster_types") or []
    importance = query.get("disaster_importance") or {}
    title = str(case.get("title") or "")
    title_terms = [term for term in TITLE_DISASTER_TERMS if term in title]
    matched_weight = 0.0
    total_weight = 0.0
    default_weight = 1.0 / len(query_types) if query_types else 0.0
    for term in query_types:
        item = importance.get(term) if isinstance(importance, dict) else None
        weight = float(item.get("weight") or default_weight) if isinstance(item, dict) else default_weight
        total_weight += weight
        family = DISASTER_FAMILY.get(str(term), (str(term),))
        if any(value in title for value in family):
            matched_weight += weight
    # 复合灾害标题覆盖四分之一以上查询权重即可进入候选，后续仍由覆盖率和专业维度精排。
    return bool(title_terms and matched_weight / (total_weight or 1.0) >= 0.25)


def _temporal_score(query: dict[str, Any], case: dict[str, Any]) -> float:
    """计算时间/季节匹配得分。

    主分：月份距离（环形，12 月和 1 月只差 1）
      同月 → 1.0，差 1 月 → 0.72，差 2 月 → 0.5，差 3 月 → 0.3，更远 → 0.12
    附加：过程时长相似度（最多贡献 0.15 的权重）
      时长越接近，得分越高
    """
    if not query.get("months"):
        return 0.0
    # 从个例的日期范围、标题、PDF 名中提取月份
    case_months = _months_from_text(" ".join([str(case.get("date_range") or ""), str(case.get("title") or ""), str(case.get("source_pdf") or "")]))
    if not case_months:
        return 0.0

    # 计算最小月份距离（环形：12月到1月只差 1，不是 11）
    month_distance = min(min(abs(left - right), 12 - abs(left - right)) for left in query["months"] for right in case_months)
    # 按距离查表给分
    score = {0: 1.0, 1: 0.72, 2: 0.5, 3: 0.3}.get(month_distance, 0.12)

    # 过程时长相似度（如果两边都能算出天数）
    query_duration = _duration_days(query.get("start_date"), query.get("end_date"))
    case_duration = _duration_from_text(str(case.get("date_range") or ""))
    if query_duration and case_duration:
        # 月份分占 85%，时长相似度占 15%
        score = 0.85 * score + 0.15 * max(0.0, 1 - abs(query_duration - case_duration) / max(query_duration, case_duration))
    return score


def _structured_reasons(query: dict[str, Any], case: dict[str, Any], breakdown: dict[str, float]) -> list[str]:
    """根据各维度得分生成人类可读的匹配理由文案，供前端展示。"""
    reasons = []
    # 区域维度：高分/低分各对应一句话
    if breakdown["area"] >= 0.7:
        reasons.append("影响区域与历史个例有较高地市重合")
    elif breakdown["area"] > 0:
        reasons.append("影响区域存在部分重合或区域泛化匹配")
    # 灾种维度
    if breakdown["disaster"] >= 0.7:
        reasons.append("灾害类型高度相关")
    elif breakdown["disaster"] > 0:
        reasons.append("灾害类型存在相关性")
    # 时间维度
    if breakdown["temporal"] >= 0.7:
        reasons.append("发生月份接近，季节背景相似")
    elif breakdown["temporal"] > 0:
        reasons.append("发生季节存在一定相似性")
    # 一个维度都没命中 → 给一条兜底说明
    return reasons or ["结构化条件信息有限，保留作为补充候选"]


def _expand_areas(values: list[str], treat_shanxi_as_all: bool = False) -> set[str]:
    """把区域列表展开为具体地市集合。

    - "全省" / "山西省" → 展开为全部 11 个地市
    - "晋北""中部"等别名 → 展开为对应的地市
    - 具体市名 → 去掉"市"字后保留
    treat_shanxi_as_all=True 时"山西省"按全省处理，否则忽略
    """
    result: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text == "全省":
            # 全省 → 全部 11 个地市
            result.update(SHANXI_CITIES)
        elif text == "山西省" and not treat_shanxi_as_all:
            # 不把"山西省"当全省时，跳过
            continue
        elif text == "山西省":
            # 把"山西省"当全省 → 全部地市
            result.update(SHANXI_CITIES)
        elif text in AREA_ALIASES:
            # 区域别名 → 展开为对应地市
            result.update(AREA_ALIASES[text])
        else:
            # 具体市名 → 去掉"市"字后加入（统一格式）
            result.add(text.replace("市", ""))
    return result


def _is_province_wide(values: list[str], treat_shanxi_as_all: bool = False) -> bool:
    """判断区域列表是否表示"全省范围"。

    包含"全省"即为全省级tr；eat_shanxi_as_all=True 时"山西省"也算。
    """
    return any(str(value).strip() == "全省" or (treat_shanxi_as_all and str(value).strip() == "山西省") for value in values)


def _months_from_text(text: str) -> list[int]:
    """从文本中正则提取所有月份（1-12），去重保序。

    匹配形如 "7月""12 月" 等格式，用 (?<!\d) 防止匹配到日期里的数字。
    """
    return _dedupe_int([int(value) for value in re.findall(r"(?<!\d)(\d{1,2})\s*月", text or "") if 1 <= int(value) <= 12])


def _parse_calendar_date(text: str) -> date_type | None:
    """从文本中尝试解析完整年月日。

    依次尝试两种格式：
    1. "2023年7月20日" / "2023-07-20" / "2023/7/20"
    2. "7月20日"（无年份，默认 2000 年）
    解析失败返回 None。
    """
    value = str(text or "").strip()
    # 格式 1：年月日 + 中文/横线/斜线分隔
    match = re.search(r"(20\d{2})\s*[年/-]\s*(\d{1,2})\s*[月/-]\s*(\d{1,2})", value)
    if match:
        return _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    # 格式 2：月日，无年份 → 默认 2000 年
    match = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日", value)
    if match:
        return _safe_date(2000, int(match.group(1)), int(match.group(2)))
    return None


def _safe_date(year: int, month: int, day: int) -> date_type | None:
    """安全创建 date 对象，非法日期（如 2 月 30 日）返回 None 而非抛异常。"""
    try:
        return date_type(year, month, day)
    except ValueError:
        return None


def _duration_days(start: str, end: str) -> int:
    """根据起止日期字符串计算过程持续天数（含首尾）。

    解析失败或结束早于开始时返回 0。
    """
    left, right = _parse_calendar_date(start), _parse_calendar_date(end)
    return (right - left).days + 1 if left and right and right >= left else 0


def _duration_from_text(text: str) -> int:
    """从日期范围文本中提取过程持续天数。

    匹配 "20日-25日""7月10日至12日" 等格式，取第一个匹配。
    只有单个日期 → 1 天；匹配不到 → 0 天。
    """
    value = str(text or "")
    # 优先按带月份的范围计算，使用 date 相减可以正确处理 7 月 31 日至 8 月 2 日和跨年范围。
    dated_range = re.search(
        r"(?:(?P<year>20\d{2})\s*年\s*)?(?P<start_month>\d{1,2})\s*月\s*(?P<start_day>\d{1,2})\s*(?:日|号)?"
        r"\s*[-~～至到]\s*(?:(?P<end_year>20\d{2})\s*年\s*)?"
        r"(?:(?P<end_month>\d{1,2})\s*月\s*)?(?P<end_day>\d{1,2})\s*(?:日|号)?",
        value,
    )
    if dated_range:
        year = int(dated_range.group("year") or 2000)
        start_month, start_day = int(dated_range.group("start_month")), int(dated_range.group("start_day"))
        end_month, end_day = int(dated_range.group("end_month") or start_month), int(dated_range.group("end_day"))
        start = _safe_date(year, start_month, start_day)
        explicit_end_year = dated_range.group("end_year")
        end_year = int(explicit_end_year) if explicit_end_year else (year + 1 if (end_month, end_day) < (start_month, start_day) else year)
        end = _safe_date(end_year, end_month, end_day)
        if start and end:
            return (end - start).days + 1

    matches = re.findall(r"(\d{1,2})\s*(?:日|号)?\s*[-~～至到]\s*(\d{1,2})\s*(?:日|号)?", value)
    if not matches:
        return 1 if re.search(r"\d{1,2}\s*日", text or "") else 0
    start, end = map(int, matches[0])
    return max(1, end - start + 1)


def _parse_range_end(text: str, start: date_type | None) -> date_type | None:
    """补齐结构化日期文本中的范围结束日，避免自然语言和表单入口的时长特征不一致。"""
    if not start:
        return None
    value = str(text or "")
    cross_month = re.search(
        r"\d{1,2}\s*月\s*\d{1,2}\s*(?:日|号)?\s*[-~～至到]\s*"
        r"(?:(?P<end_year>20\d{2})\s*年\s*)?(?P<month>\d{1,2})\s*月\s*(?P<day>\d{1,2})",
        value,
    )
    if cross_month:
        end_month, end_day = int(cross_month.group("month")), int(cross_month.group("day"))
        # 结构化表单也允许跨年范围，避免 12 月到 1 月时长被错误压缩为一天。
        explicit_end_year = cross_month.group("end_year")
        end_year = int(explicit_end_year) if explicit_end_year else (start.year + 1 if (end_month, end_day) < (start.month, start.day) else start.year)
        return _safe_date(end_year, end_month, end_day)
    same_month = re.search(
        r"\d{1,2}\s*月\s*\d{1,2}\s*(?:日|号)?\s*[-~～至到]\s*(?P<day>\d{1,2})\s*(?:日|号)?",
        value,
    )
    if same_month:
        day = int(same_month.group("day"))
        return _safe_date(start.year, start.month, day) if day >= start.day else None
    return None


def _cosine_score(left: list[float], right: list[float]) -> float:
    """计算两个向量的余弦相似度，并线性映射到 [0, 1] 区间。

    标准余弦相似度范围是 [-1, 1]，这里做 (x+1)/2 映射到 [0, 1]，
    方便和其他 0-1 分数加权融合。零向量直接返回 0。
    """
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    # 余弦相似度 [-1, 1] → 线性映射到 [0, 1]
    return max(0.0, min(1.0, (numerator / (left_norm * right_norm) + 1) / 2))


def _dedupe(values: list[str]) -> list[str]:
    """字符串列表去重，保持原顺序。用 dict.fromkeys 利用了 Python 3.7+ 字典保序的特性。"""
    return list(dict.fromkeys(values))


def _dedupe_int(values: list[int]) -> list[int]:
    """整数列表去重，保持原顺序。"""
    return list(dict.fromkeys(values))

