"""相似个例匹配的灾种指标配置。

匹配主流程不应该假定所有灾种都使用冷空气、降水和积雪作为判断依据。
本模块把共用的匹配维度与灾种差异放进配置，供结构化召回、候选重排和结果展示共同使用。
"""
from __future__ import annotations

from typing import Any


# 所有灾种都可以使用的基础配置。未识别到具体灾种时使用它，保证新增灾种也有稳定的退化路径。
GENERIC_PROFILE = {
    "profile_id": "generic",
    "label": "通用天气过程",
    "dimensions": (
        {"key": "hazard_match", "label": "灾种", "weight": 0.38, "required": True},
        {"key": "area", "label": "落区", "weight": 0.22, "required": False},
        {"key": "temporal", "label": "时段", "weight": 0.12, "required": False},
        {"key": "mechanism", "label": "主导机制", "weight": 0.16, "required": False},
        {"key": "intensity", "label": "强度量级", "weight": 0.12, "required": False},
    ),
    "structured_weights": {"area": 0.30, "disaster": 0.50, "temporal": 0.20},
    "evidence_topics": ("降水", "风", "温度", "雪", "环流", "雷达", "沙尘", "对流"),
    "metrics": ("过程强度", "持续时间", "影响范围", "主导系统", "演变过程"),
    # 这些是跨灾种都可能出现的诊断信号，仅用于判断理由是否具体，不参与硬编码排序。
    "signal_terms": ("水汽", "急流", "切变", "槽", "脊", "冷平流", "暖平流", "回波", "能见度", "温度", "风速", "持续时间", "影响范围", "演变"),
    "coverage_floor": 0.45,
    "coverage_span": 0.55,
    "penalty_floor": 0.55,
    "penalty_span": 0.45,
}


# 只在指标层表达灾种差异，不改变召回、向量、Rerank、逐例提炼和综合研判的流程。
DISASTER_PROFILES = {
    "snow": {
        "types": ("雨雪", "降雪", "暴雪"),
        "label": "雨雪过程",
        "dimensions": (
            ("hazard_match", "灾种", 0.38, True),
            ("mechanism", "相态与主导机制", 0.24, False),
            ("intensity", "降水与积雪量级", 0.16, False),
            ("temporal", "过程时段", 0.10, False),
            ("area", "落区", 0.07, False),
            ("evolution", "相态演变", 0.05, False),
        ),
        "structured_weights": {"area": 0.22, "disaster": 0.56, "temporal": 0.22},
        # 普通降水图不能自动作为雪类经验证据，只有经验文字明确落到可解释交集时才允许进入。
        "evidence_topics": ("雪", "温度", "环流", "雷达"),
        "metrics": ("降水量", "积雪深度", "相态", "低层温度", "持续时间", "相态演变"),
        "signal_terms": ("回流", "冷垫", "偏东风", "偏西风", "横槽", "冷空气路径", "切变", "相态转换", "积雪深度", "降水量", "降水强度", "低层温度"),
        "coverage_floor": 0.45,
        "coverage_span": 0.55,
        "penalty_floor": 0.55,
        "penalty_span": 0.45,
    },
    "precipitation": {
        "types": ("暴雨", "大暴雨", "强降水", "短时强降水"),
        "label": "降水过程",
        "dimensions": (
            ("hazard_match", "灾种", 0.40, True),
            ("intensity", "累计与小时雨强", 0.22, False),
            ("mechanism", "水汽与影响系统", 0.16, False),
            ("area", "落区", 0.12, False),
            ("temporal", "持续时间", 0.06, False),
            ("evolution", "降水演变", 0.04, False),
        ),
        "structured_weights": {"area": 0.27, "disaster": 0.53, "temporal": 0.20},
        "evidence_topics": ("降水", "环流", "雷达", "对流", "风"),
        "metrics": ("累计降水量", "最大小时雨强", "持续时间", "水汽输送", "回波强度", "降水演变"),
        "signal_terms": ("水汽输送", "急流", "小时雨强", "累计降水", "回波", "对流", "地形抬升", "切变", "降水演变"),
        "coverage_floor": 0.45,
        "coverage_span": 0.55,
        "penalty_floor": 0.55,
        "penalty_span": 0.45,
    },
    "wind": {
        "types": ("大风", "雷暴大风"),
        "label": "大风过程",
        "dimensions": (
            ("hazard_match", "灾种", 0.40, True),
            ("intensity", "最大风速与风力", 0.22, False),
            ("mechanism", "气压梯度与影响系统", 0.18, False),
            ("temporal", "持续时间", 0.08, False),
            ("area", "落区", 0.08, False),
            ("evolution", "风力演变", 0.04, False),
        ),
        "structured_weights": {"area": 0.25, "disaster": 0.55, "temporal": 0.20},
        "evidence_topics": ("风", "环流", "对流", "雷达", "沙尘"),
        "metrics": ("最大阵风", "平均风速", "风力等级", "持续时间", "气压梯度", "风力演变"),
        "signal_terms": ("气压梯度", "阵风", "冷平流", "下沉气流", "急流", "风力演变", "持续时间"),
        "coverage_floor": 0.45,
        "coverage_span": 0.55,
        "penalty_floor": 0.55,
        "penalty_span": 0.45,
    },
    "convection": {
        "types": ("强对流", "冰雹"),
        "label": "强对流过程",
        "dimensions": (
            ("hazard_match", "灾种", 0.40, True),
            ("mechanism", "对流触发机制", 0.21, False),
            ("intensity", "对流强度", 0.20, False),
            ("evolution", "回波演变", 0.08, False),
            ("area", "落区", 0.07, False),
            ("temporal", "影响时段", 0.04, False),
        ),
        "structured_weights": {"area": 0.25, "disaster": 0.55, "temporal": 0.20},
        "evidence_topics": ("对流", "雷达", "降水", "风", "环流"),
        "metrics": ("回波强度", "冰雹直径", "雷暴大风", "短时雨强", "影响时段", "回波演变"),
        "signal_terms": ("对流触发", "回波", "冰雹", "雷暴大风", "短时雨强", "不稳定能量", "垂直风切变", "影响时段"),
        "coverage_floor": 0.45,
        "coverage_span": 0.55,
        "penalty_floor": 0.55,
        "penalty_span": 0.45,
    },
    "cold": {
        "types": ("寒潮", "低温", "霜冻"),
        "label": "低温过程",
        "dimensions": (
            ("hazard_match", "灾种", 0.40, True),
            ("intensity", "温度及距平", 0.22, False),
            ("temporal", "持续时间", 0.16, False),
            ("mechanism", "环流与温度平流", 0.12, False),
            ("area", "落区", 0.07, False),
            ("evolution", "温度演变", 0.03, False),
        ),
        "structured_weights": {"area": 0.25, "disaster": 0.55, "temporal": 0.20},
        "evidence_topics": ("温度", "环流", "风", "雪"),
        "metrics": ("最高温度", "最低温度", "降温幅度", "距平", "持续时间", "温度演变"),
        "signal_terms": ("降温幅度", "温度距平", "冷平流", "寒潮", "霜冻", "持续时间", "最低温度"),
        "coverage_floor": 0.45,
        "coverage_span": 0.55,
        "penalty_floor": 0.55,
        "penalty_span": 0.45,
    },
    "heat": {
        "types": ("高温",),
        "label": "高温过程",
        "dimensions": (
            ("hazard_match", "灾种", 0.40, True),
            ("intensity", "最高温度与距平", 0.24, False),
            ("temporal", "高温持续时间", 0.20, False),
            ("mechanism", "高压脊与暖平流", 0.10, False),
            ("area", "落区", 0.06, False),
        ),
        "structured_weights": {"area": 0.25, "disaster": 0.55, "temporal": 0.20},
        "evidence_topics": ("温度", "环流", "风"),
        "metrics": ("最高温度", "温度距平", "高温持续天数", "暖高压", "影响范围", "高温演变"),
        "signal_terms": ("高压脊", "暖平流", "温度距平", "最高温度", "持续时间", "副高", "下沉气流"),
        "coverage_floor": 0.45,
        "coverage_span": 0.55,
        "penalty_floor": 0.55,
        "penalty_span": 0.45,
    },
    "visibility": {
        "types": ("雾",),
        "label": "低能见度过程",
        "dimensions": (
            ("hazard_match", "灾种", 0.40, True),
            ("intensity", "能见度", 0.22, False),
            ("mechanism", "湿度与逆温", 0.18, False),
            ("temporal", "持续时间", 0.12, False),
            ("area", "落区", 0.08, False),
        ),
        "structured_weights": {"area": 0.27, "disaster": 0.53, "temporal": 0.20},
        "evidence_topics": ("能见度", "温度", "环流", "风"),
        "metrics": ("能见度", "相对湿度", "逆温", "持续时间", "风速", "能见度演变"),
        "signal_terms": ("能见度", "相对湿度", "逆温", "静稳", "近地层", "风速", "持续时间"),
        "coverage_floor": 0.45,
        "coverage_span": 0.55,
        "penalty_floor": 0.55,
        "penalty_span": 0.45,
    },
    "sand": {
        "types": ("沙尘",),
        "label": "沙尘过程",
        "dimensions": (
            ("hazard_match", "灾种", 0.40, True),
            ("intensity", "沙尘强度与能见度", 0.22, False),
            ("mechanism", "上游输送与地面风", 0.18, False),
            ("area", "落区", 0.10, False),
            ("temporal", "持续时间", 0.10, False),
        ),
        "structured_weights": {"area": 0.27, "disaster": 0.53, "temporal": 0.20},
        "evidence_topics": ("沙尘", "能见度", "风", "环流", "温度"),
        "metrics": ("能见度", "地面风速", "沙尘强度", "输送路径", "持续时间", "沙尘演变"),
        "signal_terms": ("上游输送", "输送路径", "起沙", "地面风", "能见度", "沙尘强度", "持续时间"),
        "coverage_floor": 0.45,
        "coverage_span": 0.55,
        "penalty_floor": 0.55,
        "penalty_span": 0.45,
    },
}


def _materialize_profile(profile_id: str, matched_types: list[str]) -> dict[str, Any]:
    """将内部元组配置转换为可安全放入请求和日志的普通字典。"""
    source = DISASTER_PROFILES.get(profile_id, GENERIC_PROFILE)
    dimensions = source["dimensions"]
    if dimensions and isinstance(dimensions[0], tuple):
        dimensions = [
            {"key": key, "label": label, "weight": weight, "required": required}
            for key, label, weight, required in dimensions
        ]
    return {
        "profile_id": profile_id,
        "label": source["label"],
        "disaster_types": matched_types,
        "dimensions": [dict(item) for item in dimensions],
        "structured_weights": dict(source["structured_weights"]),
        "evidence_topics": list(source["evidence_topics"]),
        "metrics": list(source["metrics"]),
        "signal_terms": list(source.get("signal_terms") or GENERIC_PROFILE["signal_terms"]),
        "coverage_floor": float(source.get("coverage_floor", GENERIC_PROFILE["coverage_floor"])),
        "coverage_span": float(source.get("coverage_span", GENERIC_PROFILE["coverage_span"])),
        "penalty_floor": float(source.get("penalty_floor", GENERIC_PROFILE["penalty_floor"])),
        "penalty_span": float(source.get("penalty_span", GENERIC_PROFILE["penalty_span"])),
    }


def resolve_dimension_profile(query: dict[str, Any] | None) -> dict[str, Any]:
    """根据全部查询灾种选择指标配置，多灾种时合并维度而不是套用单一灾种规则。"""
    query = query or {}
    # 主次灾种只控制结构化灾种得分，不应让伴随灾种的阵风、能见度等指标失去深入比较机会。
    # 因此动态配置覆盖全部已识别灾种，最终排序仍由灾种重要性和有证据的维度共同约束。
    requested = [
        str(value).strip()
        for value in (query.get("disaster_types") or query.get("primary_disaster_types") or [])
        if str(value).strip()
    ]
    matched_profiles: list[str] = []
    matched_types: list[str] = []
    for profile_id, profile in DISASTER_PROFILES.items():
        overlap = [value for value in requested if value in profile["types"]]
        if overlap:
            matched_profiles.append(profile_id)
            matched_types.extend(overlap)
    if not matched_profiles:
        return _materialize_profile("generic", requested)
    if len(matched_profiles) == 1:
        return _materialize_profile(matched_profiles[0], list(dict.fromkeys(matched_types)))

    # 多灾种请求取各灾种维度的最大权重，再统一归一化，确保主灾种不会被平均稀释。
    merged: dict[str, dict[str, Any]] = {}
    evidence_topics: set[str] = set()
    metrics: list[str] = []
    signal_terms: set[str] = set()
    structured_weights = {"area": 0.0, "disaster": 0.0, "temporal": 0.0}
    for profile_id in matched_profiles:
        profile = DISASTER_PROFILES[profile_id]
        evidence_topics.update(profile["evidence_topics"])
        metrics.extend(profile["metrics"])
        signal_terms.update(profile.get("signal_terms") or [])
        for key, value in profile["structured_weights"].items():
            structured_weights[key] = max(structured_weights[key], float(value))
        for key, label, weight, required in profile["dimensions"]:
            current = merged.get(key)
            if current is None or float(weight) > float(current["weight"]):
                merged[key] = {"key": key, "label": label, "weight": float(weight), "required": required}
    total = sum(float(item["weight"]) for item in merged.values()) or 1.0
    dimensions = [
        {**item, "weight": round(float(item["weight"]) / total, 4)}
        for item in merged.values()
    ]
    return {
        "profile_id": "+".join(matched_profiles),
        "label": "、".join(_materialize_profile(item, [])["label"] for item in matched_profiles),
        "disaster_types": list(dict.fromkeys(matched_types)),
        "dimensions": dimensions,
        "structured_weights": structured_weights,
        "evidence_topics": sorted(evidence_topics),
        "metrics": list(dict.fromkeys(metrics)),
        "signal_terms": sorted(signal_terms),
        "coverage_floor": float(GENERIC_PROFILE["coverage_floor"]),
        "coverage_span": float(GENERIC_PROFILE["coverage_span"]),
        "penalty_floor": float(GENERIC_PROFILE["penalty_floor"]),
        "penalty_span": float(GENERIC_PROFILE["penalty_span"]),
    }


def dimension_weight_map(profile: dict[str, Any] | None) -> dict[str, float]:
    """返回已归一化的动态维度权重。"""
    result = {
        str(item.get("key")): float(item.get("weight") or 0.0)
        for item in (profile or {}).get("dimensions") or []
        if item.get("key")
    }
    total = sum(value for value in result.values()) or 1.0
    return {key: value / total for key, value in result.items()}


def calculate_dimension_compatibility(
    assessment: dict[str, Any] | None,
    profile: dict[str, Any] | None,
) -> tuple[float | None, dict[str, float]]:
    """仅使用有证据的维度计算兼容度，缺失指标不会被错误当成零分。"""
    assessment = assessment or {}
    scores = assessment.get("dimension_scores") or {}
    if not isinstance(scores, dict):
        scores = {}
    # 兼容旧版模型，只把旧的两项字段映射到同名动态维度。
    if not scores:
        for key in ("hazard_match", "area", "temporal", "mechanism", "intensity", "evolution"):
            value = assessment.get(key)
            if value is not None:
                scores[key] = value
        if assessment.get("mechanism_score") is not None:
            scores.setdefault("mechanism", assessment.get("mechanism_score"))
        if assessment.get("intensity_score") is not None:
            scores.setdefault("intensity", assessment.get("intensity_score"))
    weights = dimension_weight_map(profile)
    if not weights:
        # 旧调用方没有传配置时维持原机制/强度权重，保证灰度升级期间排序不变。
        weights = {"mechanism": 0.70, "intensity": 0.30}
    usable: dict[str, float] = {}
    for key, value in scores.items():
        if key not in weights:
            continue
        try:
            usable[key] = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            continue
    if not usable:
        return None, {}
    weight_sum = sum(weights[key] for key in usable)
    score = sum(weights[key] * value for key, value in usable.items()) / weight_sum
    return round(score, 4), usable


def calculate_dimension_coverage(
    assessment: dict[str, Any] | None,
    profile: dict[str, Any] | None,
) -> float:
    """计算动态维度证据覆盖度，避免缺失高权重指标的候选与完整候选等价。"""
    weights = dimension_weight_map(profile)
    if not weights:
        return 0.0
    assessment = assessment or {}
    scores = assessment.get("dimension_scores") or {}
    if not isinstance(scores, dict):
        scores = {}
    usable_keys = set()
    for key, value in scores.items():
        if key not in weights:
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            continue
        usable_keys.add(key)
    return round(sum(weights[key] for key in usable_keys), 4)


def effective_dimension_compatibility(
    assessment: dict[str, Any] | None,
    profile: dict[str, Any] | None,
) -> tuple[float | None, dict[str, float], float]:
    """将原始兼容度和证据覆盖度合成为可用于排序的保守兼容度。"""
    raw, used = calculate_dimension_compatibility(assessment, profile)
    if raw is None:
        return None, used, 0.0
    coverage = calculate_dimension_coverage(assessment, profile)
    profile = profile or {}
    floor = max(0.0, min(1.0, float(profile.get("coverage_floor", 0.45))))
    span = max(0.0, min(1.0 - floor, float(profile.get("coverage_span", 0.55))))
    factor = floor + span * coverage
    return round(raw * factor, 4), used, coverage


def legacy_dimension_scores(scores: dict[str, float]) -> tuple[float | None, float | None]:
    """给旧页面保留机制分和强度分，动态维度仍以 dimension_scores 为准。"""
    mechanism = scores.get("mechanism")
    intensity = scores.get("intensity")
    return mechanism, intensity


def all_supported_metrics() -> list[str]:
    """返回所有配置中的指标名称，供自然语言解析阶段建立受控词表。"""
    values: list[str] = []
    for profile in [GENERIC_PROFILE, *DISASTER_PROFILES.values()]:
        values.extend(str(value) for value in profile.get("metrics") or [])
    return list(dict.fromkeys(values))


def terminology_profile_context(query: dict[str, Any] | None) -> dict[str, Any]:
    """从现有灾种配置推导术语上下文，避免维护一份不断膨胀的禁用词表。"""
    query = query or {}
    requested = {
        str(value).strip()
        for value in (query.get("disaster_types") or query.get("primary_disaster_types") or [])
        if str(value).strip()
    }
    active_ids = [
        profile_id
        for profile_id, profile in DISASTER_PROFILES.items()
        if requested.intersection(profile.get("types") or ())
    ] or ["generic"]

    def profile_terms(profile: dict[str, Any]) -> set[str]:
        """指标、信号、证据主题共同构成该配置的专业词边界。"""
        return {
            str(value).strip()
            for key in ("metrics", "signal_terms", "evidence_topics")
            for value in profile.get(key) or ()
            if str(value).strip()
        }

    generic_terms = profile_terms(GENERIC_PROFILE)
    active_terms = set(generic_terms)
    selected_profile_terms: set[str] = set()
    for profile_id in active_ids:
        current_terms = profile_terms(DISASTER_PROFILES.get(profile_id, GENERIC_PROFILE))
        active_terms.update(current_terms)
        selected_profile_terms.update(current_terms)

    owners: dict[str, set[str]] = {}
    for profile_id, profile in DISASTER_PROFILES.items():
        for term in profile_terms(profile):
            owners.setdefault(term, set()).add(profile_id)
    # 只标记其他灾种专属术语；跨灾种共用词仍由证据约束，不会被误禁。
    foreign_terms = sorted(
        term
        for term, profile_ids in owners.items()
        if profile_ids.isdisjoint(active_ids) and term not in generic_terms
    )
    profile = resolve_dimension_profile(query)
    return {
        "profile_id": str(profile.get("profile_id") or "generic"),
        "label": str(profile.get("label") or "通用天气过程"),
        "active_terms": sorted(active_terms),
        # 单独保留当前灾种自身的术语，避免通用证据主题把所有物理主体都视为适用。
        "profile_terms": sorted(selected_profile_terms or generic_terms),
        "generic_terms": sorted(generic_terms),
        "foreign_terms": foreign_terms,
        "months": list(query.get("months") or []),
        "disaster_types": list(query.get("disaster_types") or []),
    }
