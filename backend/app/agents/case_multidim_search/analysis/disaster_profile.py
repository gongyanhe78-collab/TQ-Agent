"""灾种视角和强度指标的统一配置。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricProfile:
    """定义一个灾种相关指标在抽取、统计和绘图中的统一口径。"""

    metric_name: str
    unit: str
    chart_title: str
    chart_key: str
    business_focus: str


@dataclass(frozen=True)
class DisasterProfile:
    """定义灾种的业务分析视角，避免报告仍按固定模板写。"""

    key: str
    names: tuple[str, ...]
    metrics: tuple[MetricProfile, ...]
    narrative_focus: tuple[str, ...]
    chart_note: str


RAIN_METRICS = (
    MetricProfile("过程最大降水量", "mm", "代表个例过程最大降水量对比", "intensity_rain_total", "过程雨量极值"),
    MetricProfile("最大小时雨强", "mm/h", "代表个例最大小时雨强对比", "intensity_rain_hourly", "短时雨强极值"),
    MetricProfile("最大降水量", "mm", "代表个例最大降水量对比", "intensity_rain_total", "降水极值"),
)
WIND_METRICS = (
    MetricProfile("极大风速", "m/s", "代表个例极大风速对比", "intensity_wind_gust", "极大风速"),
    MetricProfile("最大风速", "m/s", "代表个例最大风速对比", "intensity_wind_max", "最大风速"),
    MetricProfile("阵风风力", "级", "代表个例阵风风力等级对比", "intensity_wind_level", "阵风等级"),
)
TEMP_METRICS = (
    MetricProfile("最高气温", "℃", "代表个例最高气温对比", "intensity_high_temperature", "最高气温"),
    MetricProfile("高温持续天数", "天", "代表个例高温持续时间对比", "intensity_high_temperature_days", "持续时间"),
)
COLD_METRICS = (
    MetricProfile("最低气温", "℃", "代表个例最低气温对比", "intensity_low_temperature", "低温极值"),
    MetricProfile("过程降温幅度", "℃", "代表个例过程降温幅度对比", "intensity_cooling", "降温幅度"),
)
SNOW_METRICS = (
    MetricProfile("最大积雪深度", "cm", "代表个例最大积雪深度对比", "intensity_snow_depth", "积雪深度"),
    MetricProfile("过程最大降雪量", "mm", "代表个例过程最大降雪量对比", "intensity_snowfall", "降雪量"),
)
CONVECTION_METRICS = RAIN_METRICS + WIND_METRICS + (
    MetricProfile("最大冰雹直径", "mm", "代表个例最大冰雹直径对比", "intensity_hail", "冰雹直径"),
    MetricProfile("雷达回波强度", "dBZ", "代表个例雷达回波强度对比", "intensity_radar", "回波强度"),
)
VISIBILITY_METRICS = (
    MetricProfile("最低能见度", "km", "代表个例最低能见度对比", "intensity_visibility", "能见度低值"),
)


DISASTER_PROFILES = (
    DisasterProfile(
        key="rain",
        names=("暴雨", "大暴雨", "强降水", "短时强降水"),
        metrics=RAIN_METRICS,
        narrative_focus=("降水时段", "累计雨量", "小时雨强", "落区移动", "短临订正", "内涝和地质灾害风险"),
        chart_note="降水类检索重点用过程雨量和小时雨强描述强度差异。",
    ),
    DisasterProfile(
        key="wind",
        names=("大风", "雷暴大风"),
        metrics=WIND_METRICS,
        narrative_focus=("极大风速", "阵风等级", "大风持续时间", "动量下传", "冷锋或对流触发", "交通和设施风险"),
        chart_note="大风类检索重点用最大风速、极大风速和阵风等级体现影响强度。",
    ),
    DisasterProfile(
        key="severe_convection",
        names=("强对流", "雷暴", "冰雹"),
        metrics=CONVECTION_METRICS,
        narrative_focus=("触发条件", "雷达回波", "短时强降水", "雷暴大风", "冰雹", "短临预警提前量"),
        chart_note="强对流检索应同时观察雨强、风速、冰雹和雷达回波，体现复合性。",
    ),
    DisasterProfile(
        key="heat",
        names=("高温",),
        metrics=TEMP_METRICS,
        narrative_focus=("最高气温", "高温持续时间", "副热带高压或大陆高压", "电力负荷", "人体健康和农业影响"),
        chart_note="高温类检索重点用最高气温和持续天数表达过程强度。",
    ),
    DisasterProfile(
        key="cold",
        names=("寒潮", "低温", "霜冻"),
        metrics=COLD_METRICS,
        narrative_focus=("最低气温", "降温幅度", "冷空气路径", "冷高压强度", "农业霜冻和能源保供"),
        chart_note="寒潮低温类检索重点用最低气温和降温幅度表达冷空气强度。",
    ),
    DisasterProfile(
        key="snow",
        names=("雨雪", "降雪", "暴雪", "雪灾", "道路结冰"),
        metrics=SNOW_METRICS + RAIN_METRICS,
        narrative_focus=("雨雪相态转换", "积雪深度", "降雪量", "道路结冰", "交通影响", "水汽和冷空气配合"),
        chart_note="雨雪暴雪类检索重点用积雪深度、降雪量和相态转换说明影响。",
    ),
    DisasterProfile(
        key="visibility",
        names=("雾", "大雾", "雾霾", "沙尘", "扬沙", "浮尘", "沙尘暴"),
        metrics=VISIBILITY_METRICS + WIND_METRICS,
        narrative_focus=("最低能见度", "持续时间", "传输路径", "起沙机制", "交通影响", "预警服务"),
        chart_note="低能见度类检索重点用最低能见度和大风条件说明影响程度。",
    ),
)


def profiles_for_disasters(disaster_names: list[str] | tuple[str, ...] | None) -> list[DisasterProfile]:
    """根据灾种标签返回需要启用的业务视角，保持原始检索顺序。"""
    text = "、".join(str(item or "") for item in (disaster_names or []))
    selected: list[DisasterProfile] = []
    seen: set[str] = set()
    for profile in DISASTER_PROFILES:
        if any(name in text for name in profile.names) and profile.key not in seen:
            selected.append(profile)
            seen.add(profile.key)
    return selected


def metric_names_for_disasters(disaster_names: list[str] | tuple[str, ...] | None) -> list[str]:
    """返回当前灾种最应关注的指标名，用于强度章节和图表筛选。"""
    names: list[str] = []
    for profile in profiles_for_disasters(disaster_names):
        for metric in profile.metrics:
            if metric.metric_name not in names:
                names.append(metric.metric_name)
    return names


def dominant_disasters_from_query_or_counts(query, aggregations: dict) -> list[str]:
    """优先使用检索条件；未指定灾种时使用命中样本中的高频灾种。"""
    query_disasters = list(getattr(query, "disaster_types", []) or [])
    if query_disasters:
        return query_disasters
    counts = aggregations.get("disaster_counts", {}) if isinstance(aggregations, dict) else {}
    return [str(name) for name in list(counts)[:3]]


def disaster_view_text(disaster_names: list[str] | tuple[str, ...] | None) -> str:
    """生成给大模型和规则报告使用的灾种分析视角说明。"""
    profiles = profiles_for_disasters(disaster_names)
    if not profiles:
        return "当前检索未指定单一主导灾种，报告应以时间、灾种共现和地市分布为主线，避免套用固定强度指标。"
    parts = []
    for profile in profiles:
        focuses = "、".join(profile.narrative_focus)
        metrics = "、".join(metric.metric_name for metric in profile.metrics[:4])
        parts.append(f"{'、'.join(profile.names[:2])}视角重点关注{focuses}；优先核查{metrics}")
    return "；".join(parts) + "。"
