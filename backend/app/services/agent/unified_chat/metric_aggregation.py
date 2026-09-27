"""从文档证据中提取可比较的气象指标事实。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricSpec:
    """描述一个可确定性聚合的指标及其常见文本表达。"""

    name: str
    aliases: tuple[str, ...]
    units: tuple[str, ...]


METRIC_SPECS = {
    "air_temperature": MetricSpec(
        "气温", ("最低气温", "最高气温", "最低温度", "最高温度", "气温"), ("℃", "°C"),
    ),
    "precipitation": MetricSpec(
        "降水量", ("累计降水量", "过程降水量", "降水量", "雨量"), ("毫米", "mm"),
    ),
    "wind_speed": MetricSpec(
        "风速", ("最大风速", "极大风速", "极大风", "阵风风速", "风速"), ("m/s", "米/秒"),
    ),
    "visibility": MetricSpec(
        "能见度", ("最低能见度", "最小水平能见度", "能见度"), ("千米", "公里", "km", "米", "m"),
    ),
}

METRIC_ALIASES = {
    "air_temperature.minimum": ("air_temperature", "min", ("最低气温", "最低温度")),
    "air_temperature.maximum": ("air_temperature", "max", ("最高气温", "最高温度")),
    "precipitation.maximum": ("precipitation", "max", METRIC_SPECS["precipitation"].aliases),
    "wind_speed.maximum": ("wind_speed", "max", METRIC_SPECS["wind_speed"].aliases),
    "visibility.minimum": ("visibility", "min", ("最低能见度", "最小水平能见度")),
}


def aggregate_metric_facts(
    chunks: list[Any],
    *,
    metric: str,
    operator: str,
    chunk_case_ids: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """抽取带出处的数值事实并执行最小值或最大值聚合。"""
    canonical, default_operator, aliases = METRIC_ALIASES.get(
        str(metric or "").strip(),
        (str(metric or "").split(".", 1)[0], str(operator or "").lower(), ()),
    )
    spec = METRIC_SPECS.get(canonical)
    operation = str(operator or default_operator or "").lower()
    if spec is None or operation not in {"min", "max"}:
        return {"status": "unsupported", "metric": metric, "operator": operation, "facts": []}
    active_aliases = tuple(aliases or spec.aliases)
    facts: list[dict[str, Any]] = []
    for chunk in chunks:
        facts.extend(_facts_from_chunk(chunk, spec, active_aliases, chunk_case_ids or {}))
    if not facts:
        return {"status": "no_evidence", "metric": metric, "operator": operation, "facts": []}
    winner = (min if operation == "min" else max)(facts, key=lambda item: item["normalized_value"])
    selected_case_ids = list((chunk_case_ids or {}).get(str(winner["chunk_id"]), []))
    return {
        "status": "ok",
        "metric": metric,
        "metric_name": spec.name,
        "operator": operation,
        "winner": winner,
        "facts": sorted(facts, key=lambda item: item["normalized_value"], reverse=operation == "max")[:50],
        "selected_case_ids": selected_case_ids,
    }


def metric_result_text(result: dict[str, Any]) -> str:
    """把结构化极值结果转换为最终回答模型可引用的证据文本。"""
    if result.get("status") != "ok":
        return "没有从指定范围的文档证据中抽取到满足指标契约的数值事实。"
    winner = dict(result.get("winner") or {})
    direction = "最低值" if result.get("operator") == "min" else "最高值"
    rows = [
        "指标聚合结果（由程序对同口径数值进行确定性比较）：",
        f"- 指标：{result.get('metric_name') or result.get('metric')}",
        f"- {direction}：{winner.get('value_text')}",
        f"- 时间：{winner.get('time') or '证据片段未明确'}",
        f"- 地点：{winner.get('location') or '证据片段未明确'}",
        f"- 来源：{winner.get('source_pdf')}；证据片段：{winner.get('chunk_id')}",
    ]
    return "\n".join(rows)


def _facts_from_chunk(
    chunk: Any,
    spec: MetricSpec,
    aliases: tuple[str, ...],
    chunk_case_ids: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """只在包含指标语义的完整分句中提取数值，避免跨段串联。"""
    content = str(getattr(chunk, "content", "") or "")
    # PDF 正文中的换行通常只是版面折行，不能把指标名称与下一行数值拆成两句。
    content = re.sub(r"[ \t]*\n[ \t]*", "", content)
    unit_pattern = "|".join(re.escape(unit) for unit in sorted(spec.units, key=len, reverse=True))
    value_pattern = re.compile(rf"(-?\d+(?:\.\d+)?)\s*({unit_pattern})(?:[（(]([^）)]{{1,30}})[）)])?", re.I)
    facts: list[dict[str, Any]] = []
    for sentence in re.split(r"(?<=[。；;])", content):
        if not any(alias in sentence for alias in aliases):
            continue
        # “风速核/急流”描述的是高空环流，不与地面站最大阵风实况混合比较。
        if (
            spec.name == "风速"
            and any(term in sentence for term in ("风速核", "高空急流", "低空急流"))
            and not any(term in sentence for term in ("实况", "国家站", "区域站", "观测站", "阵风"))
        ):
            continue
        for match in value_pattern.finditer(sentence):
            value = float(match.group(1))
            unit = match.group(2)
            if _is_non_surface_value(spec, sentence, match.start()):
                continue
            normalized_value, normalized_unit = _normalize_value(value, unit, spec)
            time_text = (
                _time_near_value(
                    sentence,
                    match.start(),
                    content,
                    str(getattr(chunk, "source_pdf", "") or ""),
                )
                or _time_text(sentence)
                or _time_text(content)
            )
            location = str(match.group(3) or "").strip()
            if location and re.search(r"\d|级|%|～|~", location):
                # “（13级）”等强度注记不是地点，继续检查数值后的“出现在”表达。
                location = ""
            if not location:
                location = _nearby_location(sentence, match.start(), match.end())
            facts.append({
                "value": value,
                "unit": unit,
                "normalized_value": normalized_value,
                "normalized_unit": normalized_unit,
                "value_text": f"{value:g}{unit}",
                "time": time_text,
                "location": location,
                "chunk_id": str(getattr(chunk, "chunk_id", "") or ""),
                "source_pdf": str(getattr(chunk, "source_pdf", "") or ""),
                "case_ids": list(chunk_case_ids.get(str(getattr(chunk, "chunk_id", "") or ""), [])),
                "evidence": sentence.strip()[:500],
            })
    return facts


def _normalize_value(value: float, unit: str, spec: MetricSpec) -> tuple[float, str]:
    """统一可换算单位，保证数值比较使用同一量纲。"""
    lowered = unit.lower()
    if spec.name == "能见度" and lowered in {"千米", "公里", "km"}:
        return value * 1000.0, "米"
    if spec.name == "降水量":
        return value, "毫米"
    if spec.name == "风速":
        return value, "m/s"
    return value, "℃"


def _is_non_surface_value(spec: MetricSpec, sentence: str, value_start: int) -> bool:
    """排除高空环流数值，确保气温和风速极值使用地面观测口径。"""
    nearby = sentence[max(0, value_start - 65):value_start + 18]
    if spec.name == "气温" and re.search(
        r"(?:\d{3,4}\s*hPa|冷涡|冷中心|暖中心|等温线|高空|对流层)",
        nearby,
        re.I,
    ):
        return True
    if spec.name == "风速" and re.search(r"(?:风速核|高空急流|低空急流)", nearby):
        return True
    return False


def _time_text(text: str) -> str:
    """保留证据附近最具体的中文时间表达。"""
    # PDF 抽取经常在数字与“年/月/日”之间插入空格，先归一化再匹配日期。
    text = re.sub(r"\s+", "", text)
    month = r"(?:1[0-2]|0?[1-9])"
    day = r"(?:3[01]|[12]\d|0?[1-9])"
    clock = r"(?:2[0-3]|[01]?\d)(?:时(?:[0-5]?\d分)?|[:：][0-5]\d)"
    year = r"(?:20\d{2}年)?"
    patterns = (
        rf"{year}{month}月{day}日{clock}(?:至|到|[-~～—])(?:{month}月)?{day}日{clock}",
        rf"{year}{month}月{day}日{clock}",
        rf"{year}{month}月{day}日?(?:至|到|[-~～—])(?:{month}月)?{day}日",
        rf"{year}{month}月{day}日",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).replace(" ", "")
    return ""


def _time_near_value(sentence: str, value_start: int, content: str, source_pdf: str = "") -> str:
    """把逐日序列中数值前最近的日号与 chunk 年月上下文组合。"""
    before = re.sub(r"\s+", "", sentence[max(0, value_start - 50):value_start])
    if re.search(
        r"(?:1[0-2]|0?[1-9])月(?:3[01]|[12]\d|0?[1-9])日?"
        r"(?:\d{1,2}(?:时|[:：]\d{2}))?(?:至|到|[-~～—])"
        r"(?:(?:1[0-2]|0?[1-9])月)?(?:3[01]|[12]\d|0?[1-9])日",
        before,
    ):
        # 数值描述的是整个过程范围时保留范围，不误取末日作为唯一发生日。
        return ""
    matches = list(re.finditer(
        r"(?:(20\d{2})年)?(?:(1[0-2]|0?[1-9])月)?"
        r"(3[01]|[12]\d|0?[1-9])日"
        r"((?:2[0-3]|[01]?\d)(?:时(?:[0-5]?\d分)?|[:：][0-5]\d))?",
        before,
    ))
    if not matches:
        return ""
    local = matches[-1]
    year = local.group(1) or ""
    month = local.group(2) or ""
    if not month:
        context_time = _time_text(content)
        context_match = re.search(r"(?:(20\d{2})年)?(1[0-2]|0?[1-9])月", context_time)
        if context_match:
            year = year or str(context_match.group(1) or "")
            month = str(context_match.group(2) or "")
    if not month:
        # 相邻 chunk 省略年月时，规范来源名仍能提供不含业务推测的年月上下文。
        source_match = re.search(
            r"(20\d{2})(?:年|[-_])\s*(1[0-2]|0?[1-9])(?:月|\D|$)",
            str(source_pdf or ""),
        )
        if source_match:
            year = year or source_match.group(1)
            month = source_match.group(2)
    if not month:
        return ""
    clock = str(local.group(4) or "")
    return f"{year + '年' if year else ''}{int(month)}月{int(local.group(3))}日{clock}"


def _nearby_location(sentence: str, value_start: int, value_end: int) -> str:
    """从数值附近的“地点+数值”或“数值+地点”结构中提取短地点名。"""
    before = sentence[max(0, value_start - 35):value_start]
    match = re.search(
        r"(?:为|达|出现在|分别为)[:：]?\s*([\u4e00-\u9fff]{2,12})(?:\s*[（(])?\s*$",
        before,
    )
    if match:
        return match.group(1)
    after = sentence[value_end:value_end + 80]
    # 实况材料常使用“数值，出现在某观测站”的后置地点表达。
    match = re.search(
        r"(?:\s*[（(][^）)]{1,20}[）)])?\s*(?:，|,)?\s*(?:最高值)?"
        r"(?:出现在|位于)\s*([\u4e00-\u9fff]{2,24})",
        after,
    )
    return match.group(1) if match else ""
