"""把 LLM 的自然语言抽取结果校验为稳定的结构化匹配请求。"""
from __future__ import annotations

import re
from datetime import date as date_type, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .query_taxonomy import AREA_ALIASES, SHANXI_CITIES, STANDARD_AREA_TYPES, STANDARD_DISASTER_TYPES


SNOW_MIXED_TERMS = ("雨夹雪", "雨雪", "雨转雪", "雪转雨", "雨转雨夹雪", "雪转雨夹雪")
SNOW_ONLY_TERMS = ("纯雪", "降雪", "下雪", "转为雪", "转成雪", "转雪", "积雪", "雪深")
BUSINESS_TIMEZONE = "Asia/Shanghai"
RELATIVE_DAY_OFFSETS = {
    "前天": -2,
    "前日": -2,
    "昨天": -1,
    "昨日": -1,
    "今天": 0,
    "今日": 0,
    "明天": 1,
    "明日": 1,
    "后天": 2,
}
RELATIVE_DAY_PATTERN = "|".join(sorted(RELATIVE_DAY_OFFSETS, key=len, reverse=True))
RELATIVE_DATE_RANGE_PATTERN = re.compile(
    rf"(?P<start>{RELATIVE_DAY_PATTERN})\s*[-~～至到]\s*(?P<end>{RELATIVE_DAY_PATTERN})"
)


def normalize_natural_extraction(
    message: str,
    extracted: dict[str, Any],
    date_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """合并原文规则和受限模型结果，不允许模型覆盖用户明确表达。"""
    source = str(message or "").strip()
    result = dict(extracted or {})
    result["disaster_types"] = _normalize_disasters(source, result.get("disaster_types"))
    result["affected_areas"] = _normalize_areas(source, result.get("affected_areas"))
    # 日期必须以用户原文规则为事实源；模型只提供表达和证据，不直接决定年月日。
    # 同一次请求只读取一次业务日期，避免恰逢午夜时解析结果与审计日期跨天。
    business_date = _business_today()
    rule_dates = _normalize_dates(source, today=business_date)
    model_dates = {
        "start_date": str(result.get("start_date") or "").strip(),
        "end_date": str(result.get("end_date") or "").strip(),
        "date_expression": str(result.get("date_expression") or result.get("date") or "").strip(),
        "date_evidence": str(result.get("date_evidence") or "").strip(),
    }
    conflict = _date_resolution_conflict(model_dates, rule_dates, source)
    result.update(rule_dates)
    # 模糊日期没有可计算端点时，只保留模型识别出的原文短语，禁止模型自行落地精确日期。
    if not rule_dates["start_date"] and not rule_dates["end_date"] and model_dates["date_expression"]:
        if model_dates["date_expression"] in source:
            result["date"] = model_dates["date_expression"]
    if date_audit is not None:
        date_audit.update({
            "source": "rule",
            "business_timezone": BUSINESS_TIMEZONE,
            "business_date": business_date.isoformat(),
            "rule_result": dict(rule_dates),
            "model_result": model_dates,
            "conflict": conflict,
        })

    # 原始文本始终保留，后续关键词补充、Embedding 和审计都可以回看用户原话。
    model_extra = str(result.get("raw_query") or "").strip()
    result["raw_query"] = source if not model_extra or model_extra in source else f"{source}；{model_extra}"
    return result


def _normalize_dates(source: str, today: date_type | None = None) -> dict[str, str]:
    """确定性解析日期；原文没有年份时按服务器当前年份补齐。"""
    current = today or _business_today()
    text = str(source or "")

    # 先解析“明天到后天”这类相对日期范围，不能因先命中“明天”而丢失结束日期。
    relative_range = RELATIVE_DATE_RANGE_PATTERN.search(text)
    if relative_range:
        start_term, end_term = relative_range.group("start"), relative_range.group("end")
        start = current + timedelta(days=RELATIVE_DAY_OFFSETS[start_term])
        end = current + timedelta(days=RELATIVE_DAY_OFFSETS[end_term])
        if end >= start:
            return {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "date": relative_range.group(0),
            }
        # 逆序表达没有可靠的时间区间，不能退化成其中一个单日。
        return {"start_date": "", "end_date": "", "date": relative_range.group(0)}

    # 相对日期不交给模型猜测，避免模型知识截止时间造成错误年份。
    relative_day = next((match for match in re.finditer(RELATIVE_DAY_PATTERN, text)), None)
    if relative_day:
        term = relative_day.group(0)
        target = current + timedelta(days=RELATIVE_DAY_OFFSETS[term])
        value = target.isoformat()
        return {"start_date": value, "end_date": value, "date": term}

    explicit_year = re.search(r"(?P<year>20\d{2})\s*[年/-]\s*(?P<month>\d{1,2})\s*[月/-]\s*(?P<day>\d{1,2})", text)
    if explicit_year:
        start = _safe_date(int(explicit_year.group("year")), int(explicit_year.group("month")), int(explicit_year.group("day")))
        if start:
            end = _range_end(text, start.year, start.month, start.day) or start
            return {"start_date": start.isoformat(), "end_date": end.isoformat(), "date": explicit_year.group(0)}

    month_day = re.search(r"(?<!\d)(?P<month>\d{1,2})\s*月\s*(?P<day>\d{1,2})\s*(?:日|号)?", text)
    if month_day:
        # 用户没有给年份时，业务约定使用当前年，而不是采用模型自行补全的年份。
        start = _safe_date(current.year, int(month_day.group("month")), int(month_day.group("day")))
        if start:
            end = _range_end(text, start.year, start.month, start.day) or start
            return {"start_date": start.isoformat(), "end_date": end.isoformat(), "date": month_day.group(0)}

    month = re.search(r"(?<!\d)(?P<month>1[0-2]|0?[1-9])\s*月", text)
    if month:
        # 只有月份时不伪造具体日，只保留季节检索所需的原文日期片段。
        return {"start_date": "", "end_date": "", "date": month.group(0)}
    return {"start_date": "", "end_date": "", "date": ""}


def _business_today() -> date_type:
    """按气象业务所在的上海时区取当天，避免服务器时区导致日期边界漂移。"""
    return datetime.now(ZoneInfo(BUSINESS_TIMEZONE)).date()


def _date_resolution_conflict(
    model_dates: dict[str, str],
    rule_dates: dict[str, str],
    source: str,
) -> bool:
    """记录模型日期幻觉或精确端点冲突，缺失字段不算冲突。"""
    model_expression = model_dates.get("date_expression") or ""
    if model_expression and model_expression not in source:
        return True
    model_start = model_dates.get("start_date") or ""
    model_end = model_dates.get("end_date") or ""
    rule_start = rule_dates.get("start_date") or ""
    rule_end = rule_dates.get("end_date") or ""
    if not (model_start or model_end):
        return False
    if not (rule_start and rule_end):
        return True
    return model_start != rule_start or model_end != rule_end


def _range_end(text: str, year: int, month: int, start_day: int) -> date_type | None:
    """解析“2月21日至23日”或“2月21日-3月1日”的结束日期。"""
    cross_month = re.search(
        r"\d{1,2}\s*月\s*\d{1,2}\s*(?:日|号)?\s*[-~～至到]\s*"
        r"(?:(?P<end_year>20\d{2})\s*年\s*)?(?P<month>\d{1,2})\s*月\s*(?P<day>\d{1,2})",
        text,
    )
    if cross_month:
        end_month, end_day = int(cross_month.group("month")), int(cross_month.group("day"))
        # 12 月末到次年 1 月初必须跨年补齐，否则结束日期会被错误判为早于开始日期。
        explicit_end_year = cross_month.group("end_year")
        end_year = int(explicit_end_year) if explicit_end_year else (year + 1 if (end_month, end_day) < (month, start_day) else year)
        return _safe_date(end_year, end_month, end_day)
    same_month = re.search(
        r"\d{1,2}\s*月\s*\d{1,2}\s*(?:日|号)?\s*[-~～至到]\s*(?P<day>\d{1,2})\s*(?:日|号)?",
        text,
    )
    if same_month:
        end_day = int(same_month.group("day"))
        return _safe_date(year, month, end_day) if end_day >= start_day else None
    return None


def _safe_date(year: int, month: int, day: int) -> date_type | None:
    """非法日期返回空值，避免自然语言输入导致整个图异常。"""
    try:
        return date_type(year, month, day)
    except ValueError:
        return None


def _normalize_disasters(source: str, model_values: Any) -> list[str]:
    """先从原文提取明确灾种，再补充模型从固定枚举中选择的结果。"""
    selected: list[str] = []
    for term in sorted(STANDARD_DISASTER_TYPES, key=len, reverse=True):
        if term in source:
            _append_specific(selected, term)
    if any(term in source for term in SNOW_MIXED_TERMS):
        _append_specific(selected, "雨雪")
    if any(term in source for term in SNOW_ONLY_TERMS):
        _append_specific(selected, "降雪")

    for value in _extract_enum_values(model_values):
        if value in STANDARD_DISASTER_TYPES:
            _append_specific(selected, value)
    return selected


def _normalize_areas(source: str, model_values: Any) -> list[str]:
    """保留原始方向区域，并展开为用于评分的地级市集合。"""
    labels: list[str] = []
    if "全省" in source or "山西省" in source:
        labels.append("全省")

    # 先匹配较长区域名，避免“山西北部”又重复识别为“北部”。
    for alias in sorted(AREA_ALIASES, key=len, reverse=True):
        if alias in source and not any(alias in existing or existing in alias for existing in labels):
            labels.append(alias)
    direct_cities = [city for city in SHANXI_CITIES if city in source]

    for value in _extract_enum_values(model_values):
        if value not in STANDARD_AREA_TYPES:
            continue
        normalized = "全省" if value == "山西省" else value
        if normalized in SHANXI_CITIES:
            if normalized not in direct_cities:
                direct_cities.append(normalized)
        elif normalized not in labels and not any(normalized in item or item in normalized for item in labels):
            labels.append(normalized)

    city_set = set(direct_cities)
    for label in labels:
        if label == "全省":
            city_set.update(SHANXI_CITIES)
        else:
            city_set.update(AREA_ALIASES.get(label, ()))

    province_wide = set(SHANXI_CITIES).issubset(city_set)
    output_labels = (["全省"] if province_wide else []) + [label for label in labels if label != "全省"]
    return _dedupe(output_labels + [city for city in SHANXI_CITIES if city in city_set])


def _extract_enum_values(value: Any) -> list[str]:
    """兼容模型返回字符串数组或带 value/evidence 的对象数组。"""
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        raw = item.get("value") if isinstance(item, dict) else item
        text = str(raw or "").strip()
        if text:
            result.append(text)
    return result


def _append_specific(values: list[str], value: str) -> None:
    """具体灾种已存在时，不再加入其名称中的宽泛子标签。"""
    if value in values or any(value in selected for selected in values):
        return
    values.append(value)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
