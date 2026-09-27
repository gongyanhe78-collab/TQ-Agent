"""把检索事实转化为可追溯的报告分析结论。"""
from __future__ import annotations

import re
import logging
import math
from collections import Counter
from statistics import mean

from backend.app.models import StandardCase
from backend.app.services.agent.case_multidim_search.retrieval.case_evidence import CaseEvidenceBundle
from backend.app.services.agent.case_multidim_search.analysis.disaster_profile import (
    DISASTER_PROFILES,
    disaster_view_text,
    metric_names_for_disasters,
)
from backend.app.services.agent.case_multidim_search.schemas import CaseSearchQuery, IntensityMetric
from backend.app.services.agent.case_multidim_search.retrieval.structured_retriever import COMPOUND_DISASTER_TERMS


LOGGER = logging.getLogger("uvicorn.error")


class ReportAnalyzer:
    """基于确定性统计、灾种共现和个例证据生成报告级研判。"""

    # 46℃左右的高温在历史极端过程中可能真实出现，超过 55℃ 才按明显异常处理。
    SUSPICIOUS_MAX_TEMPERATURE = 55.0

    def analyze(
        self,
        cases: list[StandardCase],
        aggregations: dict,
        intensity_metrics: dict[str, list[IntensityMetric]],
        image_count: int,
        query: CaseSearchQuery | None = None,
        displayed_case_count: int | None = None,
        evidence_by_case: dict[str, CaseEvidenceBundle] | None = None,
        case_text_by_case: dict[str, str] | None = None,
    ) -> dict:
        """生成查询感知的摘要、分维度研判、结论和建议。"""
        if not cases:
            return self._empty_analysis()
        evidence_by_case = evidence_by_case or {}
        case_text_by_case = case_text_by_case or {}
        case_count = len(cases)
        displayed_count = case_count if displayed_case_count is None else displayed_case_count
        focus_disasters = self._focus_disasters(query)
        view_disasters = list(getattr(query, "disaster_types", []) or focus_disasters)
        temporal_text, temporal_finding = self._temporal_analysis(
            aggregations.get("month_counts", {}), case_count
        )
        disaster_text, disaster_finding = self._disaster_analysis(
            cases,
            aggregations.get("disaster_counts", {}),
            focus_disasters,
        )
        spatial_text, spatial_finding = self._spatial_analysis(
            aggregations.get("city_counts", {}), case_count
        )
        intensity_text, intensity_finding = self._intensity_analysis(intensity_metrics, view_disasters)
        overview_tables = self._overview_intensity_tables(
            cases, intensity_metrics, query, view_disasters, case_text_by_case
        )
        overview_columns = overview_tables[0]["columns"] if overview_tables else self._overview_intensity_columns(intensity_metrics, view_disasters)[:3]
        overview_table = [row for table in overview_tables for row in table.get("rows", [])]
        overview_intensity_finding = self._overview_intensity_sentence_from_tables(overview_tables)
        if overview_intensity_finding:
            # 总体概况优先采用逐个例 chunk 抽取后的极值汇总，避免只引用某一个弱过程指标。
            intensity_finding = overview_intensity_finding
        evidence_text, evidence_finding = self._evidence_analysis(cases, image_count, evidence_by_case)
        representative_cases = self._representative_cases(cases, intensity_metrics, evidence_by_case)
        recommendations = self._recommendations(cases, intensity_metrics, view_disasters)
        key_findings = [
            temporal_finding,
            disaster_finding,
            spatial_finding,
            intensity_finding,
        ]
        key_findings = [item for item in key_findings if item]
        focus_text = "、".join(view_disasters) if view_disasters else "当前灾害过程"
        strength_assessments = self._query_strength_assessments(query, intensity_metrics)
        companion_names = [
            name for name, _count in Counter(aggregations.get("disaster_counts", {})).most_common()
            if not any(name == term or name in term or term in name for term in view_disasters)
        ][:4]
        companion_text = "、".join(companion_names) if companion_names else "其他伴随灾种"
        assessment_text = "".join(strength_assessments)
        executive_summary = (
            f"本次检索围绕{focus_text}获得 {case_count} 个有效个例，并对其中 {displayed_count} 个过程展开逐例分析。"
            "检索条件用于定位相关过程，不等同于每个命中个例均达到相应灾种的强度标准。"
            f"从现有材料看，过程风险通常由主导天气与{companion_text}共同构成，影响落区还受到山西山地、盆地及南北差异的调制。"
            f"{assessment_text}总体研判应以逐例实况强度、发生时段和影响区域为依据，避免仅凭标签扩大灾种等级。"
        )
        conclusion = (
            f"综合研判，本次命中材料反映的核心特征不是{focus_text}单独出现，而是主导天气与{companion_text}在不同区域、不同阶段相互叠加。"
            "山西山地与盆地相间的地形使降水、风速、温度和能见度的局地差异明显，同一过程内的风险重点也会随天气系统演变而转换。"
            "\n\n"
            "现有复盘对过程实况的描述相对充分，但强对流触发、短时强降水落区偏差、冷空气动量下传及雨雪相态转换等关键环节的诊断深度仍不均衡。"
            "部分个例尚未把雷达回波、站点极值、模式预报与预警发布时间串联起来，因而难以准确判断预警提前量、落区偏差来源和服务响应效果。"
            "\n\n"
            "这些材料能够支撑对主要天气过程和重点影响区域的判断，也表明吕梁山、五台山、太行山迎风坡、晋中盆地及城市敏感区是业务误差容易被放大的场景。"
            "当前最突出的业务短板，是局地突发性风险的识别证据尚未形成统一、可量化、可连续复盘的闭环。"
        )
        return {
            "executive_summary": executive_summary,
            "analysis_case_count": case_count,
            "displayed_case_count": displayed_count,
            "key_findings": key_findings[:6],
            "disaster_view": disaster_view_text(view_disasters),
            "sections": {
                "temporal": temporal_text,
                "disaster": disaster_text,
                "spatial": spatial_text,
                "intensity": intensity_text,
            },
            "representative_cases": representative_cases,
            "overview_intensity_tables": overview_tables,
            "overview_intensity_columns": overview_columns,
            "overview_intensity_table": overview_table,
            "intensity_extremes": overview_intensity_finding,
            "query_strength_assessments": strength_assessments,
            "recommendations": recommendations,
            "conclusion": conclusion,
            "limitations": self._limitations(cases, intensity_metrics, image_count),
        }

    def _temporal_analysis(self, counts: dict, case_count: int) -> tuple[str, str]:
        """解释样本覆盖月份、季节集中性和峰值月份。"""
        if not counts:
            text = "现有材料缺少可解析月份，无法判断过程的季节集中性。"
            return text, text
        ordered = sorted(
            ((int(str(name).rstrip("月")), int(count)) for name, count in counts.items()),
            key=lambda item: item[0],
        )
        months = [month for month, _ in ordered]
        season = self._season_label(months)
        if len(months) == 1:
            span_text = f"{months[0]}月"
        else:
            span_text = f"{months[0]}—{months[-1]}月"
        top_count = max(count for _, count in ordered)
        top_months = [f"{month}月" for month, count in ordered if count == top_count]
        if len(top_months) == len(ordered):
            concentration = "各月样本数量接近，未出现单一月份明显占优"
        else:
            share = top_count / max(case_count, 1) * 100
            concentration = f"{'、'.join(top_months)}样本最多，占全部个例的 {share:.1f}%"
        text = (
            f"时间演变上，命中过程覆盖{span_text}，整体集中于{season}；{concentration}。"
            "这种分布说明当前材料中的过程具有明确时段集中性，后续对比应优先选择相同时段的个例。"
        )
        finding = f"时间证据表明过程集中在{season}的{span_text}，{concentration}。"
        return text, finding

    def _season_label(self, months: list[int]) -> str:
        """根据月份集合归纳季节时段而不外推气候规律。"""
        values = set(months)
        if values and values.issubset({4, 5, 6, 7, 8, 9}):
            return "暖季"
        if values and values.issubset({11, 12, 1, 2}):
            return "冬季"
        if values and values.issubset({3, 4, 5}):
            return "春季"
        if values and values.issubset({9, 10, 11}):
            return "秋季"
        return "跨季节时段"

    def _disaster_analysis(
        self,
        cases: list[StandardCase],
        counts: dict,
        focus_disasters: list[str],
    ) -> tuple[str, str]:
        """用业务化语言分析核心灾种和伴随灾种关系，避免输出数据库分母式表达。"""
        case_count = len(cases)
        if not counts:
            text = "现有材料缺少灾种标签，无法判断灾种共现关系。"
            return text, text
        prevalence = Counter(counts)
        core_text = "、".join(focus_disasters)
        if focus_disasters:
            core_count = sum(
                all(any(term == value or term in value for value in case.disaster_types) for term in focus_disasters)
                for case in cases
            )
            if len(focus_disasters) == 1:
                first_sentence = f"{core_text}是本次检索的主分析对象，命中个例可围绕其发生时段、强度证据和伴随灾种展开复盘。"
            elif core_count == 0:
                first_sentence = f"本次多灾种检索未见{core_text}在同一过程完整叠加，但不同个例分别覆盖其中一种或多种灾种，说明样本更适合按主导灾种分组比较。"
            elif core_count == case_count:
                first_sentence = f"命中样本均呈现{core_text}共同出现的特征，复合天气过程较为集中，适合重点分析多要素同步增强机制。"
            else:
                first_sentence = f"{core_text}在部分过程中共同出现，其余个例表现为单灾种主导或局地伴随，反映出本次样本具有明显的多路径触发和局地差异。"
        else:
            ordered_core = prevalence.most_common(2)
            core_text = "、".join(name for name, _ in ordered_core)
            first_sentence = f"高频灾种主要集中在{core_text}，材料呈现多灾种共现而非单一灾种孤立发展的特征。"
        companions = [(name, count) for name, count in prevalence.most_common() if name not in focus_disasters][:4]
        if companions:
            top_names = "、".join(name for name, _ in companions[:3])
            tail = "等" if len(companions) > 3 else ""
            second_sentence = f"伴随灾种中，{top_names}{tail}出现较为突出，提示过程复盘不能只看检索勾选灾种本身，还要同步核对大风、短时强降水、低能见度或雨雪低温等伴随风险是否改变服务重点。"
        else:
            second_sentence = "当前样本未形成稳定的其他伴随灾种，复合关系仍需更多材料验证。"
        third_sentence = "业务上建议按“主导灾种—伴随灾种—影响区域—强度证据”的顺序拆解，先判断每个个例的主风险，再评估多灾种叠加是否会放大交通、农业、电力或城市运行影响。"
        text = f"灾种关系上，{first_sentence}{second_sentence}{third_sentence}"
        view_text = disaster_view_text(focus_disasters)
        if view_text:
            text += " 本次报告的业务视角为：" + view_text
        finding = f"灾种关系显示，{first_sentence}{second_sentence}"
        return text, finding

    def _spatial_analysis(self, counts: dict, case_count: int) -> tuple[str, str]:
        """解释高频影响区域、覆盖范围和省级描述边界。"""
        if not counts:
            text = "现有材料既缺少地市标签，也没有可回退的影响区域，无法形成可靠空间判断。"
            return text, text
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        top_count = ordered[0][1]
        top_names = [name for name, count in ordered if count == top_count][:3]
        other_names = [name for name, _ in ordered if name not in top_names][:3]
        coverage = len(ordered)
        top_text = "、".join(top_names)
        if other_names:
            extension = f"，并延伸至{'、'.join(other_names)}等区域"
        else:
            extension = ""
        text = (
            f"空间影响涉及 {coverage} 个地市或区域，其中{top_text}在 {top_count} 个命中个例中反复出现"
            f"{extension}。这里统计的是地市被个例涉及的次数，同一个例可以同时计入多个地市，"
            "因此只用于识别高频影响区，不能解释为互斥占比或各地受灾强度。"
        )
        finding = f"空间影响以{top_text}出现最为频繁，并延伸至多个山地、盆地和城市区域。"
        return text, finding

    def _intensity_analysis(self, metrics_by_case: dict[str, list[IntensityMetric]], disaster_names: list[str] | None = None) -> tuple[str, str]:
        """用业务化语言解释强度极值，避免写成数据库统计日志。"""
        grouped: dict[tuple[str, str], list[tuple[str, float]]] = {}
        for case_id, metrics in metrics_by_case.items():
            for metric in metrics:
                grouped.setdefault((metric.metric_name, metric.unit), []).append((case_id, metric.value))
        desired_metrics = metric_names_for_disasters(disaster_names)
        if desired_metrics:
            ordered_grouped = {key: grouped[key] for wanted in desired_metrics for key in grouped if key[0] == wanted}
            if ordered_grouped:
                grouped = ordered_grouped
        if not grouped:
            text = ('现有材料尚未抽取到可核验的关键强度指标，报告不对过程强弱作排序；' + '后续应优先回看原文图表和站点实况，补齐降水、风速、温度等核心记录。')
            return text, text
        parts = []
        for (name, unit), records in grouped.items():
            prefer_min = '最低' in name or '能见度' in name
            case_id, value = (min(records, key=lambda item: item[1]) if prefer_min else max(records, key=lambda item: item[1]))
            verb = '低至' if prefer_min else '达'
            parts.append(name + verb + f"{value:g}" + unit)
            if len(parts) >= 5:
                break
        text = ('各过程的实况强度表明，' + "，".join(parts) + '。相关数值均可在对应天气过程的正式业务材料中核验；材料未明确记录的指标在汇总表中保留为“—”，不作推断。')
        finding = '强度证据显示，' + "，".join(parts[:3]) + "。"
        return text, finding


    OVERVIEW_METRIC_COLUMNS = (
        {"key": "max_hourly_precip", "label": '最大小时雨强', "metric_names": ('最大小时雨强',), "prefer_min": False},
        {"key": "max_precip", "label": '过程最大降水量', "metric_names": ('过程最大降水量', '最大降水量'), "prefer_min": False},
        {"key": "max_snow_depth", "label": '最大积雪深度', "metric_names": ('最大积雪深度',), "prefer_min": False},
        {"key": "max_snowfall", "label": '过程最大降雪量', "metric_names": ('过程最大降雪量',), "prefer_min": False},
        {"key": "max_wind", "label": '最大风速', "metric_names": ('极大风速', '最大风速'), "prefer_min": False},
        {"key": "max_wind_level", "label": '阵风等级', "metric_names": ('阵风风力',), "prefer_min": False},
        {"key": "wind_impact_scope", "label": '8级以上大风范围', "metric_names": (), "prefer_min": False, "text_metric": "wind_impact_scope", "non_numeric": True},
        {"key": "max_temperature", "label": '最高气温', "metric_names": ('最高气温',), "prefer_min": False},
        {"key": "hot_days", "label": '高温持续天数', "metric_names": ('高温持续天数',), "prefer_min": False},
        {"key": "min_temperature", "label": '最低气温', "metric_names": ('最低气温',), "prefer_min": True},
        {"key": "max_temp_drop", "label": '过程降温幅度', "metric_names": ('过程降温幅度',), "prefer_min": False},
        {"key": "max_hail_diameter", "label": '最大冰雹直径', "metric_names": ('最大冰雹直径',), "prefer_min": False},
        {"key": "max_radar_reflectivity", "label": "最大雷达回波强度", "metric_names": ("雷达回波强度",), "prefer_min": False},
        {"key": "min_visibility", "label": '最低能见度', "metric_names": ('最低能见度',), "prefer_min": True},
    )

    OVERVIEW_TABLE_GROUPS = (
        {
            "key": "wind",
            "title": "大风类",
            "terms": ("雷暴大风", "大风"),
            # 大风独立成表，避免和寒潮、雨雪过程竞争温度、积雪等不适用字段。
            "metric_keys": ("max_wind", "max_wind_level", "wind_impact_scope"),
        },
        {
            "key": "convection_hail",
            "title": "强对流、雷暴、冰雹类",
            "terms": ("强对流", "雷暴", "冰雹"),
            "metric_keys": ("max_hourly_precip", "max_wind", "max_wind_level", "max_hail_diameter", "max_radar_reflectivity"),
        },
        {
            "key": "rain",
            "title": "暴雨、大暴雨、强降水、短时强降水类",
            "terms": ("暴雨", "大暴雨", "强降水", "短时强降水"),
            "metric_keys": ("max_hourly_precip", "max_precip", "max_wind", "max_wind_level", "max_radar_reflectivity"),
        },
        {
            "key": "cold_wave",
            "title": "寒潮、低温类",
            "terms": ("寒潮", "低温"),
            "metric_keys": ("min_temperature", "max_temp_drop", "max_wind"),
        },
        {
            "key": "rain_snow",
            "title": "雨雪类",
            "terms": ("雨雪",),
            "metric_keys": ("max_precip", "max_snowfall", "max_snow_depth", "min_temperature"),
        },
        {
            "key": "snowfall",
            "title": "降雪类",
            "terms": ("降雪",),
            "metric_keys": ("max_snowfall", "max_snow_depth", "max_precip", "min_temperature"),
        },
        {
            "key": "blizzard",
            "title": "暴雪类",
            "terms": ("暴雪", "雪灾"),
            "metric_keys": ("max_snow_depth", "max_snowfall", "max_precip", "min_temperature", "max_temp_drop"),
        },
        {
            "key": "heat",
            "title": "高温类",
            "terms": ("高温",),
            "metric_keys": ("max_temperature", "hot_days", "min_temperature"),
        },
        {
            "key": "dust",
            "title": "沙尘类",
            "terms": ("沙尘",),
            "metric_keys": ("min_visibility", "max_wind", "max_wind_level"),
        },
        {
            "key": "fog",
            "title": "雾类",
            "terms": ("雾",),
            "metric_keys": ("min_visibility", "min_temperature", "max_wind"),
        },
        {
            "key": "frost",
            "title": "霜冻类",
            "terms": ("霜冻",),
            "metric_keys": ("min_temperature", "max_temp_drop", "min_visibility"),
        },
    )

    def _overview_intensity_columns(
        self,
        metrics_by_case: dict[str, list[IntensityMetric]],
        view_disasters: list[str],
    ) -> list[dict]:
        """按检索灾种和实际抽取指标动态返回强度列，缺测项由表格单元格显示破折号。"""
        desired = set(metric_names_for_disasters(view_disasters))
        available = {
            metric.metric_name
            for metrics in metrics_by_case.values()
            for metric in metrics
            if isinstance(metric.value, (int, float)) and not self._metric_needs_review(metric)
        }
        selected: list[dict] = []
        for column in self.OVERVIEW_METRIC_COLUMNS:
            names = set(column["metric_names"])
            if names.intersection(desired) or names.intersection(available):
                selected.append(dict(column))
        return selected or [dict(column) for column in self.OVERVIEW_METRIC_COLUMNS[:4]]

    def _overview_intensity_table(
        self,
        cases: list[StandardCase],
        metrics_by_case: dict[str, list[IntensityMetric]],
        columns: list[dict],
        query: CaseSearchQuery | None = None,
        primary_by_case: dict[str, str] | None = None,
        case_text_by_case: dict[str, str] | None = None,
    ) -> list[dict]:
        """生成放在总体概况后的逐例强度汇总表，列随灾种和实际指标动态变化。"""
        rows: list[dict] = []
        for case in cases:
            metrics = metrics_by_case.get(case.case_id, [])
            primary_disaster = (primary_by_case or {}).get(case.case_id) or self._primary_disaster_text(case, query)
            metric_cells: dict[str, str] = {}
            metric_values: dict[str, float | None] = {}
            for column in columns:
                if column.get("text_metric") == "wind_impact_scope":
                    # 大风过程第三项使用实况覆盖范围，不能借用伴随沙尘的最低能见度。
                    metric_cells[column["key"]] = self._wind_impact_scope(
                        (case_text_by_case or {}).get(case.case_id, "")
                    )
                    metric_values[column["key"]] = None
                    continue
                metric = self._case_metric_extreme(
                    metrics,
                    set(column["metric_names"]),
                    prefer_min=bool(column.get("prefer_min")),
                )
                metric_cells[column["key"]] = self._format_metric_cell(metric)
                metric_values[column["key"]] = metric.value if metric and not self._metric_needs_review(metric) else None
            valid_count = sum(1 for value in metric_cells.values() if value and value != '—')
            row = {
                "case_id": case.case_id,
                "case": self._case_short_name(case),
                "title": case.title,
                "main_disasters": primary_disaster,
                "concurrent_disasters": self._concurrent_disaster_text(case, primary_disaster),
                "metrics": metric_cells,
                "metric_values": metric_values,
                "main_areas": self._short_area_text(
                    case, metrics, (case_text_by_case or {}).get(case.case_id, "")
                ),
                "data_completeness": self._data_completeness_text(valid_count, len(columns)),
                "max_hourly_precip": metric_cells.get("max_hourly_precip", '—'),
                "max_hourly_precip_value": metric_values.get("max_hourly_precip"),
                "max_wind": metric_cells.get("max_wind", '—'),
                "max_wind_value": metric_values.get("max_wind"),
                "max_temperature": metric_cells.get("max_temperature", '—'),
                "max_temperature_value": metric_values.get("max_temperature"),
            }
            rows.append(row)
        return rows

    def _concurrent_disaster_text(self, case: StandardCase, primary_disaster: str) -> str:
        """列出主导灾种以外的并发灾害，避免把同义的主导标签重复写入表格。"""
        primary_terms = [item.strip() for item in str(primary_disaster or "").split("、") if item.strip()]
        values: list[str] = []
        for raw_value in case.disaster_types or []:
            value = str(raw_value or "").strip()
            if not value:
                continue
            if any(value == primary or value in primary or primary in value for primary in primary_terms):
                continue
            if value not in values:
                values.append(value)
        return "、".join(values) or "—"


    def _overview_intensity_tables(
        self,
        cases: list[StandardCase],
        metrics_by_case: dict[str, list[IntensityMetric]],
        query: CaseSearchQuery | None,
        view_disasters: list[str],
        case_text_by_case: dict[str, str] | None = None,
    ) -> list[dict]:
        """按主导灾种拆分重点个例强度表，每张表固定三列基础信息加最多三列业务指标。"""
        if not cases:
            return []
        column_map = self._overview_metric_column_map()
        fallback_columns = self._overview_intensity_columns(metrics_by_case, view_disasters)[:3]
        grouped: dict[str, dict] = {}
        primary_by_case: dict[str, str] = {}
        fallback_group = {
            "key": "other",
            "title": "其他灾种类",
            "terms": (),
            "metric_keys": tuple(column["key"] for column in fallback_columns),
        }
        for case in cases:
            primary = self._primary_disaster_text(case, query)
            primary_by_case[case.case_id] = primary
            group = self._overview_group_for_case(case, primary) or fallback_group
            bucket = grouped.setdefault(group["key"], {"group": group, "cases": []})
            bucket["cases"].append(case)

        tables: list[dict] = []
        group_order = [group["key"] for group in self.OVERVIEW_TABLE_GROUPS] + ["other"]
        for group_key in group_order:
            bucket = grouped.get(group_key)
            if not bucket:
                continue
            group = bucket["group"]
            group_cases = bucket["cases"]
            columns = self._overview_columns_for_group(
                group, group_cases, metrics_by_case, column_map, fallback_columns, case_text_by_case
            )
            if not columns:
                continue
            rows = self._overview_intensity_table(
                group_cases,
                metrics_by_case,
                columns,
                query,
                primary_by_case=primary_by_case,
                case_text_by_case=case_text_by_case,
            )
            if rows:
                tables.append({
                    "key": group["key"],
                    "title": f"{group['title']}强度汇总",
                    # columns 仅存动态三列，PDF/前端统一在其前面追加固定的个例、主导灾种、并发灾害三列。
                    "columns": columns[:3],
                    "rows": rows,
                })
        return tables

    def _overview_metric_column_map(self) -> dict[str, dict]:
        """把强度指标列定义转成 key 映射，便于不同灾种组复用同一套字段口径。"""
        return {str(column["key"]): dict(column) for column in self.OVERVIEW_METRIC_COLUMNS}

    def _overview_group_for_case(self, case: StandardCase, primary_disaster: str) -> dict | None:
        """先按主导灾种分组；兜底时再看个例并发灾种，确保不限灾种检索也能归类。"""
        for disaster in [item.strip() for item in str(primary_disaster or "").split("、") if item.strip()]:
            group = self._overview_group_for_disaster(disaster)
            if group:
                return group
        for term in case.disaster_types or []:
            group = self._overview_group_for_disaster(str(term or ""))
            if group:
                return group
        return None

    def _overview_group_for_disaster(self, disaster: str) -> dict | None:
        """把灾种映射到用户要求的强度汇总表类别。"""
        name = str(disaster or "").strip()
        if not name:
            return None
        for group in self.OVERVIEW_TABLE_GROUPS:
            terms = [str(term) for term in group.get("terms", ())]
            if self._term_in_disasters(name, terms):
                return group
        return None

    def _overview_columns_for_group(
        self,
        group: dict,
        cases: list[StandardCase],
        metrics_by_case: dict[str, list[IntensityMetric]],
        column_map: dict[str, dict],
        fallback_columns: list[dict],
        case_text_by_case: dict[str, str] | None,
    ) -> list[dict]:
        """为单个强度汇总子表选择覆盖个例最多的三个真实指标列。"""
        preferred = [column_map[key] for key in group.get("metric_keys", ()) if key in column_map]
        if not preferred:
            preferred = [dict(column) for column in fallback_columns]
        preferred_keys = {str(column["key"]): index for index, column in enumerate(preferred)}
        # 除灾种预设指标外，再把当前个例确实抽取到的指标加入候选，避免预设过窄导致列长期为空。
        observed_names = {
            metric.metric_name
            for case in cases
            for metric in metrics_by_case.get(case.case_id, [])
            if isinstance(metric.value, (int, float)) and not self._metric_needs_review(metric)
        }
        candidates = list(preferred)
        for column in self.OVERVIEW_METRIC_COLUMNS:
            if str(column["key"]) in {str(item["key"]) for item in candidates}:
                continue
            if observed_names.intersection(set(column["metric_names"])):
                candidates.append(dict(column))
        value_counts = self._overview_group_metric_value_counts(
            cases, metrics_by_case, candidates, case_text_by_case
        )
        available = [column for column in candidates if value_counts.get(str(column["key"]), 0) > 0]
        # 优先选择达到70%覆盖率的真实字段，避免把跨个例或推断值塞进表格。
        minimum_count = max(1, math.ceil(len(cases) * 0.7))
        fully_covered = [
            column for column in available
            if value_counts.get(str(column["key"]), 0) == len(cases)
        ]
        qualified = [
            column for column in available
            if value_counts.get(str(column["key"]), 0) >= minimum_count
        ]
        # 先取全覆盖，再取达到阈值的字段；不足时才按真实覆盖数补齐，绝不生成虚假数值。
        selectable = fully_covered or qualified or available
        selectable.sort(
            key=lambda column: (
                -value_counts.get(str(column["key"]), 0),
                preferred_keys.get(str(column["key"]), len(preferred_keys)),
            )
        )
        # 只展示至少有一个真实值的指标，避免为了凑列数生成整列“—”。
        selected = [dict(column) for column in selectable[:3]]
        fill_rates = {
            str(column["key"]): round(value_counts.get(str(column["key"]), 0) / max(1, len(cases)), 3)
            for column in selected
        }
        LOGGER.info(
            "[多维检索][强度汇总列选择] group=%s case_count=%s minimum_count=%s full_coverage=%s candidates=%s selected=%s fill_rates=%s",
            group.get("key"),
            len(cases),
            minimum_count,
            [str(column["key"]) for column in fully_covered],
            {str(column["key"]): value_counts.get(str(column["key"]), 0) for column in candidates},
            [(str(column["key"]), value_counts.get(str(column["key"]), 0)) for column in selected],
            fill_rates,
        )
        return selected

    def _overview_group_metric_value_counts(
        self,
        cases: list[StandardCase],
        metrics_by_case: dict[str, list[IntensityMetric]],
        columns: list[dict],
        case_text_by_case: dict[str, str] | None = None,
    ) -> dict[str, int]:
        """统计当前子表内各指标有有效值的个例数量。"""
        value_counts: dict[str, int] = {}
        for column in columns:
            if column.get("text_metric") == "wind_impact_scope":
                value_counts[str(column["key"])] = sum(
                    self._wind_impact_scope((case_text_by_case or {}).get(case.case_id, "")) != '—'
                    for case in cases
                )
                continue
            names = set(column["metric_names"])
            count = 0
            for case in cases:
                metric = self._case_metric_extreme(
                    metrics_by_case.get(case.case_id, []),
                    names,
                    prefer_min=bool(column.get("prefer_min")),
                )
                if metric and not self._metric_needs_review(metric):
                    count += 1
            value_counts[str(column["key"])] = count
        return value_counts

    def _wind_impact_scope(self, case_text: str) -> str:
        """从大风实况正文提取8级以上覆盖范围，兼容比例和站数两种正式表达。"""
        text = re.sub(r"\s+", "", str(case_text or ""))
        if not text:
            return '—'
        percent = re.search(r"8级以上(?:阵风)?[^。；]{0,35}?(?:影响我省|影响全省)[^。；]{0,8}?(\d+(?:\.\d+)?%)", text)
        if percent:
            return f"全省{percent.group(1)}"
        stations = re.search(r"共\s*(\d+)\s*(?:个)?站[^。；]{0,45}?出现\s*8级以上(?:阵风)?大风", text)
        if stations:
            return f"{stations.group(1)}站"
        return '—'


    def _matched_query_disasters(self, case: StandardCase, query: CaseSearchQuery | None) -> list[str]:
        """按查询顺序返回当前个例命中的全部检索灾种，避免被标题中的首个灾种带偏。"""
        wanted = [str(item or "").strip() for item in (getattr(query, "disaster_types", None) or []) if str(item or "").strip()]
        if not wanted:
            return []
        values = [str(item or "").strip() for item in (case.disaster_types or []) if str(item or "").strip()]
        text = " ".join([
            str(case.title or ""), str(case.date_range or ""), str(case.summary or ""),
            str(case.weather_facts or ""), str(case.forecast_focus or ""), " ".join(values),
        ])
        matched: list[str] = []
        for term in wanted:
            if term in text or any(term == value or term in value for value in values):
                matched.append(term)
                continue
            atomics = COMPOUND_DISASTER_TERMS.get(term, ())
            if atomics and all(atomic in text or any(atomic == value or atomic in value for value in values) for atomic in atomics):
                matched.append(term)
        return matched

    def _title_matched_query_disaster(self, title: str, matched: list[str]) -> str:
        """仅在检索已命中的灾种中，按标题出现位置和词长选择唯一主导灾种。"""
        candidates: list[tuple[int, int, str]] = []
        title_text = str(title or "")
        for term in matched:
            position = title_text.find(term)
            if position >= 0:
                candidates.append((position, -len(term), term))
                continue
            atomics = COMPOUND_DISASTER_TERMS.get(term, ())
            if atomics and all(atomic in title_text for atomic in atomics):
                candidates.append((min(title_text.find(atomic) for atomic in atomics), -len(term), term))
        candidates.sort()
        return candidates[0][2] if candidates else ""

    def _primary_disaster_text(self, case: StandardCase, query: CaseSearchQuery | None = None) -> str:
        """主导灾种保持单值，并优先采用标题中明确出现的检索命中灾种。"""
        matched = self._matched_query_disasters(case, query)
        if matched:
            return self._title_matched_query_disaster(str(case.title or ""), matched) or matched[0]
        title = str(case.title or "")
        case_terms = [str(item or "").strip() for item in (case.disaster_types or []) if str(item or "").strip()]
        all_terms = list(dict.fromkeys(name for profile in DISASTER_PROFILES for name in profile.names))
        title_match = self._title_disaster_match(title, [], all_terms)
        if title_match:
            return title_match
        process_hint = self._title_process_focus_disaster(title, case_terms)
        if process_hint:
            return process_hint
        for term in case_terms:
            if self._term_in_disasters(term, all_terms):
                return term
        return case_terms[0] if case_terms else "未标注"
    def _title_process_focus_disaster(self, title: str, values: list[str]) -> str:
        """标题明确涉及降水过程时，按降水类灾种的具体程度确定归属。"""
        title_text = str(title or "")
        if not any(token in title_text for token in ("\u964d\u6c34", "\u964d\u96e8", "\u96e8")):
            return ""
        priority = ("\u77ed\u65f6\u5f3a\u964d\u6c34", "\u5f3a\u964d\u6c34", "\u5927\u66b4\u96e8", "\u66b4\u96e8", "\u96e8\u96ea", "\u964d\u96ea", "\u66b4\u96ea")
        for term in priority:
            if any(term == value or term in value or value in term for value in values):
                return term
        return ""

    def _title_disaster_match(self, title: str, query_terms: list[str], all_terms: list[str]) -> str:
        """标题命中灾种时优先采用标题里的业务灾种，例如“暴雨天气过程”归为暴雨。"""
        wanted = list(dict.fromkeys([*query_terms, *all_terms]))
        candidates: list[tuple[int, int, str]] = []
        for term in wanted:
            if not term:
                continue
            position = title.find(term)
            if position >= 0:
                candidates.append((position, -len(term), term))
                continue
            # “大暴雨”筛选条件下，标题只写“暴雨”时仍按标题归为暴雨，避免退回强对流。
            if term == "大暴雨" and "暴雨" in title:
                candidates.append((title.find("暴雨"), -len("暴雨"), "暴雨"))
            elif term == "雷暴大风" and "雷暴" in title and "大风" in title:
                candidates.append((min(title.find("雷暴"), title.find("大风")), -len(term), term))
        if not candidates:
            return ""
        candidates.sort()
        return candidates[0][2]

    def _term_in_disasters(self, term: str, values: list[str]) -> bool:
        """宽松判断灾种是否属于同一并发灾种集合。"""
        return any(term == value or term in value or value in term for value in values)

    def _data_completeness_text(self, valid_count: int, total_count: int) -> str:
        """给强度汇总表补充数据完整性说明，便于业务人员快速发现缺测。"""
        if total_count <= 0 or valid_count <= 0:
            return '缺少强度记录'
        if valid_count >= total_count:
            return '强度记录完整'
        return f"已提取{valid_count}/{total_count}项"

    def _case_metric_extreme(
        self,
        metrics: list[IntensityMetric],
        names: set[str],
        prefer_min: bool = False,
    ) -> IntensityMetric | None:
        """在单个个例内挑选某类指标的业务极值，缺少风力等级时允许按风速折算。"""
        candidates = [metric for metric in metrics if metric.metric_name in names]
        # 风级始终与当前个例最大风速折算结果比较，避免“12级以上站数”等门槛值覆盖真正极值。
        if "阵风风力" in names:
            derived_level = self._wind_level_metric_from_speed(metrics)
            if derived_level is not None:
                candidates.append(derived_level)
        if not candidates:
            return None
        # 过滤超过常识阈值的高温值，避免 OCR 或模型误读污染概况极值。
        reliable_candidates = [metric for metric in candidates if not self._metric_needs_review(metric)]
        if not reliable_candidates:
            return None
        selector = min if prefer_min else max
        # 只有小时雨强在“范围+具体站点”并存时优先站点实测；过程雨量和气温必须比较全部值。
        if names == {"最大小时雨强"}:
            exact_candidates = [metric for metric in reliable_candidates if metric.relation != "范围上限"]
            selected_candidates = exact_candidates or reliable_candidates
        else:
            selected_candidates = reliable_candidates
        return selector(selected_candidates, key=lambda item: item.value)

    def _wind_level_metric_from_speed(self, metrics: list[IntensityMetric]) -> IntensityMetric | None:
        """原文缺少风力等级但已有风速时，按蒲福风级阈值补一个可追溯的计算值。"""
        wind_metrics = [
            metric for metric in metrics
            if metric.metric_name in {"极大风速", "最大风速"} and not self._metric_needs_review(metric)
        ]
        if not wind_metrics:
            return None
        source = max(wind_metrics, key=lambda item: item.value)
        level = self._wind_level_from_speed(float(source.value))
        if level is None:
            return None
        return IntensityMetric(
            metric_name="阵风风力",
            value=float(level),
            unit="级",
            location=source.location,
            relation="按风速折算",
            source_chunk_id=source.source_chunk_id,
            source_text=source.source_text,
            confidence=min(float(source.confidence or 0.8), 0.78),
        )

    def _wind_level_from_speed(self, speed: float) -> int | None:
        """按常用蒲福风级风速下限把 m/s 折算为风力等级。"""
        thresholds = [
            (56.1, 17), (51.0, 16), (46.2, 15), (41.5, 14), (37.0, 13),
            (32.7, 12), (28.5, 11), (24.5, 10), (20.8, 9), (17.2, 8),
            (13.9, 7), (10.8, 6), (8.0, 5), (5.5, 4), (3.4, 3),
            (1.6, 2), (0.3, 1), (0.0, 0),
        ]
        for lower, level in thresholds:
            if speed >= lower:
                return level
        return None

    def _format_metric_cell(self, metric: IntensityMetric | None) -> str:
        """把强度指标格式化成表格单元格，缺失时用破折号。"""
        if metric is None:
            return '—'
        if self._metric_needs_review(metric):
            return '—（疑似异常，需复核）'
        return f"{metric.value:g} {metric.unit}".strip()

    def _metric_needs_review(self, metric: IntensityMetric | None) -> bool:
        """识别明显超出业务常识阈值、需要人工回看原文或图片的强度值。"""
        return bool(
            metric
            and metric.metric_name == '最高气温'
            and float(metric.value) > self.SUSPICIOUS_MAX_TEMPERATURE
        )

    def _case_short_name(self, case: StandardCase) -> str:
        """表格首列优先使用日期范围，缺失时回退到标题，并尽量补齐月份。"""
        text = (case.date_range or case.title or case.case_id).strip()
        if "月" not in text:
            month = None
            if case.months:
                month = case.months[0]
            else:
                title_match = re.search(r"(\d{1,2})\s*月", case.title or "")
                pdf_match = re.search(r"20\d{2}[-_](\d{1,2})", case.source_pdf or "")
                if title_match:
                    month = int(title_match.group(1))
                elif pdf_match:
                    month = int(pdf_match.group(1))
            if month:
                text = f"{month}月{text}"
        text = text.replace("2025年", "").replace("2024年", "")
        day_char = "\u65e5"
        text = re.sub(r"(\d{1,2})\s*[-~]\s*(\d{1,2})" + day_char, lambda match: f"{match.group(1)}-{match.group(2)}{day_char}", text)
        return text[:22]


    def _short_area_text(
        self,
        case: StandardCase,
        metrics: list[IntensityMetric] | None = None,
        case_text: str = "",
    ) -> str:
        """从完整个例正文统计影响区域，正文不足时再回退到指标证据和标准个例字段。"""
        values = self._areas_from_case_text(case_text)
        if not values:
            metric_text = "。".join(str(metric.source_text or "") for metric in (metrics or []))
            values = self._areas_from_case_text(metric_text)
        if not values:
            values = list(case.city_tags or case.affected_areas)
        values = [value for value in dict.fromkeys(values) if value not in {"山西", "山西省"}]
        if "全省" in values:
            return "全省"
        if not values:
            return "—"
        return "、".join(values[:5]) + ("等" if len(values) > 5 else "")

    def _areas_from_case_text(self, text: str) -> list[str]:
        """按影响语句中的出现频次选择地市；没有地市时使用山西方位区域。"""
        normalized = re.sub(r"\s*\r?\n\s*", "", str(text or ""))
        normalized = re.sub(r"[\t ]+", " ", normalized)
        if not normalized:
            return []
        sentences = [item.strip() for item in re.split(r"[。；]", normalized) if item.strip()]
        impact_terms = (
            "出现", "发生", "影响", "覆盖", "降水", "降雨", "暴雨", "大风", "高温",
            "降雪", "雨雪", "寒潮", "沙尘", "雾", "霜冻", "冰雹", "雷暴",
        )
        area_terms = (
            "太原", "大同", "朔州", "忻州", "吕梁", "晋中", "阳泉", "长治", "晋城", "临汾", "运城",
            "全省", "北部", "中部", "南部", "东部", "西部", "北中部", "中南部",
        )
        relevant = [
            sentence for sentence in sentences
            if any(term in sentence for term in impact_terms) and any(term in sentence for term in area_terms)
        ]
        source_sentences = relevant or sentences
        source_text = "。".join(source_sentences)
        # 只有明确表达全省出现或覆盖该过程时才压缩成“全省”，避免普通背景描述误判。
        if re.search(r"全省[^。；]{0,28}(?:出现|发生|均有|普遍|共\s*\d+\s*站|降水|大风|高温|降雪|雨雪|沙尘|雾)", source_text) or re.search(
            r"(?:影响|覆盖)[^。；]{0,18}全省", source_text
        ):
            return ["全省"]

        cities = ("太原", "大同", "朔州", "忻州", "吕梁", "晋中", "阳泉", "长治", "晋城", "临汾", "运城")
        city_counts: Counter[str] = Counter()
        city_first: dict[str, int] = {}
        for city in cities:
            matches = list(re.finditer(re.escape(city) + r"市?", source_text))
            if matches:
                city_counts[city] = len(matches)
                city_first[city] = matches[0].start()
        if city_counts:
            return sorted(city_counts, key=lambda city: (-city_counts[city], city_first[city], cities.index(city)))

        direction_pattern = re.compile(
            r"(?:山西)?(?:北中部|中南部|东北部|西北部|东南部|西南部|北部|中部|南部|东部|西部)"
            r"(?:大部|部分地区|部分|局部|局地)?"
        )
        direction_counts: Counter[str] = Counter()
        direction_first: dict[str, int] = {}
        for sentence_index, sentence in enumerate(source_sentences):
            for match in direction_pattern.finditer(sentence):
                value = match.group(0)
                direction_counts[value] += 1
                direction_first.setdefault(value, sentence_index)
        return sorted(direction_counts, key=lambda value: (-direction_counts[value], direction_first[value], value))
    def _overview_intensity_sentence_from_tables(self, tables: list[dict]) -> str:
        """从多张分组强度表统一抽取概况极值，防止概况只被某个弱过程主导。"""
        if not tables:
            return ""
        rows: list[dict] = []
        columns: list[dict] = []
        seen_keys: set[str] = set()
        for table in tables:
            rows.extend(table.get("rows") or [])
            for column in table.get("columns") or []:
                key = str(column.get("key") or "")
                if key and key not in seen_keys:
                    columns.append(column)
                    seen_keys.add(key)
        return self._overview_intensity_sentence(rows, columns)


    def _overview_intensity_sentence(self, rows: list[dict], columns: list[dict] | None = None) -> str:
        """根据逐例汇总表生成概况可直接引用的强度极值句。"""
        if not rows:
            return ""
        items = []
        for column in columns or []:
            key = column["key"]
            candidates = [row for row in rows if isinstance((row.get("metric_values") or {}).get(key), (int, float))]
            if not candidates:
                continue
            # 概况句优先引用未标记“需复核”的值，避免异常 OCR 或原文疑值压过可靠极值。
            reliable_candidates = [row for row in candidates if '（需复核）' not in str((row.get("metrics") or {}).get(key) or "")]
            rank_candidates = reliable_candidates or candidates
            prefer_min = bool(column.get("prefer_min"))
            best = min(rank_candidates, key=lambda row: float((row.get("metric_values") or {})[key])) if prefer_min else max(rank_candidates, key=lambda row: float((row.get("metric_values") or {})[key]))
            value_text = (best.get("metrics") or {}).get(key) or ""
            verb = "低至" if prefer_min else "达"
            # 极值归属优先使用规范化短日期名，避免“14-15日”这类缺月份标题进入概况。
            case_label = best.get("case") or best.get("title")
            items.append(f"{column['label']}{verb} {str(value_text).replace(' ', '')}（{case_label}）")
            if len(items) >= 6:
                break
        if not items:
            return ""
        return "强度特征表现为：" + "、".join(items) + "。"
    def _evidence_analysis(
        self,
        cases: list[StandardCase],
        image_count: int,
        evidence_by_case: dict[str, CaseEvidenceBundle],
    ) -> tuple[str, str]:
        """解释原文、图片索引和可展示图片的证据覆盖能力。"""
        with_chunks = sum(bool(case.source_chunk_ids) for case in cases)
        with_image_ids = sum(bool(case.evidence_image_ids) for case in cases)
        evidence_samples = self._evidence_samples(cases, evidence_by_case, limit=3)
        sample_text = f" 代表性材料包括：{'；'.join(evidence_samples)}。" if evidence_samples else ""
        text = (
            f"证据链方面，{with_chunks}/{len(cases)} 个个例关联正文材料，"
            f"{with_image_ids}/{len(cases)} 个个例具备图片索引，报告代表个例解析出 {image_count} 张可展示原图。"
            "文字结论可以回到来源材料核查，图像用于验证过程形态和落区，但不能替代缺失的标准化强度字段。"
            f"{sample_text}"
        )
        finding = (
            f"证据覆盖达到原文片段 {with_chunks}/{len(cases)}、图片索引 {with_image_ids}/{len(cases)}，"
            f"并有 {image_count} 张原图可供复核。"
        )
        return text, finding

    def _evidence_summary(
        self,
        cases: list[StandardCase],
        evidence_by_case: dict[str, CaseEvidenceBundle],
    ) -> str:
        """生成摘要中使用的证据补充说明，不暴露底层编号。"""
        samples = self._evidence_samples(cases, evidence_by_case, limit=2)
        if not samples:
            return ""
        return " 原文和图片证据进一步显示，" + "；".join(samples) + "。"

    def _evidence_samples(
        self,
        cases: list[StandardCase],
        evidence_by_case: dict[str, CaseEvidenceBundle],
        limit: int,
    ) -> list[str]:
        """提取可放入报告正文的证据句。"""
        samples = []
        for case in cases:
            bundle = evidence_by_case.get(case.case_id)
            if bundle is None:
                continue
            text = bundle.public_text(limit=1)
            if text:
                samples.append(text)
            if len(samples) >= limit:
                break
        return samples

    def _query_strength_assessments(
        self,
        query: CaseSearchQuery | None,
        metrics_by_case: dict[str, list[IntensityMetric]],
    ) -> list[str]:
        """核对检索灾种与可量化强度是否一致，防止把检索标签直接当作达标结论。"""
        requested = set(getattr(query, "disaster_types", None) or [])
        assessments: list[str] = []
        if "大暴雨" not in requested:
            return assessments
        rainfall_values = [
            float(metric.value)
            for metrics in metrics_by_case.values()
            for metric in metrics
            if metric.metric_name in {"过程最大降水量", "最大降水量"}
            and str(metric.unit or "").lower() in {"mm", "毫米"}
            and not self._metric_needs_review(metric)
        ]
        if not rainfall_values:
            assessments.append("检索条件包含大暴雨，但现有材料未形成可核验的过程累计雨量，不能据此判定达到大暴雨标准。")
            return assessments
        maximum = max(rainfall_values)
        if maximum < 100:
            assessments.append(
                f"检索条件包含大暴雨，但当前材料可核验的过程最大降水量为 {maximum:g} mm，"
                "低于24小时100 mm的大暴雨量级，现阶段只能视为相关标签命中，不能表述为强度达标。"
            )
        return assessments
    def _focus_disasters(self, query: CaseSearchQuery | None) -> list[str]:
        """把查询中的复合灾种展开为用于共现分析的原子灾种。"""
        if query is None:
            return []
        values = [
            atomic
            for term in query.disaster_types
            for atomic in COMPOUND_DISASTER_TERMS.get(term, (term,))
        ]
        return list(dict.fromkeys(values))

    def _representative_cases(
        self,
        cases: list[StandardCase],
        metrics_by_case: dict[str, list[IntensityMetric]],
        evidence_by_case: dict[str, CaseEvidenceBundle],
    ) -> list[dict]:
        """选择证据较完整的代表个例并生成干净的结构化概况。"""
        ranked = sorted(
            cases,
            key=lambda case: (
                bool(metrics_by_case.get(case.case_id)),
                bool(case.evidence_image_ids),
                len(case.weather_facts),
            ),
            reverse=True,
        )
        results = []
        for case in ranked[:3]:
            areas = "、".join(case.city_tags or case.affected_areas) or "影响区域未明确"
            disasters = "、".join(case.disaster_types) or "灾种未标注"
            results.append(
                {
                    "case_id": case.case_id,
                    "title": case.title,
                    "reason": "材料包含可核验量化记录" if metrics_by_case.get(case.case_id) else "文字与图像材料相对完整",
                    "summary": self._representative_summary(case, areas, disasters, evidence_by_case),
                }
            )
        return results

    def _representative_summary(
        self,
        case: StandardCase,
        areas: str,
        disasters: str,
        evidence_by_case: dict[str, CaseEvidenceBundle],
    ) -> str:
        """生成代表个例概况，优先使用证据内容但隐藏底层编号。"""
        base = f"{case.date_range or '时间未明确'}，{areas}出现以{disasters}为主的天气过程。"
        bundle = evidence_by_case.get(case.case_id)
        evidence_text = bundle.public_text(limit=1) if bundle else ""
        return f"{base}{evidence_text}" if evidence_text else base

    def _business_finding_text(self, *findings: str) -> str:
        """把统计发现转换成更接近业务复盘材料的自然表述，减少模板化痕迹。"""
        text = "".join(item for item in findings if item)
        replacements = {
            "时间证据表明": "从过程时间看",
            "灾种证据显示": "从灾种配置看",
            "空间证据显示": "从影响落区看",
            "强度证据中，": "强度资料方面，",
            "证据覆盖达到": "证据链方面，",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _representative_case_text(self, item: dict) -> str:
        """把结构化代表个例转成可写入规则兜底结论的自然语言短句。"""
        title = str(item.get("title") or item.get("case_id") or "代表个例").strip()
        reason = str(item.get("reason") or "证据相对完整").strip()
        summary = str(item.get("summary") or "").strip()
        if summary:
            return f"{title}（{reason}）提示：{summary}"
        return f"{title}（{reason}）可作为后续复核样本"

    def _recommendations(
        self,
        cases: list[StandardCase],
        metrics_by_case: dict[str, list[IntensityMetric]],
        disaster_names: list[str] | None = None,
    ) -> list[str]:
        """生成四条单一主题、审慎且可落地的业务建议。"""
        return [
            (
                "建议探索建立短临触发信号联合识别流程，把雷达回波增强、站点雨强跃增和大风站数扩展与预警发布时间对应起来，"
                "优先在太原、晋中盆地和北部山区检验提前量与命中效果。"
            ),
            (
                "建议在吕梁山、五台山和太行山迎风坡试点建立分区模式偏差档案，按天气形势分别检验降水、风速和温度误差，"
                "通过滚动复盘筛选稳定的本地订正因子。"
            ),
            (
                "建议探索多灾种叠加风险研判清单，围绕大风与沙尘低能见度、雨雪降温与道路结冰、强降水与城市内涝等场景，"
                "明确风险升级条件、重点区域和服务对象。"
            ),
            (
                "建议统一个例库的过程边界、灾种标签、影响区域和强度指标口径，并关联预报结论、预警发布、站点实况和服务反馈，"
                "形成可追溯、可检验的复盘记录。"
            ),
        ]
    def _limitations(
        self,
        cases: list[StandardCase],
        metrics_by_case: dict[str, list[IntensityMetric]],
        image_count: int,
    ) -> list[str]:
        """列出影响报告解释范围的数据缺口。"""
        limitations = []
        if not any(case.city_tags for case in cases) and any(case.affected_areas for case in cases):
            limitations.append("空间分析使用影响区域回退，尚未形成统一的地市级标准标签。")
        elif not any(case.city_tags or case.affected_areas for case in cases):
            limitations.append("地市和影响区域均缺失，无法形成空间分布判断。")
        if not any(metrics_by_case.values()):
            limitations.append("关键强度指标缺失，无法进行强度排序和比较。")
        if image_count == 0:
            limitations.append("未解析到可展示的原始图片，图像证据部分为空。")
        limitations.append("统计和共现关系仅代表当前检索命中样本，不代表完整气候统计口径。")
        return limitations

    def _empty_analysis(self) -> dict:
        """返回无命中个例时的完整分析结构。"""
        message = "当前条件下没有命中个例，无法形成有效的材料分析报告。"
        return {
            "executive_summary": message,
            "key_findings": [message],
            "sections": {
                "temporal": message,
                "disaster": message,
                "spatial": message,
                "intensity": message,
            },
            "representative_cases": [],
            "recommendations": ["调整检索条件或补充标准化个例数据。"],
            "conclusion": message,
            "limitations": ["没有可分析的命中样本。"],
        }





