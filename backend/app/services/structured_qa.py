"""
结构化问答模块
基于标准化个例层实现业务全扫描问答，支持统计汇总、证据搜索、
个例复盘、对比分析等多类结构化查询意图。
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable

from backend.app.models import StandardCase
from backend.app.services.image_extraction import ImageEvidence, ImageEvidenceStore


# 灾害类型关键词（用于规则匹配识别查询灾种
DISASTER_TERMS = (
    "雷暴大风",
    "强对流",
    "大暴雨",
    "暴雨",
    "强降水",
    "短时强降水",
    "雷暴",
    "雷雨",
    "大风",
    "冰雹",
    "暴雪",
    "雨雪",
    "降雪",
    "寒潮",
    "低温",
    "霜冻",
    "高温",
    "沙尘",
    "雾",
)

# 城市/地市关键词（用于规则匹配识别影响区域
CITY_TERMS = (
    "太原",
    "大同",
    "朔州",
    "忻州",
    "阳泉",
    "晋中",
    "吕梁",
    "长治",
    "晋城",
    "临汾",
    "运城",
)

# 区域别名映射（用于泛化区域匹配
AREA_ALIASES = {
    "山西": ("太原", "大同", "朔州", "忻州", "阳泉", "晋中", "吕梁", "长治", "晋城", "临汾", "运城", "山西"),
    "山西北部": ("大同", "朔州", "忻州", "北部", "山西北部"),
    "北部": ("大同", "朔州", "忻州", "北部", "山西北部"),
    "山西中部": ("太原", "阳泉", "晋中", "吕梁", "中部", "山西中部"),
    "中部": ("太原", "阳泉", "晋中", "吕梁", "中部", "山西中部"),
    "山西南部": ("长治", "晋城", "临汾", "运城", "南部", "山西南部"),
    "南部": ("长治", "晋城", "临汾", "运城", "南部", "山西南部"),
}

# 图片类型关键词映射（用于识别查询中的图片需求
IMAGE_KEYWORDS = {
    "radar": ("雷达", "回波", "组合反射率", "风雷"),
    "satellite": ("卫星", "云图", "红外", "可见光"),
    "precipitation": ("降水图", "雨量图", "累计降水", "降水"),
    "wind": ("大风图", "风速", "阵风"),
    "sounding": ("探空", "TlnP", "TInP"),
    "synoptic": ("环流", "形势图", "海平面气压"),
    "temperature": ("高温图", "气温图", "温度图"),
    "warning": ("预警", "风险图"),
}

# 图片类型中文标签映射
IMAGE_LABELS = {
    "radar": "雷达图",
    "satellite": "卫星图",
    "precipitation": "降水图",
    "wind": "大风图",
    "sounding": "探空图",
    "synoptic": "环流形势图",
    "temperature": "温度图",
    "warning": "预警图",
}


@dataclass
class StructuredIntent:
    """
    结构化查询意图

    Attributes:
        intent: 意图类型（statistics/evidence_search/case_review/comparison/rag
        months: 月份列表（如 [5, 6]
        disasters: 灾种列表
        areas: 区域列表
        metrics: 指标列表
        image_type: 图片类型（如 radar/satellite
        focus: 查询关注点
        raw_date: 原始日期字符串（如 "5月10日"
        need_table: 是否需要表格展示
    """
    intent: str
    months: list[int] = field(default_factory=list)
    disasters: list[str] = field(default_factory=list)
    areas: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    image_type: str = ""
    focus: str = ""
    raw_date: str = ""
    need_table: bool = False


@dataclass
class EnrichedCase:
    """
    增强型标准化个例（添加派生字段

    Attributes:
        case: 原始标准化个例
        year: 年份
        months: 月份列表
        city_tags: 城市标签列表
        image_types: 图片类型列表
        images: 关联图片证据列表
    """
    case: StandardCase
    year: int | None
    months: list[int]
    city_tags: list[str]
    image_types: list[str]
    images: list[ImageEvidence]


@dataclass
class StructuredAnswer:
    """
    结构化问答结果

    Attributes:
        intent: 识别的查询意图
        answer: 生成的答案文本
        cases: 匹配的增强型个例列表
        images: 关联图片证据列表
        visuals: 可视化组件列表（表格、图表等
    """
    intent: StructuredIntent
    answer: str
    cases: list[EnrichedCase] = field(default_factory=list)
    images: list[ImageEvidence] = field(default_factory=list)
    visuals: list[dict] = field(default_factory=list)


class StructuredQuestionAnswerer:
    """
    结构化问答器
    基于标准化个例层实现全扫描业务问答，支持统计汇总、证据搜索、
    个例复盘、对比分析等多类结构化查询意图。
    """

    def __init__(
        self,
        image_store: ImageEvidenceStore,
        image_to_response: Callable[[ImageEvidence], dict],
    ):
        self.image_store = image_store
        self.image_to_response = image_to_response

    def answer(self, question: str, cases: list[StandardCase]) -> StructuredAnswer | None:
        intent = self.route(question)
        if intent.intent == "rag":
            return None
        enriched = [self.enrich_case(case) for case in cases]
        matched = self.filter_cases(enriched, intent)
        if intent.focus == "case_features":
            return self._case_features_answer(intent, matched)
        if intent.intent == "statistics":
            return self._statistics_answer(intent, matched)
        if intent.intent == "evidence_search":
            return self._evidence_answer(intent, matched)
        if intent.intent == "comparison":
            return self._comparison_answer(intent, matched)
        if intent.intent == "case_review":
            return self._case_review_answer(intent, matched)
        return None

    def route(self, question: str) -> StructuredIntent:
        intent = StructuredIntent(
            intent="rag",
            months=parse_months(question),
            disasters=parse_disasters(question),
            areas=parse_areas(question),
            metrics=parse_metrics(question),
            image_type=parse_image_type(question) if asks_for_image(question) else "",
            focus=parse_focus(question),
            raw_date=parse_raw_date(question),
            need_table=any(term in question for term in ("表", "清单", "列表", "导出", "明细")),
        )
        if any(term in question for term in ("相似", "像哪些", "类似", "历史过程", "当前", "今天", "服务提示", "决策")):
            return intent
        if intent.focus == "case_features":
            intent.intent = "case_review"
        elif intent.focus in ("missing_images", "disaster_data_limit", "time_distribution", "common_features"):
            intent.intent = "statistics"
        elif any(term in question for term in ("对比", "比较", "差异", "不同", "相比", "区别")):
            intent.intent = "comparison"
        elif intent.image_type:
            intent.intent = "evidence_search"
        elif intent.raw_date and any(term in question for term in ("分析", "复盘", "过程", "影响哪里", "影响区域", "主要影响")):
            intent.intent = "case_review"
        elif intent.focus or any(term in question for term in ("多少", "几次", "几个", "统计", "汇总", "分别", "最多", "最严重", "受灾", "灾情", "分布", "主要灾种", "哪个月份")):
            intent.intent = "statistics"
        elif any(term in question for term in ("哪些", "清单", "列表", "有没有", "有无")) or intent.image_type:
            intent.intent = "evidence_search"
        return intent

    def enrich_case(self, case: StandardCase) -> EnrichedCase:
        images = self.image_store.list_by_image_ids(case.evidence_image_ids)
        image_types = sorted({self.image_store.classify(image)[0] for image in images})
        months = sorted(set(parse_months(f"{case.date_range} {case.title} {case.source_pdf}")))
        return EnrichedCase(
            case=case,
            year=parse_year(case.source_pdf),
            months=months,
            city_tags=parse_case_cities(case),
            image_types=image_types,
            images=images,
        )

    def filter_cases(self, cases: list[EnrichedCase], intent: StructuredIntent) -> list[EnrichedCase]:
        matched = cases
        if intent.months:
            wanted = set(intent.months)
            matched = [item for item in matched if wanted & set(item.months)]
        if intent.raw_date:
            matched = [item for item in matched if date_matches_case(intent.raw_date, item.case)]
        if intent.disasters:
            wanted_disasters = expand_disasters(intent.disasters)
            matched = [item for item in matched if wanted_disasters & set(expand_disasters(item.case.disaster_types))]
        if intent.areas:
            area_candidates = set()
            for area in intent.areas:
                area_candidates.update(AREA_ALIASES.get(area, (area,)))
            matched = [
                item for item in matched
                if area_candidates & set(item.city_tags) or any(area in case_text(item.case) for area in area_candidates)
            ]
        if intent.image_type:
            matched = [item for item in matched if intent.image_type in item.image_types]
        return sorted(matched, key=lambda item: (min(item.months or [99]), first_day(item.case), item.case.case_id))

    def _statistics_answer(self, intent: StructuredIntent, cases: list[EnrichedCase]) -> StructuredAnswer:
        month_counter = Counter(month for item in cases for month in item.months)
        disaster_counter = Counter(disaster for item in cases for disaster in item.case.disaster_types)
        city_counter = Counter(city for item in cases for city in item.city_tags if city != "山西")
        if intent.focus == "disaster_data_limit":
            lines = [
                "结论：不能。没有灾情数据时，只能判断天气过程频次、影响区域和灾种分布，不能判定“受灾最严重”。",
                "结构化事实：标准化个例库记录的是天气过程和影响线索，不等同于灾情损失排行。",
                "口径说明：若要比较受灾程度，还需要灾情损失、人口影响、农作物受灾面积和基础设施损毁等数据。",
            ]
            return StructuredAnswer(intent=intent, answer="\n".join(lines), cases=cases)
        if intent.focus == "missing_images":
            missing = [item for item in cases if not item.case.evidence_image_ids]
            lines = [
                f"结论：当前标准化个例中有 {len(missing)} 个缺少图片证据。",
                "检索证据：",
            ]
            if missing:
                for item in dedupe_cases(missing)[:12]:
                    disasters = "、".join(item.case.disaster_types[:5]) or "未标注灾种"
                    lines.append(f"- {item.case.case_id}：{item.case.date_range} {item.case.title}；{disasters}；来源：{item.case.source_pdf}")
                if len(missing) > 12:
                    lines.append(f"- 其余 {len(missing) - 12} 个缺图个例未展开。")
            else:
                lines.append("- 暂未发现缺少图片证据的个例。")
            lines.append("模型分析：这里只按标准化个例的 evidence_image_ids 判断；若图片元数据还未入库，也会表现为缺图。")
            return StructuredAnswer(
                intent=intent,
                answer="\n".join(lines),
                cases=missing,
                visuals=[self._case_table_visual("缺少图片证据的个例", missing[:12])] if missing else [],
            )
        if intent.focus == "time_distribution":
            lines = [
                f"结论：共找到 {len(cases)} 个相关过程，主要出现时段如下。",
                "结构化事实：",
            ]
            lines.extend(self._case_lines(cases, limit=12) if cases else ["- 当前没有匹配到相关过程。"])
            if month_counter:
                lines.append("按月份：" + "；".join(f"{month}月 {count} 个" for month, count in sorted(month_counter.items())))
            lines.append("模型分析：时间分布按标准化个例日期统计，适合判断集中月份和代表过程。")
            return StructuredAnswer(intent=intent, answer="\n".join(lines), cases=cases, visuals=self._statistics_visuals(month_counter, cases))
        if intent.focus == "common_features":
            lines = [
                f"结论：共找到 {len(cases)} 个相关过程；共同特征主要体现在灾种组合、影响区域和服务关注点上。",
                "结构化事实：",
            ]
            if disaster_counter:
                lines.append("- 高频灾种：" + "；".join(f"{name} {count} 次" for name, count in disaster_counter.most_common(6)))
            if city_counter:
                lines.append("- 高频影响地市：" + "；".join(f"{name} {count} 次" for name, count in city_counter.most_common(6)))
            lines.append("检索证据：")
            lines.extend(self._case_lines(cases, limit=6) if cases else ["- 当前没有匹配到相关过程。"])
            lines.append("模型分析：这类过程应重点关注短时强降水、雷暴大风、冰雹等强对流要素的叠加和落区重合。")
            return StructuredAnswer(intent=intent, answer="\n".join(lines), cases=cases, visuals=self._statistics_visuals(month_counter, cases))
        if intent.focus == "month_ranking" and month_counter:
            top_month, top_count = month_counter.most_common(1)[0]
            lines = [
                f"结论：按标准化个例统计，{top_month}月天气过程最多，共 {top_count} 个。",
                "结构化事实：",
                "- 按月份：" + "；".join(f"{month}月 {count} 个" for month, count in sorted(month_counter.items())),
                "检索证据：已整理代表个例表，可继续要求列出完整明细。",
                "模型分析：这是按个例数量统计的活跃月份，不等同于灾损最重月份。",
            ]
            return StructuredAnswer(intent=intent, answer="\n".join(lines), cases=cases, visuals=self._statistics_visuals(month_counter, cases))
        if intent.focus in ("city_ranking", "city_severity"):
            if not city_counter:
                lines = ["当前标准化个例库没有足够的地市标注，不能判断哪个地级市受影响更突出。"]
                return StructuredAnswer(intent=intent, answer="\n".join(lines), cases=cases)
            top_city, top_count = city_counter.most_common(1)[0]
            if intent.focus == "city_severity":
                lines = [
                    f"结论：现有标准化个例不能直接判定受灾最严重的地级市；如果按个例中出现次数作为受影响频次口径，{top_city} 最突出，出现 {top_count} 次。",
                    "结构化事实：地市出现次数排行如下。",
                ]
            else:
                lines = [
                    f"按标准化个例统计，共涉及 {len(city_counter)} 个地市；出现次数最多的地市是 {top_city}，共 {top_count} 次。",
                    "结构化事实：地市出现次数排行如下。",
                ]
            for index, (city, count) in enumerate(city_counter.most_common(10), start=1):
                lines.append(f"{index}. {city}：{count} 次")
            lines.append("证据口径：同一个标准化个例中出现某地市，计 1 次；省级“山西”不计入地市排行。")
            if intent.focus == "city_severity":
                lines.append("补充判断：真正的“受灾最严重”还需要灾情、损失、人口影响、农作物受灾面积等民政或应急管理数据支撑。")
            return StructuredAnswer(intent=intent, answer="\n".join(lines), cases=cases, visuals=self._ranking_visuals(city_counter))

        lines = [f"结论：本次按标准化个例库统计，共筛选到 {len(cases)} 个灾害天气个例。"]
        if cases:
            # 先逐个例说明实际发生了什么，再给出灾种聚合，避免用户只看到标签数量却不知道对应过程。
            lines.append("具体灾害过程：")
            lines.extend(self._case_overview_lines(intent, cases, limit=12))
        lines.append("结构化事实：")
        if month_counter:
            lines.append("- 按月份：" + "；".join(f"{month}月 {count} 个" for month, count in sorted(month_counter.items())))
        if disaster_counter:
            lines.append("- 主要灾种：" + "；".join(f"{name} {count} 次" for name, count in disaster_counter.most_common(8)))
        if city_counter:
            lines.append("- 涉及地市较多：" + "；".join(f"{name} {count} 次" for name, count in city_counter.most_common(8)))
        if cases and intent.need_table:
            lines.append("检索证据：")
            lines.append("个例明细：")
            lines.extend(self._case_lines(cases))
        elif not cases:
            lines.append("检索证据：当前标准化个例库里没有匹配到这组条件。")
        elif not intent.need_table:
            lines.append("检索证据：已整理代表个例表；如需逐条个例，可继续问“列出明细”或“导出清单”。")
        lines.append("模型分析：这个结论来自标准化个例的全量聚合，适合回答数量、月份和灾种分布；若要判断灾损轻重，还需要灾情损失数据。")
        return StructuredAnswer(
            intent=intent,
            answer="\n".join(lines),
            cases=cases,
            visuals=self._statistics_visuals(month_counter, cases),
        )

    def _evidence_answer(self, intent: StructuredIntent, cases: list[EnrichedCase]) -> StructuredAnswer:
        images = []
        lines = [f"结论：共找到 {len(cases)} 个相关标准化个例。"]
        if intent.image_type:
            label = IMAGE_LABELS.get(intent.image_type, intent.image_type)
            lines.append(f"结构化事实：图片条件为 {label}。")
        if cases:
            lines.append("检索证据：")
            display_cases = cases[:8]
            for item in display_cases:
                selected_images = dedupe_images_by_caption([
                    image for image in item.images
                    if not intent.image_type or self.image_store.classify(image)[0] == intent.image_type
                ])
                images.extend(selected_images[:2])
                captions = "；".join((image.caption or f"第{image.page_no}页图片") for image in selected_images[:2])
                suffix = f"；图像证据：{captions}" if captions else ""
                lines.append(f"- {item.case.date_range} {item.case.title}（{item.case.source_pdf}）{suffix}")
            if len(cases) > len(display_cases):
                lines.append(f"- 其余 {len(cases) - len(display_cases)} 个相关个例未展开，可继续要求“列出完整清单”。")
        else:
            lines.append("检索证据：当前标准化个例库里没有匹配到对应证据。")
        lines.append("模型分析：证据搜索类问题优先返回可追溯个例和图片线索，未命中的部分不做推断。")
        return StructuredAnswer(intent=intent, answer="\n".join(lines), cases=cases, images=dedupe_images(images))

    def _case_review_answer(self, intent: StructuredIntent, cases: list[EnrichedCase]) -> StructuredAnswer:
        if not cases:
            lines = [
                "结论：当前标准化个例库里没有匹配到这个具体过程。",
                "说明：如需分析该过程，需要补充对应时段的标准化个例或原文材料。",
            ]
            return StructuredAnswer(intent=intent, answer="\n".join(lines), cases=[])
        item = cases[0]
        disasters = "、".join(item.case.disaster_types[:6]) or "未标注"
        areas = "、".join(item.case.affected_areas[:8]) or "未标注"
        fact = compact_text(item.case.weather_facts or item.case.summary or "标准化个例未提供详细实况描述。", limit=180)
        focus = compact_text(item.case.forecast_focus or "需要结合实况监测、雷达和降水资料继续研判。", limit=140)
        lines = [
            f"结论：{item.case.date_range} {item.case.title} 主要影响区域为 {areas}，主要灾种为 {disasters}。",
            "结构化事实：",
            f"- 时段：{item.case.date_range}",
            f"- 影响区域：{areas}",
            f"- 灾种：{disasters}",
            f"- 实况线索：{fact}",
            "检索证据：",
            f"- 来源：{item.case.source_pdf}；标准化个例：{item.case.case_id}",
            "模型分析：",
            f"- 服务关注：{focus}",
        ]
        if len(cases) > 1:
            lines.append(f"- 另有 {len(cases) - 1} 个同日或同月候选过程未展开。")
        return StructuredAnswer(
            intent=intent,
            answer="\n".join(lines),
            cases=cases[:3],
            images=dedupe_images(item.images[:4]),
            visuals=[self._case_table_visual("相关过程", cases[:6])],
        )

    def _case_features_answer(self, intent: StructuredIntent, cases: list[EnrichedCase]) -> StructuredAnswer:
        lines = [
            f"结论：这里按上下文中的 {len(cases)} 个灾害过程分别整理特点。",
            "结构化事实：",
        ]
        for item in dedupe_cases(cases):
            disasters = "、".join(item.case.disaster_types[:5]) or "未标注灾种"
            areas = "、".join(item.case.affected_areas[:6]) or "未标注区域"
            summary = compact_text(item.case.summary or item.case.weather_facts or "暂无详细特点描述。", limit=120)
            focus = compact_text(item.case.forecast_focus or "需要结合监测实况继续研判。", limit=100)
            lines.append(f"- {item.case.date_range} {item.case.title}：{disasters}；影响区域：{areas}；特点：{summary}；关注：{focus}")
        lines.append("检索证据：以上来自上一轮上下文锁定的标准化个例。")
        lines.append("模型分析：这里只解释这几个上下文个例，不扩展到全库统计。")
        return StructuredAnswer(
            intent=intent,
            answer="\n".join(lines),
            cases=cases,
            visuals=[self._case_table_visual("上下文个例", cases)],
        )

    def _comparison_answer(self, intent: StructuredIntent, cases: list[EnrichedCase]) -> StructuredAnswer:
        if intent.metrics:
            return self._metric_comparison_gap_answer(intent, cases)
        if len(intent.disasters) >= 2:
            return self._disaster_comparison_answer(intent, cases)
        groups: dict[int, list[EnrichedCase]] = defaultdict(list)
        for item in cases:
            for month in item.months:
                if not intent.months or month in intent.months:
                    groups[month].append(item)
        lines = []
        if not groups:
            lines.append("当前标准化个例库里没有匹配到可对比的个例。")
            return StructuredAnswer(intent=intent, answer="\n".join(lines), cases=cases)

        profiles = {
            month: self._comparison_profile(dedupe_cases(groups[month]))
            for month in sorted(groups)
        }
        lines.append(self._comparison_conclusion(profiles))
        lines.append("模型分析：")
        lines.append("主要差异：")
        lines.extend(self._comparison_difference_lines(profiles))
        lines.append("结构化事实：")
        lines.append("分月概况：")
        for month in sorted(groups):
            profile = profiles[month]
            lines.append(
                f"- {month}月：{profile['count']} 个个例；主要灾种为 {profile['top_disasters']}；"
                f"主要影响区域为 {profile['top_cities']}。"
            )
        lines.append("检索证据：")
        lines.append("代表个例：")
        lines.extend(self._representative_case_lines(groups))
        return StructuredAnswer(
            intent=intent,
            answer="\n".join(lines),
            cases=cases,
            visuals=self._comparison_visuals(profiles, groups),
        )

    def _metric_comparison_gap_answer(self, intent: StructuredIntent, cases: list[EnrichedCase]) -> StructuredAnswer:
        metric = intent.metrics[0]
        months = intent.months[:2]
        month_text = "、".join(f"{month}月" for month in months) if months else "指定月份"
        # 核心指标完全无证据时只说明结论边界，不暴露内部任务拆解和失败状态。
        lines = [
            f"当前知识库中没有检索到可支持{month_text}{metric}对比结论的同口径数据。",
            "现有标准化个例可用于统计灾害过程，但不能替代该指标的月度观测数据。",
        ]
        return StructuredAnswer(
            intent=intent,
            answer="\n".join(lines),
            cases=cases,
            visuals=self._metric_gap_visual(metric, months),
        )

    def _disaster_comparison_answer(self, intent: StructuredIntent, cases: list[EnrichedCase]) -> StructuredAnswer:
        disaster_groups = {
            disaster: [item for item in cases if disaster in expand_disasters(item.case.disaster_types)]
            for disaster in intent.disasters[:4]
        }
        lines = ["总体来看：不同灾种的服务关注点不一样，暴雨偏向累计雨量和次生灾害，强对流偏向短临突发性。"]
        lines.append("模型分析：")
        lines.append("主要差异：")
        for disaster, group in disaster_groups.items():
            city_counter = Counter(city for item in group for city in item.city_tags if city != "山西")
            areas = "、".join(city for city, _ in city_counter.most_common(5)) or "未标注"
            if "暴雨" in disaster or "强降水" in disaster:
                focus = "重点看累计雨量、短时雨强、落区重叠以及山洪地质灾害风险。"
            elif "强对流" in disaster or "雷暴" in disaster or "大风" in disaster or "冰雹" in disaster:
                focus = "重点看雷达回波发展、短时大风、冰雹和强降水突发落区。"
            else:
                focus = "重点结合该灾种的监测指标和影响对象研判。"
            lines.append(f"- {disaster}：匹配 {len(group)} 个个例；主要区域 {areas}；{focus}")
        lines.append("结构化事实：")
        for disaster, group in disaster_groups.items():
            lines.append(f"- {disaster}代表过程：")
            lines.extend(self._case_lines(group, limit=3) if group else ["- 当前没有匹配个例。"])
        lines.append("检索证据：代表个例见上方；如需更细，可继续按某个灾种展开。")
        return StructuredAnswer(
            intent=intent,
            answer="\n".join(lines),
            cases=dedupe_cases([item for group in disaster_groups.values() for item in group]),
            visuals=[self._case_table_visual("灾种对比代表个例", dedupe_cases([item for group in disaster_groups.values() for item in group])[:8])],
        )

    def _comparison_profile(self, cases: list[EnrichedCase]) -> dict:
        disaster_counter = Counter(disaster for item in cases for disaster in item.case.disaster_types)
        city_counter = Counter(city for item in cases for city in item.city_tags if city != "山西")
        return {
            "count": len(cases),
            "top_disasters": "、".join(name for name, _ in disaster_counter.most_common(5)) or "未标注",
            "top_disaster_set": {name for name, _ in disaster_counter.most_common(5)},
            "top_cities": "、".join(name for name, _ in city_counter.most_common(5)) or "未标注",
        }

    def _comparison_conclusion(self, profiles: dict[int, dict]) -> str:
        ordered = sorted(profiles.items())
        busiest_month, busiest_profile = max(ordered, key=lambda item: item[1]["count"])
        quietest_month, quietest_profile = min(ordered, key=lambda item: item[1]["count"])
        if busiest_month == quietest_month:
            return (
                f"总体来看：{busiest_month}月匹配到 {busiest_profile['count']} 个标准化个例，"
                f"主要表现为 {busiest_profile['top_disasters']}。"
            )
        return (
            f"总体来看：{busiest_month}月过程更多、类型更丰富，共 {busiest_profile['count']} 个；"
            f"{quietest_month}月共 {quietest_profile['count']} 个，主要集中在 {quietest_profile['top_disasters']}。"
        )

    def _comparison_difference_lines(self, profiles: dict[int, dict]) -> list[str]:
        ordered = sorted(profiles.items())
        counts = "；".join(f"{month}月 {profile['count']} 个" for month, profile in ordered)
        disaster_parts = []
        for month, profile in ordered:
            disaster_parts.append(f"{month}月以 {profile['top_disasters']} 为主")
        shared = set.intersection(*(profile["top_disaster_set"] for _, profile in ordered)) if len(ordered) >= 2 else set()
        shared_line = (
            f"- 共性：共同出现的主要灾种有 {'、'.join(sorted(shared))}。"
            if shared
            else "- 共性：主要灾种交集不明显，更适合按月份分别研判。"
        )
        return [
            f"- 数量差异：{counts}。",
            "- 灾种结构：" + "；".join(disaster_parts) + "。",
            shared_line,
        ]

    def _representative_case_lines(self, groups: dict[int, list[EnrichedCase]], per_month: int = 2) -> list[str]:
        lines = []
        for month in sorted(groups):
            for item in dedupe_cases(groups[month])[:per_month]:
                disasters = "、".join(item.case.disaster_types[:5]) or "未标注灾种"
                areas = "、".join(item.case.affected_areas[:4]) or "未标注区域"
                lines.append(
                    f"- {month}月代表：{item.case.date_range} {item.case.title}："
                    f"{disasters}；影响区域：{areas}；来源：{item.case.source_pdf}"
                )
        return lines

    def _statistics_visuals(self, month_counter: Counter, cases: list[EnrichedCase]) -> list[dict]:
        visuals = []
        if month_counter:
            visuals.append(
                {
                    "type": "bar",
                    "title": "按月份统计",
                    "items": [
                        {"label": f"{month}月", "value": count}
                        for month, count in sorted(month_counter.items())
                    ],
                }
            )
        if cases:
            visuals.append(self._case_table_visual("代表个例", dedupe_cases(cases)[:8]))
        return visuals

    def _comparison_visuals(self, profiles: dict[int, dict], groups: dict[int, list[EnrichedCase]]) -> list[dict]:
        bar_items = [
            {"label": f"{month}月", "value": profile["count"]}
            for month, profile in sorted(profiles.items())
        ]

    def _metric_gap_visual(self, metric: str, months: list[int]) -> list[dict]:
        rows = [
            {
                "month": f"{month}月",
                "metric": metric,
                "status": "缺少结构化指标数据",
            }
            for month in months
        ]
        if not rows:
            return []
        return [
            {
                "type": "table",
                "title": "指标数据检查",
                "columns": [
                    {"key": "month", "label": "月份"},
                    {"key": "metric", "label": "指标"},
                    {"key": "status", "label": "数据状态"},
                ],
                "rows": rows,
            }
        ]
        profile_rows = [
            {
                "month": f"{month}月",
                "count": profile["count"],
                "disasters": profile["top_disasters"],
                "areas": profile["top_cities"],
            }
            for month, profile in sorted(profiles.items())
        ]
        representatives = []
        for month in sorted(groups):
            representatives.extend(dedupe_cases(groups[month])[:2])
        return [
            {"type": "bar", "title": "月份过程数量对比", "items": bar_items},
            {
                "type": "table",
                "title": "分月差异摘要",
                "columns": [
                    {"key": "month", "label": "月份"},
                    {"key": "count", "label": "个例数"},
                    {"key": "disasters", "label": "主要灾种"},
                    {"key": "areas", "label": "主要区域"},
                ],
                "rows": profile_rows,
            },
            self._case_table_visual("代表个例", representatives),
        ]

    def _ranking_visuals(self, city_counter: Counter) -> list[dict]:
        if not city_counter:
            return []
        return [
            {
                "type": "bar",
                "title": "地市出现次数排行",
                "items": [
                    {"label": city, "value": count}
                    for city, count in city_counter.most_common(10)
                ],
            }
        ]

    def _case_table_visual(self, title: str, cases: list[EnrichedCase]) -> dict:
        return {
            "type": "table",
            "title": title,
            "columns": [
                {"key": "date", "label": "时段"},
                {"key": "title", "label": "过程"},
                {"key": "disasters", "label": "灾种"},
                {"key": "areas", "label": "影响区域"},
                {"key": "source", "label": "来源"},
            ],
            "rows": [
                {
                    "date": item.case.date_range,
                    "title": item.case.title,
                    "disasters": "、".join(item.case.disaster_types[:5]) or "未标注",
                    "areas": "、".join(item.case.affected_areas[:5]) or "未标注",
                    "source": item.case.source_pdf,
                }
                for item in cases
            ],
        }

    def _case_lines(self, cases: list[EnrichedCase], limit: int = 30) -> list[str]:
        lines = []
        for item in dedupe_cases(cases)[:limit]:
            disasters = "、".join(item.case.disaster_types[:5]) or "未标注灾种"
            areas = "、".join(item.case.affected_areas[:6]) or "未标注区域"
            lines.append(f"- {item.case.date_range} {item.case.title}：{disasters}；影响区域：{areas}；来源：{item.case.source_pdf}")
        if len(cases) > limit:
            lines.append(f"- 其余 {len(cases) - limit} 个个例未展开。")
        return lines

    def _case_overview_lines(
        self,
        intent: StructuredIntent,
        cases: list[EnrichedCase],
        limit: int = 12,
    ) -> list[str]:
        """按“主灾种 + 并发灾种”先介绍每个过程，供普通统计答案使用。"""
        lines = []
        for index, item in enumerate(dedupe_cases(cases)[:limit], start=1):
            primary = self._primary_disaster(item, intent)
            concurrent = [
                value for value in item.case.disaster_types
                if value and value != primary
            ]
            primary_text = primary or "未标注"
            concurrent_text = "、".join(dict.fromkeys(concurrent)) or "无明确并发灾种"
            areas = "、".join(item.case.affected_areas[:6]) or "未标注区域"
            lines.append(
                f"{index}. {item.case.date_range} {item.case.title}；"
                f"主要灾种：{primary_text}；并发灾种：{concurrent_text}；"
                f"影响区域：{areas}。"
            )
        if len(cases) > limit:
            lines.append(f"其余 {len(cases) - limit} 个个例未展开。")
        return lines

    @staticmethod
    def _primary_disaster(item: EnrichedCase, intent: StructuredIntent) -> str:
        """确定展示用主灾种，不改变标准个例原始灾种数组。"""
        disasters = list(dict.fromkeys(item.case.disaster_types or []))
        if not disasters:
            return ""
        # 用户明确筛选的灾种优先作为当前过程的主灾种。
        requested = set(expand_disasters(intent.disasters))
        for disaster in disasters:
            if disaster in requested:
                return disaster
        # 标题中出现的灾种通常是材料标题的主语，优先于并发标签。
        title = str(item.case.title or "")
        for disaster in disasters:
            if disaster in title:
                return disaster
        return disasters[0]

    def response_images(self, answer: StructuredAnswer, limit: int = 6) -> list[dict]:
        return [self.image_to_response(image) for image in dedupe_images(answer.images)[:limit]]


def parse_months(text: str) -> list[int]:
    months: set[int] = set()
    chinese_months = {
        "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
        "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12,
    }
    for start, end in re.findall(r"(\d{1,2})\s*[-~～至到]\s*(\d{1,2})\s*月", text):
        start_month, end_month = int(start), int(end)
        if 1 <= start_month <= end_month <= 12:
            months.update(range(start_month, end_month + 1))
    for month in re.findall(r"(\d{1,2})\s*月", text):
        value = int(month)
        if 1 <= value <= 12:
            months.add(value)
    # 支持“三月份”“一月至三月”等自然中文表达，主 RAG 可直接完成简单聚合。
    chinese_pattern = r"(十二|十一|十|[一二三四五六七八九])"
    for start, end in re.findall(chinese_pattern + r"\s*月?\s*[-~～至到]\s*" + chinese_pattern + r"\s*月份?", text):
        start_month, end_month = chinese_months[start], chinese_months[end]
        if start_month <= end_month:
            months.update(range(start_month, end_month + 1))
    for month in re.findall(chinese_pattern + r"\s*月份?", text):
        months.add(chinese_months[month])
    pdf_match = re.search(r"(?:FST)?2025[-_年]?(\d{1,2})", text)
    if pdf_match:
        value = int(pdf_match.group(1))
        if 1 <= value <= 12:
            months.add(value)
    return sorted(months)


def parse_year(text: str) -> int | None:
    match = re.search(r"(20\d{2})", text)
    return int(match.group(1)) if match else None


def parse_disasters(text: str) -> list[str]:
    return [term for term in DISASTER_TERMS if term in text]


def parse_metrics(text: str) -> list[str]:
    metrics = []
    if any(term in text for term in ("平均降水量", "平均雨量", "平均降水")):
        metrics.append("平均降水量")
    elif any(term in text for term in ("累计降水量", "累计雨量", "总降水量")):
        metrics.append("累计降水量")
    elif "降水量" in text or "雨量" in text:
        metrics.append("降水量")
    return metrics


def parse_areas(text: str) -> list[str]:
    areas = [area for area in AREA_ALIASES if area in text]
    areas.extend(city for city in CITY_TERMS if city in text)
    if "山西" in text and "山西" not in areas:
        areas.append("山西")
    return dedupe_strings(areas)


def parse_image_type(text: str) -> str:
    for image_type, keywords in IMAGE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return image_type
    return ""


def parse_focus(text: str) -> str:
    if "没有灾情数据" in text or "无灾情数据" in text:
        return "disaster_data_limit"
    if any(term in text for term in ("缺少图片", "没有图片", "无图片", "缺图")):
        return "missing_images"
    if any(term in text for term in ("出现在哪些时间", "出现时间", "主要出现在哪些时间", "哪些时间")):
        return "time_distribution"
    if any(term in text for term in ("共同特征", "主要特征")):
        return "common_features"
    if any(
        term in text
        for term in (
            "特点分别",
            "分别是什么",
            "分别有哪些特点",
            "特点是什么",
            "分别详细解释",
            "分别解释",
            "详细解释",
            "详细说明",
            "详细讲讲",
            "详细讲",
            "讲讲",
            "展开讲",
            "分别说明",
            "逐个解释",
            "逐个说明",
            "逐条解释",
            "逐条说明",
        )
    ):
        return "case_features"
    if "哪个月份" in text and any(term in text for term in ("最多", "最频繁", "最集中")):
        return "month_ranking"
    if any(term in text for term in ("地市", "地级市", "城市", "区域", "地区", "县", "县区")) and any(
        term in text for term in ("受灾", "最严重", "灾情")
    ):
        return "city_severity"
    if any(term in text for term in ("地市", "地级市", "城市", "区域", "地区", "县", "县区")) and any(
        term in text for term in ("最多", "排名", "排行", "受影响", "影响最多")
    ):
        return "city_ranking"
    return ""


def asks_for_image(text: str) -> bool:
    return any(term in text for term in ("图", "雷达", "卫星", "云图", "回波", "TlnP", "探空"))


def parse_raw_date(text: str) -> str:
    range_match = re.search(r"(\d{1,2}\s*月\s*\d{1,2}\s*[-~～至到]\s*\d{1,2}\s*日?)", text)
    if range_match:
        return re.sub(r"\s+", "", range_match.group(1))
    month_day = re.search(r"(\d{1,2}\s*月\s*\d{1,2}\s*日?)", text)
    if month_day:
        return re.sub(r"\s+", "", month_day.group(1))
    return ""


def date_matches_case(raw_date: str, case: StandardCase) -> bool:
    month_days = _month_days(raw_date)
    if not month_days:
        return True
    case_month_days = _month_days(f"{case.date_range} {case.title}")
    if not case_month_days:
        return False
    query_months = {month for month, _ in month_days}
    case_months = {month for month, _ in case_month_days}
    query_days = {day for _, day in month_days}
    case_days = {day for _, day in case_month_days}
    return bool(query_months & case_months and query_days & case_days)


def _month_days(text: str) -> set[tuple[int, int]]:
    results: set[tuple[int, int]] = set()
    for month, start, end in re.findall(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[-~～至到]\s*(\d{1,2})", text):
        month_value = int(month)
        start_day = int(start)
        end_day = int(end)
        if 1 <= month_value <= 12 and 1 <= start_day <= end_day <= 31:
            results.update((month_value, day) for day in range(start_day, end_day + 1))
    for month, day in re.findall(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日?", text):
        month_value = int(month)
        day_value = int(day)
        if 1 <= month_value <= 12 and 1 <= day_value <= 31:
            results.add((month_value, day_value))
    return results


def parse_case_cities(case: StandardCase) -> list[str]:
    text = case_text(case)
    cities = [city for city in CITY_TERMS if city in text]
    if "山西" in text:
        cities.append("山西")
    return dedupe_strings(cities)


def expand_disasters(values: list[str]) -> set[str]:
    expanded: set[str] = set()
    for value in values:
        if value == "雷暴大风":
            expanded.update(("雷暴", "大风"))
        elif value == "雨雪":
            expanded.update(("雨", "雪", "雨雪"))
        else:
            expanded.add(value)
    return expanded


def case_text(case: StandardCase) -> str:
    return " ".join(
        [
            case.title,
            case.date_range,
            case.summary,
            case.weather_facts,
            case.forecast_focus,
            " ".join(case.disaster_types),
            " ".join(case.affected_areas),
            case.source_pdf,
        ]
    )


def compact_text(text: str, limit: int = 160) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip("，；、。") + "。"


def dedupe_strings(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def dedupe_cases(cases: list[EnrichedCase]) -> list[EnrichedCase]:
    result = []
    seen = set()
    for item in cases:
        if item.case.case_id in seen:
            continue
        seen.add(item.case.case_id)
        result.append(item)
    return result


def dedupe_images(images: list[ImageEvidence]) -> list[ImageEvidence]:
    result = []
    seen = set()
    for image in images:
        if image.image_id in seen:
            continue
        seen.add(image.image_id)
        result.append(image)
    return result


def dedupe_images_by_caption(images: list[ImageEvidence]) -> list[ImageEvidence]:
    result = []
    seen = set()
    for image in images:
        caption_key = re.sub(r"\s+", "", image.caption or "")
        key = (image.source_pdf, image.page_no, caption_key or "page_without_caption")
        if key in seen:
            continue
        seen.add(key)
        result.append(image)
    return result


def first_day(case: StandardCase) -> int:
    text = f"{case.date_range} {case.title}"
    match = re.search(r"(?:\d{1,2}\s*月\s*)?(\d{1,2})\s*(?:日|[-~～至])", text)
    return int(match.group(1)) if match else 99
