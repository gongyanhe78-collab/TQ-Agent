"""用大模型对报告级分析文字做轻量综合增强。"""
from __future__ import annotations

import json
import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from backend.app.services.agent.case_multidim_search.analysis.disaster_profile import disaster_view_text


LOGGER = logging.getLogger("uvicorn.error")


class ReportLlmEnhancer:
    """在规则统计和逐例分析摘要基础上调用一次大模型，避免报告综合阶段撑爆显存。"""

    SECTION_KEYS = ("temporal", "disaster", "spatial", "intensity")
    MAX_CASE_ANALYSES = 12
    CASE_ANALYSIS_CHARS = 420
    SECTION_CHARS = 360
    CACHE_VERSION = "report-llm-v1"

    def __init__(self, llm_client: Any, cache_dir: Path | None = None):
        """保存大模型客户端，并初始化报告级模型结果缓存目录。"""
        self.llm_client = llm_client
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def enhance(
        self,
        analysis: dict[str, Any],
        cases: list[Any],
        aggregations: dict[str, Any],
        hits: list[Any],
        query: Any,
        max_output_tokens: int = 2200,
    ) -> tuple[dict[str, Any], str]:
        """基于逐例摘要和统计事实生成报告级文字；失败时返回原规则分析。"""
        if self.llm_client is None or not hits:
            return analysis, "not_available"
        try:
            if not self.llm_client.is_available():
                return analysis, "not_available"
        except Exception:
            return analysis, "not_available"

        payload = self._payload(analysis, cases, aggregations, hits, query)
        context = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        question = (
            "请基于给定的规则统计事实和逐个例分析摘要，生成正式的省级气象业务报告文字。"
            "必须只输出合法 JSON，不要输出 Markdown、思考过程或 JSON 之外的任何文字。"
            "JSON 必须包含 executive_summary、sections、conclusion、recommendations；"
            "sections 必须包含 temporal、disaster、spatial、intensity 四个字段。"
            "所有数值必须直接使用输入事实，不得重新计算、猜测、放大或混用其他个例的数据。"
            "executive_summary 写一段220到320字的总体概况，先说明检索条件只用于定位相关过程，不代表所有命中个例均达到对应强度标准，再概括过程类型、主要影响区和代表性强度，不写建议、图号或图题。若 query_strength_assessments 非空，必须自然、完整地写入其中的阈值核对结论。"
            "每个 section 写120到180字，使用自然业务语言分析时间、灾种组合、空间落区和强度特征，不使用X/N、覆盖率日志或机械模板句。"
            "conclusion 必须写成300到430字、正好三个自然段，段落之间使用两个换行符，并采用总—分结构。"
            "第一段只作整体特征判断；第二段提炼短临识别、模式偏差、地形效应和复合灾害研判中的主要短板；"
            "第三段归纳证据边界和最需要回应的业务问题。结论负责研判和问题提炼，不得写具体改进措施、不得使用‘建议’句式，也不得出现“后续业务需要提出改进路径”等建议性过渡。"
            "结论不得复述前文统计数字、覆盖率和极值清单，只允许自然融入1到2个确有必要的代表性事实；不得展开具体个例、标题、章节编号、图号或图题。"
            "绝对禁止出现‘原始材料显示’‘图片证据显示’‘图片附近说明显示’‘来源chunk’‘关联原文’‘具备强度证据’‘代表个例显示’等生成或拼接痕迹。"
            "recommendations 必须正好输出4条完整中文建议，每条55到110字并以句号结束，每条只聚焦一个核心问题。"
            "四条建议依次对应：短临预警触发信号与提前量、地形敏感区模式偏差订正、多灾种叠加风险研判、个例库与证据链规范化。"
            "每条按‘问题—措施—目标区域或业务场景’组织，并自然体现吕梁山、五台山、太行山迎风坡、晋中盆地、太原或北部山区。"
            "涉及尚未经本地检验的阈值或指标时，只能使用‘建议探索’‘可参考’‘试点建立’等审慎表述，不得写成未经验证的硬性标准。"
            "不得使用空泛的‘加强监测’，不得截断建议，不得把原始标题和图片编号塞入建议。"
            "输出前逐项自检：无生成痕迹、无大段重复、结论与建议分工清晰、四条建议完整、语言专业精炼且体现山西业务特点；任一项不满足都必须重写。"
        )
        model_name = str(
            getattr(self.llm_client, "model_uid", "")
            or getattr(self.llm_client, "model", "")
            or "unknown"
        )
        cached = self._read_cache("report", payload, model_name, max_output_tokens)
        if cached is not None:
            LOGGER.info("[多维检索][报告增强] 命中缓存：model=%s", model_name)
            return self._merge_analysis(analysis, cached), "cache_hit"
        LOGGER.info(
            "[多维检索][报告增强] 开始调用大模型：model=%s case_count=%s hit_summary_count=%s context_chars=%s max_output_tokens=%s",
            model_name,
            len(cases),
            min(len(hits), self.MAX_CASE_ANALYSES),
            len(context),
            max_output_tokens,
        )
        try:
            answer = str(
                self.llm_client.answer_with_context(
                    question,
                    [context],
                    max_tokens=max_output_tokens,
                )
                or ""
            ).strip()
            LOGGER.info(
                "[多维检索][报告增强] 大模型结束原因：finish_reason=%s response_chars=%s max_output_tokens=%s",
                self._llm_finish_reason(),
                len(answer),
                int(max_output_tokens),
            )
        except Exception:
            LOGGER.exception("[多维检索][报告增强] 大模型调用失败：model=%s", model_name)
            return analysis, "llm_failed"
        # 日志和解析都只保留最终回答，避免思考链泄漏到终端或干扰 JSON 定位。
        answer = self._strip_thinking(answer)
        LOGGER.info(
            "[多维检索][报告增强] 大模型返回：model=%s response_chars=%s\n%s",
            model_name,
            len(answer),
            answer,
        )
        data = self._parse_json(answer)
        if not data:
            LOGGER.warning("[多维检索][报告增强] 大模型未返回合法 JSON，保留规则报告。")
            return analysis, "invalid_json"
        self._write_cache("report", payload, model_name, max_output_tokens, data)
        return self._merge_analysis(analysis, data), "llm_generated"

    def enhance_chart_insights(
        self,
        analysis: dict[str, Any],
        charts: list[Any],
        query: Any,
        max_output_tokens: int = 640,
    ) -> tuple[list[Any], str]:
        """为统计图生成更贴近业务复盘的启示，失败时保留规则文本。"""
        if self.llm_client is None or not charts:
            return charts, "not_available"
        try:
            if not self.llm_client.is_available():
                return charts, "not_available"
        except Exception:
            return charts, "not_available"
        payload = self._chart_payload(analysis, charts, query)
        context = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        question = '请基于检索条件、报告概况和图表数据，为每张图生成一条气象业务启示。必须只输出 JSON，字段为 insights，其中每项包含 chart_key 和 insight。每条 insight 写 40 到 90 个中文字，要说清该图对短临监测、证据复核或风险研判的实际作用，不要重复图面数字，不要空泛说“作为参考”。spatial 图表示各地市命中的个例数，同一个例可同时计入多个地市，严禁把它解释为占比、构成比或互斥份额。'
        model_name = str(
            getattr(self.llm_client, "model_uid", "")
            or getattr(self.llm_client, "model", "")
            or "unknown"
        )
        output_tokens = max(480, min(640, int(max_output_tokens)))
        cached = self._read_cache("chart", payload, model_name, output_tokens)
        if cached is not None:
            LOGGER.info("[多维检索][图表启示] 命中缓存：model=%s", model_name)
            return self._apply_chart_insights(charts, cached), "cache_hit"
        try:
            answer = str(
                self.llm_client.answer_with_context(
                    question,
                    [context],
                    max_tokens=output_tokens,
                )
                or ""
            ).strip()
            LOGGER.info(
                "[多维检索][图表启示] 大模型结束原因：chart_count=%s finish_reason=%s response_chars=%s max_output_tokens=%s",
                len(charts),
                self._llm_finish_reason(),
                len(answer),
                output_tokens,
            )
        except Exception:
            LOGGER.exception("[case-multidim][chart-insight] llm call failed")
            return charts, "llm_failed"
        data = self._parse_json(answer)
        if not data:
            return charts, "invalid_json"
        self._write_cache("chart", payload, model_name, output_tokens, data)
        return self._apply_chart_insights(charts, data), "llm_generated"

    def _apply_chart_insights(self, charts: list[Any], data: dict[str, Any]) -> list[Any]:
        """将模型或缓存中的图表启示应用到当前生成的图表对象。"""
        by_key = {str(item.get("chart_key") or ""): str(item.get("insight") or "").strip() for item in data.get("insights", []) if isinstance(item, dict)}
        for chart in charts:
            key = str(getattr(chart, "chart_key", "") or "")
            insight = by_key.get(key)
            if insight and not (key == "spatial" and ("占比" in insight or "%" in insight or "份额" in insight)):
                chart.interpretation = insight
        return charts

    def _cache_path(self, kind: str, payload: dict[str, Any], model_name: str, max_output_tokens: int) -> Path | None:
        """根据实际模型输入生成稳定缓存键，条件、资料或模型变化时自动失效。"""
        if self.cache_dir is None:
            return None
        cache_input = {
            "cache_version": self.CACHE_VERSION,
            "kind": kind,
            "model": model_name,
            "max_output_tokens": int(max_output_tokens),
            "payload": payload,
        }
        raw = json.dumps(cache_input, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{kind}-{digest}.json"

    def _read_cache(self, kind: str, payload: dict[str, Any], model_name: str, max_output_tokens: int) -> dict[str, Any] | None:
        """读取完整报告或图表启示缓存；损坏或旧版本缓存自动忽略。"""
        path = self._cache_path(kind, payload, model_name, max_output_tokens)
        if path is None or not path.is_file():
            return None
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(cached, dict):
                return None
            data = cached.get("data")
            if cached.get("cache_version") == self.CACHE_VERSION and isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("[多维检索][报告缓存] 读取失败，重新调用模型：%s", path)
        return None

    def _write_cache(self, kind: str, payload: dict[str, Any], model_name: str, max_output_tokens: int, data: dict[str, Any]) -> None:
        """只缓存解析成功的模型 JSON，避免失败结果污染后续查询。"""
        path = self._cache_path(kind, payload, model_name, max_output_tokens)
        if path is None:
            return
        try:
            path.write_text(
                json.dumps({"cache_version": self.CACHE_VERSION, "data": data}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            LOGGER.warning("[多维检索][报告缓存] 写入失败，不影响本次结果：%s", path)

    def _llm_finish_reason(self) -> str:
        """读取当前线程的生成结束原因，兼容旧客户端。"""
        getter = getattr(self.llm_client, "get_last_finish_reason", None)
        if callable(getter):
            try:
                return str(getter() or "unknown")
            except Exception:
                return "unknown"
        return str(getattr(self.llm_client, "last_finish_reason", "unknown") or "unknown")

    def _chart_payload(self, analysis: dict[str, Any], charts: list[Any], query: Any) -> dict[str, Any]:
        """给图表启示生成构造小型事实包，避免再次塞入原文长片段。"""
        return {
            "query": self._compact_query(query),
            "executive_summary": self._clip(analysis.get("executive_summary", ""), 260),
            "sections": {key: self._clip(value, 180) for key, value in (analysis.get("sections") or {}).items()},
            "charts": [
                {
                    "chart_key": str(getattr(chart, "chart_key", "") or ""),
                    "title": str(getattr(chart, "title", "") or ""),
                    "chart_type": str(getattr(chart, "chart_type", "") or ""),
                    "labels": list(getattr(chart, "labels", []) or [])[:10],
                    "values": list(getattr(chart, "values", []) or [])[:10],
                    "x_label": str(getattr(chart, "x_label", "") or ""),
                    "y_label": str(getattr(chart, "y_label", "") or ""),
                    "sample_size": getattr(chart, "sample_size", 0),
                }
                for chart in charts
            ],
        }

    def _payload(
        self,
        analysis: dict[str, Any],
        cases: list[Any],
        aggregations: dict[str, Any],
        hits: list[Any],
        query: Any,
    ) -> dict[str, Any]:
        """构造瘦身事实包：统计覆盖全量，逐例只传摘要，避免长上下文爆显存。"""
        selected_hits = hits[: self.MAX_CASE_ANALYSES]
        return {
            "query": self._compact_query(query),
            "case_count": len(cases),
            "displayed_case_count": len(hits),
            "omitted_case_analysis_count": max(0, len(hits) - len(selected_hits)),
            "disaster_view": analysis.get("disaster_view") or disaster_view_text(list(getattr(query, "disaster_types", []) or [])),
            "rule_analysis": self._compact_rule_analysis(analysis),
            "aggregations": self._compact_aggregations(aggregations),
            "case_analyses": [self._hit_dict(hit) for hit in selected_hits],
            "overview_intensity_tables": analysis.get("overview_intensity_tables") or [],
            "overview_intensity_columns": analysis.get("overview_intensity_columns") or [],
            "overview_intensity_table": analysis.get("overview_intensity_table") or [],
            "intensity_extremes": analysis.get("intensity_extremes") or "",
            "query_strength_assessments": analysis.get("query_strength_assessments") or [],
        }

    def _compact_query(self, query: Any) -> dict[str, Any]:
        """只保留报告写作需要的检索条件，减少无关字段。"""
        data = self._to_dict(query)
        # 报告级提示词只保留真正参与检索的条件，避免中心点强度、关键词等旧条件继续影响大模型表述。
        keep = ("start_date", "end_date", "years", "months", "disaster_types", "cities", "areas")
        return {key: data.get(key) for key in keep if data.get(key)}

    def _compact_rule_analysis(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """保留规则结论作为事实底稿，但截断长段落。"""
        sections = analysis.get("sections") or {}
        return {
            # 报告级概况必须由逐例摘要和极值清单重新生成，旧规则概况只作兜底，不进入模型主事实。
            "executive_summary": "",
            "sections": {key: self._clip(sections.get(key, ""), self.SECTION_CHARS) for key in self.SECTION_KEYS},
            "conclusion": self._clip(analysis.get("conclusion", ""), 420),
            "recommendations": [self._clip(item, 90) for item in (analysis.get("recommendations") or [])[:4]],
            "limitations": [self._clip(item, 90) for item in (analysis.get("limitations") or [])[:5]],
        }

    def _compact_aggregations(self, aggregations: dict[str, Any]) -> dict[str, Any]:
        """只传 top 统计和共现矩阵，具体频次仍由规则统计提供。"""
        aggregations = aggregations or {}
        return {
            "case_count": aggregations.get("case_count", 0),
            "month_counts": self._top_items(aggregations.get("month_counts", {}), 12),
            "disaster_counts": self._top_items(aggregations.get("disaster_counts", {}), 10),
            "city_counts": self._top_items(aggregations.get("city_counts", {}), 10),
            "intensity_values": self._compact_intensity_values(aggregations.get("intensity_values", {})),
            "intensity_case_values": self._compact_intensity_case_values(aggregations.get("intensity_case_values", {})),
            "disaster_cooccurrence": self._compact_cooccurrence(aggregations.get("disaster_cooccurrence", {})),
        }

    def _top_items(self, value: Any, limit: int) -> dict[str, Any]:
        """按原有顺序保留前若干项，避免把完整统计表塞给模型。"""
        if not isinstance(value, dict):
            return {}
        return dict(list(value.items())[:limit])

    def _compact_intensity_values(self, value: Any) -> dict[str, dict[str, float]]:
        """把强度数组压缩成数量、范围和均值，模型不需要看到全部原始数组。"""
        if not isinstance(value, dict):
            return {}
        result: dict[str, dict[str, float]] = {}
        for name, values in list(value.items())[:8]:
            numbers = [float(item) for item in values if isinstance(item, (int, float))]
            if not numbers:
                continue
            result[str(name)] = {
                "count": len(numbers),
                "min": min(numbers),
                "max": max(numbers),
                "mean": round(sum(numbers) / len(numbers), 1),
            }
        return result

    def _compact_intensity_case_values(self, value: Any) -> dict[str, list[dict[str, Any]]]:
        """压缩指标对应个例列表，给报告模型提供可引用的代表指标。"""
        if not isinstance(value, dict):
            return {}
        result: dict[str, list[dict[str, Any]]] = {}
        for name, rows in list(value.items())[:8]:
            if not isinstance(rows, list):
                continue
            clean_rows = []
            for row in rows[:8]:
                if not isinstance(row, dict):
                    continue
                clean_rows.append(
                    {
                        "title": self._clip(row.get("title", ""), 40),
                        "value": row.get("value", ""),
                        "unit": row.get("unit", ""),
                        "location": self._clip(row.get("location", ""), 20),
                    }
                )
            if clean_rows:
                result[str(name)] = clean_rows
        return result
    def _compact_cooccurrence(self, value: Any) -> dict[str, Any]:
        """共现热力图只保留前 6 个灾种，降低矩阵体积。"""
        if not isinstance(value, dict):
            return {}
        labels = list(value.get("labels") or [])[:6]
        matrix = value.get("matrix") or []
        return {"labels": labels, "matrix": [list(row)[:6] for row in matrix[:6]]}

    def _hit_dict(self, hit: Any) -> dict[str, Any]:
        """整理逐个例分析的压缩摘要，供总报告综合结论引用。"""
        case = getattr(hit, "case", {}) or {}
        analysis = str(getattr(hit, "analysis", "") or "")
        return {
            "case_id": case.get("case_id", ""),
            "title": case.get("title", ""),
            "date_range": case.get("date_range", ""),
            "disaster_types": case.get("disaster_types", []),
            "analysis_focus_disaster": case.get("matched_disaster") or case.get("analysis_focus_disaster", ""),
            "affected_areas": case.get("city_tags") or case.get("affected_areas", []),
            "analysis_summary": self._summarize_case_analysis(analysis),
            "intensity_metrics": [self._metric_dict(metric) for metric in getattr(hit, "intensity_metrics", [])[:4]],
            "evidence_image_count": len(getattr(hit, "evidence_images", []) or []),
        }

    def _summarize_case_analysis(self, text: str) -> str:
        """从逐例大模型长文本中抽取少量关键句，既保留质量又控制显存。"""
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if len(cleaned) <= self.CASE_ANALYSIS_CHARS:
            return cleaned
        sentences = [item.strip() for item in re.split(r"(?<=[。！？；])", cleaned) if item.strip()]
        keywords = ("过程", "主", "影响", "强度", "缺口", "演变", "环流", "机制", "风险", "证据", "预警", "模式")
        picked: list[str] = []
        for sentence in sentences:
            if any(keyword in sentence for keyword in keywords):
                picked.append(sentence)
            if len("".join(picked)) >= self.CASE_ANALYSIS_CHARS:
                break
        if not picked:
            picked = sentences[:3]
        return self._clip("".join(picked), self.CASE_ANALYSIS_CHARS)

    def _metric_dict(self, metric: Any) -> dict[str, Any]:
        """把强度指标转成普通字典，并截断原文证据。"""
        if hasattr(metric, "model_dump"):
            data = metric.model_dump()
        else:
            data = dict(getattr(metric, "__dict__", {}) or {})
        return {
            "metric_name": data.get("metric_name", ""),
            "value": data.get("value", ""),
            "unit": data.get("unit", ""),
            "location": data.get("location", ""),
            "source_text": self._clip(data.get("source_text", ""), 90),
        }

    def _to_dict(self, value: Any) -> dict[str, Any]:
        """兼容 Pydantic 对象和普通字典，便于写入提示上下文。"""
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, dict):
            return value
        return {}

    def _clip(self, value: Any, limit: int) -> str:
        """按字符数安全截断，避免提示词上下文无意膨胀。"""
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text if len(text) <= limit else text[:limit].rstrip() + "…"

    def _strip_thinking(self, text: str) -> str:
        """清理模型返回里的 <think> 思考链，只保留最终 JSON。"""
        cleaned = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _parse_json(self, answer: str) -> dict[str, Any]:
        """从模型返回中提取 JSON，并修复报告字段之间常见的漏逗号。"""
        # 报告模型可能仍返回思考链，必须在定位 JSON 边界前统一清理。
        cleaned = self._strip_thinking(answer)
        cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
        try:
            data = json.loads(cleaned)
        except Exception:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start < 0 or end <= start:
                return {}
            candidate = cleaned[start:end + 1]
            try:
                data = json.loads(candidate)
            except Exception:
                # 仅修复报告协议已知字段在换行处漏写逗号的情况，不做任意 JSON 猜测。
                repaired = re.sub(
                    r'([}\]\"])\s*\r?\n\s*("(?:executive_summary|sections|conclusion|recommendations|temporal|disaster|spatial|intensity)"\s*:)',
                    r'\1,\n  \2',
                    candidate,
                )
                try:
                    data = json.loads(repaired)
                except Exception:
                    return {}
        return data if isinstance(data, dict) else {}

    def _merge_analysis(self, analysis: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """只接受完整、无生成痕迹的白名单字段，防止模型文本污染最终报告。"""
        merged = dict(analysis)
        if isinstance(data.get("executive_summary"), str) and data["executive_summary"].strip():
            merged["executive_summary"] = self._ensure_summary_assessments(
                data["executive_summary"].strip(),
                list(analysis.get("query_strength_assessments") or []),
            )
        sections = dict(merged.get("sections") or {})
        llm_sections = data.get("sections") if isinstance(data.get("sections"), dict) else {}
        for key in self.SECTION_KEYS:
            value = llm_sections.get(key)
            if isinstance(value, str) and value.strip():
                sections[key] = value.strip()
        merged["sections"] = sections

        conclusion = str(data.get("conclusion") or "").strip()
        if self._valid_conclusion(conclusion):
            merged["conclusion"] = self._paragraphize_conclusion(conclusion)

        recommendations = data.get("recommendations")
        if isinstance(recommendations, list):
            cleaned = [str(item).strip() for item in recommendations if str(item).strip()]
            if len(cleaned) == 4 and all(self._valid_recommendation(item) for item in cleaned):
                merged["recommendations"] = cleaned
        merged["llm_report_enhanced"] = True
        return merged

    def _ensure_summary_assessments(self, summary: str, assessments: list[str]) -> str:
        """把确定性阈值核对结论补回总体概况，防止模型把检索标签写成强度达标。"""
        text = str(summary or "").strip()
        for assessment in assessments:
            sentence = str(assessment or "").strip()
            if sentence and sentence[:18] not in text:
                text = f"{text}{sentence}"
        return text
    def _valid_conclusion(self, value: str) -> bool:
        """验收综合结论的完整性和业务化表达，命中生成痕迹时保留规则兜底稿。"""
        forbidden = (
            "原始材料显示", "图片证据显示", "图片附近说明显示", "来源chunk",
            "关联原文", "具备强度证据", "代表个例显示", "后续业务部分需要", "提出改进路径",
        )
        text = str(value or "").strip()
        return bool(
            300 <= len(text) <= 900
            and text.endswith(("。", "！", "？"))
            and not any(token in text for token in forbidden)
            and not re.search(r"图\s*\d+|第[一二三四五六七八九十]+章", text)
            and not re.search(r"(?:[^；。]{0,36}；){3,}", text)
            and "建议" not in text
        )

    def _valid_recommendation(self, value: str) -> bool:
        """只接受长度合理、语义完整且不含图号或底层术语的业务建议。"""
        text = str(value or "").strip()
        forbidden = ("原始材料显示", "图片证据显示", "来源chunk", "关联原文", "具备强度证据")
        return bool(
            55 <= len(text) <= 140
            and text.endswith(("。", "！", "？"))
            and not any(token in text for token in forbidden)
            and not re.search(r"图\s*\d+", text)
            and any(token in text for token in ("建议", "应", "需", "可参考", "试点", "探索"))
        )

    def _paragraphize_conclusion(self, value: str) -> str:
        """把结论稳定整理为三个自然段，网页和 PDF 都可直接按换行排版。"""
        text = re.sub(r"[ \t]+", " ", str(value or "")).strip()
        paragraphs = [item.strip() for item in re.split(r"\n+", text) if item.strip()]
        if len(paragraphs) >= 3:
            return "\n\n".join(paragraphs[:3])
        sentences = [item.strip() for item in re.findall(r"[^。！？]+[。！？]", text) if item.strip()]
        if len(sentences) < 3:
            return text
        groups: list[list[str]] = [[], [], []]
        for index, sentence in enumerate(sentences):
            group_index = min(2, index * 3 // len(sentences))
            groups[group_index].append(sentence)
        return "\n\n".join("".join(group) for group in groups if group)


