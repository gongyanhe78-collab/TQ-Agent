
"""LLM assisted extraction of structured intensity metrics from all case chunks."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from backend.app.models import DocumentChunk, StandardCase
from backend.app.services.agent.case_multidim_search.analysis.disaster_profile import metric_names_for_disasters
from backend.app.services.agent.case_multidim_search.schemas import IntensityMetric

LOGGER = logging.getLogger("uvicorn.error")

MAX_HOURLY_PRECIP = "\u6700\u5927\u5c0f\u65f6\u96e8\u5f3a"
MAX_PROCESS_PRECIP = "\u8fc7\u7a0b\u6700\u5927\u964d\u6c34\u91cf"
MAX_SNOW_DEPTH = "\u6700\u5927\u79ef\u96ea\u6df1\u5ea6"
MAX_WIND_SPEED = "\u6700\u5927\u98ce\u901f"
MAX_WIND_GUST = "\u6781\u5927\u98ce\u901f"
WIND_LEVEL = "\u9635\u98ce\u98ce\u529b"
MAX_TEMPERATURE = "\u6700\u9ad8\u6c14\u6e29"
MIN_TEMPERATURE = "\u6700\u4f4e\u6c14\u6e29"
TEMP_DROP = "\u8fc7\u7a0b\u964d\u6e29\u5e45\u5ea6"
MAX_SNOWFALL = "\u8fc7\u7a0b\u6700\u5927\u964d\u96ea\u91cf"
MIN_VISIBILITY = "\u6700\u4f4e\u80fd\u89c1\u5ea6"
MAX_HAIL = "\u6700\u5927\u51b0\u96f9\u76f4\u5f84"
RADAR_REFLECTIVITY = "\u96f7\u8fbe\u56de\u6ce2\u5f3a\u5ea6"


class LlmIntensityExtractor:
    """Use an LLM to supplement rule-based intensity extraction with a strict backend whitelist."""

    # 指标证据协议变化后必须重新生成缓存，避免继续使用旧的缺失字段结果。
    PROMPT_VERSION = "llm-intensity-v8-reject-wind-warning-values"
    METRIC_UNITS = {
        MAX_HOURLY_PRECIP: "mm/h",
        MAX_PROCESS_PRECIP: "mm",
        MAX_SNOW_DEPTH: "cm",
        MAX_WIND_SPEED: "m/s",
        MAX_WIND_GUST: "m/s",
        WIND_LEVEL: "\u7ea7",
        MAX_TEMPERATURE: "\u2103",
        MIN_TEMPERATURE: "\u2103",
        TEMP_DROP: "\u2103",
        MAX_SNOWFALL: "mm",
        MIN_VISIBILITY: "km",
        MAX_HAIL: "mm",
        RADAR_REFLECTIVITY: "dBZ",
    }
    ALIASES = {
        MAX_HOURLY_PRECIP: ("\u5c0f\u65f6\u96e8\u5f3a", "\u5c0f\u65f6\u6700\u5927\u96e8\u91cf", "\u6700\u5927\u5c0f\u65f6\u964d\u6c34\u91cf"),
        MAX_PROCESS_PRECIP: ("\u8fc7\u7a0b\u964d\u6c34\u91cf", "\u7d2f\u8ba1\u964d\u6c34\u91cf", "\u6700\u5927\u964d\u6c34\u91cf"),
        MAX_SNOW_DEPTH: ("\u79ef\u96ea\u6df1\u5ea6", "\u96ea\u6df1"),
        MAX_WIND_SPEED: ("\u6700\u5927\u5e73\u5747\u98ce\u901f", "\u98ce\u901f\u6700\u5927"),
        MAX_WIND_GUST: ("\u6700\u5927\u9635\u98ce\u98ce\u901f", "\u9635\u98ce\u98ce\u901f", "\u77ac\u65f6\u98ce\u901f"),
        WIND_LEVEL: ("\u9635\u98ce\u7b49\u7ea7", "\u98ce\u529b\u7b49\u7ea7", "\u6700\u5927\u98ce\u529b"),
        MAX_TEMPERATURE: ("\u65e5\u6700\u9ad8\u6c14\u6e29", "\u6700\u9ad8\u6e29\u5ea6"),
        MIN_TEMPERATURE: ("\u65e5\u6700\u4f4e\u6c14\u6e29", "\u6700\u4f4e\u6e29\u5ea6"),
        TEMP_DROP: ("\u964d\u6e29\u5e45\u5ea6", "\u6700\u5927\u964d\u6e29"),
        MAX_SNOWFALL: ("\u964d\u96ea\u91cf", "\u6700\u5927\u964d\u96ea\u91cf"),
        MIN_VISIBILITY: ("\u6700\u5c0f\u80fd\u89c1\u5ea6", "\u80fd\u89c1\u5ea6\u4f4e\u503c"),
        MAX_HAIL: ("\u51b0\u96f9\u76f4\u5f84", "\u51b0\u96f9\u6700\u5927\u76f4\u5f84"),
        RADAR_REFLECTIVITY: ("\u7ec4\u5408\u53cd\u5c04\u7387", "\u56de\u6ce2\u5f3a\u5ea6"),
    }
    METRIC_NAMES = set(METRIC_UNITS)

    def __init__(self, llm_client=None, cache_dir: Path | None = None):
        self.llm_client = llm_client
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def collect_numeric_evidence(self, chunks: list[DocumentChunk]) -> list[dict[str, Any]]:
        """从全部正式正文收集含数字的完整句和必要承接句，不提前判断指标极值。"""
        evidence: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        connector_pattern = re.compile(r"^(?:其中|分别|最大值|最小值|极值|该站|上述|同期|过程|区域站|国家站)")
        for chunk in sorted(chunks, key=lambda item: getattr(item, "chunk_no", 0)):
            sentences = self._complete_sentences(str(chunk.content or ""))
            selected_indexes: set[int] = set()
            for index, sentence in enumerate(sentences):
                if self._contains_number(sentence):
                    selected_indexes.add(index)
                    # “最大值出现在……”等承接句可能没有数字，需要与数值句一起交给模型。
                    if index + 1 < len(sentences) and connector_pattern.search(sentences[index + 1]):
                        selected_indexes.add(index + 1)
            for index in sorted(selected_indexes):
                sentence = sentences[index].strip()
                key = (chunk.chunk_id, re.sub(r"\s+", "", sentence))
                if not sentence or key in seen:
                    continue
                seen.add(key)
                evidence.append(
                    {
                        "source_chunk_id": chunk.chunk_id,
                        "chunk_no": getattr(chunk, "chunk_no", 0),
                        "sentence_index": index,
                        "text": sentence,
                    }
                )
        return evidence

    def metrics_from_payload(
        self,
        payload: dict[str, Any],
        chunks: list[DocumentChunk],
        disaster_names=None,
    ) -> list[IntensityMetric]:
        """校验一次逐例调用返回的指标，拒绝跨个例来源和不存在的证据原句。"""
        raw_metrics = payload.get("metrics") if isinstance(payload, dict) else None
        if not isinstance(raw_metrics, list):
            return []
        focus_metrics = set(metric_names_for_disasters(disaster_names) or self.METRIC_NAMES)
        allowed_chunk_ids = {chunk.chunk_id for chunk in chunks}
        metrics: list[IntensityMetric] = []
        for index, item in enumerate(raw_metrics):
            metric, reason = self._metric_from_payload_with_reason(item, allowed_chunk_ids, chunks)
            if metric is None:
                LOGGER.info(
                    "[多维检索][强度指标过滤] rejected index=%s reason=%s raw=%s",
                    index,
                    reason,
                    item,
                )
                continue
            if metric.metric_name in focus_metrics:
                metrics.append(metric)
            else:
                LOGGER.info(
                    "[多维检索][强度指标过滤] rejected index=%s reason=not_in_disaster_focus metric=%s focus=%s",
                    index,
                    metric.metric_name,
                    sorted(focus_metrics),
                )
        return self._merge_metrics([], metrics)

    def _complete_sentences(self, text: str) -> list[str]:
        """恢复 PDF 排版换行后按中文句末标点切分，避免小数点被误拆。"""
        normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"(?<=[\u4e00-\u9fff0-9０-９，、；：℃%])\n\s*(?=[\u4e00-\u9fff0-9０-９])", "", normalized)
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n+", "\n", normalized).strip()
        sentences: list[str] = []
        for paragraph in normalized.split("\n"):
            parts = re.findall(r"[^。！？]+[。！？]?", paragraph)
            sentences.extend(part.strip() for part in parts if part.strip())
        return sentences

    def _contains_number(self, text: str) -> bool:
        """识别阿拉伯数字、全角数字和常见中文数词。"""
        return bool(re.search(r"[0-9０-９]|(?:[一二三四五六七八九十百]+\s*级)", str(text or "")))

    def enrich(self, case: StandardCase, chunks: list[DocumentChunk], rule_metrics: list[IntensityMetric], disaster_names=None, enabled: bool = True) -> list[IntensityMetric]:
        if not enabled or self.llm_client is None or not chunks:
            return list(rule_metrics)
        usable_chunks = [chunk for chunk in chunks if str(chunk.content or "").strip()]
        if not usable_chunks:
            return list(rule_metrics)
        cached = self._read_cache(case, usable_chunks, disaster_names)
        if cached is not None:
            return self._merge_metrics(rule_metrics, cached)
        try:
            llm_metrics = self._extract_with_llm(case, usable_chunks, disaster_names)
        except Exception:
            LOGGER.exception("[case-multidim][intensity] llm extraction failed: case_id=%s", case.case_id)
            llm_metrics = []
        self._write_cache(case, usable_chunks, disaster_names, llm_metrics)
        return self._merge_metrics(rule_metrics, llm_metrics)

    def extract_batch_from_evidence(self, case: StandardCase, chunks: list[DocumentChunk]) -> list[IntensityMetric]:
        """仅根据当前六个 chunk 的数字证据补充指标，不参与正文生成或跨批次取极值。"""
        if self.llm_client is None or not chunks:
            return []
        evidence = self.collect_numeric_evidence(chunks)
        if not evidence:
            return []
        metric_lines = [f"- {name}: unit={unit}" for name, unit in self.METRIC_UNITS.items()]
        context_blocks = [
            f"case_id={case.case_id}\ntitle={case.title}\ndate={case.date_range}",
            "允许指标：\n" + "\n".join(metric_lines),
            "本批全部数字证据句：\n" + "\n".join(
                f"[{item.get('source_chunk_id')}|句{item.get('sentence_index', 0)}] {item.get('text', '')}"
                for item in evidence
            ),
        ]
        question = (
            "你只负责从当前批次数字证据句中穷尽提取气象过程实况指标，不写分析正文，不引用图片，不跨批次推理。"
            "逐句检查每个数值，保留所有明确的过程实况、站点实测和范围端点；日期、时次、站数、图号、预报阈值、月气候概况不得作为指标。"
            "指标名称和单位必须严格使用允许指标；降雪量与降水量不得混淆，小时雨量归入最大小时雨强。"
            "source_text 必须逐字使用本批给出的完整证据句，source_chunk_id 必须使用该句方括号内编号。"
            "只输出 JSON：{\"metrics\":[{\"metric_name\":\"过程最大降水量\",\"value\":0,\"unit\":\"mm\",\"location\":\"\",\"relation\":\"站点实测\",\"source_chunk_id\":\"chunk-id\",\"source_text\":\"原始完整句\",\"confidence\":0.9}]}。"
        )
        try:
            answer = str(self.llm_client.answer_with_context(question, context_blocks, max_tokens=640) or "").strip()
        except Exception:
            LOGGER.exception("[多维检索][批次指标补充] 大模型调用失败：case_id=%s", case.case_id)
            return []
        LOGGER.info(
            "[多维检索][批次指标补充] 结束原因：case_id=%s finish_reason=%s evidence_count=%s response_chars=%s",
            case.case_id,
            self._llm_finish_reason(),
            len(evidence),
            len(answer),
        )
        payload = self._parse_json_object(answer)
        return self.metrics_from_payload(payload, chunks, disaster_names=None)

    def _extract_with_llm(self, case: StandardCase, chunks: list[DocumentChunk], disaster_names) -> list[IntensityMetric]:
        focus_metrics = [name for name in (metric_names_for_disasters(disaster_names) or []) if name in self.METRIC_NAMES]
        if not focus_metrics:
            focus_metrics = list(self.METRIC_UNITS)
        metric_lines = [f"- {name}: unit={self.METRIC_UNITS[name]}, aliases={','.join(self.ALIASES.get(name, ())) }" for name in focus_metrics]
        numeric_evidence = self.collect_numeric_evidence(chunks)
        context_blocks = [
            f"case_id={case.case_id}\ntitle={case.title}\ndate={case.date_range}\ndisasters={','.join(case.disaster_types or [])}",
            "metrics:\n" + "\n".join(metric_lines),
            "全部数字证据句（仅供指标抽取）：\n"
            + "\n".join(
                f"[{item.get('source_chunk_id')}|句{item.get('sentence_index', 0)}] {item.get('text', '')}"
                for item in numeric_evidence
            ),
        ]
        question = (
            "Extract weather disaster intensity metrics from all supplied complete numeric evidence sentences of this single case. "
            "Use only the provided text. Do not infer missing values. Use controlled metric names and units only. "
            "Ignore monthly/provincial climate overview values such as full-month Shanxi precipitation ranges; extract only metrics belonging to this case process. "
            "Read ALL supplied evidence sentences before deciding each metric. When a sentence gives a value range, maximum metrics use the upper endpoint and minimum visibility uses the lower endpoint. "
            "When both national-station max and regional-station max are given for the same metric, keep the larger value as the case extreme. "
            "Return strict JSON only: {\"metrics\":[{\"metric_name\":\"" + MAX_HOURLY_PRECIP + "\",\"value\":38.7,\"unit\":\"mm/h\","
            "\"location\":\"\",\"relation\":\"max\",\"source_chunk_id\":\"chunk-id\",\"source_text\":\"original sentence\",\"confidence\":0.9}]}"
        )
        # 独立指标调用作为备用链路时只接收数字证据，预算控制在 500～800 token 区间。
        answer = str(self.llm_client.answer_with_context(question, context_blocks, max_tokens=640) or "").strip()
        finish_reason = self._llm_finish_reason()
        LOGGER.info(
            "[多维检索][独立强度抽取] 大模型结束原因：case_id=%s finish_reason=%s response_chars=%s",
            case.case_id,
            finish_reason,
            len(answer),
        )
        payload = self._parse_json_object(answer)
        raw_metrics = payload.get("metrics") if isinstance(payload, dict) else None
        if not isinstance(raw_metrics, list):
            return []
        allowed_chunk_ids = {chunk.chunk_id for chunk in chunks}
        metrics = []
        for index, item in enumerate(raw_metrics):
            metric, reason = self._metric_from_payload_with_reason(item, allowed_chunk_ids, chunks)
            if metric is not None:
                metrics.append(metric)
            else:
                LOGGER.info(
                    "[多维检索][强度指标过滤] rejected index=%s reason=%s raw=%s",
                    index,
                    reason,
                    item,
                )
        return metrics

    def _llm_finish_reason(self) -> str:
        """读取当前线程的生成结束原因，兼容尚未提供元信息的客户端。"""
        getter = getattr(self.llm_client, "get_last_finish_reason", None)
        if callable(getter):
            try:
                return str(getter() or "unknown")
            except Exception:
                return "unknown"
        return str(getattr(self.llm_client, "last_finish_reason", "unknown") or "unknown")

    def _chunk_blocks(self, chunks: list[DocumentChunk]) -> list[str]:
        blocks = []
        for chunk in chunks:
            blocks.append(f"chunk_id={chunk.chunk_id}\nchunk_no={getattr(chunk, 'chunk_no', '')}\ntext:\n{chunk.content}")
        return blocks

    def _metric_from_payload(self, item: Any, allowed_chunk_ids: set[str], chunks: list[DocumentChunk]) -> IntensityMetric | None:
        """兼容旧调用方，仅返回校验通过的指标对象。"""
        metric, _ = self._metric_from_payload_with_reason(item, allowed_chunk_ids, chunks)
        return metric

    def _metric_from_payload_with_reason(
        self,
        item: Any,
        allowed_chunk_ids: set[str],
        chunks: list[DocumentChunk],
    ) -> tuple[IntensityMetric | None, str]:
        """校验指标并返回可观测的拒绝原因，便于定位字段丢失环节。"""
        if not isinstance(item, dict):
            return None, "invalid_item_type"
        metric_name = self._normalize_metric_name(str(item.get("metric_name") or ""))
        if metric_name not in self.METRIC_NAMES:
            return None, "metric_name_not_allowed"
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            return None, "value_not_numeric"
        unit = self._normalize_unit(str(item.get("unit") or ""), metric_name)
        value, unit = self._convert_unit(metric_name, value, unit)
        chunk_id = str(item.get("source_chunk_id") or "").strip()
        source_text = str(item.get("source_text") or "").strip()
        if unit != self.METRIC_UNITS[metric_name]:
            return None, "unit_mismatch"
        if self._should_skip_metric(metric_name, value, source_text):
            return None, "business_rule_rejected"
        if chunk_id not in allowed_chunk_ids:
            chunk_id = self._infer_source_chunk_id(source_text, chunks)
        if not chunk_id or not source_text:
            return None, "missing_source_reference"
        # 校验模型提供的来源片段，确保它属于当前个例并且能在原文中核验。
        source_chunk = next((chunk for chunk in chunks if chunk.chunk_id == chunk_id), None)
        if source_chunk is None:
            return None, "source_chunk_not_found"
        if not self._source_text_matches_chunk(source_text, source_chunk.content):
            return None, "source_text_not_found_in_chunk"
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.88))))
        except (TypeError, ValueError):
            confidence = 0.88
        return IntensityMetric(
            metric_name=metric_name,
            value=value,
            unit=unit,
            location=str(item.get("location") or "").strip(),
            relation=str(item.get("relation") or "").strip(),
            source_chunk_id=chunk_id,
            source_text=source_text[:300],
            confidence=confidence,
        ), "accepted"

    def _source_text_matches_chunk(self, source_text: str, chunk_text: str) -> bool:
        """校验模型引用的原文片段是否确实存在于对应正文中。"""
        source = re.sub(r"\s+", "", str(source_text or ""))
        target = re.sub(r"\s+", "", str(chunk_text or ""))
        if not source or not target:
            return False
        # 使用去空白后的前缀核验原文，兼容模型少量截断但拒绝跨个例数值。
        return source[:24] in target

    def _normalize_metric_name(self, value: str) -> str:
        text = str(value or "").strip()
        if text in self.METRIC_NAMES:
            return text
        for metric_name, aliases in self.ALIASES.items():
            if text in aliases:
                return metric_name
        if text == "\u6700\u5927\u964d\u6c34\u91cf":
            return MAX_PROCESS_PRECIP
        if text in {"\u9635\u98ce\u7b49\u7ea7", "\u6700\u5927\u98ce\u529b\u7b49\u7ea7"}:
            return WIND_LEVEL
        return text

    def _normalize_unit(self, unit: str, metric_name: str) -> str:
        text = str(unit or "").replace(" ", "").strip()
        mapping = {
            "\u6beb\u7c73": "mm", "\u6beb\u7c73/\u5c0f\u65f6": "mm/h", "\u6beb\u7c73\u6bcf\u5c0f\u65f6": "mm/h",
            "mm/hr": "mm/h", "\u7c73/\u79d2": "m/s", "\u7c73\u6bcf\u79d2": "m/s",
            "\u00b0C": "\u2103", "C": "\u2103", "\u5398\u7c73": "cm", "\u7ea7\u4ee5\u4e0a": "\u7ea7",
            "\u516c\u91cc": "km", "\u5343\u7c73": "km", "\u7c73": "m", "DBZ": "dBZ", "dBz": "dBZ",
        }
        if metric_name == MAX_HOURLY_PRECIP and text == "mm":
            return "mm/h"
        return mapping.get(text, text)

    def _convert_unit(self, metric_name: str, value: float, unit: str) -> tuple[float, str]:
        if metric_name == MIN_VISIBILITY and unit == "m":
            return value / 1000.0, "km"
        if metric_name == MAX_HAIL and unit == "cm":
            return value * 10.0, "mm"
        return value, unit

    def _should_skip_metric(self, metric_name: str, value: float, source_text: str = "") -> bool:
        positive = {MAX_HOURLY_PRECIP, MAX_PROCESS_PRECIP, MAX_SNOW_DEPTH, MAX_WIND_SPEED, MAX_WIND_GUST, WIND_LEVEL, MAX_SNOWFALL, MAX_HAIL, RADAR_REFLECTIVITY}
        if metric_name in positive and value <= 0:
            return True
        if metric_name in {MAX_WIND_SPEED, MAX_WIND_GUST, WIND_LEVEL} and self._is_wind_forecast_or_warning(source_text):
            return True
        if metric_name in {MAX_HOURLY_PRECIP, MAX_PROCESS_PRECIP} and self._is_monthly_precip_overview(source_text):
            return True
        if metric_name == MAX_TEMPERATURE and not (25 <= value <= 55):
            return True
        if metric_name == MIN_TEMPERATURE and value >= 5:
            return True
        if metric_name == TEMP_DROP and not (0 < value <= 30):
            return True
        if metric_name in {MAX_WIND_SPEED, MAX_WIND_GUST} and value > 80:
            return True
        if metric_name == WIND_LEVEL and value > 18:
            return True
        if metric_name in {MAX_HOURLY_PRECIP, MAX_PROCESS_PRECIP} and value > 1000:
            return True
        return False

    def _is_wind_forecast_or_warning(self, source_text: str) -> bool:
        """拒绝预警预报里的风速或风级阈值，只接受已发生过程实况。"""
        text = re.sub(r"\s+", "", str(source_text or ""))
        return bool(
            "\u9884\u8b66\u533a\u57df" in text
            or ("\u9884\u8b66" in text and any(token in text for token in ("\u9884\u8ba1", "\u53ef\u8fbe", "\u672a\u6765")))
            or ("\u9884\u8ba1" in text and any(token in text for token in ("\u9635\u98ce", "\u98ce\u529b", "\u98ce\u901f")))
        )

    def _is_monthly_precip_overview(self, source_text: str) -> bool:
        """\u8bc6\u522b\u6708\u5ea6\u6216\u5168\u7701\u6c14\u5019\u6982\u51b5\u4e2d\u7684\u964d\u6c34\u80cc\u666f\u503c\u3002"""
        compact_text = re.sub(r"\s+", "", str(source_text or ""))
        if not compact_text:
            return False
        if re.search(r"\d{4}\u5e74\d{1,2}\u6708", compact_text) and "\u5c71\u897f\u7701\u964d\u6c34\u91cf\u4ecb\u4e8e" in compact_text:
            return True
        overview_terms = ("\u6708\u964d\u6c34\u91cf", "\u5e73\u5747\u964d\u6c34\u91cf", "\u5168\u7701\u5e73\u5747", "\u8f83\u5e38\u5e74", "\u6c14\u5019\u6982\u51b5", "\u6c14\u8c61\u6982\u51b5")
        return "\u5c71\u897f\u7701\u964d\u6c34\u91cf" in compact_text and any(term in compact_text for term in overview_terms)

    def _infer_source_chunk_id(self, source_text: str, chunks: list[DocumentChunk]) -> str:
        compact = re.sub(r"\s+", "", source_text or "")
        if not compact:
            return ""
        for chunk in chunks:
            chunk_text = re.sub(r"\s+", "", chunk.content or "")
            if compact[:30] and compact[:30] in chunk_text:
                return chunk.chunk_id
        return ""

    def _merge_metrics(self, rule_metrics: list[IntensityMetric], llm_metrics: list[IntensityMetric]) -> list[IntensityMetric]:
        merged = []
        seen = set()
        for metric in [*rule_metrics, *llm_metrics]:
            key = (metric.metric_name, round(float(metric.value), 4), metric.unit, metric.source_chunk_id)
            if key in seen:
                continue
            seen.add(key)
            merged.append(metric)
        return merged

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        value = re.sub(r"<think>[\s\S]*?</think>", "", str(text or ""), flags=re.IGNORECASE).strip()
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE).strip()
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            start = value.find("{")
            end = value.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(value[start:end + 1])
                except json.JSONDecodeError:
                    return {}
        return {}

    def _cache_path(self, case: StandardCase, chunks: list[DocumentChunk], disaster_names) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256()
        digest.update(self.PROMPT_VERSION.encode("utf-8"))
        digest.update(str(case.case_id).encode("utf-8"))
        digest.update("|".join(disaster_names or []).encode("utf-8"))
        for chunk in chunks:
            digest.update(str(chunk.chunk_id).encode("utf-8"))
            digest.update(str(chunk.content or "").encode("utf-8"))
        safe_id = re.sub(r"[^0-9A-Za-z_-]+", "-", str(case.case_id)).strip("-") or "case"
        return self.cache_dir / f"{safe_id}-{digest.hexdigest()[:16]}.json"

    def _read_cache(self, case: StandardCase, chunks: list[DocumentChunk], disaster_names) -> list[IntensityMetric] | None:
        path = self._cache_path(case, chunks, disaster_names)
        if path is None or not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [IntensityMetric.model_validate(item) for item in data.get("metrics", [])]
        except Exception:
            return None

    def _write_cache(self, case: StandardCase, chunks: list[DocumentChunk], disaster_names, metrics: list[IntensityMetric]) -> None:
        path = self._cache_path(case, chunks, disaster_names)
        if path is None:
            return
        path.write_text(json.dumps({"metrics": [metric.model_dump() for metric in metrics]}, ensure_ascii=False, indent=2), encoding="utf-8")
