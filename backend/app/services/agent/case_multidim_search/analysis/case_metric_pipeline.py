"""按全部关联 chunk 分批构建个例强度指标事实库。"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from threading import BoundedSemaphore
from dataclasses import dataclass
from pathlib import Path

from backend.app.models import DocumentChunk, StandardCase
from backend.app.services.agent.case_multidim_search.analysis.intensity import IntensityExtractor
from backend.app.services.agent.case_multidim_search.analysis.llm_intensity_extractor import LlmIntensityExtractor
from backend.app.services.agent.case_multidim_search.schemas import IntensityMetric


LOGGER = logging.getLogger("uvicorn.error")

# 规则指标只做本地文本扫描，不应被外部模型请求阻塞；全进程最多同时处理四个个例。
_RULE_METRIC_SEMAPHORE = BoundedSemaphore(4)


@dataclass
class CaseMetricFacts:
    """保存某个例全部可追溯指标及其扫描状态。"""

    metrics: list[IntensityMetric]
    status_by_metric: dict[str, str]
    batch_count: int
    rule_metric_count: int
    llm_metric_count: int

    def audit(self) -> dict:
        """生成面向接口审计和日志的轻量摘要。"""
        return {
            "batch_size": 6,
            "batch_count": self.batch_count,
            "confirmed_metric_count": len(self.metrics),
            "rule_metric_count": self.rule_metric_count,
            "llm_metric_count": self.llm_metric_count,
            "status_by_metric": dict(self.status_by_metric),
        }


class CaseMetricPipeline:
    """将完整指标抽取与正文生成解耦，确保六 chunk 限制不造成指标漏扫。"""

    # 正文在进入事实库前已完成共享 chunk 边界切分，升级缓存版本避免旧结果继续展示。
    CACHE_VERSION = "case-metric-facts-v3-boundary-sliced-wind-warning-filter"
    BATCH_SIZE = 6

    def __init__(
        self,
        rule_extractor: IntensityExtractor,
        llm_extractor: LlmIntensityExtractor,
        cache_dir: Path,
    ):
        self.rule_extractor = rule_extractor
        self.llm_extractor = llm_extractor
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract(
        self,
        case: StandardCase,
        chunks: list[DocumentChunk],
        llm_enabled: bool,
    ) -> CaseMetricFacts:
        """扫描全部关联 chunk；每批最多六个 chunk，最后统一合并全个例指标。"""
        ordered_chunks = sorted(
            [chunk for chunk in chunks if str(chunk.content or "").strip()],
            key=lambda item: (getattr(item, "chunk_no", 0), str(item.chunk_id)),
        )
        if not ordered_chunks:
            return self._facts([], 0, 0, 0)
        cached = self._read_cache(case, ordered_chunks, llm_enabled)
        if cached is not None:
            LOGGER.info(
                "[多维检索][指标事实库] 命中缓存：case_id=%s chunk_count=%s batch_count=%s",
                case.case_id,
                len(ordered_chunks),
                cached.batch_count,
            )
            return cached

        all_metrics: list[IntensityMetric] = []
        rule_metric_count = 0
        llm_metric_count = 0
        batches = list(self._batches(ordered_chunks))
        for batch_index, batch in enumerate(batches, start=1):
            # 规则先扫描每一批完整原文，避免模型漏读时造成指标永久缺失。
            # 独立限制本地规则扫描并发，避免多个请求同时读取大批文本时争抢 CPU。
            with _RULE_METRIC_SEMAPHORE:
                rule_metrics = self.rule_extractor.extract(case, batch)
            rule_metric_count += len(rule_metrics)
            llm_metrics: list[IntensityMetric] = []
            if llm_enabled:
                # 模型只处理当前六个 chunk 的数字证据，不承担正文、图片和全局极值任务。
                llm_metrics = self.llm_extractor.extract_batch_from_evidence(case, batch)
                llm_metric_count += len(llm_metrics)
            all_metrics.extend(rule_metrics)
            all_metrics.extend(llm_metrics)
            LOGGER.info(
                "[多维检索][指标事实库] 批次完成：case_id=%s batch=%s/%s chunk_ids=%s rule_metrics=%s llm_metrics=%s",
                case.case_id,
                batch_index,
                len(batches),
                [chunk.chunk_id for chunk in batch],
                len(rule_metrics),
                len(llm_metrics),
            )

        facts = self._facts(
            self._merge_metrics(all_metrics),
            len(batches),
            rule_metric_count,
            llm_metric_count,
        )
        self._write_cache(case, ordered_chunks, llm_enabled, facts)
        LOGGER.info(
            "[多维检索][指标事实库] 全量扫描完成：case_id=%s chunk_count=%s batch_count=%s confirmed=%s status=%s",
            case.case_id,
            len(ordered_chunks),
            facts.batch_count,
            len(facts.metrics),
            facts.status_by_metric,
        )
        return facts

    def _batches(self, chunks: list[DocumentChunk]):
        """按稳定顺序拆分，严格保证单次模型最多接收六个 chunk。"""
        for start in range(0, len(chunks), self.BATCH_SIZE):
            yield chunks[start:start + self.BATCH_SIZE]

    def _facts(
        self,
        metrics: list[IntensityMetric],
        batch_count: int,
        rule_metric_count: int,
        llm_metric_count: int,
    ) -> CaseMetricFacts:
        """为每个标准指标显式给出确认或未发现状态，避免沉默丢失。"""
        names = set(self.llm_extractor.METRIC_NAMES)
        names.update(metric.metric_name for metric in metrics)
        confirmed_names = {metric.metric_name for metric in metrics}
        statuses = {
            name: "confirmed" if name in confirmed_names else "not_present_after_all_chunks_scanned"
            for name in sorted(names)
        }
        return CaseMetricFacts(
            metrics=metrics,
            status_by_metric=statuses,
            batch_count=batch_count,
            rule_metric_count=rule_metric_count,
            llm_metric_count=llm_metric_count,
        )

    def _merge_metrics(self, metrics: list[IntensityMetric]) -> list[IntensityMetric]:
        """仅消除同一来源的重复候选，保留不同站点和不同数值以便后续取业务极值。"""
        merged: list[IntensityMetric] = []
        seen: set[tuple] = set()
        for metric in metrics:
            canonical_name = "过程最大降水量" if metric.metric_name == "最大降水量" else metric.metric_name
            normalized = metric.model_copy(update={"metric_name": canonical_name})
            key = (
                normalized.metric_name,
                round(float(normalized.value), 4),
                normalized.unit,
                normalized.source_chunk_id,
                re.sub(r"\s+", "", normalized.source_text)[:80],
            )
            if key not in seen:
                seen.add(key)
                merged.append(normalized)
        return merged

    def _cache_path(self, case: StandardCase, chunks: list[DocumentChunk], llm_enabled: bool) -> Path:
        """缓存键覆盖全部 chunk、模型模式和流水线版本，杜绝复用旧联合调用结果。"""
        digest = hashlib.sha256()
        digest.update(self.CACHE_VERSION.encode("utf-8"))
        digest.update(str(case.case_id).encode("utf-8"))
        digest.update(str(bool(llm_enabled)).encode("utf-8"))
        digest.update(str(getattr(self.llm_extractor.llm_client, "model_uid", "")).encode("utf-8"))
        for chunk in chunks:
            digest.update(str(chunk.chunk_id).encode("utf-8"))
            digest.update(str(chunk.content or "").encode("utf-8"))
        safe_case_id = re.sub(r"[^0-9A-Za-z_-]+", "-", str(case.case_id)).strip("-") or "case"
        return self.cache_dir / f"{safe_case_id}-{digest.hexdigest()[:20]}.json"

    def _read_cache(self, case: StandardCase, chunks: list[DocumentChunk], llm_enabled: bool) -> CaseMetricFacts | None:
        """读取新版事实库缓存；任何格式异常都重新扫描原文。"""
        path = self._cache_path(case, chunks, llm_enabled)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("cache_version") != self.CACHE_VERSION:
                return None
            return CaseMetricFacts(
                metrics=[IntensityMetric.model_validate(item) for item in data.get("metrics", [])],
                status_by_metric={str(key): str(value) for key, value in (data.get("status_by_metric") or {}).items()},
                batch_count=int(data.get("batch_count") or 0),
                rule_metric_count=int(data.get("rule_metric_count") or 0),
                llm_metric_count=int(data.get("llm_metric_count") or 0),
            )
        except Exception:
            return None

    def _write_cache(self, case: StandardCase, chunks: list[DocumentChunk], llm_enabled: bool, facts: CaseMetricFacts) -> None:
        """写入独立事实库缓存，不与逐例正文缓存共享。"""
        path = self._cache_path(case, chunks, llm_enabled)
        path.write_text(
            json.dumps(
                {
                    "cache_version": self.CACHE_VERSION,
                    "metrics": [metric.model_dump() for metric in facts.metrics],
                    "status_by_metric": facts.status_by_metric,
                    "batch_count": facts.batch_count,
                    "rule_metric_count": facts.rule_metric_count,
                    "llm_metric_count": facts.llm_metric_count,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
