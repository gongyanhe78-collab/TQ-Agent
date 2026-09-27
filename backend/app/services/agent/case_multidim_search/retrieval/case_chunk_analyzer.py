"""按个例懒加载、筛选并分析关联文档片段。"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from backend.app.models import DocumentChunk, StandardCase
from backend.app.services.agent.case_multidim_search.analysis.disaster_profile import disaster_view_text
from backend.app.services.agent.case_multidim_search.schemas import CaseSearchQuery
from backend.app.services.model_client import get_rerank_client


LOGGER = logging.getLogger("uvicorn.error")


@dataclass
class ChunkRerankHit:
    """传给通用
     客户端的 chunk 命中对象。

    这里只包装当前个例的 chunk 正文，便于调用已有 rerank 客户端统一精排。
    """

    chunk: DocumentChunk
    document_text: str
    score: float = 0.0


class CaseChunkAnalyzer:
    """每次只处理一个个例，限制传给大模型的片段数量与总字符数。"""

    # 提示词或输出要求改变时递增版本，避免复用旧逻辑生成的缓存。
    # 缓存版本随上下文协议和生成元信息变化，确保旧的缺失指标结果不会复用。
    # 正文已与指标事实库解耦；升级版本避免复用旧的联合生成缓存。
    PROMPT_VERSION = "case-analysis-v23-four-slots-image-citations"

    ROLE_KEYWORDS = {
        "overview": ("天气过程", "过程概况", "主要天气过程", "天气实况", "实况特征", "过程特征"),
        "intensity": ("最大风速", "阵风", "风力", "降水量", "小时雨强", "雷达回波", "dBZ", "中心点", "极值"),
        "circulation": ("环流形势", "影响系统", "高空槽", "切变线", "冷高压", "冷锋", "副高", "低涡", "槽线"),
        "forecast": ("预报服务", "预警", "重要气象信息", "模式", "CMA_GFS", "EC", "预报偏差", "订正"),
        "impact": ("影响", "灾情", "风险", "道路结冰", "城市内涝", "设施农业", "交通"),
    }

    def __init__(
        self,
        document_store,
        llm_client,
        cache_dir: Path,
        metric_extractor=None,
        standard_case_store=None,
    ):
        """保存片段存储、大模型客户端和逐例分析缓存目录。"""
        self.document_store = document_store
        self.llm_client = llm_client
        self.cache_dir = Path(cache_dir)
        self.rerank_client = get_rerank_client()
        self.metric_extractor = metric_extractor
        # 标准个例用于识别共享 chunk 内的相邻个例标题，不参与跨个例扩展检索。
        self.standard_case_store = standard_case_store

    def load_case_chunks(self, case: StandardCase) -> list[DocumentChunk]:
        """读取并裁剪当前个例正式关联范围内的正文，不读取全库向量。"""
        chunk_ids = list(dict.fromkeys(case.source_chunk_ids))
        if not chunk_ids:
            return []
        if not hasattr(self.document_store, "get_chunks"):
            raise RuntimeError("当前文档数据源未提供 get_chunks 接口，无法读取个例证据。")
        # 公司接口按 chunk_id 精确回查，避免加载本地 Chroma 或扫描全库。
        chunks = list(self.document_store.get_chunks(chunk_ids))
        by_id = {chunk.chunk_id: chunk for chunk in chunks if chunk is not None}
        ordered_chunks = [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]
        # source_chunk_ids 是候选范围；共享首尾 chunk 还必须按标题裁掉相邻个例正文。
        return self._slice_case_owned_content(case, ordered_chunks)

    def _slice_case_owned_content(
        self,
        case: StandardCase,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """在正式关联 chunk 内按相邻标题切出只属于当前个例的正文。

        建库时一个 chunk 可能同时承接前一例的结尾和下一例的标题。若直接把完整
        chunk 送进指标事实库，前后个例的数值会被误当成当前个例证据。这里仅在既有
        source_chunk_ids 范围内定位标题，绝不把整份 PDF 的其他内容追加进来。
        """
        ordered_chunks = sorted(
            [chunk for chunk in chunks if str(chunk.content or "").strip()],
            key=lambda item: (item.chunk_no, item.chunk_id),
        )
        if not ordered_chunks or self.standard_case_store is None:
            return ordered_chunks
        if not self._has_shared_candidate_chunk(case, ordered_chunks):
            # 无共享 chunk 时正式关联范围已经足够精确，无须额外标题匹配或产生告警。
            return ordered_chunks

        text, positions = self._joined_chunk_text(ordered_chunks)
        current_start = self._find_title_position(text, case.title)
        if current_start is None:
            LOGGER.warning(
                "[多维检索][个例边界] 未找到当前标题，保留正式关联范围：case_id=%s title=%s chunks=%s",
                case.case_id,
                case.title,
                [chunk.chunk_id for chunk in ordered_chunks],
            )
            return ordered_chunks

        next_start, next_case_id = self._next_case_title_position(text, case, current_start)
        end = next_start if next_start is not None else len(text)
        sliced = self._slice_chunks_by_offsets(ordered_chunks, positions, current_start, end)
        if not sliced:
            LOGGER.warning(
                "[多维检索][个例边界] 标题切片为空，保留正式关联范围：case_id=%s",
                case.case_id,
            )
            return ordered_chunks

        removed_chars = len(text) - sum(len(chunk.content) for chunk in sliced)
        LOGGER.info(
            "[多维检索][个例边界] case_id=%s start=%s end=%s next_case_id=%s removed_chars=%s chunks=%s",
            case.case_id,
            current_start,
            end,
            next_case_id or "",
            removed_chars,
            [chunk.chunk_id for chunk in sliced],
        )
        return sliced

    def _has_shared_candidate_chunk(self, case: StandardCase, chunks: list[DocumentChunk]) -> bool:
        """确认当前候选范围确有被同 PDF 其他个例复用的 chunk，避免无谓切片。"""
        candidate_ids = {chunk.chunk_id for chunk in chunks}
        try:
            all_cases = self.standard_case_store.list_cases()
        except Exception:
            LOGGER.exception("[多维检索][个例边界] 读取标准个例失败：case_id=%s", case.case_id)
            return False
        return any(
            other.case_id != case.case_id
            and other.source_pdf == case.source_pdf
            and candidate_ids.intersection(other.source_chunk_ids)
            for other in all_cases
        )

    def _joined_chunk_text(self, chunks: list[DocumentChunk]) -> tuple[str, list[tuple[int, int]]]:
        """按 PDF 分块序号拼接正文，并保存每个 chunk 在拼接文本中的字符区间。"""
        parts: list[str] = []
        positions: list[tuple[int, int]] = []
        cursor = 0
        for chunk in chunks:
            content = str(chunk.content or "")
            start = cursor
            cursor += len(content)
            positions.append((start, cursor))
            parts.append(content)
        return "".join(parts), positions

    def _next_case_title_position(
        self,
        text: str,
        case: StandardCase,
        current_start: int,
    ) -> tuple[int | None, str | None]:
        """查找当前标题之后最早出现的同 PDF 其他个例标题。"""
        try:
            all_cases = self.standard_case_store.list_cases()
        except Exception:
            LOGGER.exception("[多维检索][个例边界] 读取标准个例失败：case_id=%s", case.case_id)
            return None, None

        candidates: list[tuple[int, str]] = []
        for other in all_cases:
            if other.case_id == case.case_id or other.source_pdf != case.source_pdf:
                continue
            position = self._find_title_position(text, other.title, start=current_start + 1)
            if position is not None:
                candidates.append((position, other.case_id))
        return min(candidates, default=(None, None), key=lambda item: item[0])

    def _find_title_position(self, text: str, title: str, start: int = 0) -> int | None:
        """宽容匹配 PDF 标题中的空白和日期连接符差异，返回原文字符位置。"""
        normalized_title = str(title or "").strip()
        if not normalized_title:
            return None
        pattern_parts: list[str] = []
        for char in normalized_title:
            if char.isspace():
                pattern_parts.append(r"\s*")
            elif char in "-~～－—–至到":
                # PDF 提取时日期范围的横线、波浪线和“至”经常互相替换。
                pattern_parts.append(r"[-~～－—–至到]\s*")
            else:
                pattern_parts.append(re.escape(char))
        matches = list(re.finditer("".join(pattern_parts), text[start:]))
        if not matches:
            return None
        # 同一标题可能先在“本月主要过程”目录中出现、后在正式章节中出现；优先章节标题。
        for match in matches:
            title_start = start + match.start()
            prefix_start = max(0, title_start - 12)
            heading = re.search(r"[一二三四五六七八九十0-9]+[、.．]\s*$", text[prefix_start:title_start])
            if heading:
                return prefix_start + heading.start()
        return start + matches[0].start()

    def _slice_chunks_by_offsets(
        self,
        chunks: list[DocumentChunk],
        positions: list[tuple[int, int]],
        start: int,
        end: int,
    ) -> list[DocumentChunk]:
        """把拼接文本中的正文区间重新映射回原 chunk，保留溯源编号和顺序。"""
        result: list[DocumentChunk] = []
        for chunk, (chunk_start, chunk_end) in zip(chunks, positions):
            content_start = max(start, chunk_start)
            content_end = min(end, chunk_end)
            if content_start >= content_end:
                continue
            content = str(chunk.content or "")[content_start - chunk_start:content_end - chunk_start].strip()
            if not content:
                continue
            result.append(
                DocumentChunk(
                    source_pdf=chunk.source_pdf,
                    chunk_id=chunk.chunk_id,
                    chunk_no=chunk.chunk_no,
                    content=content,
                    file_path=chunk.file_path,
                    embedding=chunk.embedding,
                )
            )
        return result

    def select_relevant_chunks(
        self,
        case: StandardCase,
        chunks: list[DocumentChunk],
        query: CaseSearchQuery,
        limit: int,
        char_limit: int,
    ) -> list[DocumentChunk]:
        """在当前个例内部选择最适合送给大模型的 chunk。

        个例已经由公司标准化个例接口确定；这里不重新检索个例，只在当前个例关联的
        source_chunk_ids 内部排序。关联 chunk 数量超过上限时调用 rerank 模型选择前 N 个，
        少于等于上限时保留原文顺序，避免无意义的模型调用。
        """
        safe_limit = min(6, max(3, int(limit)))
        usable_chunks = [chunk for chunk in chunks if str(chunk.content or "").strip()]
        if not usable_chunks:
            return []
        if len(usable_chunks) <= safe_limit:
            # chunk 数量本来不多时不做排序截断，保留完整过程上下文。
            # 不超过六个关联片段时完整保留，字符上限只在送入模型的上下文组装阶段生效。
            return sorted(usable_chunks, key=lambda item: item.chunk_no)

        selected = self._select_by_rerank_model(case, usable_chunks, query, safe_limit, char_limit)
        if selected:
            return self._ensure_analysis_role_coverage(selected, usable_chunks, safe_limit, char_limit)

        LOGGER.warning(
            "[多维检索][chunk重排序] rerank 不可用或返回为空，使用原文顺序兜底：case_id=%s candidate_chunks=%s limit=%s",
            case.case_id,
            len(usable_chunks),
            safe_limit,
        )
        selected = self._take_chunks_by_original_order(usable_chunks, safe_limit, char_limit)
        return self._ensure_analysis_role_coverage(selected, usable_chunks, safe_limit, char_limit)

    def _ensure_analysis_role_coverage(
        self,
        selected: list[DocumentChunk],
        candidates: list[DocumentChunk],
        limit: int,
        char_limit: int,
    ) -> list[DocumentChunk]:
        """在六 chunk 上限内补齐复盘所需的不同证据角色，避免只选到多段实况。"""
        result = list(selected)
        # 强度由全量事实库负责；六个正文名额优先覆盖过程、致灾机理和复合风险证据。
        required_roles = ("overview", "circulation", "impact")
        for role in required_roles:
            if any(self._chunk_role(chunk) == role for chunk in result):
                continue
            replacement = next(
                (chunk for chunk in sorted(candidates, key=lambda item: item.chunk_no)
                 if self._chunk_role(chunk) == role and chunk.chunk_id not in {item.chunk_id for item in result}),
                None,
            )
            if replacement is None:
                continue
            role_counts = Counter(self._chunk_role(chunk) for chunk in result)
            removable = next(
                (chunk for chunk in reversed(result) if role_counts[self._chunk_role(chunk)] > 1),
                None,
            )
            if removable is None:
                continue
            used_chars = sum(len(str(chunk.content or "")) for chunk in result)
            next_chars = used_chars - len(str(removable.content or "")) + len(str(replacement.content or ""))
            if next_chars > char_limit:
                continue
            result.remove(removable)
            result.append(replacement)
        return sorted(result[:limit], key=lambda item: item.chunk_no)

    def _take_chunks_by_original_order(
        self,
        chunks: list[DocumentChunk],
        limit: int,
        char_limit: int,
    ) -> list[DocumentChunk]:
        """对少量 chunk 只按原文顺序和字符上限裁剪，不做额外相关性排序。"""
        selected: list[DocumentChunk] = []
        selected_ids: set[str] = set()
        used_chars = 0
        for chunk in sorted(chunks, key=lambda item: item.chunk_no):
            if len(selected) >= limit or used_chars >= char_limit:
                break
            used_chars = self._add_chunk_if_room(chunk, selected, selected_ids, used_chars, char_limit)
        return selected

    def _select_by_rerank_model(
        self,
        case: StandardCase,
        chunks: list[DocumentChunk],
        query: CaseSearchQuery,
        limit: int,
        char_limit: int,
    ) -> list[DocumentChunk]:
        """调用配置中的 rerank 模型，在当前个例候选 chunk 中选择最相关的前 N 个。

        这里的 rerank 查询不是用户原始问题的简单复述，而是把个例标题、灾种、地市和检索条件
        合在一起，让 Qwen3-VL-Reranker-2B 优先挑出能支撑逐例报告的实况、强度、环流和服务片段。
        """
        client = self.rerank_client
        if client is None or not hasattr(client, "rerank"):
            return []

        hits = [
            ChunkRerankHit(chunk=chunk, document_text=self._chunk_document_text(case, chunk))
            for chunk in chunks
        ]
        question = self._chunk_rerank_query(case, query)
        model_name = str(
            getattr(client, "model_uid", "")
            or getattr(client, "model", "")
            or type(client).__name__
        )
        try:
            ranked_hits = client.rerank(question, hits, top_n=min(limit, len(hits)))
        except Exception:
            LOGGER.exception(
                "[多维检索][chunk重排序] rerank 模型调用失败：case_id=%s model=%s candidate_chunks=%s",
                case.case_id,
                model_name,
                len(chunks),
            )
            return []

        ranked_chunks = [
            hit.chunk
            for hit in ranked_hits
            if isinstance(getattr(hit, "chunk", None), DocumentChunk)
        ]
        if not ranked_chunks:
            return []

        LOGGER.info(
            "[多维检索][chunk重排序] case_id=%s model=%s candidate_chunks=%s selected_limit=%s top_chunks=%s",
            case.case_id,
            model_name,
            len(chunks),
            limit,
            [chunk.chunk_id for chunk in ranked_chunks[:limit]],
        )
        return self._take_ranked_chunks_then_restore_order(ranked_chunks, limit, char_limit)

    def _take_ranked_chunks_then_restore_order(
        self,
        ranked_chunks: list[DocumentChunk],
        limit: int,
        char_limit: int,
    ) -> list[DocumentChunk]:
        """按 rerank 得分取前 N 个，再恢复 PDF 原文顺序交给大模型阅读。"""
        selected: list[DocumentChunk] = []
        selected_ids: set[str] = set()
        used_chars = 0
        for chunk in ranked_chunks:
            if len(selected) >= limit or used_chars >= char_limit:
                break
            if chunk.chunk_id in selected_ids:
                continue
            used_chars = self._add_chunk_if_room(chunk, selected, selected_ids, used_chars, char_limit)
        return sorted(selected, key=lambda chunk: chunk.chunk_no)

    def _chunk_rerank_query(self, case: StandardCase, query: CaseSearchQuery) -> str:
        """构造 chunk 精排查询，让模型知道当前个例分析最需要哪些证据。"""
        query_parts = [
            f"个例标题：{case.title}",
            f"发生时段：{case.date_range}",
            f"灾种：{'、'.join(case.disaster_types)}",
            f"影响区域：{'、'.join(case.city_tags or case.affected_areas)}",
        ]
        if query.years:
            query_parts.append(f"检索年份：{'、'.join(str(year) for year in query.years)}")
        if query.months:
            query_parts.append(f"检索月份：{'、'.join(str(month) for month in query.months)}")
        if query.disaster_types:
            query_parts.append(f"用户筛选灾种：{'、'.join(query.disaster_types)}")
        if query.cities or query.areas:
            query_parts.append(f"用户筛选地区：{'、'.join(query.cities + query.areas)}")
        query_parts.append(f"本次灾种分析视角：{disaster_view_text(query.disaster_types or case.disaster_types)}")
        query_parts.append(
            "请优先选择能支撑逐例报告的片段，包括过程概况、天气实况、灾种匹配的强度数值、环流与温湿结构、地形或雷达诊断、实际影响和复合风险。"
        )
        return "\n".join(part for part in query_parts if part)

    def _chunk_document_text(self, case: StandardCase, chunk: DocumentChunk) -> str:
        """把单个 chunk 包装成 rerank 模型可理解的文档文本。

        个例标题、灾种、地区已放在 query 里，这里只放 chunk 自身内容，
        避免每个候选文档都携带相同元数据而干扰模型判断。
        """
        role = self._chunk_role(chunk)
        return "\n".join(
            part
            for part in [
                f"source_pdf: {chunk.source_pdf}",
                f"chunk_id: {chunk.chunk_id}",
                f"chunk_no: {chunk.chunk_no}",
                f"chunk_role: {role}",
                f"content: {chunk.content}",
            ]
            if part
        )

    def _add_chunk_if_room(
        self,
        chunk: DocumentChunk,
        selected: list[DocumentChunk],
        selected_ids: set[str],
        used_chars: int,
        char_limit: int,
    ) -> int:
        """在字符上限允许时加入片段，并返回新的累计字符数。"""
        content = str(chunk.content or "").strip()
        if not content:
            return used_chars
        if selected and used_chars + len(content) > char_limit:
            return used_chars
        selected.append(chunk)
        selected_ids.add(chunk.chunk_id)
        return used_chars + len(content)

    def _chunk_role(self, chunk: DocumentChunk) -> str:
        """判断片段在个例分析中的主要业务角色，供模型上下文配比使用。"""
        text = str(chunk.content or "")
        for role, keywords in self.ROLE_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return role
        return "other"

    def analyze_case(
        self,
        case: StandardCase,
        chunks: list[DocumentChunk],
        query: CaseSearchQuery,
        fallback_text: str,
        enabled: bool,
        char_limit: int,
        max_output_tokens: int,
        displayed_images: list[dict] | None = None,
    ) -> tuple[str, str, list[str]]:
        """兼容旧调用方，只返回正文和图片引用；新流程同时读取指标。"""
        answer, status, cited_image_ids, _ = self.analyze_case_with_metrics(
            case,
            chunks,
            query,
            fallback_text,
            enabled,
            char_limit,
            max_output_tokens,
            displayed_images=displayed_images,
        )
        return answer, status, cited_image_ids

    def analyze_case_with_metrics(
        self,
        case: StandardCase,
        chunks: list[DocumentChunk],
        query: CaseSearchQuery,
        fallback_text: str,
        enabled: bool,
        char_limit: int,
        max_output_tokens: int,
        displayed_images: list[dict] | None = None,
        metric_evidence: list[dict] | None = None,
        all_chunks: list[DocumentChunk] | None = None,
        extract_metrics: bool = True,
        verified_metrics: list | None = None,
    ) -> tuple[str, str, list[str], list]:
        """生成个例正文和图片引用；新主流程可关闭旧的联合指标输出。"""
        source_chunks = list(all_chunks or chunks)
        evidence = list(metric_evidence or [])
        if not enabled or not chunks or self.llm_client is None:
            LOGGER.info(
                "[多维检索][逐例分析] 使用规则降级：case_id=%s enabled=%s chunk_count=%s client=%s",
                case.case_id,
                enabled,
                len(chunks),
                type(self.llm_client).__name__ if self.llm_client is not None else "None",
            )
            return fallback_text, "rule_fallback", [], []
        cache_path = self._cache_path(
            case, chunks, query, char_limit, max_output_tokens, displayed_images or [], evidence, extract_metrics, verified_metrics or []
        )
        cached, cached_image_ids, cached_metrics = self._read_cache(cache_path)
        if cached:
            LOGGER.info(
                "[多维检索][逐例分析] 命中缓存，不调用大模型：case_id=%s cache=%s",
                case.case_id,
                cache_path,
            )
            return cached, "cache_hit", cached_image_ids, cached_metrics
        # 指标由独立事实库完成；正文模型只读取精选分析 chunk，避免数字证据误导业务叙述。
        context_blocks = self._context_blocks(
            case,
            chunks,
            char_limit,
            displayed_images or [],
            evidence if extract_metrics else None,
            verified_metrics=verified_metrics or [],
        )
        question = (
            "Use a concise, evidence-led operational review. Do not expose JSON, chunk-extraction language, internal evidence labels, or repetitive missing-data boilerplate. "
            "Connect paragraphs through the event evolution, forecast challenge, or risk transition. Each operational action must be specific to this case, its primary hazard, location, and observed evidence. "
            "请只分析当前这一个气象灾害个例，写成简洁清晰的报告段落，质量要高于规则模板。"
            "Write exactly 3 balanced paragraphs, about 360 to 480 Chinese characters in total; each paragraph must contain at least two complete sentences, and diagnostic depth must remain similar across cases. "
            "三段分别承担明确任务：第一段归纳过程演变、主导灾种、重点落区和可核验强度；"
            "第二段解释关键天气系统、动力热力条件、地形调制及其与强度峰值时段的对应关系；"
            "第三段复盘预报偏差、预警提前量或服务针对性并给出完整业务动作。"
            f"本次灾种分析视角：{disaster_view_text(query.disaster_types or case.disaster_types)}"
            "写作必须采用稳定的业务复盘结构：过程核心和主导灾种—定量实况和异常或缺测说明—触发机制或诊断证据—复合风险与业务动作。"
            "如果原文没有不稳定度指数、垂直风切变、冷池强度、雷达回波演变或预警提前量，不得编造；只有缺失确实影响关键判断时才自然说明一次材料边界，不要反复使用“证据缺口”“需补充资料”等模板句。"
            "复合灾害不能只写‘叠加影响’，应依据当前材料说明大风加沙尘对能见度、大风加降雪对吹雪、降水加低温对道路结冰、强对流中雨强与雷暴大风或冰雹的联动效应；材料不支持的灾种标签不要在正文中展开，关键边界最多自然说明一次。"
            "业务建议尽量指向山西地形和本地业务流程，例如吕梁山、五台山、太行山迎风坡、晋中盆地、太原及北部山区的监测和预警关注点；不要只写加强监测。"
            "不得只写两句话，不得用半句话收尾；第三段必须给出与本个例灾种、落区和地形相匹配的完整业务动作。"
            "不要输出 <think>、</think> 或任何思考过程、推理草稿、分析计划，只输出最终报告正文。"
            "上下文会给出‘候选图片图注’，这些图还没有全部展示到页面。"
            "如果某个判断确实需要图像支撑，可以引用候选图片；图片必须嵌入过程逻辑，正文不得连续用‘图X显示、图Y表明’开头罗列图注，也不得让图号成为句子主体；"
            "每一处被正文引用的图片，都必须说明这张图具体支撑了什么判断，例如落区、强度、环流或演变依据；最终引用图片统一放在该个例分析正文之后展示。"
            "引用图片时必须逐字使用候选图片图注中的图号或完整图题；不要写候选列表里不存在的图号、子图编号或图题。"
            "如果无法确认某张图在候选图片图注中，就不要提这张图；如果没有必要引用图片，就完全不要写图号或图片标题。"
            "不得引用候选图片列表之外的图，不得引用其他个例，不得补造原文中没有的数值。冷平流、动量下传、地形加速等机制应尽量与已有时段或强度变化建立半定量对应；原文提供预报、预警或服务记录时，应复盘提前量、准确性和服务对象，原文未提供时不得编造。"
            "强度指标必须综合全部数字证据句，不能只看精选片段；若证据未明确给出指标则不要猜测。"
            "指标名和单位仅允许：最大小时雨强(mm/h)、过程最大降水量(mm)、最大积雪深度(cm)、最大风速(m/s)、极大风速(m/s)、阵风风力(级)、最高气温(℃)、最低气温(℃)、过程降温幅度(℃)、过程最大降雪量(mm)、最低能见度(km)、最大冰雹直径(mm)、雷达回波强度(dBZ)。"
            "同一指标必须看完全部数字证据后再确定：范围取正确端点，明确站点值优先于阈值描述，国家站与区域站同时出现时取全体最大值；过程降温幅度不得写成最低气温，降雪量不得写成降水量。"
            "最大风速明确为米每秒时，阵风风力按蒲福风级换算：17.2-20.7为8级、20.8-24.4为9级、24.5-28.4为10级、28.5-32.6为11级、32.7-36.9为12级、37.0-41.4为13级、41.5-46.1为14级、46.2-50.9为15级、51.0-56.0为16级、56.1-61.2为17级。"
            "日期、时次、站点数量、图号和预报阈值不是过程极值；无法确认的指标不要输出。"
            "最后只输出 JSON：{\"metrics\":[{\"metric_name\":\"过程最大降水量\",\"value\":0,\"unit\":\"mm\",\"location\":\"\",\"relation\":\"max\",\"source_chunk_id\":\"chunk-id\",\"source_text\":\"原始完整句\",\"confidence\":0.9}],\"analysis\":\"完整正文\",\"cited_image_ids\":[\"正文实际引用的图片ID\"]}。metrics 的来源句必须逐字来自当前个例数字证据；cited_image_ids 只能填写候选图列表中的ID；没有引用图片时必须为空数组。"
        )
        if not extract_metrics:
            question = self._analysis_only_question(case, query)
        model_name = str(
            getattr(self.llm_client, "model_uid", "")
            or getattr(self.llm_client, "model", "")
            or "unknown"
        )
        base_url = str(getattr(self.llm_client, "base_url", "") or "")
        context_chars = sum(len(block) for block in context_blocks)
        LOGGER.info(
            "[多维检索][逐例分析] 开始调用大模型：case_id=%s model=%s base_url=%s "
            "chunk_count=%s context_chars=%s max_output_tokens=%s",
            case.case_id,
            model_name,
            base_url,
            len(chunks),
            context_chars,
            max_output_tokens,
        )
        try:
            answer = str(
                self.llm_client.answer_with_context(
                    question,
                    context_blocks,
                    max_tokens=max_output_tokens,
                )
                or ""
            ).strip()
            finish_reason = self._llm_finish_reason()
            LOGGER.info(
                "[多维检索][逐例分析] 大模型结束原因：case_id=%s finish_reason=%s response_chars=%s",
                case.case_id,
                finish_reason,
                len(answer),
            )
        except Exception:
            LOGGER.exception(
                "[多维检索][逐例分析] 大模型调用失败：case_id=%s model=%s",
                case.case_id,
                model_name,
            )
            return fallback_text, "llm_failed", [], []
        answer = self._strip_thinking(answer)
        structured = self._parse_json_object(answer)
        cited_image_ids = []
        metrics = []
        if extract_metrics and isinstance(structured, dict) and self.metric_extractor is not None:
            try:
                metrics = self.metric_extractor.metrics_from_payload(
                    structured,
                    source_chunks,
                    disaster_names=query.disaster_types or case.disaster_types,
                )
            except Exception:
                LOGGER.exception("[多维检索][逐例分析] 强度指标协议校验失败：case_id=%s", case.case_id)
        if not extract_metrics and structured:
            answer = self._compose_diagnostic_analysis(case, verified_metrics or [], structured)
            allowed_ids = {str(image.get("image_id") or "") for image in (displayed_images or [])}
            cited_image_ids = [
                str(item).strip()
                for item in (structured.get("cited_image_ids") or [])
                if str(item or "").strip() in allowed_ids
            ]
        elif structured.get("analysis"):
            answer = str(structured.get("analysis") or "").strip()
            allowed_ids = {str(image.get("image_id") or "") for image in (displayed_images or [])}
            cited_image_ids = [
                str(item).strip()
                for item in (structured.get("cited_image_ids") or [])
                if str(item or "").strip() in allowed_ids
            ]
        # 兼容模型返回被截断或二次包装的 JSON，不把协议原文渗漏到页面。
        if extract_metrics:
            answer = self._extract_analysis_text(answer, structured)
        if not answer:
            LOGGER.warning(
                "[多维检索][逐例分析] 大模型返回空内容：case_id=%s model=%s",
                case.case_id,
                model_name,
            )
            return fallback_text, "llm_empty", [], metrics
        answer = self._complete_analysis_text(answer)
        answer = self._normalize_analysis_paragraphs(answer)
        if not self._analysis_is_complete(answer):
            LOGGER.warning(
                "[多维检索][逐例分析] 大模型正文过短或不完整，使用规则结果：case_id=%s response_chars=%s finish_reason=%s",
                case.case_id,
                len(answer),
                finish_reason,
            )
            return fallback_text, "llm_incomplete", [], metrics
        # 用户需要在后端终端直接核对模型实际返回内容，因此这里完整记录逐例回答；思考链会先清洗掉。
        LOGGER.info(
            "[多维检索][逐例分析] 大模型返回成功：case_id=%s model=%s response_chars=%s\n%s",
            case.case_id,
            model_name,
            len(answer),
            answer,
        )
        self._write_cache(cache_path, answer, cited_image_ids, metrics)
        return answer, "llm_generated", cited_image_ids, metrics

    def _analysis_only_question(self, case: StandardCase, query: CaseSearchQuery) -> str:
        """构造诊断槽位提示词，让模型只做有证据支撑的个例复盘。"""
        return (
            "请只分析当前这一个气象灾害个例。上下文中的‘已核验强度事实’来自全部关联正文，"
            "只能引用，不能改写数值、补造地点或再做极值判断；其余材料最多六个精选分析片段。"
            f"本次灾种分析视角：{disaster_view_text(query.disaster_types or case.disaster_types)}。"
            "请分别填写过程画像、关键致灾链、复合风险、可迁移启示四个槽位；"
            "每个非空槽位必须给出当前材料可直接支撑的判断，不要按原文顺序复述，也不要输出完整三段正文。"
            "所有非空槽位合计约260至330个汉字，供后端结合已核验事实组成约360至480个汉字的三段复盘。"
            "过程画像只概括过程演变、主导灾种、重点落区或已核验实况。"
            "关键致灾链仅在材料给出环流、热力、地形、雷达或明确演变证据时填写；没有证据必须为空字符串。"
            "复合风险仅在材料给出并发灾害、影响对象或风险演变证据时填写；不得把灾种标签直接当作风险事实。"
            "可迁移启示必须从前三部分已经证实的信号、区域、演变或阈值推导，不能写‘加强监测’等泛化口号；无法推导时必须为空字符串。"
            "上下文已经同时提供当前个例关联的全部候选图片图注。候选图片仅在确有必要时引用，"
            "图号或图题必须逐字来自候选图注，并在同一句明确说明图片支撑的过程演变、致灾链或风险判断。"
            "禁止输出‘可参考图N’、‘图N...’、‘图1、图2、...’或任何带省略号、等等字样的开放式图片引用；"
            "不能完整说明图片作用时不要在任何槽位中提图，并从 cited_image_ids 中排除该图片。不得输出思考过程或 Markdown。"
            "最后只输出 JSON：{\"process_profile\":\"\",\"hazard_chain\":\"\",\"compound_risk\":\"\","
            "\"transferable_insight\":\"\",\"cited_image_ids\":[\"正文实际引用的图片ID\"]}。"
        )

    def _llm_finish_reason(self) -> str:
        """读取当前线程的生成结束原因，兼容旧客户端和测试桩。"""
        getter = getattr(self.llm_client, "get_last_finish_reason", None)
        if callable(getter):
            try:
                return str(getter() or "unknown")
            except Exception:
                return "unknown"
        return str(getattr(self.llm_client, "last_finish_reason", "unknown") or "unknown")


    def _complete_analysis_text(self, text: str) -> str:
        """去掉模型达到输出上限时遗留的残句，避免半句话进入页面和 PDF。"""
        value = str(text or "").strip()
        if not value or value.endswith(("。", "！", "？")):
            return value
        last_stop = max(value.rfind("。"), value.rfind("！"), value.rfind("？"))
        return value[:last_stop + 1].strip() if last_stop >= 0 else value

    def _normalize_analysis_paragraphs(self, text: str) -> str:
        """模型未返回换行时按完整句整理为三段，避免正常回答因格式差异被误判。"""
        value = str(text or "").strip()
        paragraphs = [item.strip() for item in re.split(r"\n+", value) if item.strip()]
        if len(paragraphs) >= 3:
            return "\n\n".join(paragraphs[:3])
        sentences = [item.strip() for item in re.findall(r"[^。！？]+[。！？]", value) if item.strip()]
        if len(sentences) < 6:
            return value
        groups = [sentences[:2], sentences[2:4], sentences[4:]]
        return "\n\n".join("".join(group) for group in groups if group)

    def _compose_diagnostic_analysis(self, case: StandardCase, verified_metrics: list, slots: dict) -> str:
        """把四个诊断槽位和全量核验事实组合为稳定的三段逐例复盘。"""
        profile = self._diagnostic_sentence(slots.get("process_profile"))
        hazard_chain = self._diagnostic_sentence(slots.get("hazard_chain"))
        compound_risk = self._diagnostic_sentence(slots.get("compound_risk"))
        transferable = self._diagnostic_sentence(slots.get("transferable_insight"))

        date_text = str(case.date_range or "发生时段未明确")
        disaster_text = "、".join(case.disaster_types) or "灾种未标注"
        area_text = "、".join(case.city_tags or case.affected_areas) or "影响区域未明确"
        metric_summary = self._verified_metric_summary(verified_metrics)
        first_paragraph = (
            f"{date_text}，{area_text}发生以{disaster_text}为主的天气过程。"
            f"已核验的强度事实包括{metric_summary}。"
            + (profile or "精选材料未给出更完整的过程演变描述，本段仅保留可核验的过程归属和实况事实。")
        )
        diagnostic_parts = [part for part in (hazard_chain, compound_risk) if part]
        # 两类诊断都缺失时只保留一次合并边界，不在正文中连续罗列多个“材料未给出”。
        second_paragraph = "".join(diagnostic_parts) or (
            "当前精选材料未形成可核验的致灾机理或复合影响链条，本例不采用常见形势作经验外推。"
            "具体研判仅依据已核验实况，不按灾种标签补写未经证实的风险。"
        )
        third_paragraph = transferable or (
            "现有证据只能支持本次过程事实回顾，尚不足以提炼可迁移的触发信号、重点区域或业务阈值。"
            "后续结论应在获得同类诊断证据后再形成，避免把一般性建议包装成本例启示。"
        )
        paragraphs = [first_paragraph, second_paragraph, third_paragraph]
        # 槽位内容较短时只补充数据口径和证据边界，绝不补写原文没有的天气机制或服务动作。
        return self._pad_diagnostic_analysis(paragraphs)

    def _pad_diagnostic_analysis(self, paragraphs: list[str]) -> str:
        """在材料真实但槽位较短时补足复盘篇幅，仍保持三段和可核验边界。"""
        additions = (
            "各项强度均按来源句所示的时段和站点理解，未将预警阈值与过程实况混用。",
            "未列入原文的成因条件不参与本例解释，避免用经验推断替代诊断证据。",
            "可迁移判断只从本例已证实的信号、落区和风险链条推导，不额外扩展材料范围。",
        )
        result = list(paragraphs)
        for index, addition in enumerate(additions):
            if len("\n\n".join(result)) >= 320:
                break
            result[index] = result[index].rstrip() + addition
        return "\n\n".join(result)

    def _diagnostic_sentence(self, value) -> str:
        """清理单个槽位，并保证后端拼接时保持完整句边界。"""
        text = re.sub(r"\s+", "", str(value or "")).strip("。；;，, ")
        if not text:
            return ""
        return text + "。"

    def _verified_metric_summary(self, metrics: list) -> str:
        """将事实库中的每种指标按业务极值压缩成适合逐例正文的实况摘要。"""
        selected = self._select_verified_metric_extremes(metrics)
        if not selected:
            return "当前全部关联正文中未形成可核验的统一强度指标"
        values = []
        for metric in selected:
            name = self._metric_value(metric, "metric_name")
            value = self._metric_value(metric, "value")
            unit = self._metric_value(metric, "unit")
            location = self._metric_value(metric, "location")
            try:
                value_text = f"{float(value):g}"
            except (TypeError, ValueError):
                continue
            location_text = f"（{location}）" if location else ""
            values.append(f"{name}{value_text}{unit}{location_text}")
        return "、".join(values) if values else "当前全部关联正文中未形成可核验的统一强度指标"

    def _verified_metric_lines(self, metrics: list) -> list[str]:
        """保留每种指标的业务极值和原始来源句，供模型作事实引用。"""
        lines = []
        for metric in self._select_verified_metric_extremes(metrics):
            name = self._metric_value(metric, "metric_name")
            value = self._metric_value(metric, "value")
            unit = self._metric_value(metric, "unit")
            location = self._metric_value(metric, "location")
            source_text = self._metric_value(metric, "source_text")
            try:
                value_text = f"{float(value):g}"
            except (TypeError, ValueError):
                continue
            location_text = f"，地点：{location}" if location else ""
            source_suffix = f"；来源句：{source_text}" if source_text else ""
            lines.append(f"- {name}：{value_text}{unit}{location_text}{source_suffix}")
        return lines

    def _select_verified_metric_extremes(self, metrics: list) -> list:
        """每个标准指标只保留一个业务极值，避免同类站点值挤占正文上下文。"""
        groups: dict[str, list] = {}
        for metric in metrics or []:
            name = self._metric_value(metric, "metric_name")
            value = self._metric_value(metric, "value")
            if not name:
                continue
            try:
                float(value)
            except (TypeError, ValueError):
                continue
            groups.setdefault(name, []).append(metric)
        prefer_min = {"最低能见度", "最低气温"}
        selected = []
        for name, candidates in groups.items():
            key = lambda item: float(self._metric_value(item, "value"))
            selected.append(min(candidates, key=key) if name in prefer_min else max(candidates, key=key))
        return selected

    @staticmethod
    def _metric_value(metric, field: str) -> str:
        """兼容 Pydantic 指标对象和测试中的字典指标。"""
        if isinstance(metric, dict):
            return str(metric.get(field) or "").strip()
        return str(getattr(metric, field, "") or "").strip()

    def _analysis_is_complete(self, text: str) -> bool:
        """按总篇幅、三段结构和完整句数量验收逐例分析。"""
        value = str(text or "").strip()
        paragraphs = [item.strip() for item in re.split(r"\n+", value) if item.strip()]
        return bool(
            320 <= len(value) <= 760
            and len(paragraphs) == 3
            and all(len(paragraph) >= 45 for paragraph in paragraphs)
            and all(len(re.findall(r"[。！？]", paragraph)) >= 2 for paragraph in paragraphs)
            and value.endswith(("。", "！", "？"))
        )
    def _context_blocks(
        self,
        case: StandardCase,
        chunks: list[DocumentChunk],
        char_limit: int,
        displayed_images: list[dict] | None = None,
        metric_evidence: list[dict] | None = None,
        verified_metrics: list | None = None,
    ) -> list[str]:
        """构造精选正文和全部数字证据句上下文，并在正文上限处安全截断。"""
        header = (
            f"个例ID：{case.case_id}\n标题：{case.title}\n时段：{case.date_range}\n"
            f"灾种：{'、'.join(case.disaster_types)}\n区域：{'、'.join(case.city_tags or case.affected_areas)}"
        )
        remaining = max(0, int(char_limit) - len(header))
        blocks = [header]
        image_lines = self._displayed_image_lines(displayed_images or [])
        if image_lines:
            # 把候选图片标题交给模型；后续只展示正文实际引用且匹配成功的图片。
            image_block = "候选图片图注：\n" + "\n".join(image_lines)
            blocks.append(image_block)
            # 图注属于图片选择协议，不占用 6 个正文 chunk 的字符预算。
        evidence_lines = [
            f"[{item.get('source_chunk_id')}|句{item.get('sentence_index', 0)}] {item.get('text', '')}"
            for item in (metric_evidence or [])
            if str(item.get("text") or "").strip()
        ]
        if evidence_lines:
            # 全量数字证据不做极值判断，独立于精选正文传给模型。
            blocks.append("全部数字证据句（仅供指标核对）：\n" + "\n".join(evidence_lines))
        verified_metric_lines = self._verified_metric_lines(verified_metrics or [])
        if verified_metric_lines:
            # 事实库已扫描全部关联 chunk；正文模型只能引用这些已核验数值，不能重新猜测。
            blocks.append("已核验强度事实（只能引用，不得改写或补造）：\n" + "\n".join(verified_metric_lines))
        for chunk in chunks:
            if remaining <= 0:
                break
            content = str(chunk.content or "").strip()[:remaining]
            blocks.append(f"片段{chunk.chunk_no}：{content}")
            remaining -= len(content)
        return blocks

    def _displayed_image_lines(self, images: list[dict]) -> list[str]:
        """把候选图片标题整理进提示词，约束模型只引用可匹配的图。"""
        lines: list[str] = []
        for index, image in enumerate(images, start=1):
            title = str(image.get("caption") or image.get("image_id") or "原始证据图").strip()
            image_id = str(image.get("image_id") or "").strip()
            page_no = image.get("page_no")
            source_pdf = str(image.get("source_pdf") or "").strip()
            page_text = f"，原文第{page_no}页" if page_no else ""
            source_text = f"，来源{source_pdf}" if source_pdf else ""
            lines.append(f"\u56fe\u7247{index}\uff5c\u56fe\u7247ID={image_id}\uff5c\u5b8c\u6574\u56fe\u9898={title}{page_text}{source_text}")
        return lines

    def _cache_path(
        self,
        case: StandardCase,
        chunks: list[DocumentChunk],
        query: CaseSearchQuery,
        char_limit: int,
        max_output_tokens: int,
        displayed_images: list[dict] | None = None,
        metric_evidence: list[dict] | None = None,
        extract_metrics: bool = True,
        verified_metrics: list | None = None,
    ) -> Path:
        """以个例、查询条件和实际片段内容生成稳定缓存键。"""
        payload = {
            "prompt_version": self.PROMPT_VERSION,
            "model": str(
                getattr(self.llm_client, "model_uid", "")
                or getattr(self.llm_client, "model", "")
            ),
            "case_id": case.case_id,
            "context_char_limit": int(char_limit),
            "max_output_tokens": int(max_output_tokens),
            "extract_metrics": bool(extract_metrics),
            "query": query.model_dump(),
            "chunks": [(chunk.chunk_id, chunk.content) for chunk in chunks],
            "candidate_images": [
                (image.get("image_id"), image.get("caption"))
                for image in (displayed_images or [])
            ],
            "metric_evidence": [
                (item.get("source_chunk_id"), item.get("sentence_index"), item.get("text"))
                for item in (metric_evidence or [])
            ],
            "verified_metrics": [
                (
                    self._metric_value(metric, "metric_name"),
                    self._metric_value(metric, "value"),
                    self._metric_value(metric, "unit"),
                    self._metric_value(metric, "location"),
                    self._metric_value(metric, "source_text"),
                )
                for metric in (verified_metrics or [])
            ],
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]
        safe_case_id = re.sub(r"[^0-9A-Za-z_-]+", "_", case.case_id).strip("_") or "case"
        return self.cache_dir / f"{safe_case_id}-{digest}.json"

    def _read_cache(self, path: Path) -> tuple[str, list[str], list]:
        """读取逐例分析缓存，并兼容旧缓存里嵌套的 JSON 协议文本。"""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            analysis = self._strip_thinking(str(data.get("analysis") or "").strip())
            cited_image_ids = [
                str(item).strip()
                for item in (data.get("cited_image_ids") or [])
                if str(item or "").strip()
            ]
            metrics = []
            if self.metric_extractor is not None:
                try:
                    from backend.app.services.agent.case_multidim_search.schemas import IntensityMetric
                    metrics = [IntensityMetric.model_validate(item) for item in (data.get("metrics") or [])]
                except Exception:
                    metrics = []
            nested = self._parse_json_object(analysis)
            if nested.get("analysis"):
                cited_image_ids = [
                    str(item).strip()
                    for item in (nested.get("cited_image_ids") or cited_image_ids)
                    if str(item or "").strip()
                ]
            analysis = self._extract_analysis_text(analysis, nested)
            return analysis, list(dict.fromkeys(cited_image_ids)), metrics
        except Exception:
            return "", [], []

    def _extract_analysis_text(self, text: str, parsed: dict | None = None) -> str:
        """从正常或被截断的 JSON 响应中恢复可展示的分析正文。"""
        payload = parsed if isinstance(parsed, dict) else self._parse_json_object(text)
        value = payload.get("analysis") if isinstance(payload, dict) else None
        if isinstance(value, str) and value.strip():
            return self._strip_thinking(value).strip()
        raw = self._strip_thinking(str(text or "")).strip()
        match = re.search(r'"analysis"\s*:\s*"((?:\\.|[^"\\])*)"', raw, flags=re.DOTALL)
        if match:
            try:
                return json.loads('"' + match.group(1) + '"').strip()
            except json.JSONDecodeError:
                return match.group(1).replace(r'\n', '\n').replace(r'\"', '"').strip()
        # 最后一道兼容只处理明确的 analysis 协议前缀，避免把正常正文误删。
        if raw.startswith('{"analysis":'):
            raw = raw[len('{"analysis":'):].lstrip().lstrip('"')
            raw = re.split(r'"\s*,\s*"cited_image_ids"', raw, maxsplit=1)[0]
            return raw.rstrip('}"').strip()
        return raw

    def _strip_thinking(self, text: str) -> str:
        """清理模型返回里的 <think> 思考链，只保留最终报告正文。"""
        cleaned = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _parse_json_object(self, text: str) -> dict:
        """读取逐例分析缓存，并恢复模型声明的图片引用列表。"""
        value = str(text or "").strip()
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", value, flags=re.DOTALL)
            if not match:
                return {}
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}

    def _write_cache(self, path: Path, analysis: str, cited_image_ids: list[str], metrics: list | None = None) -> None:
        """缓存最终正文和图片引用协议，避免下次重新猜测图文关系。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "analysis": analysis,
                    "cited_image_ids": cited_image_ids,
                    "metrics": [item.model_dump() if hasattr(item, "model_dump") else dict(item) for item in (metrics or [])],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )






