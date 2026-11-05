"""对向量命中的候选 chunk 执行专用模型精排。"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from ..infrastructure.data_store import LocalCaseDataStore


logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class ChunkRerankHit:
    """保存 Rerank 所需正文以及向量召回分。

    frozen=True 表示不可变数据类，创建后不能修改字段，
    作为 Rerank 的输入/输出数据结构，保证数据完整性。
    """

    chunk_id: str       # 段落 chunk 的唯一标识
    case_id: str        # 这个 chunk 所属的个例 ID
    source_pdf: str     # 来源 PDF 文件名
    content: str        # chunk 的正文内容（已截断，避免太长）
    cosine_score: float # 向量检索得到的余弦相似度分

    @property
    def document_text(self) -> str:
        """转换为通用 Rerank 客户端可读取的文档文本。

        把元数据（case_id、chunk_id、来源）和正文拼在一起，
        方便 Rerank 模型在知道上下文的情况下判断相关性。
        不同的 Rerank SDK 都可以直接消费这个字符串。
        """
        return (
            f"case_id: {self.case_id}\n"
            f"chunk_id: {self.chunk_id}\n"
            f"source_pdf: {self.source_pdf}\n"
            f"content: {self.content}"
        )


def rerank_semantic_hits(
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    store: LocalCaseDataStore,
    rerank_client: Any,
    limit: int = 100,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    """精排候选内的向量命中，并把精排结果融合回语义分。

    整体流程：
    1. 从候选个例中汇总所有向量命中的 chunk，构建 Rerank 输入
    2. 可用性检查：没命中 / Rerank 不可用 → 直接返回原结果 + 状态码
    3. 调用 Rerank 模型，对所有 chunk 做精排
    4. 调用成功 → 把精排结果融合回个例的语义分中
    5. 调用失败 → 降级，保留原向量排序，加警告

    Returns:
        (更新后的候选列表, 状态码, 警告列表)
        状态码：called / not_available / failed / skipped_no_hits
    """
    # 第一步：从候选个例中构建所有待 Rerank 的 chunk 列表
    raw_hits = _build_hits(candidates, store, limit)

    # 没有可 Rerank 的 chunk → 直接跳过，返回原候选
    if not raw_hits:
        return candidates, "skipped_no_hits", []

    # 第二步：检查 Rerank 客户端是否可用（不可用就降级，不调用）
    try:
        if not rerank_client or not rerank_client.is_available():
            return candidates, "not_available", ["Rerank 模型不可用，保留向量相似度排序。"]
    except Exception as exc:
        # 状态检查本身抛异常也算不可用
        return candidates, "not_available", [f"Rerank 模型状态检查失败，保留向量相似度排序：{exc}"]

    # 第三步：调用 Rerank 模型
    model = str(getattr(rerank_client, "model", None) or getattr(rerank_client, "model_uid", "unknown"))
    started = perf_counter()
    logger.info("[SmartCaseMatch][Rerank] 开始调用 model=%s documents=%d", model, len(raw_hits))
    try:
        # 调用 Rerank：传入查询文本 + 待排文档列表 + 返回数量
        ordered = rerank_client.rerank(query.get("query_text") or "", raw_hits, len(raw_hits))
        if not ordered:
            raise RuntimeError("Rerank 未返回有效排序")
    except Exception as exc:
        # 调用失败 → 降级，保留原向量排序，加警告
        logger.exception("[SmartCaseMatch][Rerank] 调用失败 model=%s error=%s", model, exc)
        return candidates, "failed", [f"Rerank 模型调用失败，保留向量相似度排序：{exc}"]

    # 第四步：调用成功 → 把 Rerank 结果融合回候选个例的语义分
    elapsed_ms = round((perf_counter() - started) * 1000, 2)
    logger.info("[SmartCaseMatch][Rerank] 调用成功 model=%s results=%d elapsed_ms=%.2f", model, len(ordered), elapsed_ms)
    return _merge_rerank_order(candidates, ordered), "called", []


def _build_hits(
    candidates: list[dict[str, Any]],
    store: LocalCaseDataStore,
    limit: int,
) -> list[ChunkRerankHit]:
    """按向量分汇总结构化候选中的命中 chunk，最多保留一百条。

    把所有候选个例的 semantic_hits 汇总成一个去重后的 chunk 列表，
    供 Rerank 模型统一排序。
    处理要点：
    - 同一个 chunk 可能被多个个例引用，只保留一份（去重）
    - 内容截断到 1800 字，避免 Rerank 输入过长
    - 最后按向量分从高到低排序，截取前 limit 条
    """
    hits: list[ChunkRerankHit] = []
    seen: set[str] = set()  # 已加入的 chunk_id，用于去重

    for candidate in candidates:
        case_id = str(candidate.get("case_id") or "")
        # 遍历这个个例的所有向量命中 chunk
        for item in candidate.get("semantic_hits") or []:
            chunk_id = str(item.get("chunk_id") or "")
            # 空 ID 或已经加过了就跳过
            if not chunk_id or chunk_id in seen:
                continue
            # 从 store 里取 chunk 完整内容
            chunk = store.get_chunk(chunk_id)
            if not chunk:
                continue
            seen.add(chunk_id)
            hits.append(ChunkRerankHit(
                chunk_id=chunk_id,
                case_id=case_id,
                source_pdf=str(chunk.get("source_pdf") or ""),
                content=str(chunk.get("content") or "")[:1800],  # 内容截断到 1800 字
                cosine_score=float(item.get("score") or 0.0),
            ))
    # 按向量相似度从高到低排序
    hits.sort(key=lambda item: item.cosine_score, reverse=True)
    # 最多保留 limit 条（默认 100），控制 Rerank 成本
    return hits[:limit]


def _merge_rerank_order(
    candidates: list[dict[str, Any]],
    ordered_hits: list[ChunkRerankHit],
) -> list[dict[str, Any]]:
    """使用 Rerank 顺序确定各个例代表 chunk，并小幅修正语义分。

    Rerank 输出的是 chunk 级别的排序，但我们需要的是个例级别的语义分。
    融合策略：
    1. 对每个个例，取它的前 2 个最高排名的 chunk 作为代表
    2. 计算 Rerank 分：1 - 排名位置 / 总数量（排名越靠前分越高，0~1 之间）
    3. 新的语义分 = 0.7 * 原向量分 + 0.3 * Rerank 分
       （向量分为主，Rerank 只做微调，权重 30%）
    4. 更新 semantic_score、semantic_chunk_ids 等字段
    """
    # 兼容客户端返回原对象、(对象, 分数)或带 hit/score 的字典；新客户端可直接保留相关度。
    normalized = [_normalize_ordered_hit(item) for item in ordered_hits]
    normalized = [item for item in normalized if item is not None]
    normalized_hits = [item[0] for item in normalized]
    # 总文档数以成功解析后的结果为准，避免异常条目改变位置分母。
    total = max(1, len(normalized_hits) - 1)
    raw_scores = {item[0].chunk_id: item[1] for item in normalized if item[1] is not None}
    # chunk_id → ChunkRerankHit 对象的映射，方便查表
    hit_by_id = {item.chunk_id: item for item in normalized_hits}
    # chunk_id → Rerank 后的排名位置（0 = 最相关）
    position = {item.chunk_id: index for index, item in enumerate(normalized_hits)}

    updated = []
    for candidate in candidates:
        # 找出这个个例的所有命中 chunk 在 Rerank 结果中的对象
        candidate_hits = [
            hit_by_id[str(item.get("chunk_id"))]
            for item in candidate.get("semantic_hits") or []
            if str(item.get("chunk_id")) in hit_by_id
        ]
        # 按 Rerank 排名重新排序（越靠前越相关）
        candidate_hits.sort(key=lambda item: position[item.chunk_id])
        # 取前 2 个作为这个个例的代表 chunk
        representatives = candidate_hits[:2]

        # 这个个例没有任何 Rerank 命中 → 原样返回
        if not representatives:
            updated.append(candidate)
            continue

        # 原向量分：取代表 chunk 的平均分
        cosine_score = sum(item.cosine_score for item in representatives) / len(representatives)
        if raw_scores and all(item.chunk_id in raw_scores for item in representatives):
            # 优先使用模型返回的真实相关度，避免同一 chunk 的分数随批次大小变化。
            score_values = list(raw_scores.values())
            low, high = min(score_values), max(score_values)
            normalized_scores = {
                chunk_id: (score - low) / (high - low) if high > low else 1.0
                for chunk_id, score in raw_scores.items()
            }
            rerank_score = sum(normalized_scores[item.chunk_id] for item in representatives) / len(representatives)
            score_source = "model_relevance"
        else:
            # 兼容当前只返回顺序的客户端；位置分仅作为没有真实相关度时的保守降级。
            rerank_score = sum(1.0 - position[item.chunk_id] / total for item in representatives) / len(representatives)
            score_source = "rank_position_fallback"
        # 融合语义分：向量分 70% + Rerank 分 30%（向量分为主，Rerank 微调）
        semantic_score = 0.7 * cosine_score + 0.3 * rerank_score

        updated.append({
            **candidate,
            "semantic_score": round(semantic_score, 4),        # 更新后的语义分
            "semantic_rerank_score": round(rerank_score, 4),  # Rerank 分（单独存，供审计）
            "semantic_rerank_score_source": score_source,
            "semantic_chunk_ids": [item.chunk_id for item in representatives],  # 更新代表 chunk
            "semantic_rerank_available": True,                 # 标记 Rerank 成功参与
        })
    return updated


def _normalize_ordered_hit(value: Any) -> tuple[ChunkRerankHit, float | None] | None:
    """读取带真实相关度的Rerank结果，同时兼容既有仅排序客户端。"""
    if isinstance(value, ChunkRerankHit):
        return value, None
    if isinstance(value, (tuple, list)) and len(value) == 2 and isinstance(value[0], ChunkRerankHit):
        try:
            return value[0], float(value[1])
        except (TypeError, ValueError):
            return value[0], None
    if isinstance(value, dict) and isinstance(value.get("hit"), ChunkRerankHit):
        try:
            return value["hit"], float(value.get("score"))
        except (TypeError, ValueError):
            return value["hit"], None
    return None
