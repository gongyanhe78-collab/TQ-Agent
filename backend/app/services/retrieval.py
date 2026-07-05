"""
检索与问答服务模块
提供关键词重排序、答案生成、RAG（检索增强生成）完整流程
"""
from __future__ import annotations

import re

from backend.app.models import RagAnswer


# 匹配中文日期范围的正则表达式（如"2025年1月1～3日"）
DATE_RANGE_PATTERN = re.compile(
    r"(?:(?P<year>\d{4})年)?(?:(?P<month>\d{1,2})月)?(?P<start>\d{1,2})\s*[~～\-—至]\s*(?:(?P<end_month>\d{1,2})月)?(?P<end>\d{1,2})\s*日"
)
# 灾害类型关键词列表
DISASTER_TERMS = ("暴雪", "暴雨", "强对流", "高温", "寒潮", "沙尘", "雨雪", "大风", "霜冻", "降水")


class KeywordReranker:
    """
    关键词重排序器
    对向量检索的初步结果进行二次排序，基于关键词匹配度优化排序结果
    解决向量检索只看语义相似度、不看重关键词精确匹配的问题
    """

    def rerank(self, question, hits, top_n):
        """
        对检索结果进行重排序

        Args:
            question: 用户问题
            hits: 初步检索结果列表
            top_n: 返回前 N 个结果

        Returns:
            重排序后的结果列表（前 top_n 个）
        """
        # 从问题中提取关键词
        tokens = self._tokens(question)

        # 计算每个命中结果的得分：(关键词匹配数量, 向量相似度)
        # 先按关键词数量排序，数量相同则按向量相似度排序
        def score(hit):
            text = f"{hit.case.title} {hit.case.date_range} {hit.case.content}"
            # 统计有多少个关键词在文本中出现
            overlap = sum(1 for token in tokens if token and token in text)
            return (overlap, hit.score)

        # 按得分降序排序，取前 top_n 个
        return sorted(hits, key=score, reverse=True)[:top_n]

    def _tokens(self, text: str) -> list[str]:
        """
        简单分词（按标点和空白分割）

        Args:
            text: 待分词文本

        Returns:
            关键词列表
        """
        return [part for part in re.split(r"[\s，。！？、；：,.!?;:（）()]+", text) if part]


class SimpleAnswerGenerator:
    """
    简单答案生成器
    基于检索到的案例，可选择是否调用 LLM 生成回答
    """

    def __init__(self, llm_client=None):
        """
        初始化答案生成器

        Args:
            llm_client: LLM 客户端，不传则只返回摘要
        """
        self.llm_client = llm_client
        self.last_llm_used = False  # 上次是否实际调用了 LLM
        self.last_llm_status = "not_requested"  # 上次调用状态

    def answer(self, question, hits):
        """
        基于检索结果生成回答

        Args:
            question: 用户问题
            hits: 检索命中的案例列表

        Returns:
            回答字符串
        """
        self.last_llm_used = False
        self.last_llm_status = "not_requested"
        # 没有命中结果的情况
        if not hits:
            self.last_llm_status = "skipped_no_hits"
            return "未找到相关个例。"
        # 构建摘要上下文（标题 + 前 120 字内容）
        context = "\n".join(
            f"{hit.case.case_id}: {hit.case.title} {hit.case.content[:120]}"
            for hit in hits
        )
        # 如果配置了可用的 LLM 客户端，调用 LLM 生成回答
        if self.llm_client and self.llm_client.is_available():
            try:
                answer = self.llm_client.answer_with_context(
                    question,
                    # 为每个案例构建详细的上下文块
                    [
                        f"{hit.case.case_id}\n标题：{hit.case.title}\n时段：{hit.case.date_range}\n内容：{hit.case.content}"
                        for hit in hits
                    ],
                )
            except Exception as exc:
                # LLM 调用失败时降级为返回摘要
                self.last_llm_status = f"failed: {exc}"
                return f"大模型调用失败，以下为检索到的参考个例：\n{context}"
            self.last_llm_used = True
            self.last_llm_status = "called"
            return answer
        # 没有配置 API Key 时返回摘要
        self.last_llm_status = "skipped_no_api_key"
        return f"根据命中的历史个例，优先参考以下内容：\n{context}"


class RagService:
    """
    RAG（检索增强生成）问答服务
    整合向量检索、关键词重排、答案生成，提供完整的问答流程
    """

    def __init__(self, embedding_client, case_store, reranker, answer_generator):
        """
        初始化 RAG 服务

        Args:
            embedding_client: 向量嵌入客户端
            case_store: 案例存储
            reranker: 重排序器
            answer_generator: 答案生成器
        """
        self.embedding_client = embedding_client
        self.case_store = case_store
        self.reranker = reranker
        self.answer_generator = answer_generator

    def ask(self, question: str, top_k: int = 5, top_n: int = 3) -> RagAnswer:
        """
        完整的问答流程
        检索 -> 重排 -> 生成答案

        Args:
            question: 用户问题
            top_k: 向量检索取回的数量
            top_n: 重排后返回的数量

        Returns:
            RagAnswer 问答结果对象
        """
        # 第一步：检索并重排序
        retrieval_mode, reranked = self.retrieve(question, top_k=top_k, top_n=top_n)
        # 第二步：基于检索结果生成答案
        answer = self.answer_generator.answer(question, reranked)
        # 构建并返回完整结果
        return RagAnswer(
            question=question,
            answer=answer,
            hits=reranked,
            retrieval_mode=retrieval_mode,
            llm_used=getattr(self.answer_generator, "last_llm_used", False),
            llm_status=getattr(self.answer_generator, "last_llm_status", "not_requested"),
        )

    def retrieve(self, question: str, top_k: int = 5, top_n: int = 3):
        """
        检索阶段（向量检索 + 关键词重排）
        向量检索失败时自动降级为关键词匹配检索

        Args:
            question: 用户问题
            top_k: 向量检索取回数量
            top_n: 重排后返回数量

        Returns:
            (检索模式字符串, 重排后的结果列表)
        """
        retrieval_mode = "vector"
        try:
            # 检查向量索引兼容性（维度是否匹配）
            self._ensure_vector_index_compatible()
            # 将问题转为向量
            query_embedding = self.embedding_client.embed_query(question)
            # 向量检索
            recalled = self.case_store.query(query_embedding, top_k=top_k)
        except Exception as exc:
            # 向量检索失败，降级为关键词匹配检索
            retrieval_mode = f"vector_error_fallback: {exc}"
            recalled = self._lexical_recall(question, top_k=top_k)
        # 关键词重排序
        return retrieval_mode, self.reranker.rerank(question, recalled, top_n=top_n)

    def _ensure_vector_index_compatible(self) -> None:
        """
        检查向量索引是否兼容（内部方法）
        检查点：1) 嵌入客户端可用 2) 向量维度匹配

        Raises:
            RuntimeError: 如果不兼容
        """
        # 检查嵌入客户端是否可用
        if hasattr(self.embedding_client, "is_available") and not self.embedding_client.is_available():
            raise RuntimeError("embedding client is unavailable")
        # 如果存储没有 collection_info 方法则跳过检查
        if not hasattr(self.case_store, "collection_info"):
            return
        # 获取预期的向量维度
        expected = (
            self.embedding_client.expected_dimension()
            if hasattr(self.embedding_client, "expected_dimension")
            else None
        )
        if expected is None:
            return
        # 获取实际的向量维度
        actual = self.case_store.collection_info().get("dimension")
        if actual != expected:
            raise RuntimeError(
                f"vector index dimension mismatch: expected {expected}, got {actual}; click 入库 to rebuild"
            )

    def _lexical_recall(self, question: str, top_k: int):
        """
        关键词匹配检索（降级方案）（内部方法）
        当向量检索不可用时，使用关键词匹配进行检索

        Args:
            question: 用户问题
            top_k: 返回数量

        Returns:
            命中结果列表
        """
        if not hasattr(self.case_store, "list_cases"):
            return []
        # 对问题进行分词和关键词提取
        tokens = self._tokenize(question)
        hits = []
        # 遍历所有案例计算匹配得分
        for case in self.case_store.list_cases():
            score = self._score_case(tokens, case)
            # 动态创建 Hit 对象（兼容 RetrievalHit 接口）
            hits.append(type("Hit", (), {"case": case, "score": float(score)})())
        # 按得分降序排序
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """
        对问题进行分词和关键词提取（内部方法）
        提取多种匹配关键词：原文本、日期变体、灾害词、月份/日期、分词结果

        Args:
            text: 问题文本

        Returns:
            关键词列表（去重）
        """
        normalized = self._normalize_text(text)
        tokens: list[str] = [normalized]
        # 添加日期变体（如"1月1-3日"、"1-3日"、"1日"、"3日"）
        tokens.extend(self._date_variants(normalized))
        # 添加问题中包含的灾害类型关键词
        tokens.extend(term for term in DISASTER_TERMS if term in normalized)
        # 添加问题中的月份和日期（如"1月"、"3日"）
        tokens.extend(piece for piece in re.findall(r"\d{1,2}月|\d{1,2}日", normalized))
        # 添加普通分词结果
        tokens.extend(
            piece
            for piece in re.split(r"[\s，。！？、；：,.!?;:（）()]+", normalized)
            if piece
        )
        # 去重（保留顺序）并过滤空字符串
        return list(dict.fromkeys(token for token in tokens if token))

    def _score_case(self, tokens: list[str], case) -> int:
        """
        计算案例与问题的关键词匹配得分（内部方法）
        不同字段权重不同：日期范围 > 标题 > 内容

        Args:
            tokens: 问题关键词列表
            case: 案例对象

        Returns:
            匹配得分（整数，越高越匹配）
        """
        title = self._normalize_text(case.title)
        date_range = self._normalize_text(case.date_range)
        content = self._normalize_text(case.content)
        score = 0
        for token in tokens:
            # 标题匹配权重：至少 8 分或长度 * 3
            if token in title:
                score += max(8, len(token) * 3)
            # 日期范围匹配权重：至少 10 分或长度 * 4（最高权重）
            if token in date_range:
                score += max(10, len(token) * 4)
            # 内容匹配权重：至少 1 分或长度
            if token in content:
                score += max(1, len(token))
        return score

    def _normalize_text(self, text: str) -> str:
        """
        标准化文本（统一日期分隔符，移除空格）

        Args:
            text: 原始文本

        Returns:
            标准化后的文本
        """
        return (
            text.replace("～", "-")
            .replace("~", "-")
            .replace("—", "-")
            .replace("至", "-")
            .replace(" ", "")
        )

    def _date_variants(self, text: str) -> list[str]:
        """
        提取日期的多种变体形式（内部方法）
        例如从"1月1～3日"提取出"1月1-3日"、"1-3日"、"1日"、"3日"

        Args:
            text: 包含日期的文本

        Returns:
            日期变体列表
        """
        variants: list[str] = []
        for match in DATE_RANGE_PATTERN.finditer(text):
            month = match.group("month") or match.group("end_month")
            start = match.group("start")
            end = match.group("end")
            if month:
                variants.append(f"{month}月{start}-{end}日")
            variants.append(f"{start}-{end}日")
            variants.append(f"{start}日")
            variants.append(f"{end}日")
        return variants

    def _lexical_score(self, tokens: list[str], haystack: str) -> int:
        """
        计算通用文本匹配得分（备用方法）

        Args:
            tokens: 关键词列表
            haystack: 待匹配文本

        Returns:
            匹配得分
        """
        normalized = self._normalize_text(haystack)
        score = 0
        for token in tokens:
            if token in normalized:
                score += max(2, len(token))
        return score
