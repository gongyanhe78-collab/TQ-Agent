"""
数据模型模块
定义所有核心数据结构：提取的案例、检索结果、RAG回答
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExtractedCase:
    """
    从 PDF 中提取的气象灾害案例数据模型

    Attributes:
        source_pdf: 来源 PDF 文件名
        case_id: 案例唯一标识符（格式：文件名-case-序号）
        case_no: 案例在该 PDF 中的序号
        title: 案例标题（如"1月11日雨雪天气过程"）
        date_range: 案例发生的时间范围（如"1月11～12日"）
        content: 案例详细正文内容
        file_path: 提取后保存的 txt 文件路径
        embedding: 案例文本的向量嵌入（1024维浮点数数组）
    """
    source_pdf: str
    case_id: str
    case_no: int
    title: str
    date_range: str
    content: str
    file_path: str | None = None
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        将案例对象转换为字典格式
        用于 JSON 序列化存储

        Returns:
            包含所有字段的字典
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractedCase":
        """
        从字典数据还原 ExtractedCase 对象

        Args:
            data: 包含所有案例字段的字典

        Returns:
            ExtractedCase 实例
        """
        return cls(**data)


@dataclass
class DocumentChunk:
    """从 PDF 文档化后切分出的独立文本块，用于新的文档级向量库。"""

    source_pdf: str
    chunk_id: str
    chunk_no: int
    content: str
    file_path: str | None = None
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """将文档 chunk 转成可写入 JSON 备份文件的字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentChunk":
        """从 JSON 备份中的字典还原 DocumentChunk 对象。"""
        return cls(**data)


@dataclass
class StandardCase:
    """标准化灾害个例记录，是文档 chunk 之上的结构化数据层。"""

    case_id: str
    title: str
    date_range: str
    disaster_types: list[str] = field(default_factory=list)
    affected_areas: list[str] = field(default_factory=list)
    source_pdf: str = ""
    source_chunk_ids: list[str] = field(default_factory=list)
    summary: str = ""
    weather_facts: str = ""
    forecast_focus: str = ""
    evidence_image_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """将标准化个例转成可 JSON 序列化的字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StandardCase":
        """从 JSON 字典还原标准化个例，并兼容缺失字段。"""
        defaults = {
            "disaster_types": [],
            "affected_areas": [],
            "source_pdf": "",
            "source_chunk_ids": [],
            "summary": "",
            "weather_facts": "",
            "forecast_focus": "",
            "evidence_image_ids": [],
            "confidence": 0.0,
        }
        normalized = {**defaults, **data}
        return cls(**normalized)


@dataclass
class RetrievalHit:
    """
    检索命中结果数据模型

    Attributes:
        case: 命中的案例对象
        score: 相似度得分（余弦相似度，值越大越相似）
    """
    case: ExtractedCase
    score: float


@dataclass
class DocumentRetrievalHit:
    """文档 chunk 检索命中结果。"""

    chunk: DocumentChunk
    score: float


@dataclass
class RagAnswer:
    """
    RAG（检索增强生成）问答结果数据模型

    Attributes:
        question: 用户的原始问题
        answer: 生成的回答文本
        hits: 检索命中的案例列表（RetrievalHit 对象数组）
        retrieval_mode: 检索模式（"vector" 向量检索 / 其他降级模式）
        llm_used: 是否实际调用了大模型生成回答
        llm_status: LLM 调用状态（not_requested/called/skipped_no_api_key/failed等）
    """
    question: str
    answer: str
    hits: list[RetrievalHit] = field(default_factory=list)
    retrieval_mode: str = "vector"
    llm_used: bool = False
    llm_status: str = "not_requested"
