"""
API 请求数据模型模块
定义所有 REST API 接口的请求和响应数据结构，包括问答、流式问答、
智能体查询、会话管理和向量删除等接口的参数验证。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    普通 RAG 问答请求体

    Attributes:
        question: 用户问题文本
        top_k: 向量检索取回的数量
        top_n: 返回结果数量
    """

    question: str = Field(min_length=1)
    top_k: int = 5
    top_n: int = 3


class StreamQueryRequest(QueryRequest):
    """流式问答请求体，允许绑定 session_id 保存多轮会话消息。"""

    session_id: str | None = None
    regenerate_from_message_id: int | None = Field(
        default=None,
        ge=1,
        description="需要重新生成的助手消息 ID；设置后复用其原问题且排除旧回答",
    )
    regeneration_instruction: str = Field(
        default="",
        max_length=1000,
        description="用户针对当前回答提出的重生成约束，只作用于本次版本",
    )


class AgentQueryRequest(QueryRequest):
    """智能体专用问答请求体，增加会话和上下文返回开关。"""

    session_id: str | None = None
    return_context: bool = True
    metadata: dict | None = None


class CreateSessionRequest(BaseModel):
    """创建或更新会话标题的请求体。"""

    title: str | None = None


class MessageFeedbackRequest(BaseModel):
    """单条助手消息的正向或改进反馈。"""

    rating: str = Field(pattern="^(love|needs_improvement|none)$")
    client_id: str = Field(default="anonymous", min_length=1, max_length=120)
    reason: str = Field(default="", max_length=1000)


class MessageIssueRequest(BaseModel):
    """用户上报单条回答问题时提交的分类和说明。"""

    category: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=2000)
    client_id: str = Field(default="anonymous", min_length=1, max_length=120)
