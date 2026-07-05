from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """普通 RAG 问答请求体。"""

    question: str = Field(min_length=1)
    top_k: int = 5
    top_n: int = 3


class StreamQueryRequest(QueryRequest):
    """流式问答请求体，允许绑定 session_id 保存多轮会话消息。"""

    session_id: str | None = None


class AgentQueryRequest(QueryRequest):
    """智能体专用问答请求体，增加会话和上下文返回开关。"""

    session_id: str | None = None
    return_context: bool = True
    metadata: dict | None = None


class CreateSessionRequest(BaseModel):
    """创建或更新会话标题的请求体。"""

    title: str | None = None


class DeleteVectorsRequest(BaseModel):
    """批量删除向量节点请求体。"""

    keys: list[str] = Field(min_length=1)
