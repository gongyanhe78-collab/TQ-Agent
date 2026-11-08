"""第一阶段冻结的用户、会话、记忆与反馈数据模型。"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base, utc_now


def json_type():
    """为 PostgreSQL 使用 JSONB，为测试 SQLite 使用标准 JSON。"""
    return JSON().with_variant(JSONB(), "postgresql")


class User(Base):
    """本地账号及其角色、状态。"""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(16), default="user", index=True)
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class RefreshToken(Base):
    """不可逆保存的刷新令牌及令牌族状态。"""

    __tablename__ = "refresh_tokens"

    token_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    family_id: Mapped[str] = mapped_column(String(36), index=True)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class UserProfile(Base):
    """结构化用户画像，第一阶段先建立权威事实表。"""

    __tablename__ = "user_profiles"

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        primary_key=True,
    )
    display_name: Mapped[str] = mapped_column(String(120), default="")
    preferences: Mapped[dict[str, Any]] = mapped_column(json_type(), default=dict)
    profile_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ChatSession(Base):
    """归属于单一用户的聊天会话。"""

    __tablename__ = "sessions"
    __table_args__ = (
        Index("idx_sessions_user_updated", "user_id", "updated_at"),
    )

    session_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        index=True,
    )
    title: Mapped[str] = mapped_column(String(200), default="新会话")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Message(Base):
    """会话内的权威历史消息。"""

    __tablename__ = "messages"
    __table_args__ = (
        Index("idx_messages_session_id", "session_id", "message_id"),
    )

    message_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        index=True,
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    message_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", json_type(), default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AgentRun(Base):
    """Agent 运行审计、证据和 Trace 关联。"""

    __tablename__ = "agent_runs"
    __table_args__ = (
        Index("idx_agent_runs_session_created", "session_id", "created_at"),
        Index("idx_agent_runs_trace", "trace_id"),
    )

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        index=True,
    )
    agent_type: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24))
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text, default="")
    trace_id: Mapped[str] = mapped_column(String(64), default="")
    evidence_cases: Mapped[list[Any]] = mapped_column(json_type(), default=list)
    evidence_chunks: Mapped[list[Any]] = mapped_column(json_type(), default=list)
    images: Mapped[list[Any]] = mapped_column(json_type(), default=list)
    reports: Mapped[list[Any]] = mapped_column(json_type(), default=list)
    run_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", json_type(), default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SessionAgentState(Base):
    """每种 Agent 在一个会话内的可恢复状态。"""

    __tablename__ = "session_agent_states"

    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        primary_key=True,
    )
    agent_type: Mapped[str] = mapped_column(String(64), primary_key=True)
    state: Mapped[dict[str, Any]] = mapped_column(json_type(), default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class MemoryItem(Base):
    """可审计、可删除的长期记忆事实。"""

    __tablename__ = "memory_items"
    __table_args__ = (
        Index("idx_memory_items_user_status", "user_id", "status"),
    )

    memory_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        index=True,
    )
    memory_type: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    sensitivity: Mapped[str] = mapped_column(String(16), default="normal")
    status: Mapped[str] = mapped_column(String(16), default="candidate")
    source_session_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    source_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    memory_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", json_type(), default=dict)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class MemorySummary(Base):
    """会话记忆摘要及压缩覆盖边界。"""

    __tablename__ = "memory_summaries"

    summary_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        index=True,
    )
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        index=True,
    )
    summary: Mapped[str] = mapped_column(Text)
    covered_message_id: Mapped[int] = mapped_column(Integer, default=0)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class MessageFeedback(Base):
    """按用户、消息和客户端隔离的互斥反馈。"""

    __tablename__ = "message_feedback"
    __table_args__ = (
        UniqueConstraint("session_id", "message_id", "client_id", name="uq_message_feedback_client"),
    )

    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        primary_key=True,
    )
    message_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        index=True,
    )
    rating: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SessionFeedbackGuidance(Base):
    """只作用于当前会话的临时反馈要求。"""

    __tablename__ = "session_feedback_guidance"
    __table_args__ = (
        UniqueConstraint("session_id", "source_message_id", "client_id", name="uq_feedback_guidance_client"),
    )

    guidance_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        index=True,
    )
    source_message_id: Mapped[int] = mapped_column(Integer)
    client_id: Mapped[str] = mapped_column(String(120))
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        index=True,
    )
    raw_feedback: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(24), default="pending")
    guidance_prompt: Mapped[str] = mapped_column(Text, default="")
    guidance_type: Mapped[list[Any]] = mapped_column(json_type(), default=list)
    requested_dimensions: Mapped[list[Any]] = mapped_column(json_type(), default=list)
    needs_same_scope_retrieval: Mapped[bool] = mapped_column(default=False)
    scope_change: Mapped[bool] = mapped_column(default=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class MessageIssueReport(Base):
    """用户对助手消息的问题上报。"""

    __tablename__ = "message_issue_reports"

    report_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        index=True,
    )
    message_id: Mapped[int] = mapped_column(Integer)
    client_id: Mapped[str] = mapped_column(String(120))
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        index=True,
    )
    category: Mapped[str] = mapped_column(String(80))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SharedMessageSnapshot(Base):
    """不可变只读的公开分享快照。"""

    __tablename__ = "shared_message_snapshots"

    token: Mapped[str] = mapped_column(String(96), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        index=True,
    )
    message_id: Mapped[int] = mapped_column(Integer)
    owner_user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        index=True,
    )
    title: Mapped[str] = mapped_column(String(200))
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AuditLog(Base):
    """管理员敏感操作的最小审计记录。"""

    __tablename__ = "audit_logs"

    audit_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    actor_user_id: Mapped[str] = mapped_column(String(36), index=True)
    action: Mapped[str] = mapped_column(String(80))
    target_type: Mapped[str] = mapped_column(String(40))
    target_id: Mapped[str] = mapped_column(String(120))
    detail: Mapped[dict[str, Any]] = mapped_column(json_type(), default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
