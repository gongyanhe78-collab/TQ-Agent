"""按用户隔离的 SQLAlchemy 会话仓储，兼容旧 SessionStore 接口。"""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from backend.app.db.base import utc_now
from backend.app.db.models import (
    AgentRun,
    ChatSession,
    MemorySummary,
    Message,
    MessageFeedback,
    MessageIssueReport,
    SessionAgentState,
    SessionFeedbackGuidance,
    SharedMessageSnapshot,
)


def _as_utc(value: datetime | None) -> datetime | None:
    """把数据库返回的无时区时间按 UTC 解释。"""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _iso(value: datetime | None) -> str | None:
    """把时间统一序列化为带时区的 ISO 字符串。"""
    normalized = _as_utc(value)
    return normalized.isoformat() if normalized else None


class SharedMessageRepository:
    """匿名读取公开分享快照，不暴露会话归属和内部消息元数据。"""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def get_shared_snapshot(self, token: str) -> dict | None:
        """按公开 token 读取未撤销、未过期的分享快照。"""
        now = utc_now()
        with Session(self.engine) as db:
            row = db.get(SharedMessageSnapshot, token)
            if (
                row is None
                or row.revoked_at is not None
                or (
                    row.expires_at is not None
                    and _as_utc(row.expires_at) <= now
                )
            ):
                return None
            return {
                "token": row.token,
                "title": row.title,
                "question": row.question,
                "answer": row.answer,
                "created_at": _iso(row.created_at),
            }

    def shared_snapshot_meta(self, token: str) -> dict | None:
        """返回分享快照存在状态，用于区分 404 与过期/撤销 403。"""
        with Session(self.engine) as db:
            row = db.get(SharedMessageSnapshot, token)
            if row is None:
                return None
            return {
                "token": row.token,
                "expires_at": _iso(row.expires_at),
                "revoked_at": _iso(row.revoked_at),
            }


class SqlAlchemySessionStore:
    """以单个用户身份访问会话事实表，所有资源条件在 SQL 层生效。"""

    def __init__(self, engine: Engine, user_id: str) -> None:
        if not user_id:
            raise ValueError("user_id 不能为空")
        self.engine = engine
        self.user_id = user_id

    def create_session(self, title: str | None = None) -> dict:
        """为当前用户创建会话并返回稳定响应结构。"""
        now = utc_now()
        session_id = str(uuid4())
        with Session(self.engine) as db:
            row = ChatSession(
                session_id=session_id,
                user_id=self.user_id,
                title=str(title or "新会话")[:200],
                created_at=now,
                updated_at=now,
            )
            db.add(row)
            db.commit()
        return self.get_session(session_id) or {}

    def list_sessions(self) -> list[dict]:
        """只列出当前用户的会话，并按最近活动时间倒序。"""
        with Session(self.engine) as db:
            rows = db.scalars(
                select(ChatSession)
                .where(ChatSession.user_id == self.user_id)
                .order_by(ChatSession.updated_at.desc())
            )
            return [self._session_dict(row) for row in rows]

    def get_session(self, session_id: str) -> dict | None:
        """读取当前用户拥有的单个会话。"""
        with Session(self.engine) as db:
            row = self._session(db, session_id)
            return self._session_dict(row) if row else None

    def update_session_title(self, session_id: str, title: str) -> dict:
        """更新当前用户会话标题，不存在时按旧接口抛出 KeyError。"""
        now = utc_now()
        with Session(self.engine) as db:
            row = self._session(db, session_id)
            if row is None:
                raise KeyError(f"session not found: {session_id}")
            row.title = str(title or "新会话")[:200]
            row.updated_at = now
            db.commit()
        return self.get_session(session_id) or {}

    def delete_session(self, session_id: str) -> bool:
        """删除当前用户会话及依赖数据，跨用户访问返回 False。"""
        with Session(self.engine) as db:
            row = self._session(db, session_id)
            if row is None:
                return False
            for model in (
                MessageFeedback,
                SessionFeedbackGuidance,
                MessageIssueReport,
                SharedMessageSnapshot,
                SessionAgentState,
                AgentRun,
                MemorySummary,
                Message,
            ):
                db.execute(delete(model).where(model.session_id == session_id))
            db.delete(row)
            db.commit()
        return True

    def save_agent_state(self, session_id: str, agent_type: str, state: dict) -> None:
        """保存当前用户会话内某类 Agent 的可恢复状态。"""
        with Session(self.engine) as db:
            if self._session(db, session_id) is None:
                raise KeyError(f"session not found: {session_id}")
            row = db.get(SessionAgentState, (session_id, agent_type))
            if row is None:
                row = SessionAgentState(
                    session_id=session_id,
                    agent_type=agent_type,
                    state=dict(state or {}),
                    updated_at=utc_now(),
                )
                db.add(row)
            else:
                row.state = dict(state or {})
                row.updated_at = utc_now()
            db.commit()

    def get_agent_state(self, session_id: str, agent_type: str) -> dict | None:
        """读取当前用户会话内的 Agent 状态。"""
        with Session(self.engine) as db:
            if self._session(db, session_id) is None:
                return None
            row = db.get(SessionAgentState, (session_id, agent_type))
            return dict(row.state or {}) if row else None

    def delete_agent_state(self, session_id: str, agent_type: str) -> None:
        """删除当前用户会话内的 Agent 状态。"""
        with Session(self.engine) as db:
            if self._session(db, session_id) is None:
                return
            row = db.get(SessionAgentState, (session_id, agent_type))
            if row is not None:
                db.delete(row)
                db.commit()

    def start_agent_run(
        self,
        run_id: str,
        session_id: str,
        agent_type: str,
        question: str,
        *,
        status: str = "running",
        metadata: dict | None = None,
    ) -> dict:
        """创建当前用户会话的 Agent 运行记录。"""
        now = utc_now()
        with Session(self.engine) as db:
            if self._session(db, session_id) is None:
                raise KeyError(f"session not found: {session_id}")
            row = db.get(AgentRun, run_id)
            if row is not None and row.session_id != session_id:
                raise KeyError(f"agent run not found: {run_id}")
            if row is None:
                row = AgentRun(
                    run_id=run_id,
                    session_id=session_id,
                    agent_type=agent_type,
                    status=status,
                    question=question,
                    answer="",
                    trace_id=str((metadata or {}).get("trace_id") or ""),
                    evidence_cases=[],
                    evidence_chunks=[],
                    images=[],
                    reports=[],
                    run_metadata=dict(metadata or {}),
                    created_at=now,
                    updated_at=now,
                )
                db.add(row)
            else:
                row.agent_type = agent_type
                row.status = status
                row.question = question
                row.run_metadata = dict(metadata or {})
                row.trace_id = str(
                    (metadata or {}).get("trace_id")
                    or row.trace_id
                    or ""
                )
                row.updated_at = now
            db.commit()
        return self.get_agent_run(run_id) or {}

    def finish_agent_run(
        self,
        run_id: str,
        *,
        status: str,
        answer: str = "",
        evidence_cases: list[dict] | None = None,
        evidence_chunks: list[dict] | None = None,
        images: list[dict] | None = None,
        reports: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """写入当前用户 Agent 运行的最终结果。"""
        with Session(self.engine) as db:
            row = self._agent_run(db, run_id)
            if row is None:
                return {}
            row.status = status
            row.answer = answer
            row.evidence_cases = list(evidence_cases or [])
            row.evidence_chunks = list(evidence_chunks or [])
            row.images = list(images or [])
            row.reports = list(reports or [])
            row.run_metadata = dict(metadata or {})
            row.trace_id = str(row.run_metadata.get("trace_id") or row.trace_id or "")
            row.updated_at = utc_now()
            db.commit()
        return self.get_agent_run(run_id) or {}

    def get_agent_run(self, run_id: str) -> dict | None:
        """读取当前用户拥有的 Agent 运行。"""
        with Session(self.engine) as db:
            row = self._agent_run(db, run_id)
            return self._agent_run_dict(row) if row else None

    def list_agent_runs(self, session_id: str) -> list[dict]:
        """按倒序列出当前用户会话的 Agent 运行。"""
        with Session(self.engine) as db:
            if self._session(db, session_id) is None:
                return []
            rows = db.scalars(
                select(AgentRun)
                .where(AgentRun.session_id == session_id)
                .order_by(AgentRun.created_at.desc())
            )
            return [self._agent_run_dict(row) for row in rows]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> dict:
        """追加消息并同步会话最近活动时间。"""
        now = utc_now()
        with Session(self.engine) as db:
            session = self._session(db, session_id)
            if session is None:
                raise KeyError(f"session not found: {session_id}")
            row = Message(
                session_id=session_id,
                role=role,
                content=content,
                message_metadata=dict(metadata or {}),
                created_at=now,
            )
            session.updated_at = now
            db.add(row)
            db.commit()
            db.refresh(row)
            return self._message_dict(row)

    def list_messages(self, session_id: str) -> list[dict]:
        """按消息编号正序读取当前用户会话消息。"""
        with Session(self.engine) as db:
            if self._session(db, session_id) is None:
                return []
            rows = db.scalars(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.message_id.asc())
            )
            return [self._message_dict(row) for row in rows]

    def get_message(self, session_id: str, message_id: int) -> dict | None:
        """同时按会话和消息编号读取，阻止跨会话使用编号。"""
        with Session(self.engine) as db:
            if self._session(db, session_id) is None:
                return None
            row = db.scalar(
                select(Message).where(
                    Message.session_id == session_id,
                    Message.message_id == int(message_id),
                )
            )
            return self._message_dict(row) if row else None

    def previous_user_message(
        self,
        session_id: str,
        assistant_message_id: int,
    ) -> dict | None:
        """读取助手消息之前最近的一条用户消息。"""
        with Session(self.engine) as db:
            if self._session(db, session_id) is None:
                return None
            row = db.scalar(
                select(Message)
                .where(
                    Message.session_id == session_id,
                    Message.role == "user",
                    Message.message_id < int(assistant_message_id),
                )
                .order_by(Message.message_id.desc())
                .limit(1)
            )
            return self._message_dict(row) if row else None

    def list_messages_before(self, session_id: str, message_id: int) -> list[dict]:
        """读取目标消息之前的当前用户历史。"""
        with Session(self.engine) as db:
            if self._session(db, session_id) is None:
                return []
            rows = db.scalars(
                select(Message)
                .where(
                    Message.session_id == session_id,
                    Message.message_id < int(message_id),
                )
                .order_by(Message.message_id.asc())
            )
            return [self._message_dict(row) for row in rows]

    def list_messages_after(self, session_id: str, message_id: int = 0) -> list[dict]:
        """读取指定编号之后的当前用户增量消息。"""
        with Session(self.engine) as db:
            if self._session(db, session_id) is None:
                return []
            rows = db.scalars(
                select(Message)
                .where(
                    Message.session_id == session_id,
                    Message.message_id > max(0, int(message_id)),
                )
                .order_by(Message.message_id.asc())
            )
            return [self._message_dict(row) for row in rows]

    def update_message_metadata(
        self,
        session_id: str,
        message_id: int,
        patch: dict,
    ) -> dict:
        """合并更新消息元数据，不覆盖已有证据和审计字段。"""
        with Session(self.engine) as db:
            if self._session(db, session_id) is None:
                raise KeyError(f"message not found: {message_id}")
            row = db.scalar(
                select(Message).where(
                    Message.session_id == session_id,
                    Message.message_id == int(message_id),
                )
            )
            if row is None:
                raise KeyError(f"message not found: {message_id}")
            metadata = dict(row.message_metadata or {})
            metadata.update(patch or {})
            row.message_metadata = metadata
            db.commit()
            db.refresh(row)
            return self._message_dict(row)

    def set_message_feedback(
        self,
        session_id: str,
        message_id: int,
        client_id: str,
        rating: str,
        reason: str = "",
    ) -> dict:
        """按当前用户、消息和客户端保存互斥反馈。"""
        if self.get_message(session_id, message_id) is None:
            raise KeyError(f"message not found: {message_id}")
        now = utc_now()
        with Session(self.engine) as db:
            row = db.get(MessageFeedback, (session_id, int(message_id), client_id))
            if rating == "none":
                if row is not None:
                    db.delete(row)
                guidance = db.scalar(
                    select(SessionFeedbackGuidance).where(
                        SessionFeedbackGuidance.session_id == session_id,
                        SessionFeedbackGuidance.source_message_id == int(message_id),
                        SessionFeedbackGuidance.client_id == client_id,
                        SessionFeedbackGuidance.user_id == self.user_id,
                    )
                )
                if guidance is not None:
                    db.delete(guidance)
            elif row is None:
                db.add(
                    MessageFeedback(
                        session_id=session_id,
                        message_id=int(message_id),
                        client_id=client_id,
                        user_id=self.user_id,
                        rating=rating,
                        reason=str(reason or "")[:1000],
                        created_at=now,
                        updated_at=now,
                    )
                )
            else:
                row.rating = rating
                row.reason = str(reason or "")[:1000]
                row.updated_at = now
            if rating != "needs_improvement":
                guidance = db.scalar(
                    select(SessionFeedbackGuidance).where(
                        SessionFeedbackGuidance.session_id == session_id,
                        SessionFeedbackGuidance.source_message_id == int(message_id),
                        SessionFeedbackGuidance.client_id == client_id,
                        SessionFeedbackGuidance.user_id == self.user_id,
                    )
                )
                if guidance is not None:
                    db.delete(guidance)
            db.commit()
        return {"message_id": int(message_id), "rating": rating, "reason": reason}

    def get_message_feedback(
        self,
        session_id: str,
        message_id: int,
        client_id: str,
    ) -> dict:
        """读取当前用户客户端对消息的反馈。"""
        with Session(self.engine) as db:
            if self._session(db, session_id) is None:
                return {"rating": "none", "reason": ""}
            row = db.scalar(
                select(MessageFeedback).where(
                    MessageFeedback.session_id == session_id,
                    MessageFeedback.message_id == int(message_id),
                    MessageFeedback.client_id == client_id,
                    MessageFeedback.user_id == self.user_id,
                )
            )
            if row is None:
                return {"rating": "none", "reason": ""}
            return {"rating": row.rating, "reason": row.reason}

    def create_feedback_guidance(
        self,
        session_id: str,
        source_message_id: int,
        client_id: str,
        raw_feedback: str,
    ) -> dict:
        """创建或重置当前消息对应的临时反馈要求。"""
        with Session(self.engine) as db:
            if self._session(db, session_id) is None:
                raise KeyError(f"session not found: {session_id}")
            if self._message(db, session_id, source_message_id) is None:
                raise KeyError(f"message not found: {source_message_id}")
            existing = db.scalar(
                select(SessionFeedbackGuidance).where(
                    SessionFeedbackGuidance.session_id == session_id,
                    SessionFeedbackGuidance.source_message_id == int(source_message_id),
                    SessionFeedbackGuidance.client_id == client_id,
                )
            )
            if existing is not None:
                db.delete(existing)
                db.flush()
            guidance_id = f"feedback_guidance_{uuid4().hex}"
            now = utc_now()
            db.add(
                SessionFeedbackGuidance(
                    guidance_id=guidance_id,
                    session_id=session_id,
                    source_message_id=int(source_message_id),
                    client_id=client_id,
                    user_id=self.user_id,
                    raw_feedback=str(raw_feedback or "")[:1000],
                    status="pending",
                    guidance_prompt="",
                    guidance_type=[],
                    requested_dimensions=[],
                    needs_same_scope_retrieval=False,
                    scope_change=False,
                    confidence=0.0,
                    error="",
                    created_at=now,
                    updated_at=now,
                )
            )
            db.commit()
        return self.get_feedback_guidance(guidance_id) or {
            "guidance_id": guidance_id,
            "status": "pending",
        }

    def get_feedback_guidance(self, guidance_id: str) -> dict | None:
        """读取当前用户拥有的临时反馈要求。"""
        with Session(self.engine) as db:
            row = db.scalar(
                select(SessionFeedbackGuidance).where(
                    SessionFeedbackGuidance.guidance_id == guidance_id,
                    SessionFeedbackGuidance.user_id == self.user_id,
                )
            )
            return self._guidance_dict(row) if row else None

    def update_feedback_guidance(self, guidance_id: str, result: dict) -> dict | None:
        """写入模型归一化结果。"""
        payload = result if isinstance(result, dict) else {
            "status": "failed",
            "error": "归一化结果不是对象",
        }
        with Session(self.engine) as db:
            row = db.scalar(
                select(SessionFeedbackGuidance).where(
                    SessionFeedbackGuidance.guidance_id == guidance_id,
                    SessionFeedbackGuidance.user_id == self.user_id,
                )
            )
            if row is None:
                return None
            row.status = str(payload.get("status") or "failed")[:24]
            row.guidance_prompt = str(payload.get("guidance_prompt") or "")[:300]
            row.guidance_type = list(payload.get("guidance_type") or [])[:4]
            row.requested_dimensions = list(payload.get("requested_dimensions") or [])[:5]
            row.needs_same_scope_retrieval = bool(payload.get("needs_same_scope_retrieval"))
            row.scope_change = bool(payload.get("scope_change"))
            row.confidence = float(payload.get("confidence") or 0)
            row.error = str(payload.get("error") or "")[:240]
            row.updated_at = utc_now()
            db.commit()
        return self.get_feedback_guidance(guidance_id)

    def set_feedback_guidance_status(
        self,
        guidance_id: str,
        status: str,
        error: str = "",
    ) -> dict | None:
        """只更新临时反馈要求状态。"""
        with Session(self.engine) as db:
            row = db.scalar(
                select(SessionFeedbackGuidance).where(
                    SessionFeedbackGuidance.guidance_id == guidance_id,
                    SessionFeedbackGuidance.user_id == self.user_id,
                )
            )
            if row is None:
                return None
            row.status = str(status)[:24]
            row.error = str(error)[:240]
            row.updated_at = utc_now()
            db.commit()
        return self.get_feedback_guidance(guidance_id)

    def list_feedback_guidances(
        self,
        session_id: str,
        statuses: tuple[str, ...] = ("active",),
        limit: int = 5,
    ) -> list[dict]:
        """列出当前用户会话的指定状态临时要求。"""
        if not statuses:
            return []
        safe_limit = max(1, min(int(limit), 5))
        with Session(self.engine) as db:
            if self._session(db, session_id) is None:
                return []
            rows = db.scalars(
                select(SessionFeedbackGuidance)
                .where(
                    SessionFeedbackGuidance.session_id == session_id,
                    SessionFeedbackGuidance.user_id == self.user_id,
                    SessionFeedbackGuidance.status.in_(statuses),
                )
                .order_by(SessionFeedbackGuidance.updated_at.desc())
                .limit(safe_limit)
            )
            return [self._guidance_dict(row) for row in rows]

    def expire_old_feedback_guidances(self, session_id: str, keep: int = 5) -> None:
        """只保留当前用户会话最近若干条有效临时要求。"""
        safe_keep = max(1, min(int(keep), 5))
        with Session(self.engine) as db:
            if self._session(db, session_id) is None:
                return
            rows = list(
                db.scalars(
                    select(SessionFeedbackGuidance)
                    .where(
                        SessionFeedbackGuidance.session_id == session_id,
                        SessionFeedbackGuidance.user_id == self.user_id,
                        SessionFeedbackGuidance.status == "active",
                    )
                    .order_by(SessionFeedbackGuidance.updated_at.desc())
                )
            )
            now = utc_now()
            for row in rows[safe_keep:]:
                row.status = "expired"
                row.updated_at = now
            db.commit()

    def add_message_issue(
        self,
        session_id: str,
        message_id: int,
        client_id: str,
        category: str,
        description: str = "",
    ) -> dict:
        """保存当前用户对助手消息的问题上报。"""
        if self.get_message(session_id, message_id) is None:
            raise KeyError(f"message not found: {message_id}")
        report_id = f"issue_{uuid4().hex}"
        with Session(self.engine) as db:
            db.add(
                MessageIssueReport(
                    report_id=report_id,
                    session_id=session_id,
                    message_id=int(message_id),
                    client_id=client_id,
                    user_id=self.user_id,
                    category=category,
                    description=str(description or "")[:2000],
                    created_at=utc_now(),
                )
            )
            db.commit()
        return {"report_id": report_id, "message_id": int(message_id), "status": "submitted"}

    def create_shared_snapshot(
        self,
        session_id: str,
        message_id: int,
        question: str,
        answer: str,
        title: str,
        expires_at=None,
    ) -> dict:
        """创建不含内部元数据的公开分享快照。"""
        if self.get_message(session_id, message_id) is None:
            raise KeyError(f"message not found: {message_id}")
        token = uuid4().hex + uuid4().hex
        now = utc_now()
        with Session(self.engine) as db:
            db.add(
                SharedMessageSnapshot(
                    token=token,
                    session_id=session_id,
                    message_id=int(message_id),
                    owner_user_id=self.user_id,
                    title=str(title or "")[:200],
                    question=str(question or ""),
                    answer=str(answer or ""),
                    created_at=now,
                    expires_at=expires_at,
                )
            )
            db.commit()
        return {
            "token": token,
            "title": title,
            "created_at": _iso(now),
            "expires_at": _iso(expires_at),
        }

    def get_shared_snapshot(self, token: str) -> dict | None:
        """公开读取未撤销、未过期的分享快照。"""
        now = utc_now()
        with Session(self.engine) as db:
            row = db.get(SharedMessageSnapshot, token)
            if (
                row is None
                or row.revoked_at is not None
                or (
                    row.expires_at is not None
                    and _as_utc(row.expires_at) <= now
                )
            ):
                return None
            return {
                "token": row.token,
                "title": row.title,
                "question": row.question,
                "answer": row.answer,
                "created_at": _iso(row.created_at),
            }

    def shared_snapshot_meta(self, token: str) -> dict | None:
        """返回分享快照存在状态，用于区分 404 与过期/撤销 403。"""
        with Session(self.engine) as db:
            row = db.get(SharedMessageSnapshot, token)
            if row is None:
                return None
            return {
                "token": row.token,
                "expires_at": _iso(row.expires_at),
                "revoked_at": _iso(row.revoked_at),
            }

    def revoke_shared_snapshot(
        self,
        session_id: str,
        message_id: int,
    ) -> bool:
        """撤销会话内某条消息的公开分享，返回是否撤销成功。"""
        now = utc_now()
        with Session(self.engine) as db:
            row = db.scalars(
                select(SharedMessageSnapshot).where(
                    SharedMessageSnapshot.session_id == session_id,
                    SharedMessageSnapshot.message_id == int(message_id),
                    SharedMessageSnapshot.revoked_at.is_(None),
                )
            ).first()
            if row is None:
                return False
            row.revoked_at = now
            db.commit()
            return True

    @staticmethod
    def _session_dict(row: ChatSession) -> dict:
        """转换会话响应，不暴露内部用户归属。"""
        return {
            "session_id": row.session_id,
            "title": row.title,
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
        }

    @staticmethod
    def _message_dict(row: Message) -> dict:
        """把消息元数据同时保留在 metadata 和兼容顶层字段。"""
        metadata = dict(row.message_metadata or {})
        result = {
            "message_id": int(row.message_id),
            "session_id": row.session_id,
            "role": row.role,
            "content": row.content,
            "created_at": _iso(row.created_at),
            "metadata": metadata,
        }
        result.update(metadata)
        return result

    @staticmethod
    def _agent_run_dict(row: AgentRun) -> dict:
        """转换 Agent 运行记录并保留历史字段名。"""
        return {
            "run_id": row.run_id,
            "session_id": row.session_id,
            "agent_type": row.agent_type,
            "status": row.status,
            "question": row.question,
            "answer": row.answer,
            "trace_id": row.trace_id,
            "evidence_cases": list(row.evidence_cases or []),
            "evidence_chunks": list(row.evidence_chunks or []),
            "images": list(row.images or []),
            "reports": list(row.reports or []),
            "metadata": dict(row.run_metadata or {}),
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
        }

    @staticmethod
    def _guidance_dict(row: SessionFeedbackGuidance) -> dict:
        """转换临时反馈要求字段类型。"""
        return {
            "guidance_id": row.guidance_id,
            "session_id": row.session_id,
            "source_message_id": int(row.source_message_id),
            "client_id": row.client_id,
            "raw_feedback": row.raw_feedback,
            "status": row.status,
            "guidance_prompt": row.guidance_prompt,
            "guidance_type": list(row.guidance_type or []),
            "requested_dimensions": list(row.requested_dimensions or []),
            "needs_same_scope_retrieval": bool(row.needs_same_scope_retrieval),
            "scope_change": bool(row.scope_change),
            "confidence": float(row.confidence or 0),
            "error": row.error,
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
        }

    def _session(self, db: Session, session_id: str) -> ChatSession | None:
        """同时按会话和当前用户查询父资源。"""
        if not session_id:
            return None
        return db.scalar(
            select(ChatSession).where(
                ChatSession.session_id == session_id,
                ChatSession.user_id == self.user_id,
            )
        )

    def _message(self, db: Session, session_id: str, message_id: int) -> Message | None:
        """在已校验用户会话内读取消息。"""
        if self._session(db, session_id) is None:
            return None
        return db.scalar(
            select(Message).where(
                Message.session_id == session_id,
                Message.message_id == int(message_id),
            )
        )

    def _agent_run(self, db: Session, run_id: str) -> AgentRun | None:
        """通过父会话归属约束 Agent 运行查询。"""
        return db.scalar(
            select(AgentRun)
            .join(ChatSession, ChatSession.session_id == AgentRun.session_id)
            .where(
                AgentRun.run_id == run_id,
                ChatSession.user_id == self.user_id,
            )
        )
