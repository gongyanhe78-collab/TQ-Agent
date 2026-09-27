"""
会话存储服务模块
使用 SQLite 持久化存储聊天会话和消息记录
提供会话的增删改查、消息追加等功能
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class SessionStore:
    """
    会话存储服务
    使用 SQLite 数据库持久化保存聊天会话和消息记录
    支持会话创建、查询、更新、删除，以及消息追加
    """

    def __init__(self, db_path: Path):
        """
        初始化会话存储
        设置数据库文件路径，自动创建所需的数据表（如果不存在）

        Args:
            db_path: SQLite 数据库文件路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def create_session(self, title: str | None = None) -> dict:
        """
        创建新的聊天会话

        Args:
            title: 会话标题，不传则默认使用"新会话"

        Returns:
            新会话的完整数据行字典
        """
        now = self._now()
        session_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title or "新会话", now, now),
            )
        return self.get_session(session_id)

    def list_sessions(self) -> list[dict]:
        """
        列出所有会话
        按最后活动时间倒序排列（最新的在前）

        Returns:
            会话列表（字典数组）
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> dict | None:
        """
        根据会话 ID 获取单个会话信息

        Args:
            session_id: 会话 ID

        Returns:
            会话数据字典，不存在则返回 None
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id, title, created_at, updated_at FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_session_title(self, session_id: str, title: str) -> dict:
        """
        更新会话标题

        Args:
            session_id: 会话 ID
            title: 新的会话标题

        Returns:
            更新后的会话数据行

        Raises:
            KeyError: 会话不存在时抛出
        """
        if self.get_session(session_id) is None:
            raise KeyError(f"session not found: {session_id}")
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (title or "新会话", now, session_id),
            )
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        """
        删除会话及其所有消息

        Args:
            session_id: 会话 ID

        Returns:
            删除成功返回 True，会话不存在返回 False
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM message_feedback WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM session_feedback_guidance WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM message_issue_reports WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM shared_message_snapshots WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM session_agent_states WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM agent_runs WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        return cursor.rowcount > 0

    def save_agent_state(self, session_id: str, agent_type: str, state: dict) -> None:
        """保存某个会话下的 Agent 状态，多维检索的待确认条件也存放在这里。"""
        if self.get_session(session_id) is None:
            raise KeyError(f"session not found: {session_id}")
        now = self._now()
        state_json = json.dumps(state or {}, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                (
                    "INSERT INTO session_agent_states (session_id, agent_type, state, updated_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(session_id, agent_type) DO UPDATE SET "
                    "state = excluded.state, updated_at = excluded.updated_at"
                ),
                (session_id, agent_type, state_json, now),
            )

    def get_agent_state(self, session_id: str, agent_type: str) -> dict | None:
        """读取指定会话和 Agent 的持久化状态。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state FROM session_agent_states WHERE session_id = ? AND agent_type = ?",
                (session_id, agent_type),
            ).fetchone()
        if row is None:
            return None
        return self._parse_metadata(row["state"])

    def delete_agent_state(self, session_id: str, agent_type: str) -> None:
        """清理指定 Agent 状态，不影响同一会话中的聊天消息。"""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM session_agent_states WHERE session_id = ? AND agent_type = ?",
                (session_id, agent_type),
            )

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
        """创建 Agent 运行记录，运行编号由具体 Agent 或统一入口生成。"""
        if self.get_session(session_id) is None:
            raise KeyError(f"session not found: {session_id}")
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                (
                    "INSERT INTO agent_runs ("
                    "run_id, session_id, agent_type, status, question, answer, evidence_cases, "
                    "evidence_chunks, images, reports, metadata, created_at, updated_at"
                    ") VALUES (?, ?, ?, ?, ?, '', '[]', '[]', '[]', '[]', ?, ?, ?)"
                ),
                (
                    run_id,
                    session_id,
                    agent_type,
                    status,
                    question,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
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
        """写入 Agent 最终答案、证据、图片、报告和运行元数据。"""
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                (
                    "UPDATE agent_runs SET status = ?, answer = ?, evidence_cases = ?, "
                    "evidence_chunks = ?, images = ?, reports = ?, metadata = ?, updated_at = ? "
                    "WHERE run_id = ?"
                ),
                (
                    status,
                    answer,
                    json.dumps(evidence_cases or [], ensure_ascii=False),
                    json.dumps(evidence_chunks or [], ensure_ascii=False),
                    json.dumps(images or [], ensure_ascii=False),
                    json.dumps(reports or [], ensure_ascii=False),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    run_id,
                ),
            )
        return self.get_agent_run(run_id) or {}

    def get_agent_run(self, run_id: str) -> dict | None:
        """按运行编号读取完整 Agent 结果。"""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        for field in ("evidence_cases", "evidence_chunks", "images", "reports"):
            result[field] = self._parse_json_list(result.get(field))
        result["metadata"] = self._parse_metadata(result.get("metadata"))
        return result

    def list_agent_runs(self, session_id: str) -> list[dict]:
        """按时间倒序返回一个主会话下的全部 Agent 运行记录。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id FROM agent_runs WHERE session_id = ? ORDER BY created_at DESC",
                (session_id,),
            ).fetchall()
        return [run for row in rows if (run := self.get_agent_run(row["run_id"])) is not None]

    def add_message(self, session_id: str, role: str, content: str, metadata: dict | None = None) -> dict:
        """
        向会话中追加一条消息
        同时更新会话的最后活动时间戳

        Args:
            session_id: 会话 ID
            role: 消息角色（如 user、assistant）
            content: 消息内容
            metadata: 附加元数据，例如 assistant 消息关联的图片证据

        Returns:
            新增消息的数据字典

        Raises:
            KeyError: 会话不存在时抛出
        """
        if self.get_session(session_id) is None:
            raise KeyError(f"session not found: {session_id}")
        now = self._now()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at, metadata) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, now, metadata_json),
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
        message = {
            "message_id": cursor.lastrowid,
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": now,
            "metadata": metadata or {},
        }
        message.update(metadata or {})
        return message

    def list_messages(self, session_id: str) -> list[dict]:
        """
        列出指定会话的所有消息
        按插入顺序正序排列（最早的消息在前）

        Args:
            session_id: 会话 ID

        Returns:
            消息列表（字典数组）
        """
        with self._connect() as conn:
            rows = conn.execute(
                (
                    "SELECT message_id, session_id, role, content, created_at, metadata "
                    "FROM messages WHERE session_id = ? ORDER BY message_id ASC"
                ),
                (session_id,),
            ).fetchall()
        messages = []
        for row in rows:
            message = dict(row)
            metadata = self._parse_metadata(message.pop("metadata", ""))
            message["metadata"] = metadata
            message.update(metadata)
            messages.append(message)
        return messages

    def get_message(self, session_id: str, message_id: int) -> dict | None:
        """读取会话内的一条消息，防止跨会话使用消息编号。"""
        with self._connect() as conn:
            row = conn.execute(
                (
                    "SELECT message_id, session_id, role, content, created_at, metadata "
                    "FROM messages WHERE session_id = ? AND message_id = ?"
                ),
                (session_id, int(message_id)),
            ).fetchone()
        if row is None:
            return None
        message = dict(row)
        metadata = self._parse_metadata(message.pop("metadata", ""))
        message["metadata"] = metadata
        message.update(metadata)
        return message

    def previous_user_message(self, session_id: str, assistant_message_id: int) -> dict | None:
        """定位助手回答之前最近的一条用户消息，供导出标题和重新生成使用。"""
        with self._connect() as conn:
            row = conn.execute(
                (
                    "SELECT message_id, session_id, role, content, created_at, metadata "
                    "FROM messages WHERE session_id = ? AND role = 'user' AND message_id < ? "
                    "ORDER BY message_id DESC LIMIT 1"
                ),
                (session_id, int(assistant_message_id)),
            ).fetchone()
        if row is None:
            return None
        message = dict(row)
        metadata = self._parse_metadata(message.pop("metadata", ""))
        message["metadata"] = metadata
        message.update(metadata)
        return message

    def list_messages_before(self, session_id: str, message_id: int) -> list[dict]:
        """读取目标消息之前的历史，用于重新生成时隔离旧回答。"""
        with self._connect() as conn:
            rows = conn.execute(
                (
                    "SELECT message_id, session_id, role, content, created_at, metadata "
                    "FROM messages WHERE session_id = ? AND message_id < ? ORDER BY message_id ASC"
                ),
                (session_id, int(message_id)),
            ).fetchall()
        messages = []
        for row in rows:
            message = dict(row)
            metadata = self._parse_metadata(message.pop("metadata", ""))
            message["metadata"] = metadata
            message.update(metadata)
            messages.append(message)
        return messages

    def update_message_metadata(self, session_id: str, message_id: int, patch: dict) -> dict:
        """合并更新消息元数据，保留已有证据、报告和运行信息。"""
        message = self.get_message(session_id, message_id)
        if message is None:
            raise KeyError(f"message not found: {message_id}")
        metadata = dict(message.get("metadata") or {})
        metadata.update(patch or {})
        with self._connect() as conn:
            conn.execute(
                "UPDATE messages SET metadata = ? WHERE session_id = ? AND message_id = ?",
                (json.dumps(metadata, ensure_ascii=False), session_id, int(message_id)),
            )
        return self.get_message(session_id, message_id) or {}

    def set_message_feedback(
        self,
        session_id: str,
        message_id: int,
        client_id: str,
        rating: str,
        reason: str = "",
    ) -> dict:
        """按消息和客户端幂等保存互斥反馈，rating=none 时删除反馈。"""
        if self.get_message(session_id, message_id) is None:
            raise KeyError(f"message not found: {message_id}")
        now = self._now()
        with self._connect() as conn:
            if rating == "none":
                conn.execute(
                    "DELETE FROM message_feedback WHERE session_id = ? AND message_id = ? AND client_id = ?",
                    (session_id, int(message_id), client_id),
                )
                # 取消反馈后，不能继续把对应的临时要求带入后续对话。
                conn.execute(
                    "DELETE FROM session_feedback_guidance WHERE session_id = ? AND source_message_id = ? AND client_id = ?",
                    (session_id, int(message_id), client_id),
                )
            else:
                conn.execute(
                    (
                        "INSERT INTO message_feedback "
                        "(session_id, message_id, client_id, rating, reason, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(session_id, message_id, client_id) DO UPDATE SET "
                        "rating = excluded.rating, reason = excluded.reason, updated_at = excluded.updated_at"
                    ),
                    (session_id, int(message_id), client_id, rating, reason, now, now),
                )
                if rating != "needs_improvement":
                    # 点赞或切换为其他反馈时，清理同一消息留下的改进要求。
                    conn.execute(
                        "DELETE FROM session_feedback_guidance WHERE session_id = ? AND source_message_id = ? AND client_id = ?",
                        (session_id, int(message_id), client_id),
                    )
        return {"message_id": int(message_id), "rating": rating, "reason": reason}

    def get_message_feedback(self, session_id: str, message_id: int, client_id: str) -> dict:
        """读取指定客户端对消息的当前反馈，供刷新会话后恢复选中态。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT rating, reason FROM message_feedback WHERE session_id = ? AND message_id = ? AND client_id = ?",
                (session_id, int(message_id), client_id),
            ).fetchone()
        return dict(row) if row else {"rating": "none", "reason": ""}

    def create_feedback_guidance(
        self,
        session_id: str,
        source_message_id: int,
        client_id: str,
        raw_feedback: str,
    ) -> dict:
        """创建或重置当前消息对应的临时反馈归一化任务。"""
        if self.get_session(session_id) is None:
            raise KeyError(f"session not found: {session_id}")
        if self.get_message(session_id, source_message_id) is None:
            raise KeyError(f"message not found: {source_message_id}")
        now = self._now()
        raw = str(raw_feedback or "")[:1000]
        with self._connect() as conn:
            existing = conn.execute(
                (
                    "SELECT guidance_id FROM session_feedback_guidance "
                    "WHERE session_id = ? AND source_message_id = ? AND client_id = ?"
                ),
                (session_id, int(source_message_id), client_id),
            ).fetchone()
            if existing:
                # 每次重新提交生成新的任务号，旧的后台任务即使晚到也不能覆盖新反馈。
                conn.execute(
                    "DELETE FROM session_feedback_guidance WHERE guidance_id = ?",
                    (str(existing["guidance_id"]),),
                )
                guidance_id = f"feedback_guidance_{uuid.uuid4().hex}"
                conn.execute(
                    (
                        "INSERT INTO session_feedback_guidance (guidance_id, session_id, source_message_id, client_id, "
                        "raw_feedback, status, guidance_prompt, guidance_type, requested_dimensions, "
                        "needs_same_scope_retrieval, scope_change, confidence, error, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, 'pending', '', '[]', '[]', 0, 0, 0, '', ?, ?)"
                    ),
                    (guidance_id, session_id, int(source_message_id), client_id, raw, now, now),
                )
            else:
                guidance_id = f"feedback_guidance_{uuid.uuid4().hex}"
                conn.execute(
                    (
                        "INSERT INTO session_feedback_guidance (guidance_id, session_id, source_message_id, client_id, "
                        "raw_feedback, status, guidance_prompt, guidance_type, requested_dimensions, "
                        "needs_same_scope_retrieval, scope_change, confidence, error, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, 'pending', '', '[]', '[]', 0, 0, 0, '', ?, ?)"
                    ),
                    (guidance_id, session_id, int(source_message_id), client_id, raw, now, now),
                )
        return self.get_feedback_guidance(guidance_id) or {"guidance_id": guidance_id, "status": "pending"}

    def get_feedback_guidance(self, guidance_id: str) -> dict | None:
        """读取单条临时反馈要求。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM session_feedback_guidance WHERE guidance_id = ?",
                (guidance_id,),
            ).fetchone()
        return self._feedback_guidance_row(row) if row else None

    def update_feedback_guidance(self, guidance_id: str, result: dict) -> dict | None:
        """写入模型归一化结果，所有字段均来自后端校验后的结构。"""
        if not isinstance(result, dict):
            result = {"status": "failed", "error": "归一化结果不是对象"}
        now = self._now()
        status = str(result.get("status") or "failed")[:24]
        with self._connect() as conn:
            conn.execute(
                (
                    "UPDATE session_feedback_guidance SET status = ?, guidance_prompt = ?, guidance_type = ?, "
                    "requested_dimensions = ?, needs_same_scope_retrieval = ?, scope_change = ?, confidence = ?, "
                    "error = ?, updated_at = ? WHERE guidance_id = ?"
                ),
                (
                    status,
                    str(result.get("guidance_prompt") or "")[:300],
                    json.dumps(list(result.get("guidance_type") or [])[:4], ensure_ascii=False),
                    json.dumps(list(result.get("requested_dimensions") or [])[:5], ensure_ascii=False),
                    int(bool(result.get("needs_same_scope_retrieval"))),
                    int(bool(result.get("scope_change"))),
                    float(result.get("confidence") or 0),
                    str(result.get("error") or "")[:240],
                    now,
                    guidance_id,
                ),
            )
        return self.get_feedback_guidance(guidance_id)

    def set_feedback_guidance_status(self, guidance_id: str, status: str, error: str = "") -> dict | None:
        """只更新临时要求状态，避免去重或过期时覆盖已归一化内容。"""
        with self._connect() as conn:
            conn.execute(
                "UPDATE session_feedback_guidance SET status = ?, error = ?, updated_at = ? WHERE guidance_id = ?",
                (str(status)[:24], str(error)[:240], self._now(), guidance_id),
            )
        return self.get_feedback_guidance(guidance_id)

    def list_feedback_guidances(
        self,
        session_id: str,
        statuses: tuple[str, ...] = ("active",),
        limit: int = 5,
    ) -> list[dict]:
        """按会话读取临时要求，默认只返回当前有效要求。"""
        safe_limit = max(1, min(int(limit), 5))
        placeholders = ",".join("?" for _ in statuses)
        with self._connect() as conn:
            rows = conn.execute(
                (
                    "SELECT * FROM session_feedback_guidance WHERE session_id = ? "
                    f"AND status IN ({placeholders}) ORDER BY updated_at DESC LIMIT ?"
                ),
                (session_id, *statuses, safe_limit),
            ).fetchall()
        return [self._feedback_guidance_row(row) for row in rows]

    def expire_old_feedback_guidances(self, session_id: str, keep: int = 5) -> None:
        """只保留最近若干条有效要求，避免会话级 Prompt 无限增长。"""
        safe_keep = max(1, min(int(keep), 5))
        with self._connect() as conn:
            rows = conn.execute(
                (
                    "SELECT guidance_id FROM session_feedback_guidance WHERE session_id = ? AND status = 'active' "
                    "ORDER BY updated_at DESC"
                ),
                (session_id,),
            ).fetchall()
            for row in rows[safe_keep:]:
                conn.execute(
                    "UPDATE session_feedback_guidance SET status = 'expired', updated_at = ? WHERE guidance_id = ?",
                    (self._now(), row["guidance_id"]),
                )

    @staticmethod
    def _feedback_guidance_row(row) -> dict:
        """把 SQLite 行转换成前端和编排器可用的临时要求对象。"""
        item = dict(row)
        for key in ("guidance_type", "requested_dimensions"):
            try:
                value = json.loads(item.get(key) or "[]")
            except json.JSONDecodeError:
                value = []
            item[key] = value if isinstance(value, list) else []
        item["needs_same_scope_retrieval"] = bool(item.get("needs_same_scope_retrieval"))
        item["scope_change"] = bool(item.get("scope_change"))
        item["confidence"] = float(item.get("confidence") or 0)
        return item

    def add_message_issue(
        self,
        session_id: str,
        message_id: int,
        client_id: str,
        category: str,
        description: str = "",
    ) -> dict:
        """保存问题上报，并返回可追踪的报告编号。"""
        if self.get_message(session_id, message_id) is None:
            raise KeyError(f"message not found: {message_id}")
        report_id = f"issue_{uuid.uuid4().hex}"
        with self._connect() as conn:
            conn.execute(
                (
                    "INSERT INTO message_issue_reports "
                    "(report_id, session_id, message_id, client_id, category, description, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)"
                ),
                (report_id, session_id, int(message_id), client_id, category, description, self._now()),
            )
        return {"report_id": report_id, "message_id": int(message_id), "status": "submitted"}

    def create_shared_snapshot(
        self,
        session_id: str,
        message_id: int,
        question: str,
        answer: str,
        title: str,
    ) -> dict:
        """创建不可变的公开快照，只保存用户可见问题和回答。"""
        token = uuid.uuid4().hex + uuid.uuid4().hex
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                (
                    "INSERT INTO shared_message_snapshots "
                    "(token, session_id, message_id, title, question, answer, created_at, revoked_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, '')"
                ),
                (token, session_id, int(message_id), title, question, answer, now),
            )
        return {"token": token, "title": title, "created_at": now}

    def get_shared_snapshot(self, token: str) -> dict | None:
        """读取仍有效的共享快照，不返回原会话内部元数据。"""
        with self._connect() as conn:
            row = conn.execute(
                (
                    "SELECT token, title, question, answer, created_at "
                    "FROM shared_message_snapshots WHERE token = ? AND revoked_at = ''"
                ),
                (token,),
            ).fetchone()
        return dict(row) if row else None

    def list_messages_after(self, session_id: str, message_id: int = 0) -> list[dict]:
        """只读取指定消息之后的增量，供后台会话记忆刷新使用。"""
        with self._connect() as conn:
            rows = conn.execute(
                (
                    "SELECT message_id, session_id, role, content, created_at, metadata "
                    "FROM messages WHERE session_id = ? AND message_id > ? ORDER BY message_id ASC"
                ),
                (session_id, max(0, int(message_id))),
            ).fetchall()
        messages = []
        for row in rows:
            message = dict(row)
            metadata = self._parse_metadata(message.pop("metadata", ""))
            message["metadata"] = metadata
            message.update(metadata)
            messages.append(message)
        return messages

    def _init_db(self) -> None:
        """
        初始化数据库表（内部方法）
        创建 sessions 和 messages 表（如果不存在）
        """
        with self._connect() as conn:
            conn.execute(
                (
                    "CREATE TABLE IF NOT EXISTS sessions ("
                    "session_id TEXT PRIMARY KEY, "
                    "title TEXT NOT NULL, "
                    "created_at TEXT NOT NULL, "
                    "updated_at TEXT NOT NULL)"
                )
            )
            conn.execute(
                (
                    "CREATE TABLE IF NOT EXISTS messages ("
                    "message_id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "session_id TEXT NOT NULL, "
                    "role TEXT NOT NULL, "
                    "content TEXT NOT NULL, "
                    "created_at TEXT NOT NULL, "
                    "metadata TEXT NOT NULL DEFAULT '{}')"
                )
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(messages)").fetchall()
            }
            if "metadata" not in columns:
                conn.execute("ALTER TABLE messages ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")
            # Agent 运行结果独立成表，避免把大块证据挤进会话主表。
            conn.execute(
                (
                    "CREATE TABLE IF NOT EXISTS agent_runs ("
                    "run_id TEXT PRIMARY KEY, "
                    "session_id TEXT NOT NULL, "
                    "agent_type TEXT NOT NULL, "
                    "status TEXT NOT NULL, "
                    "question TEXT NOT NULL, "
                    "answer TEXT NOT NULL DEFAULT '', "
                    "evidence_cases TEXT NOT NULL DEFAULT '[]', "
                    "evidence_chunks TEXT NOT NULL DEFAULT '[]', "
                    "images TEXT NOT NULL DEFAULT '[]', "
                    "reports TEXT NOT NULL DEFAULT '[]', "
                    "metadata TEXT NOT NULL DEFAULT '{}', "
                    "created_at TEXT NOT NULL, "
                    "updated_at TEXT NOT NULL)"
                )
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_runs_session ON agent_runs(session_id, created_at)"
            )
            # 每个主会话为每种 Agent 保存一份状态，多维检索提案不再落单独 JSON 文件。
            conn.execute(
                (
                    "CREATE TABLE IF NOT EXISTS session_agent_states ("
                    "session_id TEXT NOT NULL, "
                    "agent_type TEXT NOT NULL, "
                    "state TEXT NOT NULL DEFAULT '{}', "
                    "updated_at TEXT NOT NULL, "
                    "PRIMARY KEY(session_id, agent_type))"
                )
            )
            # 用户反馈与问题上报独立存储，避免污染回答展示元数据。
            conn.execute(
                (
                    "CREATE TABLE IF NOT EXISTS message_feedback ("
                    "session_id TEXT NOT NULL, message_id INTEGER NOT NULL, client_id TEXT NOT NULL, "
                    "rating TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '', "
                    "created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
                    "PRIMARY KEY(session_id, message_id, client_id))"
                )
            )
            # 会话级临时反馈要求独立存储，不进入长期记忆、用户问题或回答正文。
            conn.execute(
                (
                    "CREATE TABLE IF NOT EXISTS session_feedback_guidance ("
                    "guidance_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, source_message_id INTEGER NOT NULL, "
                    "client_id TEXT NOT NULL, raw_feedback TEXT NOT NULL DEFAULT '', status TEXT NOT NULL, "
                    "guidance_prompt TEXT NOT NULL DEFAULT '', guidance_type TEXT NOT NULL DEFAULT '[]', "
                    "requested_dimensions TEXT NOT NULL DEFAULT '[]', needs_same_scope_retrieval INTEGER NOT NULL DEFAULT 0, "
                    "scope_change INTEGER NOT NULL DEFAULT 0, confidence REAL NOT NULL DEFAULT 0, "
                    "error TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
                    "UNIQUE(session_id, source_message_id, client_id))"
                )
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_guidance_session ON session_feedback_guidance(session_id, status, updated_at)"
            )
            conn.execute(
                (
                    "CREATE TABLE IF NOT EXISTS message_issue_reports ("
                    "report_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, message_id INTEGER NOT NULL, "
                    "client_id TEXT NOT NULL, category TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', "
                    "created_at TEXT NOT NULL)"
                )
            )
            # 共享链接只保存可见快照，不暴露证据、审计和本地路径。
            conn.execute(
                (
                    "CREATE TABLE IF NOT EXISTS shared_message_snapshots ("
                    "token TEXT PRIMARY KEY, session_id TEXT NOT NULL, message_id INTEGER NOT NULL, "
                    "title TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL, "
                    "created_at TEXT NOT NULL, revoked_at TEXT NOT NULL DEFAULT '')"
                )
            )

    @contextmanager
    def _connect(self):
        """
        创建 SQLite 数据库连接（内部方法）
        返回的连接支持通过列名访问行数据

        Returns:
            SQLite 连接对象（row_factory 已设置为 sqlite3.Row）
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _parse_metadata(self, value: str | None) -> dict:
        if not value:
            return {}
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _parse_json_list(self, value: str | None) -> list:
        """解析 SQLite 中的 JSON 数组，损坏数据按空数组处理。"""
        if not value:
            return []
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    def _now(self) -> str:
        """
        获取当前 UTC 时间（内部方法）

        Returns:
            ISO8601 格式的时间字符串（带时区）
        """
        return datetime.now(timezone.utc).isoformat()
