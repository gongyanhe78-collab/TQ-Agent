"""
会话存储服务模块
使用 SQLite 持久化存储聊天会话和消息记录
提供会话的增删改查、消息追加等功能
"""
from __future__ import annotations

import sqlite3
import uuid
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
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        return cursor.rowcount > 0

    def add_message(self, session_id: str, role: str, content: str) -> dict:
        """
        向会话中追加一条消息
        同时更新会话的最后活动时间戳

        Args:
            session_id: 会话 ID
            role: 消息角色（如 user、assistant）
            content: 消息内容

        Returns:
            新增消息的数据字典

        Raises:
            KeyError: 会话不存在时抛出
        """
        if self.get_session(session_id) is None:
            raise KeyError(f"session not found: {session_id}")
        now = self._now()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
        return {
            "message_id": cursor.lastrowid,
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": now,
        }

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
                    "SELECT message_id, session_id, role, content, created_at "
                    "FROM messages WHERE session_id = ? ORDER BY message_id ASC"
                ),
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

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
                    "created_at TEXT NOT NULL)"
                )
            )

    def _connect(self):
        """
        创建 SQLite 数据库连接（内部方法）
        返回的连接支持通过列名访问行数据

        Returns:
            SQLite 连接对象（row_factory 已设置为 sqlite3.Row）
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _now(self) -> str:
        """
        获取当前 UTC 时间（内部方法）

        Returns:
            ISO8601 格式的时间字符串（带时区）
        """
        return datetime.now(timezone.utc).isoformat()
