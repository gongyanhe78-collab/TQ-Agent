"""把旧 SQLite 业务数据一次性、可重复地迁移到 PostgreSQL 权威库。"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import secrets
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import Engine, create_engine, func, inspect, select, text, tuple_
from sqlalchemy.orm import Session

from backend.app.db import models
from backend.app.db.base import Base
from backend.app.security.passwords import hash_password

TABLES = (
    "sessions",
    "messages",
    "agent_runs",
    "session_agent_states",
    "message_feedback",
    "session_feedback_guidance",
    "message_issue_reports",
    "shared_message_snapshots",
    "user_profiles",
    "memory_items",
    "memory_summaries",
)

PRIMARY_KEYS = {
    "sessions": ("session_id",),
    "messages": ("message_id",),
    "agent_runs": ("run_id",),
    "session_agent_states": ("session_id", "agent_type"),
    "message_feedback": ("session_id", "message_id", "client_id"),
    "session_feedback_guidance": ("guidance_id",),
    "message_issue_reports": ("report_id",),
    "shared_message_snapshots": ("token",),
    "user_profiles": ("user_id",),
    "memory_items": ("memory_id",),
    "memory_summaries": ("summary_id",),
}


@dataclass(frozen=True, slots=True)
class MigrationOptions:
    """迁移入口参数，密码只用于创建尚不存在的基线账号。"""

    source: Path
    target_url: str
    legacy_user_id: str = "legacy_user"
    legacy_username: str = "legacy_user"
    legacy_password: str | None = None
    dry_run: bool = False


@dataclass(slots=True)
class TableCount:
    """单表源数据、目标数据和本次处理结果。"""

    source: int = 0
    target: int = 0
    inserted: int = 0
    skipped: int = 0


@dataclass(slots=True)
class MigrationReport:
    """迁移计数和 dry-run 状态，可供 CLI 或测试读取。"""

    dry_run: bool
    counts: dict[str, TableCount] = field(default_factory=dict)

    def reconcile_mismatches(self) -> list[str]:
        """返回源、目标主键行数不一致的表名。"""
        return [
            table
            for table, count in self.counts.items()
            if count.source != count.target
        ]


class ReadOnlySqliteSource:
    """只读访问旧 SQLite，不调用 SessionStore 的初始化写逻辑。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"SQLite 源库不存在：{self.path}")
        uri = self.path.as_uri() + "?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA query_only=ON")

    def close(self) -> None:
        """关闭只读连接。"""
        self.connection.close()

    def read_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        """读取所有已知表；旧库不存在的高级表按空表处理。"""
        return {table: self._read_table(table) for table in TABLES}

    def _read_table(self, table: str) -> list[dict[str, Any]]:
        """在确认表存在后按稳定主键顺序读取全部行。"""
        if not self._has_table(table):
            return []
        order_columns = ", ".join(PRIMARY_KEYS[table])
        rows = self.connection.execute(
            f"SELECT * FROM {table} ORDER BY {order_columns}"
        ).fetchall()
        return [dict(row) for row in rows]

    def _has_table(self, table: str) -> bool:
        """只查询系统目录，不修改源库结构。"""
        row = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        return row is not None


def normalize_database_url(value: str) -> str:
    """把常见 PostgreSQL 地址转换为 psycopg 3 驱动地址。"""
    url = str(value or "").strip()
    if not url:
        raise ValueError("目标数据库地址不能为空")
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def normalize_utc(value: Any, *, fallback: datetime) -> datetime:
    """把旧库时间统一转换为带 UTC 时区的 datetime。"""
    if value is None or (isinstance(value, str) and not value.strip()):
        return fallback
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int | float)):
        parsed = datetime.fromtimestamp(float(value), tz=UTC)
    elif isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"无法解析时间值：{value}") from exc
    else:
        raise TypeError(f"不支持的时间值类型：{type(value).__name__}")
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def optional_utc(value: Any) -> datetime | None:
    """把空字符串或空值转换为 NULL，其余值按 UTC 解析。"""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return normalize_utc(value, fallback=datetime.now(UTC))


def _json_value(
    value: Any,
    expected_type: type,
    default: Any,
) -> Any:
    """解析 SQLite JSON 文本，并要求结果类型与目标列一致。"""
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    parsed = value
    if isinstance(value, bytes):
        parsed = value.decode("utf-8")
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError as exc:
            raise ValueError("旧库包含无法解析的 JSON 字段") from exc
    if not isinstance(parsed, expected_type):
        raise TypeError(
            f"JSON 字段类型不匹配：期望 {expected_type.__name__}，"
            f"实际 {type(parsed).__name__}"
        )
    return parsed


def _json_dict(value: Any) -> dict[str, Any]:
    """解析对象型 JSON。"""
    return dict(_json_value(value, dict, {}))


def _json_list(value: Any) -> list[Any]:
    """解析数组型 JSON。"""
    return list(_json_value(value, list, []))


def _boolean(value: Any) -> bool:
    """兼容 SQLite 的 0/1、文本和原生布尔值。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int | float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _text(value: Any, default: str = "") -> str:
    """把可空旧字段转换为目标库要求的非空文本。"""
    return default if value is None else str(value)


def _required_text(value: Any, field_name: str) -> str:
    """读取必须存在的旧字段，空值直接中止迁移。"""
    result = _text(value).strip()
    if not result:
        raise ValueError(f"迁移字段 {field_name} 不能为空")
    return result


def _insert_or_skip(
    db: Session,
    model: type,
    primary_key: tuple[Any, ...],
    values: dict[str, Any],
    report: MigrationReport,
    table: str,
    validate_existing: Callable[[Any], bool] | None = None,
) -> Any:
    """按主键插入缺失行；已存在时只校验稳定归属关系。"""
    lookup = primary_key[0] if len(primary_key) == 1 else primary_key
    existing = db.get(model, lookup)
    if existing is not None:
        if validate_existing is not None and not validate_existing(existing):
            raise ValueError(f"{table} 主键已存在但归属或业务标识不一致")
        report.counts[table].skipped += 1
        return existing
    row = model(**values)
    db.add(row)
    report.counts[table].inserted += 1
    return row


def _validate_target_schema(engine: Engine) -> None:
    """要求目标库先通过 Alembic 建好完整结构。"""
    actual_tables = set(inspect(engine).get_table_names())
    missing = set(Base.metadata.tables) - actual_tables
    if missing:
        names = "、".join(sorted(missing))
        raise RuntimeError(f"目标库尚未执行 Alembic，缺少表：{names}")
    if "alembic_version" not in actual_tables:
        raise RuntimeError("目标库缺少 alembic_version，请先执行 alembic upgrade head")


def _ensure_legacy_user(
    db: Session,
    options: MigrationOptions,
    now: datetime,
) -> None:
    """创建或复用本地基线用户，不覆盖已存在账号密码。"""
    user_id = _required_text(options.legacy_user_id, "legacy_user_id")
    username = _required_text(options.legacy_username, "legacy_username")
    existing = db.get(models.User, user_id)
    if existing is not None:
        if existing.username != username:
            raise ValueError("legacy_user_id 已绑定其他用户名，拒绝自动覆盖")
        return
    conflict = db.scalar(select(models.User).where(models.User.username == username))
    if conflict is not None:
        raise ValueError("legacy_username 已被其他用户占用，拒绝自动合并")
    password_hash = (
        hash_password(options.legacy_password)
        if options.legacy_password
        else hash_password(secrets.token_urlsafe(32))
    )
    db.add(
        models.User(
            user_id=user_id,
            username=username,
            password_hash=password_hash,
            role="user",
            status="active",
            created_at=now,
            updated_at=now,
        )
    )
    db.flush()


def _migrate_profile(
    db: Session,
    rows: list[dict[str, Any]],
    options: MigrationOptions,
    now: datetime,
    report: MigrationReport,
) -> list[tuple[str]]:
    """把实际存在的历史画像绑定到基线用户。"""
    if not rows:
        return []
    user_id = options.legacy_user_id
    chosen = rows[0]
    if len(rows) > 1:
        matching = [row for row in rows if row.get("user_id") == user_id]
        if len(matching) != 1:
            raise ValueError("旧库画像行数大于 1，无法安全绑定单一基线用户")
        chosen = matching[0]
    values = {
        "user_id": user_id,
        "display_name": _text(chosen.get("display_name")),
        "preferences": _json_dict(chosen.get("preferences")),
        "profile_version": int(chosen.get("profile_version") or 1),
        "created_at": normalize_utc(chosen.get("created_at"), fallback=now),
        "updated_at": normalize_utc(chosen.get("updated_at"), fallback=now),
    }
    _insert_or_skip(
        db,
        models.UserProfile,
        (user_id,),
        values,
        report,
        "user_profiles",
    )
    return [(user_id,)]


def _migrate_sessions(
    db: Session,
    rows: list[dict[str, Any]],
    options: MigrationOptions,
    now: datetime,
    report: MigrationReport,
) -> list[tuple[str]]:
    """把全部旧会话绑定到基线用户。"""
    keys: list[tuple[str]] = []
    for row in rows:
        session_id = _required_text(row.get("session_id"), "sessions.session_id")
        values = {
            "session_id": session_id,
            "user_id": options.legacy_user_id,
            "title": _text(row.get("title"), "新会话") or "新会话",
            "created_at": normalize_utc(row.get("created_at"), fallback=now),
            "updated_at": normalize_utc(row.get("updated_at"), fallback=now),
        }
        _insert_or_skip(
            db,
            models.ChatSession,
            (session_id,),
            values,
            report,
            "sessions",
            lambda item: item.user_id == options.legacy_user_id,
        )
        keys.append((session_id,))
    return keys


def _migrate_messages(
    db: Session,
    rows: list[dict[str, Any]],
    now: datetime,
    report: MigrationReport,
) -> list[tuple[int]]:
    """保留历史消息主键并迁移清理后的 JSON 元数据。"""
    keys: list[tuple[int]] = []
    for row in rows:
        message_id = int(row.get("message_id"))
        session_id = _required_text(row.get("session_id"), "messages.session_id")
        values = {
            "message_id": message_id,
            "session_id": session_id,
            "role": _required_text(row.get("role"), "messages.role"),
            "content": _text(row.get("content")),
            "message_metadata": _json_dict(row.get("metadata")),
            "created_at": normalize_utc(row.get("created_at"), fallback=now),
        }
        _insert_or_skip(
            db,
            models.Message,
            (message_id,),
            values,
            report,
            "messages",
            lambda item, session_id=session_id: item.session_id == session_id,
        )
        keys.append((message_id,))
    return keys


def _migrate_agent_runs(
    db: Session,
    rows: list[dict[str, Any]],
    now: datetime,
    report: MigrationReport,
) -> list[tuple[str]]:
    """迁移 Agent 问题、证据、报告和 Trace 标识。"""
    keys: list[tuple[str]] = []
    for row in rows:
        run_id = _required_text(row.get("run_id"), "agent_runs.run_id")
        session_id = _required_text(row.get("session_id"), "agent_runs.session_id")
        metadata = _json_dict(row.get("metadata"))
        values = {
            "run_id": run_id,
            "session_id": session_id,
            "agent_type": _text(row.get("agent_type")),
            "status": _text(row.get("status")),
            "question": _text(row.get("question")),
            "answer": _text(row.get("answer")),
            "trace_id": _text(row.get("trace_id") or metadata.get("trace_id")),
            "evidence_cases": _json_list(row.get("evidence_cases")),
            "evidence_chunks": _json_list(row.get("evidence_chunks")),
            "images": _json_list(row.get("images")),
            "reports": _json_list(row.get("reports")),
            "run_metadata": metadata,
            "created_at": normalize_utc(row.get("created_at"), fallback=now),
            "updated_at": normalize_utc(row.get("updated_at"), fallback=now),
        }
        _insert_or_skip(
            db,
            models.AgentRun,
            (run_id,),
            values,
            report,
            "agent_runs",
            lambda item, session_id=session_id: item.session_id == session_id,
        )
        keys.append((run_id,))
    return keys


def _migrate_agent_states(
    db: Session,
    rows: list[dict[str, Any]],
    now: datetime,
    report: MigrationReport,
) -> list[tuple[str, str]]:
    """迁移可恢复 Agent 状态和待确认条件。"""
    keys: list[tuple[str, str]] = []
    for row in rows:
        session_id = _required_text(
            row.get("session_id"),
            "session_agent_states.session_id",
        )
        agent_type = _required_text(
            row.get("agent_type"),
            "session_agent_states.agent_type",
        )
        values = {
            "session_id": session_id,
            "agent_type": agent_type,
            "state": _json_dict(row.get("state")),
            "updated_at": normalize_utc(row.get("updated_at"), fallback=now),
        }
        _insert_or_skip(
            db,
            models.SessionAgentState,
            (session_id, agent_type),
            values,
            report,
            "session_agent_states",
        )
        keys.append((session_id, agent_type))
    return keys


def _migrate_feedback(
    db: Session,
    rows: list[dict[str, Any]],
    options: MigrationOptions,
    now: datetime,
    report: MigrationReport,
) -> list[tuple[str, int, str]]:
    """迁移消息反馈并补写历史缺失的 user_id。"""
    keys: list[tuple[str, int, str]] = []
    for row in rows:
        session_id = _required_text(
            row.get("session_id"),
            "message_feedback.session_id",
        )
        message_id = int(row.get("message_id"))
        client_id = _required_text(
            row.get("client_id"),
            "message_feedback.client_id",
        )
        values = {
            "session_id": session_id,
            "message_id": message_id,
            "client_id": client_id,
            "user_id": options.legacy_user_id,
            "rating": _text(row.get("rating")),
            "reason": _text(row.get("reason")),
            "created_at": normalize_utc(row.get("created_at"), fallback=now),
            "updated_at": normalize_utc(row.get("updated_at"), fallback=now),
        }
        _insert_or_skip(
            db,
            models.MessageFeedback,
            (session_id, message_id, client_id),
            values,
            report,
            "message_feedback",
            lambda item: item.user_id == options.legacy_user_id,
        )
        keys.append((session_id, message_id, client_id))
    return keys


def _migrate_guidance(
    db: Session,
    rows: list[dict[str, Any]],
    options: MigrationOptions,
    now: datetime,
    report: MigrationReport,
) -> list[tuple[str]]:
    """迁移会话级临时反馈要求及其布尔和数组字段。"""
    keys: list[tuple[str]] = []
    for row in rows:
        guidance_id = _required_text(
            row.get("guidance_id"),
            "session_feedback_guidance.guidance_id",
        )
        session_id = _required_text(
            row.get("session_id"),
            "session_feedback_guidance.session_id",
        )
        values = {
            "guidance_id": guidance_id,
            "session_id": session_id,
            "source_message_id": int(row.get("source_message_id")),
            "client_id": _required_text(
                row.get("client_id"),
                "session_feedback_guidance.client_id",
            ),
            "user_id": options.legacy_user_id,
            "raw_feedback": _text(row.get("raw_feedback")),
            "status": _text(row.get("status")),
            "guidance_prompt": _text(row.get("guidance_prompt")),
            "guidance_type": _json_list(row.get("guidance_type")),
            "requested_dimensions": _json_list(row.get("requested_dimensions")),
            "needs_same_scope_retrieval": _boolean(
                row.get("needs_same_scope_retrieval")
            ),
            "scope_change": _boolean(row.get("scope_change")),
            "confidence": float(row.get("confidence") or 0),
            "error": _text(row.get("error")),
            "created_at": normalize_utc(row.get("created_at"), fallback=now),
            "updated_at": normalize_utc(row.get("updated_at"), fallback=now),
        }
        _insert_or_skip(
            db,
            models.SessionFeedbackGuidance,
            (guidance_id,),
            values,
            report,
            "session_feedback_guidance",
            lambda item: item.user_id == options.legacy_user_id,
        )
        keys.append((guidance_id,))
    return keys


def _migrate_issue_reports(
    db: Session,
    rows: list[dict[str, Any]],
    options: MigrationOptions,
    now: datetime,
    report: MigrationReport,
) -> list[tuple[str]]:
    """迁移问题上报并绑定基线用户。"""
    keys: list[tuple[str]] = []
    for row in rows:
        report_id = _required_text(
            row.get("report_id"),
            "message_issue_reports.report_id",
        )
        session_id = _required_text(
            row.get("session_id"),
            "message_issue_reports.session_id",
        )
        values = {
            "report_id": report_id,
            "session_id": session_id,
            "message_id": int(row.get("message_id")),
            "client_id": _required_text(
                row.get("client_id"),
                "message_issue_reports.client_id",
            ),
            "user_id": options.legacy_user_id,
            "category": _text(row.get("category")),
            "description": _text(row.get("description")),
            "created_at": normalize_utc(row.get("created_at"), fallback=now),
        }
        _insert_or_skip(
            db,
            models.MessageIssueReport,
            (report_id,),
            values,
            report,
            "message_issue_reports",
            lambda item: item.user_id == options.legacy_user_id,
        )
        keys.append((report_id,))
    return keys


def _migrate_shares(
    db: Session,
    rows: list[dict[str, Any]],
    options: MigrationOptions,
    now: datetime,
    report: MigrationReport,
) -> list[tuple[str]]:
    """保持分享 token 不变，并把空撤销时间转换为 NULL。"""
    keys: list[tuple[str]] = []
    for row in rows:
        token = _required_text(
            row.get("token"),
            "shared_message_snapshots.token",
        )
        session_id = _required_text(
            row.get("session_id"),
            "shared_message_snapshots.session_id",
        )
        values = {
            "token": token,
            "session_id": session_id,
            "message_id": int(row.get("message_id")),
            "owner_user_id": options.legacy_user_id,
            "title": _text(row.get("title")),
            "question": _text(row.get("question")),
            "answer": _text(row.get("answer")),
            "expires_at": optional_utc(row.get("expires_at")),
            "revoked_at": optional_utc(row.get("revoked_at")),
            "created_at": normalize_utc(row.get("created_at"), fallback=now),
        }
        _insert_or_skip(
            db,
            models.SharedMessageSnapshot,
            (token,),
            values,
            report,
            "shared_message_snapshots",
            lambda item: item.owner_user_id == options.legacy_user_id,
        )
        keys.append((token,))
    return keys


def _migrate_memories(
    db: Session,
    rows: list[dict[str, Any]],
    options: MigrationOptions,
    now: datetime,
    report: MigrationReport,
) -> list[tuple[str]]:
    """迁移长期记忆事实并统一绑定基线用户。"""
    keys: list[tuple[str]] = []
    for row in rows:
        memory_id = _required_text(row.get("memory_id"), "memory_items.memory_id")
        values = {
            "memory_id": memory_id,
            "user_id": options.legacy_user_id,
            "memory_type": _text(row.get("memory_type")),
            "content": _text(row.get("content")),
            "confidence": float(row.get("confidence") or 0),
            "sensitivity": _text(row.get("sensitivity"), "normal"),
            "status": _text(row.get("status"), "candidate"),
            "source_session_id": row.get("source_session_id") or None,
            "expires_at": optional_utc(row.get("expires_at")),
            "created_at": normalize_utc(row.get("created_at"), fallback=now),
            "updated_at": normalize_utc(row.get("updated_at"), fallback=now),
        }
        _insert_or_skip(
            db,
            models.MemoryItem,
            (memory_id,),
            values,
            report,
            "memory_items",
            lambda item: item.user_id == options.legacy_user_id,
        )
        keys.append((memory_id,))
    return keys


def _migrate_memory_summaries(
    db: Session,
    rows: list[dict[str, Any]],
    options: MigrationOptions,
    now: datetime,
    report: MigrationReport,
) -> list[tuple[str]]:
    """迁移会话记忆摘要和已覆盖的消息边界。"""
    keys: list[tuple[str]] = []
    for row in rows:
        summary_id = _required_text(
            row.get("summary_id"),
            "memory_summaries.summary_id",
        )
        session_id = _required_text(
            row.get("session_id"),
            "memory_summaries.session_id",
        )
        values = {
            "summary_id": summary_id,
            "user_id": options.legacy_user_id,
            "session_id": session_id,
            "summary": _text(row.get("summary")),
            "covered_message_id": int(row.get("covered_message_id") or 0),
            "version": int(row.get("version") or 1),
            "created_at": normalize_utc(row.get("created_at"), fallback=now),
            "updated_at": normalize_utc(row.get("updated_at"), fallback=now),
        }
        _insert_or_skip(
            db,
            models.MemorySummary,
            (summary_id,),
            values,
            report,
            "memory_summaries",
            lambda item: item.user_id == options.legacy_user_id,
        )
        keys.append((summary_id,))
    return keys


def _count_target_rows(
    db: Session,
    model: type,
    table: str,
    keys: list[tuple[Any, ...]],
) -> int:
    """按源主键统计目标库中实际存在的行数。"""
    if not keys:
        return 0
    columns = [getattr(model, column) for column in PRIMARY_KEYS[table]]
    if len(columns) == 1:
        condition = columns[0].in_([key[0] for key in keys])
    else:
        condition = tuple_(*columns).in_(keys)
    return int(db.scalar(select(func.count()).select_from(model).where(condition)) or 0)


def _reconcile(
    db: Session,
    report: MigrationReport,
    keys_by_table: dict[str, list[tuple[Any, ...]]],
) -> None:
    """提交前核对每张表由源主键覆盖的目标行数。"""
    model_by_table = {
        "sessions": models.ChatSession,
        "messages": models.Message,
        "agent_runs": models.AgentRun,
        "session_agent_states": models.SessionAgentState,
        "message_feedback": models.MessageFeedback,
        "session_feedback_guidance": models.SessionFeedbackGuidance,
        "message_issue_reports": models.MessageIssueReport,
        "shared_message_snapshots": models.SharedMessageSnapshot,
        "user_profiles": models.UserProfile,
        "memory_items": models.MemoryItem,
        "memory_summaries": models.MemorySummary,
    }
    db.flush()
    for table in TABLES:
        report.counts[table].target = _count_target_rows(
            db,
            model_by_table[table],
            table,
            keys_by_table[table],
        )
    mismatches = report.reconcile_mismatches()
    if mismatches:
        raise RuntimeError(f"迁移行数对账失败：{'、'.join(mismatches)}")


def _reset_postgres_message_sequence(db: Session) -> None:
    """显式导入消息 ID 后同步 PostgreSQL 自增序列。"""
    connection = db.connection()
    if connection.dialect.name != "postgresql":
        return
    sequence = connection.scalar(
        text("SELECT pg_get_serial_sequence('messages', 'message_id')")
    )
    if not sequence:
        return
    maximum = connection.scalar(text("SELECT COALESCE(MAX(message_id), 0) FROM messages"))
    if maximum:
        connection.execute(
            text("SELECT setval(:sequence, :value, true)"),
            {"sequence": sequence, "value": int(maximum)},
        )
    else:
        connection.execute(
            text("SELECT setval(:sequence, 1, false)"),
            {"sequence": sequence},
        )


def _prepare_report(
    snapshot: dict[str, list[dict[str, Any]]],
    *,
    dry_run: bool,
) -> MigrationReport:
    """在写入前固定每张表的源行数基线。"""
    return MigrationReport(
        dry_run=dry_run,
        counts={
            table: TableCount(source=len(snapshot[table]))
            for table in TABLES
        },
    )


def _apply_snapshot(
    db: Session,
    snapshot: dict[str, list[dict[str, Any]]],
    options: MigrationOptions,
    report: MigrationReport,
    now: datetime,
) -> None:
    """按外键依赖顺序写入快照并收集目标主键。"""
    _ensure_legacy_user(db, options, now)
    keys_by_table = {
        "user_profiles": [],
        "sessions": [],
        "messages": [],
        "agent_runs": [],
        "session_agent_states": [],
        "message_feedback": [],
        "session_feedback_guidance": [],
        "message_issue_reports": [],
        "shared_message_snapshots": [],
        "memory_items": [],
        "memory_summaries": [],
    }

    keys_by_table["user_profiles"] = _migrate_profile(
        db,
        snapshot["user_profiles"],
        options,
        now,
        report,
    )
    keys_by_table["sessions"] = _migrate_sessions(
        db,
        snapshot["sessions"],
        options,
        now,
        report,
    )
    keys_by_table["messages"] = _migrate_messages(
        db,
        snapshot["messages"],
        now,
        report,
    )
    keys_by_table["agent_runs"] = _migrate_agent_runs(
        db,
        snapshot["agent_runs"],
        now,
        report,
    )
    keys_by_table["session_agent_states"] = _migrate_agent_states(
        db,
        snapshot["session_agent_states"],
        now,
        report,
    )
    keys_by_table["message_feedback"] = _migrate_feedback(
        db,
        snapshot["message_feedback"],
        options,
        now,
        report,
    )
    keys_by_table["session_feedback_guidance"] = _migrate_guidance(
        db,
        snapshot["session_feedback_guidance"],
        options,
        now,
        report,
    )
    keys_by_table["message_issue_reports"] = _migrate_issue_reports(
        db,
        snapshot["message_issue_reports"],
        options,
        now,
        report,
    )
    keys_by_table["shared_message_snapshots"] = _migrate_shares(
        db,
        snapshot["shared_message_snapshots"],
        options,
        now,
        report,
    )
    keys_by_table["memory_items"] = _migrate_memories(
        db,
        snapshot["memory_items"],
        options,
        now,
        report,
    )
    keys_by_table["memory_summaries"] = _migrate_memory_summaries(
        db,
        snapshot["memory_summaries"],
        options,
        now,
        report,
    )
    _reconcile(db, report, keys_by_table)
    if not options.dry_run:
        _reset_postgres_message_sequence(db)


def run_migration(options: MigrationOptions) -> MigrationReport:
    """执行一次性数据迁移；dry-run 在完成对账后回滚目标事务。"""
    target_url = normalize_database_url(options.target_url)
    source = ReadOnlySqliteSource(options.source)
    engine: Engine | None = None
    try:
        snapshot = source.read_snapshot()
        report = _prepare_report(snapshot, dry_run=options.dry_run)
        engine = create_engine(target_url, pool_pre_ping=True)
        _validate_target_schema(engine)
        now = datetime.now(UTC)
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                with Session(connection, expire_on_commit=False) as db:
                    _apply_snapshot(db, snapshot, options, report, now)
                if options.dry_run:
                    transaction.rollback()
                else:
                    transaction.commit()
            except Exception:
                if transaction.is_active:
                    transaction.rollback()
                raise
        return report
    finally:
        source.close()
        if engine is not None:
            engine.dispose()


def _build_parser() -> argparse.ArgumentParser:
    """构造带安全默认值的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="把旧 SQLite 会话数据迁移到已执行 Alembic 的 PostgreSQL。"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/sessions.sqlite3"),
        help="旧 SQLite 文件路径，默认 data/sessions.sqlite3",
    )
    parser.add_argument(
        "--target",
        default=os.getenv("DATABASE_URL", ""),
        help="目标数据库 URL；未传入时读取 DATABASE_URL",
    )
    parser.add_argument("--legacy-user-id", default="legacy_user")
    parser.add_argument("--legacy-username", default="legacy_user")
    parser.add_argument(
        "--legacy-password",
        default=None,
        help="首次创建基线用户时使用的密码；不提供则生成不可知随机密码",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="完整执行后在目标事务内回滚，不留下迁移数据",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口，只输出计数，不输出密码、消息正文或数据库地址。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.target:
        parser.error("必须提供 --target 或 DATABASE_URL")
    options = MigrationOptions(
        source=args.source,
        target_url=args.target,
        legacy_user_id=args.legacy_user_id,
        legacy_username=args.legacy_username,
        legacy_password=args.legacy_password,
        dry_run=args.dry_run,
    )
    try:
        report = run_migration(options)
    except Exception as exc:
        print(f"迁移失败：{exc}")
        return 1
    suffix = "（dry-run，目标事务已回滚）" if report.dry_run else ""
    print(f"迁移完成{suffix}")
    for table in TABLES:
        count = report.counts[table]
        print(
            f"{table}: source={count.source} target={count.target} "
            f"inserted={count.inserted} skipped={count.skipped}"
        )
    if not options.legacy_password:
        print("提示：未提供 legacy 密码，新基线账号使用不可知随机密码。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
