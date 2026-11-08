"""创建第一阶段用户、会话、记忆、反馈与审计表。"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def _json_type() -> sa.types.TypeEngine:
    """为 PostgreSQL 使用 JSONB，为测试 SQLite 使用标准 JSON。"""
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def _timestamps() -> list[sa.Column]:
    """返回创建时间和更新时间列定义。"""
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    ]


def upgrade() -> None:
    """创建全部权威业务表和访问路径所需索引。"""
    op.create_table(
        "users",
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("user_id"),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_role", "users", ["role"], unique=False)
    op.create_index("ix_users_status", "users", ["status"], unique=False)

    op.create_table(
        "refresh_tokens",
        sa.Column("token_id", sa.String(length=36), nullable=False),
        sa.Column("family_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.user_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("token_id"),
        sa.UniqueConstraint("token_hash", name="uq_refresh_tokens_token_hash"),
    )
    op.create_index(
        "ix_refresh_tokens_expires_at",
        "refresh_tokens",
        ["expires_at"],
        unique=False,
    )
    op.create_index(
        "ix_refresh_tokens_family_id",
        "refresh_tokens",
        ["family_id"],
        unique=False,
    )
    op.create_index(
        "ix_refresh_tokens_user_id",
        "refresh_tokens",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "user_profiles",
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("display_name", sa.String(length=120), nullable=False),
        sa.Column("preferences", _json_type(), nullable=False),
        sa.Column("profile_version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.user_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("user_id"),
    )

    op.create_table(
        "sessions",
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.user_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_index(
        "idx_sessions_user_updated",
        "sessions",
        ["user_id", "updated_at"],
        unique=False,
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"], unique=False)

    op.create_table(
        "messages",
        sa.Column("message_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", _json_type(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("message_id"),
    )
    op.create_index(
        "idx_messages_session_id",
        "messages",
        ["session_id", "message_id"],
        unique=False,
    )
    op.create_index("ix_messages_session_id", "messages", ["session_id"], unique=False)

    op.create_table(
        "agent_runs",
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("agent_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("evidence_cases", _json_type(), nullable=False),
        sa.Column("evidence_chunks", _json_type(), nullable=False),
        sa.Column("images", _json_type(), nullable=False),
        sa.Column("reports", _json_type(), nullable=False),
        sa.Column("metadata", _json_type(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index(
        "idx_agent_runs_session_created",
        "agent_runs",
        ["session_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "idx_agent_runs_trace",
        "agent_runs",
        ["trace_id"],
        unique=False,
    )
    op.create_index(
        "ix_agent_runs_session_id",
        "agent_runs",
        ["session_id"],
        unique=False,
    )

    op.create_table(
        "session_agent_states",
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("agent_type", sa.String(length=64), nullable=False),
        sa.Column("state", _json_type(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("session_id", "agent_type"),
    )

    op.create_table(
        "memory_items",
        sa.Column("memory_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("memory_type", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("sensitivity", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("source_session_id", sa.String(length=36), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.user_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("memory_id"),
    )
    op.create_index(
        "idx_memory_items_user_status",
        "memory_items",
        ["user_id", "status"],
        unique=False,
    )
    op.create_index(
        "ix_memory_items_user_id",
        "memory_items",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "memory_summaries",
        sa.Column("summary_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("covered_message_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.user_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("summary_id"),
    )
    op.create_index(
        "ix_memory_summaries_session_id",
        "memory_summaries",
        ["session_id"],
        unique=False,
    )
    op.create_index(
        "ix_memory_summaries_user_id",
        "memory_summaries",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "message_feedback",
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("client_id", sa.String(length=120), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("rating", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.user_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("session_id", "message_id", "client_id"),
    )
    op.create_index(
        "ix_message_feedback_user_id",
        "message_feedback",
        ["user_id"],
        unique=False,
    )
    # PostgreSQL 会合并建表语句中与主键同列的唯一约束，因此单独添加。
    with op.batch_alter_table("message_feedback") as batch_op:
        batch_op.create_unique_constraint(
            "uq_message_feedback_client",
            ["session_id", "message_id", "client_id"],
        )

    op.create_table(
        "session_feedback_guidance",
        sa.Column("guidance_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("source_message_id", sa.Integer(), nullable=False),
        sa.Column("client_id", sa.String(length=120), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("raw_feedback", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("guidance_prompt", sa.Text(), nullable=False),
        sa.Column("guidance_type", _json_type(), nullable=False),
        sa.Column("requested_dimensions", _json_type(), nullable=False),
        sa.Column("needs_same_scope_retrieval", sa.Boolean(), nullable=False),
        sa.Column("scope_change", sa.Boolean(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("error", sa.Text(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.user_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("guidance_id"),
        sa.UniqueConstraint(
            "session_id",
            "source_message_id",
            "client_id",
            name="uq_feedback_guidance_client",
        ),
    )
    op.create_index(
        "ix_session_feedback_guidance_session_id",
        "session_feedback_guidance",
        ["session_id"],
        unique=False,
    )
    op.create_index(
        "ix_session_feedback_guidance_user_id",
        "session_feedback_guidance",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "message_issue_reports",
        sa.Column("report_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("client_id", sa.String(length=120), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("category", sa.String(length=80), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.user_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("report_id"),
    )
    op.create_index(
        "ix_message_issue_reports_session_id",
        "message_issue_reports",
        ["session_id"],
        unique=False,
    )
    op.create_index(
        "ix_message_issue_reports_user_id",
        "message_issue_reports",
        ["user_id"],
        unique=False,
    )

    op.create_table(
        "shared_message_snapshots",
        sa.Column("token", sa.String(length=96), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("owner_user_id", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["owner_user_id"],
            ["users.user_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("token"),
    )
    op.create_index(
        "ix_shared_message_snapshots_owner_user_id",
        "shared_message_snapshots",
        ["owner_user_id"],
        unique=False,
    )
    op.create_index(
        "ix_shared_message_snapshots_session_id",
        "shared_message_snapshots",
        ["session_id"],
        unique=False,
    )

    op.create_table(
        "audit_logs",
        sa.Column("audit_id", sa.String(length=64), nullable=False),
        sa.Column("actor_user_id", sa.String(length=36), nullable=False),
        sa.Column("action", sa.String(length=80), nullable=False),
        sa.Column("target_type", sa.String(length=40), nullable=False),
        sa.Column("target_id", sa.String(length=120), nullable=False),
        sa.Column("detail", _json_type(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("audit_id"),
    )
    op.create_index(
        "ix_audit_logs_actor_user_id",
        "audit_logs",
        ["actor_user_id"],
        unique=False,
    )


def downgrade() -> None:
    """按依赖逆序删除第一阶段全部表。"""
    op.drop_index("ix_audit_logs_actor_user_id", table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_index(
        "ix_shared_message_snapshots_session_id",
        table_name="shared_message_snapshots",
    )
    op.drop_index(
        "ix_shared_message_snapshots_owner_user_id",
        table_name="shared_message_snapshots",
    )
    op.drop_table("shared_message_snapshots")
    op.drop_index(
        "ix_message_issue_reports_user_id",
        table_name="message_issue_reports",
    )
    op.drop_index(
        "ix_message_issue_reports_session_id",
        table_name="message_issue_reports",
    )
    op.drop_table("message_issue_reports")
    op.drop_index(
        "ix_session_feedback_guidance_user_id",
        table_name="session_feedback_guidance",
    )
    op.drop_index(
        "ix_session_feedback_guidance_session_id",
        table_name="session_feedback_guidance",
    )
    op.drop_table("session_feedback_guidance")
    op.drop_index(
        "ix_message_feedback_user_id",
        table_name="message_feedback",
    )
    op.drop_table("message_feedback")
    op.drop_index(
        "ix_memory_summaries_user_id",
        table_name="memory_summaries",
    )
    op.drop_index(
        "ix_memory_summaries_session_id",
        table_name="memory_summaries",
    )
    op.drop_table("memory_summaries")
    op.drop_index("ix_memory_items_user_id", table_name="memory_items")
    op.drop_index("idx_memory_items_user_status", table_name="memory_items")
    op.drop_table("memory_items")
    op.drop_table("session_agent_states")
    op.drop_index("ix_agent_runs_session_id", table_name="agent_runs")
    op.drop_index("idx_agent_runs_trace", table_name="agent_runs")
    op.drop_index("idx_agent_runs_session_created", table_name="agent_runs")
    op.drop_table("agent_runs")
    op.drop_index("ix_messages_session_id", table_name="messages")
    op.drop_index("idx_messages_session_id", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_sessions_user_id", table_name="sessions")
    op.drop_index("idx_sessions_user_updated", table_name="sessions")
    op.drop_table("sessions")
    op.drop_table("user_profiles")
    op.drop_index("ix_refresh_tokens_user_id", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_family_id", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_expires_at", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")
    op.drop_index("ix_users_status", table_name="users")
    op.drop_index("ix_users_role", table_name="users")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
