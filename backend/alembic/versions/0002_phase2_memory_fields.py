"""为长期记忆增加来源消息、版本、元数据与软删除字段。"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0002_phase2_memory_fields"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def _json_type() -> sa.types.TypeEngine:
    """为 PostgreSQL 使用 JSONB，为测试 SQLite 使用标准 JSON。"""
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    """为 memory_items 增加可审计、可删除的长期记忆字段。"""
    op.add_column(
        "memory_items",
        sa.Column("source_message_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "memory_items",
        sa.Column(
            "version",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
    )
    op.add_column(
        "memory_items",
        sa.Column(
            "metadata",
            _json_type(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb")
            if op.get_bind().dialect.name == "postgresql"
            else sa.text("'{}'"),
        ),
    )
    op.add_column(
        "memory_items",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    """回滚阶段二长期记忆字段。"""
    op.drop_column("memory_items", "deleted_at")
    op.drop_column("memory_items", "metadata")
    op.drop_column("memory_items", "version")
    op.drop_column("memory_items", "source_message_id")
