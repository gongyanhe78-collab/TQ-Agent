"""数据库迁移：${message}"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}


revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    """应用本次结构变更。"""
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    """回退本次结构变更。"""
    ${downgrades if downgrades else "pass"}
