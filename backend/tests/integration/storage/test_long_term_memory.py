"""长期记忆仓储的用户隔离、软删除与重建来源集成测试。"""
from __future__ import annotations

import importlib

import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.db.base import Base
from backend.app.db.models import User


def _load_module(module_name: str):
    """导入待实现模块，并把缺失实现明确报告为测试失败。"""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise AssertionError(f"待实现模块缺失：{module_name}：{exc}") from exc


class LongTermMemoryRepositoryTests(unittest.TestCase):
    """记忆仓储必须以 user_id 作为所有读删操作的范围边界。"""

    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        with Session(self.engine) as session:
            for user_id in ("user-a", "user-b"):
                session.add(
                    User(
                        user_id=user_id,
                        username=user_id,
                        password_hash="not-used",
                        role="user",
                        status="active",
                    )
                )
            session.commit()
        self.repository = _load_module(
            "backend.app.repositories.memory_repository"
        ).MemoryRepository(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_user_b_cannot_read_or_delete_user_a_memory(self) -> None:
        """用户 B 不能读取或删除用户 A 的记忆，A 的记忆保持 active。"""
        memory = self.repository.create(
            user_id="user-a",
            content="偏好关注暴雨",
        )
        assert self.repository.get(user_id="user-b", memory_id=memory["memory_id"]) is None
        assert (
            self.repository.soft_delete(user_id="user-b", memory_id=memory["memory_id"])
            is False
        )
        assert (
            self.repository.get(user_id="user-a", memory_id=memory["memory_id"])["status"]
            == "active"
        )

    def test_soft_delete_keeps_audit_row_and_excludes_from_active(self) -> None:
        """软删除保留最小审计信息，并从 active 列表和重建来源中排除。"""
        memory = self.repository.create(
            user_id="user-a",
            content="关注山西北部冰雹",
            status="active",
        )
        memory_id = memory["memory_id"]

        deleted = self.repository.soft_delete(user_id="user-a", memory_id=memory_id)
        assert deleted is True

        row = self.repository.get(user_id="user-a", memory_id=memory_id)
        assert row["status"] == "deleted"
        assert row["deleted_at"] is not None
        assert all(
            item["memory_id"] != memory_id
            for item in self.repository.list_active("user-a")
        )
        assert all(
            item["memory_id"] != memory_id
            for item in self.repository.rebuildable_active("user-a")
        )

    def test_list_active_and_rebuildable_active_only_return_active(self) -> None:
        """active 列表只返回未删除、未过期且处于 active 状态的记忆。"""
        self.repository.create(user_id="user-a", content="偏好暴雨", status="candidate")
        active = self.repository.create(
            user_id="user-a",
            content="关注区域：大同",
            status="active",
        )
        ids = [item["memory_id"] for item in self.repository.list_active("user-a")]
        assert active["memory_id"] in ids
        rebuild_ids = [
            item["memory_id"]
            for item in self.repository.rebuildable_active("user-a")
        ]
        assert active["memory_id"] in rebuild_ids
        assert len(rebuild_ids) == 1


if __name__ == "__main__":
    unittest.main()
