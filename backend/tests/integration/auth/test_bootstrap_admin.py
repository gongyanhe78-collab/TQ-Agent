"""管理员初始化脚本的业务规则测试。"""
from __future__ import annotations

import unittest
from datetime import timedelta

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from backend.app.db.base import Base
from backend.app.db.repositories import AuthRepository
from backend.app.security.tokens import AccessTokenService
from backend.app.services.admin_bootstrap import (
    AdminBootstrapConfigurationError,
    AdminBootstrapConflict,
    bootstrap_admin,
    bootstrap_admin_from_environment,
)
from backend.app.services.auth_service import AuthService, InvalidCredentials


class AdminBootstrapTests(unittest.TestCase):
    """验证管理员初始化可重复执行，且权限只能通过显式创建获得。"""

    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.repository = AuthRepository(self.engine)
        self.service = AuthService(
            repository=self.repository,
            access_tokens=AccessTokenService(
                secret="admin-bootstrap-test-secret-at-least-32",
                ttl_seconds=900,
            ),
            refresh_ttl=timedelta(days=7),
        )

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_missing_admin_is_created_once_and_existing_password_is_kept(self) -> None:
        """首次创建管理员，重复执行时复用原账号且不覆盖原密码。"""
        created = bootstrap_admin(
            self.service,
            "bootstrap-admin",
            "InitialPassword-2026",
        )
        repeated = bootstrap_admin(
            self.service,
            "bootstrap-admin",
            "ReplacementPassword-2026",
        )

        self.assertEqual(created["action"], "created")
        self.assertEqual(repeated["action"], "existing")
        self.assertEqual(created["user"]["user_id"], repeated["user"]["user_id"])
        self.assertEqual(repeated["user"]["role"], "admin")
        self.assertEqual(
            self.service.login(
                "bootstrap-admin",
                "InitialPassword-2026",
            ).user["role"],
            "admin",
        )
        with self.assertRaises(InvalidCredentials):
            self.service.login(
                "bootstrap-admin",
                "ReplacementPassword-2026",
            )

    def test_existing_regular_user_is_never_promoted(self) -> None:
        """同名普通用户已存在时初始化必须失败，并保留原角色和密码。"""
        existing = self.service.register(
            "bootstrap-admin",
            "UserPassword-2026",
        )

        with self.assertRaises(AdminBootstrapConflict):
            bootstrap_admin(
                self.service,
                "bootstrap-admin",
                "AdminPassword-2026",
            )

        current = self.repository.get_user_by_username("bootstrap-admin")
        self.assertEqual(current["user_id"], existing["user_id"])
        self.assertEqual(current["role"], "user")
        self.assertEqual(
            self.service.login(
                "bootstrap-admin",
                "UserPassword-2026",
            ).user["role"],
            "user",
        )
        with self.assertRaises(InvalidCredentials):
            self.service.login(
                "bootstrap-admin",
                "AdminPassword-2026",
            )

    def test_environment_requires_both_credentials_or_skips_without_changes(self) -> None:
        """环境变量未配置时跳过；只配置一项时明确拒绝启动。"""
        skipped = bootstrap_admin_from_environment(self.service, environ={})
        self.assertEqual(skipped["action"], "skipped")
        self.assertIsNone(self.repository.get_user_by_username("bootstrap-admin"))

        with self.assertRaises(AdminBootstrapConfigurationError):
            bootstrap_admin_from_environment(
                self.service,
                environ={
                    "BOOTSTRAP_ADMIN_USERNAME": "bootstrap-admin",
                },
            )
        with self.assertRaises(AdminBootstrapConfigurationError):
            bootstrap_admin_from_environment(
                self.service,
                environ={
                    "BOOTSTRAP_ADMIN_PASSWORD": "AdminPassword-2026",
                },
            )


if __name__ == "__main__":
    unittest.main()
