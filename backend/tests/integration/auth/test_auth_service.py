"""本地账号注册、登录和刷新令牌轮换测试。"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.db.base import Base
from backend.app.db.repositories import AuthRepository
from backend.app.security.tokens import AccessTokenService
from backend.app.services.auth_service import AuthService, InvalidCredentials


class AuthServiceTests(unittest.TestCase):
    """验证认证服务必须遵守的账号和令牌生命周期。"""

    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.now = datetime(2026, 11, 6, 8, 0, tzinfo=timezone.utc)
        self.repository = AuthRepository(self.engine)
        self.token_service = AccessTokenService(
            secret="unit-test-secret-that-is-long-enough",
            ttl_seconds=900,
            clock=lambda: self.now,
        )
        self.service = AuthService(
            repository=self.repository,
            access_tokens=self.token_service,
            refresh_ttl=timedelta(days=7),
            clock=lambda: self.now,
        )

    def test_register_login_and_me_require_the_correct_password(self) -> None:
        """正确密码可登录；错误密码统一拒绝且不泄露账号状态。"""
        user = self.service.register("weather-user", "StrongPassword-2026")
        login = self.service.login("weather-user", "StrongPassword-2026")

        self.assertEqual(user["username"], "weather-user")
        self.assertEqual(self.service.get_current_user(login.access_token)["user_id"], user["user_id"])
        with self.assertRaises(InvalidCredentials):
            self.service.login("weather-user", "wrong-password")
        with self.assertRaises(InvalidCredentials):
            self.service.login("missing-user", "wrong-password")

    def test_refresh_token_rotates_and_reuse_revokes_the_token_family(self) -> None:
        """旧刷新令牌被再次使用后，整个令牌族都不能继续刷新。"""
        user = self.service.register("rotate-user", "StrongPassword-2026")
        login = self.service.login("rotate-user", "StrongPassword-2026")
        rotated = self.service.refresh(login.refresh_token)

        self.assertNotEqual(rotated.refresh_token, login.refresh_token)
        with self.assertRaises(InvalidCredentials):
            self.service.refresh(login.refresh_token)
        with self.assertRaises(InvalidCredentials):
            self.service.refresh(rotated.refresh_token)
        self.assertEqual(self.service.get_current_user(rotated.access_token)["user_id"], user["user_id"])

    def test_logout_revokes_refresh_token(self) -> None:
        """退出后当前刷新令牌不能继续换取访问令牌。"""
        self.service.register("logout-user", "StrongPassword-2026")
        login = self.service.login("logout-user", "StrongPassword-2026")

        self.service.logout(login.refresh_token)

        with self.assertRaises(InvalidCredentials):
            self.service.refresh(login.refresh_token)

    def test_expired_refresh_token_is_rejected(self) -> None:
        """刷新令牌超过有效期后必须拒绝。"""
        self.service.register("expired-user", "StrongPassword-2026")
        login = self.service.login("expired-user", "StrongPassword-2026")
        later = AuthService(
            repository=self.repository,
            access_tokens=self.token_service,
            refresh_ttl=timedelta(days=7),
            clock=lambda: self.now + timedelta(days=8),
        )

        with self.assertRaises(InvalidCredentials):
            later.refresh(login.refresh_token)


if __name__ == "__main__":
    unittest.main()
