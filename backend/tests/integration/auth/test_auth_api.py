"""本地认证 HTTP 接口与刷新 Cookie 行为测试。"""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from backend.app.api.routes import auth
from backend.app.db.base import Base
from backend.app.db.repositories import AuthRepository
from backend.app.security.tokens import AccessTokenService
from backend.app.services.auth_service import AuthService


class AuthApiTests(unittest.TestCase):
    """验证登录、刷新、登出和当前用户接口的完整浏览器契约。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.now = datetime(2026, 11, 6, 8, 0, tzinfo=timezone.utc)
        self.token_service = AccessTokenService(
            secret="auth-api-test-secret-that-is-long-enough",
            ttl_seconds=900,
            clock=lambda: self.now,
        )
        self.auth_service = AuthService(
            repository=AuthRepository(self.engine),
            access_tokens=self.token_service,
            refresh_ttl=timedelta(days=7),
            clock=lambda: self.now,
        )
        self.auth_service.register("api-user", "StrongPassword-2026")
        self.services = SimpleNamespace(
            auth_service=self.auth_service,
            settings=SimpleNamespace(
                auth_cookie_secure=False,
                auth_cookie_samesite="lax",
                auth_refresh_cookie_name="refresh_token",
            ),
        )
        app = FastAPI()
        app.state.services = self.services
        app.include_router(auth.router)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        self.engine.dispose()

    def _login(self) -> dict:
        """执行登录并返回解析后的响应。"""
        response = self.client.post(
            "/api/auth/login",
            json={"username": "api-user", "password": "StrongPassword-2026"},
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_login_me_refresh_and_logout_use_httponly_cookie(self) -> None:
        """访问令牌只返回正文，刷新令牌必须保存在 HttpOnly Cookie。"""
        login = self._login()
        cookie = self.client.cookies.get("refresh_token")
        self.assertTrue(cookie)
        self.assertNotEqual(cookie, login["access_token"])

        me = self.client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {login['access_token']}"},
        )
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["username"], "api-user")

        refreshed = self.client.post("/api/auth/refresh")
        self.assertEqual(refreshed.status_code, 200)
        self.assertTrue(refreshed.json()["access_token"])
        self.assertNotEqual(self.client.cookies.get("refresh_token"), cookie)

        logged_out = self.client.post("/api/auth/logout")
        self.assertEqual(logged_out.status_code, 200)
        self.assertFalse(self.client.cookies.get("refresh_token"))
        self.assertEqual(self.client.post("/api/auth/refresh").status_code, 401)

    def test_wrong_password_and_expired_access_token_return_401(self) -> None:
        """错误密码和过期访问令牌使用统一 401 错误。"""
        wrong = self.client.post(
            "/api/auth/login",
            json={"username": "api-user", "password": "wrong-password"},
        )
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(wrong.json()["detail"], "用户名或密码错误")

        expired_service = AccessTokenService(
            secret="auth-api-test-secret-that-is-long-enough",
            ttl_seconds=900,
            clock=lambda: self.now - timedelta(hours=1),
        )
        expired_token = expired_service.issue(
            self.auth_service.repository.get_user_by_username("api-user")["user_id"],
            "user",
        )
        response = self.client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        self.assertEqual(response.status_code, 401)

    def test_reused_refresh_token_revokes_the_family(self) -> None:
        """旧刷新令牌复用后，新令牌也必须失效。"""
        self._login()
        old_cookie = self.client.cookies.get("refresh_token")
        first_refresh = self.client.post("/api/auth/refresh")
        self.assertEqual(first_refresh.status_code, 200)
        current_cookie = self.client.cookies.get("refresh_token")

        self.client.cookies.set("refresh_token", old_cookie)
        reused = self.client.post("/api/auth/refresh")
        self.assertEqual(reused.status_code, 401)

        self.client.cookies.set("refresh_token", current_cookie)
        revoked = self.client.post("/api/auth/refresh")
        self.assertEqual(revoked.status_code, 401)

    def test_cookie_flags_are_explicit(self) -> None:
        """刷新 Cookie 必须带 HttpOnly、SameSite 和受控 Path。"""
        response = self.client.post(
            "/api/auth/login",
            json={"username": "api-user", "password": "StrongPassword-2026"},
        )
        cookie_header = response.headers.get("set-cookie", "").lower()
        self.assertIn("httponly", cookie_header)
        self.assertIn("samesite=lax", cookie_header)
        self.assertIn("path=/api/auth", cookie_header)


if __name__ == "__main__":
    unittest.main()
