"""访问令牌签发、校验和过期行为测试。"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from backend.app.security.tokens import AccessTokenService, InvalidAccessToken


class AccessTokenTests(unittest.TestCase):
    """验证 JWT 声明、签名和过期边界。"""

    def setUp(self) -> None:
        self.now = datetime(2026, 11, 6, 8, 0, tzinfo=timezone.utc)
        self.service = AccessTokenService(
            secret="unit-test-secret-that-is-long-enough",
            ttl_seconds=900,
            clock=lambda: self.now,
        )

    def test_issue_access_token_contains_required_identity_claims(self) -> None:
        """缺少用户、角色、jti、签发或过期声明都会破坏鉴权边界。"""
        token = self.service.issue("user-a", "admin")
        claims = self.service.decode(token)

        self.assertEqual(claims["sub"], "user-a")
        self.assertEqual(claims["role"], "admin")
        self.assertEqual(claims["iat"], int(self.now.timestamp()))
        self.assertEqual(claims["exp"], int((self.now + timedelta(seconds=900)).timestamp()))
        self.assertTrue(claims["jti"])

    def test_expired_or_tampered_token_is_rejected(self) -> None:
        """过期令牌或错误签名不得被继续接受。"""
        token = self.service.issue("user-a", "user")
        later = AccessTokenService(
            secret="unit-test-secret-that-is-long-enough",
            ttl_seconds=900,
            clock=lambda: self.now + timedelta(seconds=901),
        )

        with self.assertRaises(InvalidAccessToken):
            later.decode(token)
        with self.assertRaises(InvalidAccessToken):
            self.service.decode(f"{token}tampered")


if __name__ == "__main__":
    unittest.main()
