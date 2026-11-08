"""密码哈希的安全属性测试。"""
from __future__ import annotations

import unittest

from backend.app.security.passwords import hash_password, verify_password


class PasswordHasherTests(unittest.TestCase):
    """验证密码使用 Argon2id 且不会接受错误明文。"""

    def test_password_hash_uses_argon2id_and_verifies_original_password(self) -> None:
        """生产代码若改用弱哈希或错误比较逻辑，本测试必须失败。"""
        password_hash = hash_password("Correct-Horse-Battery-Staple")

        self.assertTrue(password_hash.startswith("$argon2id$"))
        self.assertTrue(verify_password(password_hash, "Correct-Horse-Battery-Staple"))
        self.assertFalse(verify_password(password_hash, "wrong-password"))


if __name__ == "__main__":
    unittest.main()
