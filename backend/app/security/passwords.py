"""使用 Argon2id 处理本地账号密码。"""
from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError


_PASSWORD_HASHER = PasswordHasher()


def hash_password(password: str) -> str:
    """生成带随机盐的 Argon2id 密码哈希。"""
    if not isinstance(password, str) or not password:
        raise ValueError("密码不能为空")
    return _PASSWORD_HASHER.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """校验明文密码；损坏哈希或错误密码统一返回 False。"""
    if not password_hash or not isinstance(password, str):
        return False
    try:
        return _PASSWORD_HASHER.verify(password_hash, password)
    except (InvalidHashError, VerificationError, TypeError):
        return False
