"""管理员账号的安全初始化与幂等规则。"""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from backend.app.services.auth_service import AuthService


LOGGER = logging.getLogger("risk-admin-bootstrap")
USERNAME_ENV = "BOOTSTRAP_ADMIN_USERNAME"
PASSWORD_ENV = "BOOTSTRAP_ADMIN_PASSWORD"


class AdminBootstrapConflict(RuntimeError):
    """同名普通用户已存在，禁止通过初始化流程自动提权。"""


class AdminBootstrapConfigurationError(RuntimeError):
    """管理员用户名和密码必须同时配置或同时留空。"""


def bootstrap_admin(
    auth_service: AuthService,
    username: str,
    password: str,
) -> dict[str, Any]:
    """创建管理员；账号已存在时保持幂等且不覆盖原密码。"""
    normalized = str(username or "").strip()
    existing = auth_service.repository.get_user_by_username(normalized)
    if existing is not None:
        if existing.get("role") != "admin":
            raise AdminBootstrapConflict(
                "同名账号已存在但不是管理员，拒绝自动提权。"
            )
        safe_user = dict(existing)
        safe_user.pop("password_hash", None)
        LOGGER.info("管理员账号已存在，跳过创建 username=%s", normalized)
        return {"action": "existing", "user": safe_user}

    user = auth_service.register(normalized, password, role="admin")
    LOGGER.info("管理员账号创建成功 username=%s", normalized)
    return {"action": "created", "user": user}


def bootstrap_admin_from_environment(
    auth_service: AuthService,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """从环境变量读取管理员凭据；未配置时安全跳过。"""
    values = environ if environ is not None else os.environ
    username = str(values.get(USERNAME_ENV, "") or "").strip()
    password = str(values.get(PASSWORD_ENV, "") or "")
    if not username and not password:
        LOGGER.info("未配置管理员初始化凭据，跳过该步骤。")
        return {"action": "skipped", "user": None}
    if not username or not password:
        raise AdminBootstrapConfigurationError(
            "管理员初始化用户名和密码必须同时配置。"
        )
    return bootstrap_admin(auth_service, username, password)
