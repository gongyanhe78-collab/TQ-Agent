"""根据环境变量幂等创建首个管理员的命令行入口。"""
from __future__ import annotations

import logging
import os
import sys
from datetime import timedelta

from backend.app.core.config import get_settings
from backend.app.db.repositories import AuthRepository
from backend.app.db.session import build_engine
from backend.app.security.tokens import AccessTokenService
from backend.app.services.admin_bootstrap import bootstrap_admin_from_environment
from backend.app.services.auth_service import AuthService


LOGGER = logging.getLogger("risk-admin-bootstrap")


def main() -> int:
    """构造最小认证依赖并执行管理员初始化。"""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = get_settings()
    engine = build_engine(settings)
    try:
        auth_service = AuthService(
            repository=AuthRepository(engine),
            access_tokens=AccessTokenService(
                secret=settings.auth_jwt_secret,
                ttl_seconds=settings.auth_access_ttl_seconds,
            ),
            refresh_ttl=timedelta(seconds=settings.auth_refresh_ttl_seconds),
        )
        bootstrap_admin_from_environment(auth_service, environ=os.environ)
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOGGER.error("管理员初始化失败：%s", exc)
        raise SystemExit(1) from exc
