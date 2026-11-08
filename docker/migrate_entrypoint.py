#!/usr/bin/env python3
"""在一次性迁移容器中执行 Alembic 和可选的 SQLite 数据迁移。"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path


LOGGER = logging.getLogger("risk-migrate")
BASE_DIR = Path(__file__).resolve().parents[1]
SENSITIVE_OPTIONS = {"--target", "--legacy-password"}


def _find_alembic_config() -> Path | None:
    """查找项目根目录或 backend 目录下的 Alembic 配置。"""
    for candidate in (BASE_DIR / "alembic.ini", BASE_DIR / "backend" / "alembic.ini"):
        if candidate.is_file():
            return candidate
    return None


def _masked_command(command: list[str]) -> str:
    """生成脱敏后的命令文本，避免日志泄露数据库密码。"""
    masked: list[str] = []
    hide_next = False
    for argument in command:
        if hide_next:
            masked.append("***")
            hide_next = False
            continue
        option_name = argument.split("=", 1)[0]
        if option_name in SENSITIVE_OPTIONS:
            if "=" in argument:
                masked.append(f"{option_name}=***")
            else:
                masked.append(argument)
                hide_next = True
            continue
        masked.append(argument)
    return " ".join(masked)


def _run(command: list[str], *, cwd: Path) -> None:
    """同步执行子进程并把失败状态传递给 Compose。"""
    LOGGER.info("执行命令：%s", _masked_command(command))
    process_env = os.environ.copy()
    process_env.setdefault("PYTHONUTF8", "1")
    subprocess.run(command, cwd=cwd, env=process_env, check=True)


def _run_alembic() -> None:
    """存在 Alembic 配置时升级到最新版本，否则安全跳过。"""
    config_path = _find_alembic_config()
    if config_path is None:
        LOGGER.warning("未找到 Alembic 配置，当前阶段跳过数据库结构升级。")
        return
    _run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=config_path.parent,
    )


def _run_legacy_migration() -> None:
    """仅在显式开启时执行 SQLite 到 PostgreSQL 的一次性迁移。"""
    enabled = os.getenv("RUN_LEGACY_MIGRATION", "false").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        LOGGER.info("未开启 SQLite 数据迁移，跳过该步骤。")
        return

    candidates = (
        BASE_DIR / "scripts" / "migrate_sqlite_to_postgres.py",
        BASE_DIR / "backend" / "scripts" / "migrate_sqlite_to_postgres.py",
    )
    script_path = next((path for path in candidates if path.is_file()), None)
    if script_path is None:
        raise FileNotFoundError("已开启 SQLite 数据迁移，但迁移脚本尚未落盘。")

    source_path = Path(
        os.getenv("LEGACY_SQLITE_PATH", "/app/data/sessions.sqlite3")
    ).expanduser()
    target_url = os.getenv("DATABASE_URL", "").strip()
    if not source_path.is_file():
        raise FileNotFoundError(f"SQLite 源库不存在：{source_path}")
    if not target_url:
        raise ValueError("SQLite 迁移需要 DATABASE_URL。")

    command = [
        sys.executable,
        str(script_path),
        "--source",
        str(source_path),
        "--target",
        target_url,
    ]
    if os.getenv("LEGACY_MIGRATION_DRY_RUN", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        command.append("--dry-run")
    _run(command, cwd=BASE_DIR)


def _run_bootstrap_admin() -> None:
    """在数据库结构就绪后按环境变量幂等创建首个管理员。"""
    candidates = (
        BASE_DIR / "scripts" / "bootstrap_admin.py",
        BASE_DIR / "backend" / "scripts" / "bootstrap_admin.py",
    )
    script_path = next((path for path in candidates if path.is_file()), None)
    if script_path is None:
        raise FileNotFoundError("未找到管理员初始化脚本。")
    _run([sys.executable, str(script_path)], cwd=BASE_DIR)


def main() -> int:
    """按结构升级、历史数据迁移和管理员初始化的顺序执行启动前任务。"""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _run_alembic()
    _run_legacy_migration()
    _run_bootstrap_admin()
    LOGGER.info("数据库启动前任务执行完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
