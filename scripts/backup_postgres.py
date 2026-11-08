"""PostgreSQL 备份脚本：pg_dump 到 backups/ 目录并记录行数校验。"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_backup_counts import (
    build_count_runner,
    parse_database_url,
    verify_counts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKUP_DIR = PROJECT_ROOT / "backups"


def _pg_dump_command(
    parsed: dict[str, str],
    *,
    container: bool,
) -> list[str]:
    """构造 pg_dump 命令行，凭据来自解析后的数据库 URL。"""
    if container:
        return [
            "docker",
            "compose",
            "exec",
            "-T",
            "postgres",
            "pg_dump",
            "-U",
            parsed["user"],
            "-d",
            parsed["database"],
            "--format=plain",
            "--no-owner",
            "--no-privileges",
        ]
    return [
        "pg_dump",
        (
            f"postgresql://{parsed['user']}:{parsed['password']}"
            f"@{parsed['host']}:{parsed['port']}/{parsed['database']}"
        ),
        "--format=plain",
        "--no-owner",
        "--no-privileges",
    ]


def _backup_file() -> Path:
    """生成带时间戳的备份文件名。"""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return BACKUP_DIR / f"risk-{stamp}.sql"


def _counts_with_fallback(
    url: str,
    *,
    container: bool,
) -> dict[str, int] | None:
    """统计行数；校验不可用时返回 None 并提示。"""
    try:
        return verify_counts(
            build_count_runner(url, container=container)
        )
    except Exception as exc:
        print(f"[警告] 行数校验不可用：{exc}", file=sys.stderr)
        return None


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数，密钥只通过环境变量传入。"""
    parser = argparse.ArgumentParser(
        description="把 PostgreSQL 数据备份到 backups/ 目录并记录行数。"
    )
    parser.add_argument(
        "--url",
        default=os.getenv("DATABASE_URL", ""),
        help="PostgreSQL URL；未传入时读取 DATABASE_URL",
    )
    parser.add_argument(
        "--container",
        action="store_true",
        help="通过 docker compose exec postgres 执行 pg_dump",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口，只输出备份文件与行数摘要。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("必须提供 --url 或 DATABASE_URL")
    parsed = parse_database_url(args.url)
    before = _counts_with_fallback(args.url, container=bool(args.container))
    target = _backup_file()
    command = _pg_dump_command(parsed, container=bool(args.container))
    with target.open("wb") as output:
        subprocess.run(
            command,
            stdout=output,
            check=True,
        )
    after = _counts_with_fallback(args.url, container=bool(args.container))
    size = target.stat().st_size
    summary = {
        "file": target.name,
        "size_bytes": size,
        "before": before,
        "after": after,
    }
    print(json.dumps(summary, ensure_ascii=False))
    if before is not None and after is not None and before != after:
        print("[警告] 备份前后行数不一致，请检查并发写入", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
