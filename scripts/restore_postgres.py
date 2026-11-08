"""PostgreSQL 恢复脚本：psql 导入备份并校验行数。"""
from __future__ import annotations

import argparse
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


def _latest_backup() -> Path:
    """返回 backups/ 目录中时间最新的 SQL 备份。"""
    candidates = sorted(BACKUP_DIR.glob("risk-*.sql"))
    if not candidates:
        raise FileNotFoundError(
            f"{BACKUP_DIR} 中没有找到 risk-*.sql 备份文件"
        )
    return candidates[-1]


def _psql_command(
    parsed: dict[str, str],
    *,
    container: bool,
) -> list[str]:
    """构造 psql 导入命令，凭据来自解析后的数据库 URL。"""
    if container:
        return [
            "docker",
            "compose",
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            parsed["user"],
            "-d",
            parsed["database"],
            "-v",
            "ON_ERROR_STOP=1",
        ]
    return [
        "psql",
        (
            f"postgresql://{parsed['user']}:{parsed['password']}"
            f"@{parsed['host']}:{parsed['port']}/{parsed['database']}"
        ),
        "-v",
        "ON_ERROR_STOP=1",
    ]


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
        description="把 backups/ 目录中的 SQL 备份恢复到 PostgreSQL。"
    )
    parser.add_argument(
        "--url",
        default=os.getenv("DATABASE_URL", ""),
        help="PostgreSQL URL；未传入时读取 DATABASE_URL",
    )
    parser.add_argument(
        "--file",
        default="",
        help="备份文件路径；为空时使用 backups/ 中最新备份",
    )
    parser.add_argument(
        "--container",
        action="store_true",
        help="通过 docker compose exec postgres 执行 psql",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="目标库已有数据时仍继续恢复（默认拒绝覆盖）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口，只输出恢复结果与行数摘要。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("必须提供 --url 或 DATABASE_URL")
    parsed = parse_database_url(args.url)
    source = Path(args.file) if args.file else _latest_backup()
    if not source.is_file():
        parser.error(f"备份文件不存在：{source}")
    before = _counts_with_fallback(args.url, container=bool(args.container))
    if before and sum(before.values()) > 0 and not args.force:
        print(
            "[中止] 目标库已有数据，恢复前请先清空数据库或使用 --force",
            file=sys.stderr,
        )
        return 2
    command = _psql_command(parsed, container=bool(args.container))
    with source.open("rb") as dump:
        subprocess.run(
            command,
            stdin=dump,
            check=True,
        )
    after = _counts_with_fallback(args.url, container=bool(args.container))
    summary = {
        "file": source.name,
        "before": before,
        "after": after,
    }
    print(json.dumps(summary, ensure_ascii=False))
    if before is not None and after is not None:
        for table, count in before.items():
            restored = (after.get(table) or 0)
            if restored != count:
                print(
                    f"[警告] 表 {table} 行数不一致：备份 {count}，恢复后 {restored}",
                    file=sys.stderr,
                )
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
