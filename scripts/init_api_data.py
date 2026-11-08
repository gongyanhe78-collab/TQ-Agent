"""把本机知识库文件幂等同步到 API 数据卷。"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


LOGGER = logging.getLogger("risk-data-init")
SKIPPED_DIRECTORY_NAMES = {"__pycache__", "numba_cache"}
SKIPPED_FILE_NAMES = {
    "sessions.sqlite3",
    "sessions.sqlite3-shm",
    "sessions.sqlite3-wal",
}


@dataclass(frozen=True, slots=True)
class SyncResult:
    """记录一次知识库同步的文件数量。"""

    copied: int
    skipped: int


def _should_skip(relative_path: Path) -> bool:
    """排除只属于本机运行状态的缓存和旧 SQLite 文件。"""
    if any(part in SKIPPED_DIRECTORY_NAMES for part in relative_path.parts):
        return True
    return relative_path.name in SKIPPED_FILE_NAMES


def _source_files(source_dir: Path) -> list[Path]:
    """返回需要同步的普通文件，拒绝跟随符号链接。"""
    return sorted(
        (
            path
            for path in source_dir.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and not _should_skip(path.relative_to(source_dir))
        ),
        key=lambda path: path.as_posix(),
    )


def _needs_copy(source_path: Path, target_path: Path) -> bool:
    """根据文件大小和纳秒修改时间判断是否需要刷新目标副本。"""
    if not target_path.is_file():
        return True
    source_stat = source_path.stat()
    target_stat = target_path.stat()
    return (
        source_stat.st_size != target_stat.st_size
        or source_stat.st_mtime_ns != target_stat.st_mtime_ns
    )


def sync_knowledge_data(source_dir: Path, target_dir: Path) -> SyncResult:
    """把知识库文件复制到目标目录，已有且未变化的文件保持不变。"""
    source_dir = source_dir.expanduser().resolve()
    target_dir = target_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"知识库源目录不存在：{source_dir}")
    if source_dir == target_dir:
        raise ValueError("知识库源目录和目标目录不能相同。")

    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for source_path in _source_files(source_dir):
        relative_path = source_path.relative_to(source_dir)
        target_path = target_dir / relative_path
        if not _needs_copy(source_path, target_path):
            skipped += 1
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied += 1

    return SyncResult(copied=copied, skipped=skipped)


def _parse_args() -> argparse.Namespace:
    """解析 Compose 或运维命令传入的数据目录。"""
    parser = argparse.ArgumentParser(description="同步知识库到 API 命名数据卷")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(os.getenv("DATA_INIT_SOURCE", "/seed-data")),
        help="本机知识库源目录",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(os.getenv("DATA_INIT_TARGET", "/app/data")),
        help="API 数据卷目标目录",
    )
    return parser.parse_args()


def main() -> int:
    """执行一次幂等同步并输出可核对的文件数量。"""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args()
    result = sync_knowledge_data(args.source, args.target)
    LOGGER.info(
        "知识库同步完成：复制 %s 个文件，跳过 %s 个未变化文件。",
        result.copied,
        result.skipped,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOGGER.error("知识库同步失败：%s", exc)
        raise SystemExit(1) from exc
