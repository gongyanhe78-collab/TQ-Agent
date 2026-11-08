"""验证知识库数据卷初始化保持幂等并排除运行态文件。"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.init_api_data import sync_knowledge_data


def test_sync_copies_knowledge_files_and_skips_runtime_state(tmp_path: Path) -> None:
    """首次初始化应复制知识库，但不能把旧 SQLite 和 Numba 缓存带入数据卷。"""
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    (source_dir / "document_index").mkdir(parents=True)
    (source_dir / "document_images").mkdir()
    (source_dir / "numba_cache").mkdir()
    (source_dir / "document_index" / "chunks.json").write_text("[]", encoding="utf-8")
    (source_dir / "document_images" / "figure.png").write_bytes(b"image")
    (source_dir / "standard_cases1.json").write_text("{}", encoding="utf-8")
    (source_dir / "sessions.sqlite3").write_bytes(b"legacy")
    (source_dir / "numba_cache" / "cache.bin").write_bytes(b"cache")

    result = sync_knowledge_data(source_dir, target_dir)

    assert result.copied == 3
    assert result.skipped == 0
    assert (target_dir / "document_index" / "chunks.json").read_text(
        encoding="utf-8"
    ) == "[]"
    assert (target_dir / "document_images" / "figure.png").read_bytes() == b"image"
    assert (target_dir / "standard_cases1.json").is_file()
    assert not (target_dir / "sessions.sqlite3").exists()
    assert not (target_dir / "numba_cache").exists()


def test_second_sync_keeps_unchanged_target_files(tmp_path: Path) -> None:
    """重复启动初始化服务时不得反复复制未变化的大体积知识库文件。"""
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    (source_dir / "image_metadata.json").write_text("{}", encoding="utf-8")

    first_result = sync_knowledge_data(source_dir, target_dir)
    second_result = sync_knowledge_data(source_dir, target_dir)

    assert first_result.copied == 1
    assert second_result.copied == 0
    assert second_result.skipped == 1


def test_sync_rejects_same_source_and_target(tmp_path: Path) -> None:
    """错误配置不能把同一目录当作源和目标而掩盖路径问题。"""
    shared_dir = tmp_path / "data"
    shared_dir.mkdir()

    with pytest.raises(ValueError, match="不能相同"):
        sync_knowledge_data(shared_dir, shared_dir)
