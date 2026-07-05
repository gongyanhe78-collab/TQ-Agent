from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ProcessedFileStore:
    """记录已经完成抽取入库的 PDF 文件指纹，用于判断是否需要增量处理。"""

    def __init__(self, storage_path: Path):
        """初始化记录文件路径，并确保父目录存在。"""
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def is_processed(self, pdf_path: Path) -> bool:
        """判断 PDF 当前内容是否已经处理过；文件内容变化时会返回 False。"""
        fingerprint = self._fingerprint(pdf_path)
        records = self._load()
        return records.get(Path(pdf_path).name) == fingerprint

    def mark_processed(self, pdf_path: Path) -> None:
        """将 PDF 当前内容指纹写入记录文件，表示该文件已经完成处理。"""
        records = self._load()
        records[Path(pdf_path).name] = self._fingerprint(pdf_path)
        self.storage_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self) -> dict[str, str]:
        """读取已处理 PDF 的指纹记录；没有记录文件时返回空字典。"""
        if not self.storage_path.exists():
            return {}
        return json.loads(self.storage_path.read_text(encoding="utf-8"))

    def _fingerprint(self, pdf_path: Path) -> str:
        """根据文件大小和 sha256 内容摘要生成稳定指纹。"""
        path = Path(pdf_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return f"{path.stat().st_size}:{digest}"
