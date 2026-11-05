"""意图识别与任务分发的结构化审计日志。"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any


class IntentAuditWriter:
    """以 JSONL 保存意图链路，并限制单个文件大小。"""

    MAX_BYTES = 5 * 1024 * 1024

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = Lock()

    def write(self, payload: dict[str, Any]) -> None:
        """线程安全追加单轮审计；写盘失败不影响主业务分发。"""
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._rotate_if_needed()
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:
            return

    def _rotate_if_needed(self) -> None:
        """超过上限时保留一个历史文件，避免审计无限占用磁盘。"""
        if not self.path.is_file() or self.path.stat().st_size < self.MAX_BYTES:
            return
        backup = self.path.with_suffix(self.path.suffix + ".1")
        if backup.exists():
            backup.unlink()
        self.path.replace(backup)
