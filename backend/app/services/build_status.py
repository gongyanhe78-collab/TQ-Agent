from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class BuildStatusStore:
    """管理知识库构建状态，并将状态持久化到 JSON 文件。"""

    def __init__(self, status_path: Path):
        """初始化状态文件路径，并确保父目录存在。"""
        self.status_path = Path(status_path)
        self.status_path.parent.mkdir(parents=True, exist_ok=True)

    def get_status(self) -> dict:
        """读取当前构建状态；如果没有状态文件，则返回默认空闲状态。"""
        if not self.status_path.exists():
            return {
                "status": "idle",
                "last_build_time": None,
                "last_error": None,
                "last_result": None,
            }
        return json.loads(self.status_path.read_text(encoding="utf-8"))

    def mark_building(self) -> None:
        """将知识库构建状态标记为进行中，并清空上一次错误。"""
        self._write({"status": "building", "last_error": None})

    def mark_success(self, result: dict) -> None:
        """将知识库构建状态标记为成功，并保存本次构建结果摘要。"""
        self._write({"status": "idle", "last_error": None, "last_result": result})

    def mark_failed(self, error: Exception | str) -> None:
        """将知识库构建状态标记为失败，并记录失败原因。"""
        self._write({"status": "failed", "last_error": str(error)})

    def _write(self, updates: dict) -> None:
        """合并状态更新，补充更新时间，然后写回 JSON 状态文件。"""
        payload = self.get_status()
        payload.update(updates)
        payload["last_build_time"] = datetime.now(timezone.utc).isoformat()
        self.status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
