"""公司统一知识库 HTTP 接口客户端。"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin


DEFAULT_COMPANY_KB_BASE_URL = "http://127.0.0.1:8000"


class CompanyKbClientError(RuntimeError):
    """调用公司统一知识库接口时抛出的异常。"""


class CompanyKbClient:
    """封装多维个例检索所需的公司知识库 HTTP 请求。"""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ):
        """保存基础地址、认证信息和请求超时配置。"""
        self.base_url = self._normalize_base_url(
            base_url
            or os.getenv("CASE_MULTIDIM_KB_BASE_URL")
            or os.getenv("COMPANY_RAG_BASE_URL")
            or os.getenv("RAG_API_BASE_URL")
            or DEFAULT_COMPANY_KB_BASE_URL
        )
        self.api_key = api_key if api_key is not None else os.getenv("CASE_MULTIDIM_KB_API_KEY", "")
        self.timeout = float(timeout or os.getenv("CASE_MULTIDIM_KB_TIMEOUT", "30"))

    def status(self) -> dict[str, Any]:
        """读取知识库接口健康状态。"""
        data = self._get_json("/api/v1/kb/status")
        return data if isinstance(data, dict) else {}

    def list_standard_cases(self, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """读取远程标准化个例列表。"""
        data = self._get_json("/api/v1/artifacts/standard-cases", params=params)
        if isinstance(data, dict):
            cases = data.get("cases") or data.get("items") or data.get("data") or []
            return [item for item in cases if isinstance(item, dict)]
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """按 chunk_id 读取单个原文片段。"""
        safe_id = str(chunk_id or "").strip()
        if not safe_id:
            return None
        try:
            data = self._get_json(f"/api/v1/artifacts/chunk/{safe_id}")
        except CompanyKbClientError as exc:
            # 单个 chunk 缺失时允许上层跳过，其他错误继续暴露。
            if "404" in str(exc):
                return None
            raise
        if isinstance(data, dict):
            chunk = data.get("chunk") or data
            return chunk if isinstance(chunk, dict) else None
        return None

    def list_image_metadata(self, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """读取远程图片元数据列表。"""
        data = self._get_json("/api/v1/artifacts/image-metadata", params=params)
        if isinstance(data, dict):
            images = data.get("images") or data.get("items") or data.get("data") or []
            return [item for item in images if isinstance(item, dict)]
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    def absolute_url(self, value: str) -> str:
        """把接口返回的相对路径转成绝对 URL。"""
        raw = str(value or "").strip()
        if not raw:
            return ""
        if re.match(r"^https?://", raw, flags=re.IGNORECASE):
            return raw
        return urljoin(self.base_url + "/", raw.lstrip("/"))

    def download_asset(self, image_url: str, output_dir: Path, filename_hint: str = "") -> Path:
        """按需下载远程图片，供网页预览和 PDF 复用。"""
        url = self.absolute_url(image_url)
        if not url:
            raise CompanyKbClientError("图片 URL 为空，无法下载远程图片。")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(url.split("?", 1)[0]).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            suffix = ".png"
        safe_name = self._safe_filename(filename_hint or Path(url).stem or "remote-image")
        target = output_dir / f"{safe_name}{suffix}"
        if target.is_file() and target.stat().st_size > 0:
            return target

        response = self._http().get(url, headers=self._headers(), timeout=self.timeout)
        if response.status_code >= 400:
            raise CompanyKbClientError(f"远程图片下载失败：HTTP {response.status_code} {url}")
        target.write_bytes(response.content)
        return target

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """执行 GET 请求并返回 JSON 内容。"""
        if not self.base_url:
            raise CompanyKbClientError(
                "未配置 CASE_MULTIDIM_KB_BASE_URL，无法调用公司统一知识库接口。"
            )
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        response = self._http().get(url, params=params or {}, headers=self._headers(), timeout=self.timeout)
        if response.status_code >= 400:
            hint = ""
            if response.status_code == 404:
                hint = "；请检查 CASE_MULTIDIM_KB_BASE_URL 是否指向知识库服务，而不是当前业务后端"
            raise CompanyKbClientError(f"公司知识库接口返回错误：HTTP {response.status_code} {url}{hint}")
        try:
            return response.json()
        except ValueError as exc:
            raise CompanyKbClientError(f"公司知识库返回不是 JSON：{url}") from exc

    def _headers(self) -> dict[str, str]:
        """同时支持 Bearer 和 X-API-Key 两种常见认证方式。"""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key
        return headers

    def _http(self):
        """延迟导入 requests，便于应用启动和单元测试隔离。"""
        try:
            import requests
        except ImportError as exc:
            raise CompanyKbClientError("缺少 requests 依赖，无法调用 HTTP 知识库。") from exc
        return requests

    def _normalize_base_url(self, value: str | None) -> str:
        """统一移除基础地址末尾斜杠，避免拼接重复。"""
        return str(value or "").strip().rstrip("/")

    def _safe_filename(self, value: str) -> str:
        """根据远程 image_id 生成本地缓存文件名。"""
        name = re.sub(r"[^0-9A-Za-z一-鿿_.-]+", "_", str(value or "remote-image"))
        return name.strip("._-") or "remote-image"
