"""项目配置定义，模块导入时不创建目录或连接外部服务。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _env_value(name: str, default: str) -> str:
    """读取环境变量，空值按未配置处理。"""
    return os.getenv(name, "").strip() or default


def _env_path(name: str, default: Path) -> Path:
    """读取可选的路径环境变量，未配置时沿用项目默认目录。"""
    configured = os.getenv(name, "").strip()
    return Path(configured) if configured else default


def _dashscope_api_key() -> str:
    """兼容项目历史使用的三种 DashScope 密钥变量名。"""
    return (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("ALIYUN_API_KEY")
        or os.getenv("ALiYunAPI", "")
    )


@dataclass(slots=True)
class Settings:
    """项目全局配置，字段保持与旧版本一致。"""

    project_root: Path = PROJECT_ROOT
    resource_dir: Path = PROJECT_ROOT / "resource"
    data_dir: Path = PROJECT_ROOT / "data"
    smart_case_data_dir: Path = PROJECT_ROOT / "data"
    document_index_dir: Path = PROJECT_ROOT / "data" / "document_index"
    document_images_dir: Path = PROJECT_ROOT / "data" / "document_images"
    numba_cache_dir: Path = PROJECT_ROOT / "data" / "numba_cache"
    image_metadata_path: Path = PROJECT_ROOT / "data" / "image_metadata.json"
    standard_cases_path: Path = PROJECT_ROOT / "data" / "standard_cases1.json"
    session_db_path: Path = PROJECT_ROOT / "data" / "sessions.sqlite3"
    case_multidim_runtime_dir: Path = field(
        default_factory=lambda: _env_path(
            "CASE_MULTIDIM_RUNTIME_DIR",
            PROJECT_ROOT / "backend" / "app" / "agents" / "case_multidim_search" / "runtime",
        )
    )
    document_collection_name: str = "weather_document_chunks"

    dashscope_api_key: str = field(default_factory=_dashscope_api_key)
    dashscope_base_url: str = field(
        default_factory=lambda: _env_value(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    )
    dashscope_chat_model: str = field(
        default_factory=lambda: _env_value("DASHSCOPE_CHAT_MODEL", "qwen-plus")
    )
    dashscope_intent_model: str = field(
        default_factory=lambda: _env_value(
            "DASHSCOPE_INTENT_MODEL",
            _env_value("DASHSCOPE_CHAT_MODEL", "qwen-plus"),
        )
    )
    dashscope_followup_model: str = field(
        default_factory=lambda: _env_value(
            "DASHSCOPE_FOLLOWUP_MODEL",
            _env_value(
                "DASHSCOPE_INTENT_MODEL",
                _env_value("DASHSCOPE_CHAT_MODEL", "qwen-plus"),
            ),
        )
    )
    dashscope_embedding_model: str = field(
        default_factory=lambda: _env_value(
            "DASHSCOPE_EMBEDDING_MODEL",
            "text-embedding-v4",
        )
    )
    dashscope_rerank_model: str = field(
        default_factory=lambda: _env_value(
            "DASHSCOPE_RERANK_MODEL",
            "qwen3-rerank",
        )
    )
    dashscope_rerank_endpoint: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_RERANK_ENDPOINT", "")
    )

    chat_model: str = ""
    embedding_model: str = ""
    rerank_model: str = ""
    api_key: str = ""
    base_url: str = ""

    def __post_init__(self) -> None:
        """回填系统状态接口需要的统一模型字段。"""
        self.chat_model = self.dashscope_chat_model
        self.embedding_model = self.dashscope_embedding_model
        self.rerank_model = self.dashscope_rerank_model
        self.api_key = self.dashscope_api_key
        self.base_url = self.dashscope_base_url

    def ensure_directories(self) -> None:
        """创建运行所需目录，仅由显式生命周期初始化调用。"""
        for directory in (
            self.data_dir,
            self.document_index_dir,
            self.document_images_dir,
            self.numba_cache_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """加载环境变量并返回缓存的配置实例。"""
    load_dotenv()
    return Settings()
