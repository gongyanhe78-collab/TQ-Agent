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


def _trace_sample_rate(raw: str) -> float:
    """解析采样率，并在生产环境强制限制到 0.1。"""
    try:
        rate = float(raw)
    except (TypeError, ValueError):
        rate = 1.0
    rate = min(1.0, max(0.0, rate))
    environment = (
        os.getenv("APP_ENV")
        or os.getenv("ENVIRONMENT")
        or os.getenv("DEPLOYMENT_ENV")
        or ""
    ).strip().lower()
    if environment in {"prod", "production"}:
        return min(rate, 0.1)
    return rate


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
    database_url: str = field(
        default_factory=lambda: _env_value(
            "DATABASE_URL",
            f"sqlite+pysqlite:///{(PROJECT_ROOT / 'data' / 'risk.sqlite3').as_posix()}",
        )
    )
    legacy_user_id: str = "legacy_user"
    legacy_username: str = "legacy_user"
    legacy_password: str = ""
    redis_url: str = field(
        default_factory=lambda: _env_value("REDIS_URL", "redis://127.0.0.1:6379/0")
    )
    redis_cache_enabled: bool = field(
        default_factory=lambda: _env_value("REDIS_CACHE_ENABLED", "true").lower()
        in {"1", "true", "yes", "on"}
    )
    redis_context_ttl_seconds: int = field(
        default_factory=lambda: int(_env_value("REDIS_CONTEXT_TTL_SECONDS", "1800"))
    )
    redis_key_prefix: str = field(
        default_factory=lambda: _env_value("REDIS_KEY_PREFIX", "risk")
    )
    auth_jwt_secret: str = field(
        default_factory=lambda: _env_value(
            "AUTH_JWT_SECRET",
            "development-only-secret-change-before-deployment",
        )
    )
    auth_access_ttl_seconds: int = field(
        default_factory=lambda: int(_env_value("AUTH_ACCESS_TTL_SECONDS", "900"))
    )
    auth_refresh_ttl_seconds: int = field(
        default_factory=lambda: int(_env_value("AUTH_REFRESH_TTL_SECONDS", "604800"))
    )
    auth_refresh_cookie_name: str = field(
        default_factory=lambda: _env_value("AUTH_REFRESH_COOKIE_NAME", "refresh_token")
    )
    auth_cookie_secure: bool = field(
        default_factory=lambda: _env_value("AUTH_COOKIE_SECURE", "false").lower()
        in {"1", "true", "yes", "on"}
    )
    auth_cookie_samesite: str = field(
        default_factory=lambda: _env_value("AUTH_COOKIE_SAMESITE", "lax")
    )
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            item.strip()
            for item in _env_value(
                "CORS_ORIGINS",
                "http://127.0.0.1:5173,http://localhost:5173",
            ).split(",")
            if item.strip()
        )
    )
    case_multidim_runtime_dir: Path = field(
        default_factory=lambda: _env_path(
            "CASE_MULTIDIM_RUNTIME_DIR",
            PROJECT_ROOT / "backend" / "app" / "agents" / "case_multidim_search" / "runtime",
        )
    )
    document_collection_name: str = "weather_document_chunks"
    otel_service_name: str = field(
        default_factory=lambda: _env_value("OTEL_SERVICE_NAME", "risk-master")
    )
    otlp_endpoint: str = field(
        default_factory=lambda: (
            os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
            or os.getenv("OTLP_ENDPOINT", "").strip()
        )
    )
    tracing_enabled: bool = field(
        default_factory=lambda: _env_value("TRACING_ENABLED", "true").lower()
        in {"1", "true", "yes", "on"}
    )
    tracing_sample_rate: float = field(
        default_factory=lambda: _trace_sample_rate(
            _env_value("OTEL_TRACES_SAMPLER_ARG", "1.0")
        )
    )
    log_json: bool = field(
        default_factory=lambda: _env_value("LOG_JSON", "false").lower()
        in {"1", "true", "yes", "on"}
    )
    metrics_enabled: bool = field(
        default_factory=lambda: _env_value("METRICS_ENABLED", "true").lower()
        in {"1", "true", "yes", "on"}
    )
    agent_max_seconds: float = field(
        default_factory=lambda: float(_env_value("AGENT_MAX_SECONDS", "300"))
    )
    agent_max_steps: int = field(
        default_factory=lambda: int(_env_value("AGENT_MAX_STEPS", "50"))
    )
    agent_max_model_calls: int = field(
        default_factory=lambda: int(_env_value("AGENT_MAX_MODEL_CALLS", "40"))
    )
    agent_token_budget: int = field(
        default_factory=lambda: int(_env_value("AGENT_TOKEN_BUDGET", "150000"))
    )
    agent_step_timeout_seconds: float = field(
        default_factory=lambda: float(
            _env_value("AGENT_STEP_TIMEOUT_SECONDS", "90")
        )
    )

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

    # 模型与外部服务降级：统一超时和熔断参数，连续失败时快速失败。
    model_read_timeout_seconds: float = field(
        default_factory=lambda: float(
            _env_value("MODEL_READ_TIMEOUT_SECONDS", "60")
        )
    )
    model_connect_timeout_seconds: float = field(
        default_factory=lambda: float(
            _env_value("MODEL_CONNECT_TIMEOUT_SECONDS", "10")
        )
    )
    circuit_failure_threshold: int = field(
        default_factory=lambda: int(
            _env_value("CIRCUIT_FAILURE_THRESHOLD", "5")
        )
    )
    circuit_cooldown_seconds: float = field(
        default_factory=lambda: float(
            _env_value("CIRCUIT_COOLDOWN_SECONDS", "30")
        )
    )

    # 开放域联网搜索：普通聊天模型直接回答库外问题并启用联网搜索。
    web_search_enabled: bool = field(
        default_factory=lambda: _env_value("WEB_SEARCH_ENABLED", "true").lower()
        in {"1", "true", "yes", "on"}
    )
    web_search_forced: bool = field(
        default_factory=lambda: _env_value("WEB_SEARCH_FORCED", "true").lower()
        in {"1", "true", "yes", "on"}
    )
    # 本地知识库检索为空时自动升级联网搜索。
    rag_empty_upgrade_web_search: bool = field(
        default_factory=lambda: _env_value("RAG_EMPTY_UPGRADE_WEB_SEARCH", "true").lower()
        in {"1", "true", "yes", "on"}
    )
    # 预留低分升级阈值；0 表示仅空结果触发，本次不参与判断逻辑。
    rag_min_score: float = field(
        default_factory=lambda: float(_env_value("RAG_MIN_SCORE", "0"))
    )

    chat_model: str = ""
    embedding_model: str = ""
    rerank_model: str = ""
    api_key: str = ""
    base_url: str = ""
    vector_backend: str = field(
        default_factory=lambda: _env_value("VECTOR_BACKEND", "chroma").lower()
    )
    vector_fallback_backend: str = field(
        default_factory=lambda: _env_value(
            "VECTOR_FALLBACK_BACKEND",
            "chroma",
        ).lower()
    )
    vector_schema_version: str = field(
        default_factory=lambda: _env_value("VECTOR_SCHEMA_VERSION", "v1")
    )
    milvus_uri: str = field(
        default_factory=lambda: _env_value(
            "MILVUS_URI",
            "http://127.0.0.1:19530",
        )
    )
    milvus_token: str = field(
        default_factory=lambda: os.getenv("MILVUS_TOKEN", "")
    )
    milvus_database: str = field(
        default_factory=lambda: _env_value("MILVUS_DATABASE", "default")
    )
    milvus_timeout_seconds: float = field(
        default_factory=lambda: float(
            _env_value("MILVUS_TIMEOUT_SECONDS", "10")
        )
    )
    milvus_collection_name: str = field(
        default_factory=lambda: _env_value(
            "MILVUS_DOCUMENT_COLLECTION",
            "weather_document_chunks_v1",
        )
    )
    milvus_memory_collection_name: str = field(
        default_factory=lambda: _env_value(
            "MILVUS_MEMORY_COLLECTION",
            "weather_user_memories_v1",
        )
    )
    milvus_dense_top_k: int = field(
        default_factory=lambda: int(_env_value("MILVUS_DENSE_TOP_K", "30"))
    )
    milvus_sparse_top_k: int = field(
        default_factory=lambda: int(_env_value("MILVUS_SPARSE_TOP_K", "30"))
    )
    milvus_rrf_k: int = field(
        default_factory=lambda: int(_env_value("MILVUS_RRF_K", "60"))
    )

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
