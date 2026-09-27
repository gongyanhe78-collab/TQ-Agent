"""
配置文件模块
负责加载环境变量，定义项目全局配置参数
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# 加载 .env 文件中的环境变量
load_dotenv()


@dataclass
class Settings:
    """
    项目全局配置类
    使用 dataclass 定义所有配置项，支持从环境变量覆盖
    """
    # 项目根目录路径（向上两级，即 E:\risk）
    project_root: Path = Path(__file__).resolve().parents[2]
    # PDF 资源文件存放目录
    resource_dir: Path = project_root / "resource"
    # 数据根目录
    data_dir: Path = project_root / "data"
    # 三个业务 Agent 共用主项目数据目录，保留该名称兼容新替换的 Agent。
    smart_case_data_dir: Path = data_dir
    # PDF 文档化后按自然段 chunk 入库的唯一向量索引目录
    document_index_dir: Path = data_dir / "document_index"
    # PDF 图片证据保存目录
    document_images_dir: Path = data_dir / "document_images"
    # Numba 缓存目录，供 Unstructured PDF 文档化依赖使用，避免卡在系统临时目录
    numba_cache_dir: Path = data_dir / "numba_cache"
    # PDF 图片证据元数据文件
    image_metadata_path: Path = data_dir / "image_metadata.json"
    # 标准化个例 JSON 存储文件
    standard_cases_path: Path = data_dir / "standard_cases1.json"
    # 会话管理数据库
    session_db_path: Path = data_dir / "sessions.sqlite3"
    # 唯一的 PDF 自然段 chunk ChromaDB 集合名称
    document_collection_name: str = "weather_document_chunks"

    # ========== 云端模型配置 ==========
    # DashScope 千问 API Key，兼容历史 ALIYUN_API_KEY / ALiYunAPI 写法
    dashscope_api_key: str = (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("ALIYUN_API_KEY")
        or os.getenv("ALiYunAPI", "")
    )
    # DashScope OpenAI 兼容接口地址和聊天模型名称
    dashscope_base_url: str = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    dashscope_chat_model: str = os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus")
    # 意图模型允许单独配置；未配置时复用已经验证可调用的聊天模型，避免默认模型无配额后静默退回规则。
    dashscope_intent_model: str = (
        os.getenv("DASHSCOPE_INTENT_MODEL")
        or os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus")
    )
    # 后续问题属于非关键后处理，默认复用轻量意图模型，也允许部署环境独立指定更快模型。
    dashscope_followup_model: str = (
        os.getenv("DASHSCOPE_FOLLOWUP_MODEL")
        or os.getenv("DASHSCOPE_INTENT_MODEL")
        or os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus")
    )
    # Chat、Embedding、Rerank 均固定走云端 API，不再保留本地模型分支。
    dashscope_embedding_model: str = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")
    dashscope_rerank_model: str = os.getenv("DASHSCOPE_RERANK_MODEL", "qwen3-rerank")
    dashscope_rerank_endpoint: str = os.getenv("DASHSCOPE_RERANK_ENDPOINT", "")

    # 兼容系统状态页的展示字段，值始终来自云端模型配置。
    chat_model: str = ""
    embedding_model: str = ""
    rerank_model: str = ""
    api_key: str = ""
    base_url: str = ""

    def __post_init__(self) -> None:
        """回填状态接口需要的统一云端模型字段。"""
        self.chat_model = self.dashscope_chat_model
        self.embedding_model = self.dashscope_embedding_model
        self.rerank_model = self.dashscope_rerank_model
        self.api_key = self.dashscope_api_key
        self.base_url = self.dashscope_base_url

    def ensure_directories(self) -> None:
        """
        确保所有必要的数据目录都存在
        如果目录不存在则自动创建，父目录也会一并创建
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.document_index_dir.mkdir(parents=True, exist_ok=True)
        self.document_images_dir.mkdir(parents=True, exist_ok=True)
        self.numba_cache_dir.mkdir(parents=True, exist_ok=True)


# 创建全局配置实例，项目其他地方直接导入使用
settings = Settings()
# 初始化时确保所有目录都已创建
settings.ensure_directories()
# 指定 Numba 缓存目录，保证 PDF 文档化依赖在 Windows 环境下稳定导入
os.environ.setdefault("NUMBA_CACHE_DIR", str(settings.numba_cache_dir))
