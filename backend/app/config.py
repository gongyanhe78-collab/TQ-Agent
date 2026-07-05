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
    # 提取出的案例 txt 文件存放目录
    extracted_dir: Path = data_dir / "extracted_cases"
    # 样本案例存放目录（用于训练/评估）
    samples_dir: Path = data_dir / "extracted_samples"
    # 向量索引存放目录
    index_dir: Path = data_dir / "index"
    # PDF 文档化后按 chunk 入库的新向量索引目录，独立于原个例库
    document_index_dir: Path = data_dir / "document_index"
    # PDF 图片证据保存目录
    document_images_dir: Path = data_dir / "document_images"
    # PDF 图片证据元数据文件
    image_metadata_path: Path = data_dir / "image_metadata.json"
    # 标准化个例 JSON 存储文件
    standard_cases_path: Path = data_dir / "standard_cases.json"
    # JSON 格式的案例备份文件路径
    json_store_path: Path = index_dir / "cases.json"
    # 记录知识库构建状态
    build_status_path: Path = index_dir / "build_status.json"
    # 记录已完成抽取入库的 PDF 文件指纹
    processed_files_path: Path = index_dir / "processed_files.json"
    # 会话管理数据库
    session_db_path: Path = data_dir / "sessions.sqlite3"
    # ChromaDB 向量集合名称
    collection_name: str = "weather_cases"
    # PDF 文档 chunk 使用的 ChromaDB 集合名称
    document_collection_name: str = "weather_document_chunks"
    # 阿里云通义千问 API 密钥，用于嵌入、重排和 DashScope 回退聊天
    dashscope_api_key: str = os.getenv("ALIYUN_API_KEY") or os.getenv("ALiYunAPI", "")
    # DashScope API 兼容模式的基础 URL，用于嵌入、重排和 DashScope 回退聊天
    dashscope_base_url: str = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # DMXAPI OpenAI 兼容聊天接口配置，用于问答和案例提取
    dmx_api_key: str = os.getenv("DMXAPI", "")
    dmx_base_url: str = os.getenv("DMXAPI_BASE_URL", "https://www.dmxapi.cn/v1")
    # 聊天模型单独配置：默认优先使用 DMXAPI 的 gpt-5.4-nano，缺少 DMXAPI 时回退 DashScope
    chat_api_key: str = os.getenv("CHAT_API_KEY") or dmx_api_key or dashscope_api_key
    chat_base_url: str = os.getenv("CHAT_BASE_URL") or (dmx_base_url if dmx_api_key else dashscope_base_url)
    chat_model: str = (
        os.getenv("CHAT_MODEL")
        or os.getenv("DMXAPI_CHAT_MODEL")
        or ("gpt-5.4-nano" if dmx_api_key else os.getenv("DASHSCOPE_CHAT_MODEL", "qwen3.6-max-preview"))
    )
    # 兼容旧代码字段：默认仍指向 DashScope，避免 embedding/rerank 被 DMXAPI 误切走
    api_key: str = dashscope_api_key
    base_url: str = dashscope_base_url
    # 向量嵌入模型名称（用于文本向量化）
    embedding_model: str = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")
    # 重排序模型名称（用于向量召回后的二次排序）
    rerank_model: str = os.getenv("DASHSCOPE_RERANK_MODEL", "qwen3-rerank")
    # DashScope 重排序接口地址，可通过环境变量覆盖
    rerank_endpoint: str = os.getenv("DASHSCOPE_RERANK_ENDPOINT", "")

    def ensure_directories(self) -> None:
        """
        确保所有必要的数据目录都存在
        如果目录不存在则自动创建，父目录也会一并创建
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.document_index_dir.mkdir(parents=True, exist_ok=True)
        self.document_images_dir.mkdir(parents=True, exist_ok=True)


# 创建全局配置实例，项目其他地方直接导入使用
settings = Settings()
# 初始化时确保所有目录都已创建
settings.ensure_directories()
