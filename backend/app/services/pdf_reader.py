"""
PDF 文件读取服务模块
从 PDF 文件中提取文本内容并进行规范化处理
"""
from __future__ import annotations

from pathlib import Path

from backend.app.services.text_cleaning import normalize_text


class PdfReaderService:
    """
    PDF 文本读取服务
    使用 pypdf 库读取 PDF 文件，提取所有页面的文本并规范化
    """

    def read_text(self, pdf_path: Path) -> str:
        """
        读取 PDF 文件并提取规范化的文本内容

        Args:
            pdf_path: PDF 文件路径

        Returns:
            提取并规范化后的文本字符串

        Raises:
            RuntimeError: 如果 pypdf 包未安装
        """
        # 动态导入 pypdf（避免没有安装时整个程序报错）
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - depends on installed deps
            raise RuntimeError("pypdf is required to read PDF files.") from exc

        # 创建 PDF 读取器
        reader = PdfReader(str(pdf_path))
        # 遍历所有页面，提取每页文本（空页面返回空字符串）
        chunks = [(page.extract_text() or "") for page in reader.pages]
        # 用换行连接所有页面文本并规范化
        return normalize_text("\n".join(chunks))
