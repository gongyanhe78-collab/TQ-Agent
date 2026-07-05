"""
案例提取服务模块
整合 PDF 读取、案例分割、LLM 精炼，完成完整的案例提取流程
"""
from __future__ import annotations

from pathlib import Path

from backend.app.models import ExtractedCase
from backend.app.services.case_splitter import CaseSplitter
from backend.app.services.llm_client import DashScopeChatClient
from backend.app.services.pdf_reader import PdfReaderService
from backend.app.services.text_cleaning import normalize_title


class CaseExtractionService:
    """
    案例提取服务
    整合 PDF 读取、规则分割、LLM 精炼三个步骤，从 PDF 提取标准化的灾害案例
    """

    def __init__(
        self,
        pdf_reader: PdfReaderService | None = None,
        splitter: CaseSplitter | None = None,
        llm_client: DashScopeChatClient | None = None,
    ):
        """
        初始化案例提取服务

        Args:
            pdf_reader: PDF 读取器，不传则使用默认
            splitter: 案例分割器，不传则使用默认
            llm_client: LLM 客户端，不传则使用默认
        """
        self.pdf_reader = pdf_reader or PdfReaderService()
        self.splitter = splitter or CaseSplitter()
        self.llm_client = llm_client or DashScopeChatClient()

    def extract_cases(self, pdf_path: Path) -> list[ExtractedCase]:
        """
        从 PDF 文件提取所有灾害案例
        流程：读取 PDF 文本 -> 规则分割候选案例 -> LLM 精炼标准化

        Args:
            pdf_path: PDF 文件路径

        Returns:
            ExtractedCase 对象列表
        """
        # 1. 读取 PDF 文本内容
        text = self.pdf_reader.read_text(pdf_path)
        # 2. 使用规则分割出候选案例
        candidates = self.splitter.split_cases(text, pdf_path.name)
        refined_cases: list[ExtractedCase] = []
        # 3. 对每个候选案例使用 LLM 精炼
        for candidate in candidates:
            refined = self.llm_client.refine_case(candidate)
            # 如果 LLM 确认是案例，则用 LLM 输出的标准化值更新
            if refined.is_case:
                # 优先使用 LLM 精炼后的标题（如果为空则保留原值）
                candidate.title = normalize_title(refined.title) or candidate.title
                candidate.date_range = refined.date_range or candidate.date_range
                candidate.content = refined.content.strip() or candidate.content
            refined_cases.append(candidate)
        return refined_cases
