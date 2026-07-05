"""
案例提取流水线模块
整合案例提取和文件输出，将提取的案例保存为 txt 文件
"""
from __future__ import annotations

from pathlib import Path

from backend.app.models import ExtractedCase


class ExtractionPipeline:
    """
    案例提取流水线
    调用案例提取服务，将提取结果保存为独立的 txt 文件
    """

    def __init__(self, extractor, output_dir: Path):
        """
        初始化提取流水线

        Args:
            extractor: 案例提取服务实例
            output_dir: 输出文件目录
        """
        self.extractor = extractor
        self.output_dir = Path(output_dir)
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_pdf(self, pdf_path: Path) -> list[Path]:
        """
        处理单个 PDF 文件，提取案例并保存为 txt 文件

        Args:
            pdf_path: PDF 文件路径

        Returns:
            保存的 txt 文件路径列表
        """
        # 调用提取器提取所有案例
        cases = self.extractor.extract_cases(pdf_path)
        # 将每个案例写入独立的 txt 文件
        return [self._write_case(case) for case in cases]

    def _write_case(self, case: ExtractedCase) -> Path:
        """
        将单个案例写入 txt 文件（内部方法）

        Args:
            case: ExtractedCase 案例对象

        Returns:
            保存的文件路径
        """
        file_path = self.output_dir / f"{case.case_id}.txt"
        # 构建文件内容：元数据 + 空行 + 正文
        payload = "\n".join(
            [
                f"source_pdf: {case.source_pdf}",
                f"case_id: {case.case_id}",
                f"title: {case.title}",
                f"date_range: {case.date_range}",
                "",
                case.content.strip(),
                "",
            ]
        )
        file_path.write_text(payload, encoding="utf-8")
        # 更新案例对象中的文件路径
        case.file_path = str(file_path)
        return file_path
