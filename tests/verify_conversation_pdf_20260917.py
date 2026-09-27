"""生成并核验承接式简易 PDF，输出文本和渲染结果供人工复查。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.agent.case_multidim_search.reporting.conversation_report import ConversationReportBuilder


OUTPUT_DIR = ROOT / "tests" / "outputs" / "conversation_pdf_20260917"


def main() -> None:
    """使用包含段落、列表和表格的页面回答验证内容边界。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    answer = (
        "## 统计结论\n\n"
        "2025年5月至7月共记录13次主要气象灾害过程。\n\n"
        "## 主要过程\n\n"
        "| 时段 | 灾害过程 | 主要特征 |\n"
        "| --- | --- | --- |\n"
        "| 5月2日 | 雷暴大风天气过程 | 多地出现雷暴大风 |\n"
        "| 7月28日 | 强对流天气过程 | 局地短时强降水 |\n\n"
        "1. 重点关注短时强降水与雷暴大风的叠加影响。\n"
        "2. 本报告正文与上一轮页面展示内容保持一致。"
    )
    pdf_path = OUTPUT_DIR / "2025-05-to-07-weather-analysis.pdf"
    ConversationReportBuilder().build(answer, pdf_path, "2025年5月至7月气象灾害分析报告")
    extracted = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)
    text_path = OUTPUT_DIR / "extracted.txt"
    text_path.write_text(extracted, encoding="utf-8")
    checks = {
        "pdf_path": str(pdf_path),
        "page_count": len(PdfReader(str(pdf_path)).pages),
        "visible_sentence_count": extracted.count("2025年5月至7月共记录13次主要气象灾害过程"),
        "contains_formal_overview": "综合概况" in extracted,
        "contains_representative_case_section": "代表个例分析" in extracted,
        "contains_formal_conclusion": "综合结论与建议" in extracted,
    }
    (OUTPUT_DIR / "verification.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
