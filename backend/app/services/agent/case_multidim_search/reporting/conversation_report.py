"""把会话中已经展示的 Markdown 回答排版为简易 PDF。"""
from __future__ import annotations

import logging
import html
import re
import shutil
import subprocess
import time
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.app.services.agent.case_multidim_search.reporting.chart_tool import ChartTool
from backend.app.services.agent.case_multidim_search.reporting.report_document import ReferenceReportDocument


class ConversationReportBuilder:
    """只排版页面可见回答，不读取证据、图片或隐藏分析字段。"""

    def __init__(self, chart_tool: ChartTool | None = None):
        """复用正式报告的中文字体和连续分页能力。"""
        self.chart_tool = chart_tool or ChartTool()

    def build(
        self,
        answer: str,
        output_path: Path,
        report_title: str,
        progress_callback=None,
    ) -> Path:
        """按照上一轮回答的原始顺序排版一次，禁止补充或改写正文。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        title = str(report_title or "气象灾害分析报告").strip()
        content = str(answer or "").strip()
        if not content:
            raise ValueError("上一轮页面回答为空，无法生成简易报告")

        # 优先使用本机 Edge 的打印引擎，和聊天页面共享 HTML/CSS 语义；
        # 无头浏览器不可用时再回退到历史 Matplotlib 排版，保证服务仍能导出。
        edge_path = self._find_edge()
        if edge_path:
            try:
                return self._build_with_browser(
                    content,
                    output_path,
                    title,
                    progress_callback,
                    edge_path,
                )
            except Exception as exc:
                logging.getLogger(__name__).warning("HTML/CSS PDF 渲染失败，回退旧版排版：%s", exc)

        import matplotlib

        logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        self._notify(progress_callback, 20, "正在读取上一轮页面内容")
        blocks = self._parse_markdown(content)
        with PdfPages(output_path) as pdf:
            pdf.infodict()["Title"] = title
            pdf.infodict()["Subject"] = "会话分析结果简易报告"
            document = ReferenceReportDocument(pdf, plt, self.chart_tool._font_path())
            document.add_report_title(title, f"生成日期：{date.today().strftime('%Y年%m月%d日')}")
            self._notify(progress_callback, 45, "正在排版上一轮页面内容")
            for block in blocks:
                self._write_block(document, block)
            self._notify(progress_callback, 92, "正在保存 PDF 文件")
            document.finish()
        return output_path

    def _build_with_browser(
        self,
        content: str,
        output_path: Path,
        title: str,
        progress_callback,
        edge_path: str,
    ) -> Path:
        """使用与聊天页面一致的 Markdown HTML 和打印 CSS 生成 A4 PDF。"""
        self._notify(progress_callback, 20, "正在读取上一轮页面内容")
        output_path = output_path.resolve()
        html_document = self._html_document(content, title)
        # 临时文件必须以 .html 结尾，否则 Edge 会按纯文本打印 HTML/CSS 源码。
        html_path = output_path.with_name(f"{output_path.stem}.tmp.html")
        html_path.write_text(html_document, encoding="utf-8")
        try:
            self._notify(progress_callback, 48, "正在按页面样式排版 PDF")
            # 独立浏览器配置可以避免复用用户已经打开的 Edge 进程后丢失打印任务。
            with TemporaryDirectory(
                prefix="message-pdf-edge-",
                dir=output_path.parent,
                ignore_cleanup_errors=True,
            ) as profile_dir:
                if output_path.exists():
                    output_path.unlink()
                command = [
                    edge_path,
                    "--headless=new",
                    "--disable-gpu",
                    "--disable-extensions",
                    "--no-sandbox",
                    "--no-pdf-header-footer",
                    f"--user-data-dir={profile_dir}",
                    f"--print-to-pdf={output_path}",
                    html_path.resolve().as_uri(),
                ]
                subprocess.run(command, check=True, capture_output=True, timeout=90)
                # Windows 上浏览器进程退出与文件句柄刷新可能有极短时间差。
                for _ in range(20):
                    if output_path.is_file() and output_path.stat().st_size >= 100:
                        break
                    time.sleep(0.1)
            if not output_path.is_file() or output_path.stat().st_size < 100:
                raise RuntimeError("浏览器没有生成有效 PDF")
            self._notify(progress_callback, 92, "正在保存 PDF 文件")
            return output_path
        finally:
            try:
                html_path.unlink()
            except OSError:
                pass

    @staticmethod
    def _find_edge() -> str:
        """定位 Windows Edge；部署到无 Edge 环境时返回空字符串触发兼容回退。"""
        candidates = [
            shutil.which("msedge"),
            shutil.which("chrome"),
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
        return next((item for item in candidates if item and Path(item).is_file()), "")

    @classmethod
    def _html_document(cls, content: str, title: str) -> str:
        """将页面可见 Markdown 转换为安全的静态 HTML，不调用模型也不补充内容。"""
        blocks = cls._parse_markdown(content)
        body: list[str] = []
        for block in blocks:
            block_type = str(block.get("type") or "paragraph")
            if block_type == "heading":
                level = min(max(int(block.get("level") or 2), 1), 6)
                body.append(f"<h{level}>{cls._inline_html(block.get('text', ''))}</h{level}>")
            elif block_type == "table":
                headers = "".join(f"<th>{cls._inline_html(value)}</th>" for value in block.get("headers") or [])
                rows = "".join(
                    "<tr>" + "".join(f"<td>{cls._inline_html(value)}</td>" for value in row) + "</tr>"
                    for row in block.get("rows") or []
                )
                body.append(f"<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>")
            elif block_type == "list":
                body.append(f"<p class=\"list-item\">{cls._inline_html(block.get('text', ''))}</p>")
            else:
                body.append(f"<p>{cls._inline_html(block.get('text', ''))}</p>")
        css_path = Path(__file__).resolve().parents[6] / "frontend" / "src" / "styles" / "answer-content.css"
        css = css_path.read_text(encoding="utf-8") if css_path.is_file() else ""
        # 导出专用规则补足聊天容器的宽度、页边距和打印分页行为。
        css += """
        @page { size: A4; margin: 18mm 17mm 18mm; }
        html, body { background: #fff; margin: 0; padding: 0; }
        .report { max-width: 180mm; margin: 0 auto; }
        .report-title { color: #1f1f1d; font-size: 23px; font-weight: 700; line-height: 1.35; margin: 0 0 8px; }
        .report-date { color: #777771; font-size: 10pt; margin: 0 0 22px; }
        .list-item { margin: 4px 0 4px 18px; }
        """
        return (
            "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title><style>{css}</style></head><body>"
            f"<main class='report'><h1 class='report-title'>{html.escape(title)}</h1>"
            f"<p class='report-date'>生成日期：{date.today().strftime('%Y年%m月%d日')}</p>"
            f"<article class='markdown-body'>{''.join(body)}</article></main></body></html>"
        )

    @staticmethod
    def _inline_html(value: str) -> str:
        """转义正文并保留页面中使用的粗体、斜体、行内代码和链接。"""
        text = html.escape(str(value or ""), quote=False)
        text = re.sub(r"\*\*([^*]+)\*\*|__([^_]+)__", lambda m: f"<strong>{m.group(1) or m.group(2)}</strong>", text)
        text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)|(?<!_)_([^_]+)_(?!_)", lambda m: f"<em>{m.group(1) or m.group(2)}</em>", text)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"<a href=\"\2\">\1</a>", text)
        return text

    def _write_block(self, document: ReferenceReportDocument, block: dict) -> None:
        """将解析后的单个 Markdown 块写入报告。"""
        block_type = str(block.get("type") or "paragraph")
        if block_type == "heading":
            title = self._plain_text(str(block.get("text") or ""))
            if not title:
                return
            if int(block.get("level") or 2) <= 2:
                document.add_section(title)
            else:
                document.add_subsection(title)
            return
        if block_type == "table":
            headers = [self._plain_text(value) for value in block.get("headers") or []]
            rows = [
                [self._plain_text(value) for value in row]
                for row in block.get("rows") or []
            ]
            # ReferenceReportDocument 单次最多绘制十四行，分块写入才能保证不丢内容。
            for offset in range(0, len(rows), 14):
                document.add_table(headers, rows[offset:offset + 14], max_rows=14)
            return
        text = self._plain_text(str(block.get("text") or ""))
        if text:
            document.add_paragraph(text, first_line_indent=block_type != "list")

    @classmethod
    def _parse_markdown(cls, content: str) -> list[dict]:
        """识别标题、列表、段落和 Markdown 表格，同时保持原始出现顺序。"""
        lines = str(content or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        blocks: list[dict] = []
        paragraph: list[str] = []
        index = 0

        def flush_paragraph() -> None:
            """把连续普通文本合并为一个段落。"""
            text = " ".join(item.strip() for item in paragraph if item.strip()).strip()
            if text:
                blocks.append({"type": "paragraph", "text": text})
            paragraph.clear()

        while index < len(lines):
            line = lines[index].strip()
            if not line:
                flush_paragraph()
                index += 1
                continue
            heading = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading:
                flush_paragraph()
                blocks.append({"type": "heading", "level": len(heading.group(1)), "text": heading.group(2)})
                index += 1
                continue
            if cls._is_table_start(lines, index):
                flush_paragraph()
                headers = cls._table_cells(lines[index])
                index += 2
                rows: list[list[str]] = []
                while index < len(lines) and "|" in lines[index]:
                    cells = cls._table_cells(lines[index])
                    if not cells:
                        break
                    if len(cells) < len(headers):
                        cells.extend([""] * (len(headers) - len(cells)))
                    rows.append(cells[:len(headers)])
                    index += 1
                blocks.append({"type": "table", "headers": headers, "rows": rows})
                continue
            list_item = re.match(r"^\s*(?:[-*+]\s+|(\d+)[.)、]\s*)(.+)$", lines[index])
            if list_item:
                flush_paragraph()
                marker = f"{list_item.group(1)}. " if list_item.group(1) else "• "
                blocks.append({"type": "list", "text": marker + list_item.group(2).strip()})
                index += 1
                continue
            paragraph.append(line)
            index += 1
        flush_paragraph()
        return blocks

    @staticmethod
    def _is_table_start(lines: list[str], index: int) -> bool:
        """判断当前行与下一行是否构成 Markdown 表头。"""
        if index + 1 >= len(lines) or "|" not in lines[index]:
            return False
        separator = lines[index + 1].strip().strip("|")
        cells = [cell.strip() for cell in separator.split("|")]
        return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)

    @staticmethod
    def _table_cells(line: str) -> list[str]:
        """拆分 Markdown 表格单元格。"""
        return [cell.strip() for cell in str(line).strip().strip("|").split("|")]

    @staticmethod
    def _plain_text(text: str) -> str:
        """仅移除 Markdown 展示符号，不删除或概括回答内容。"""
        value = str(text or "")
        value = re.sub(r"!\[([^]]*)]\([^)]*\)", r"\1", value)
        value = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", value)
        value = re.sub(r"(`{1,3}|\*\*|__|~~)", "", value)
        return value.strip()

    @staticmethod
    def _notify(callback, percent: int, message: str) -> None:
        """把本地排版进度交给统一入口，回调异常不影响文件生成。"""
        if not callable(callback):
            return
        try:
            callback(percent, message)
        except Exception:
            return
