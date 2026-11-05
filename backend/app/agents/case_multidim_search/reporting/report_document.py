"""按照参考技术报告参数连续编排 A4 PDF 页面。"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from PIL import Image


class ReferenceReportDocument:
    """管理白底技术报告的分页、段落、图片、图注和页码。"""

    PAGE_SIZE = (8.27, 11.69)
    LEFT_MARGIN = 85.1 / 595.3
    RIGHT_MARGIN = 1 - LEFT_MARGIN
    TOP_MARGIN = 754.0 / 841.9
    BOTTOM_MARGIN = 64.0 / 841.9
    BODY_FONT_SIZE = 12.0
    SECTION_FONT_SIZE = 15.0
    SUBSECTION_FONT_SIZE = 14.04
    CAPTION_FONT_SIZE = 11.0
    BODY_LINE_STEP = 23.4 / 841.9
    PARAGRAPH_GAP = 4.0 / 841.9
    TEXT_WIDTH_UNITS = 35.0

    def __init__(self, pdf, plt, fallback_font_path: Path | None = None):
        """初始化 PDF 输出对象、中文字体和连续排版状态。"""
        from matplotlib.font_manager import FontProperties

        body_path = self._first_existing_font(
            Path("C:/Windows/Fonts/simsun.ttc"),
            Path("C:/Windows/Fonts/simfang.ttf"),
            fallback_font_path,
        )
        heading_path = self._first_existing_font(
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("C:/Windows/Fonts/msyh.ttc"),
            body_path,
        )
        self.pdf = pdf
        self.plt = plt
        self.body_font = FontProperties(fname=str(body_path)) if body_path else FontProperties()
        self.heading_font = FontProperties(fname=str(heading_path)) if heading_path else self.body_font.copy()
        self.heading_font.set_weight("bold")
        self.figure = None
        self.cursor_y = self.TOP_MARGIN
        self.page_number = 0
        self.figure_number = 0
        self._new_page()

    def add_report_title(self, title: str, date_text: str, subtitle: str = "") -> None:
        """在首页顶部绘制居中报告标题、检索条件副标题和生成日期。"""
        self.cursor_y = 0.855
        title_lines = self._wrap_text(str(title), limit=24.0)
        for line in title_lines:
            self.figure.text(
                0.5,
                self.cursor_y,
                line,
                ha="center",
                fontsize=21,
                color="#000000",
                fontproperties=self.heading_font,
            )
            self.cursor_y -= 28.0 / 841.9
        self.cursor_y -= 2.0 / 841.9
        self.figure.text(
            0.5,
            self.cursor_y,
            str(date_text),
            ha="center",
            fontsize=self.BODY_FONT_SIZE,
            color="#000000",
            fontproperties=self.body_font,
        )
        self.cursor_y -= 20.0 / 841.9
        subtitle_text = str(subtitle or "").strip()
        if subtitle_text:
            # 副标题只承载检索条件，字号小于主标题，避免报告标题与实际命中个例脱节。
            for raw_line in subtitle_text.splitlines():
                for line in self._wrap_text(raw_line, limit=42.0):
                    self.figure.text(
                        0.5,
                        self.cursor_y,
                        line,
                        ha="center",
                        fontsize=10.5,
                        color="#333333",
                        fontproperties=self.body_font,
                    )
                    self.cursor_y -= 16.0 / 841.9
            self.cursor_y -= 8.0 / 841.9
        else:
            self.cursor_y -= 17.0 / 841.9

    def add_table(self, headers: list[str], rows: list[list[str]], max_rows: int = 14) -> None:
        """绘制白底网格表格，按单元格内容动态换行和增高行高。"""
        if not headers or not rows:
            return
        rows = rows[:max_rows]
        column_count = len(headers)
        table_width = self.RIGHT_MARGIN - self.LEFT_MARGIN
        if column_count <= 3:
            col_widths = [1 / column_count] * column_count
        elif column_count >= 9:
            # PDF 宽表不能沿用网页列宽；前两列收窄，把空间让给中间指标列。
            first_width = 0.105
            second_width = 0.085
            area_width = 0.125
            completeness_width = 0.090
            metric_count = max(column_count - 4, 1)
            metric_width = max(0.064, (1.0 - first_width - second_width - area_width - completeness_width) / metric_count)
            col_widths = [first_width, second_width] + [metric_width] * metric_count + [area_width, completeness_width]
            total = sum(col_widths)
            col_widths = [value / total for value in col_widths]
        else:
            fixed = [0.13, 0.12]
            tail_count = column_count - 3
            tail_width = max(0.10, (1.0 - sum(fixed) - 0.14) / max(tail_count, 1))
            col_widths = fixed + [tail_width] * tail_count + [0.14]
            total = sum(col_widths)
            col_widths = [value / total for value in col_widths]
        wrapped_headers = self._wrap_table_row(headers, col_widths)
        wrapped_rows = [self._wrap_table_row(row, col_widths) for row in rows]
        header_lines = max(cell.count("\n") + 1 for cell in wrapped_headers)
        row_lines = [max(cell.count("\n") + 1 for cell in row) for row in wrapped_rows]
        # 宽表使用表格专用字号，并按换行后的真实行数动态增高，避免表头和内容被截断。
        header_font_size = 6.4 if column_count >= 9 else 7.8
        body_font_size = 6.2 if column_count >= 9 else 7.2
        line_step = 11.8 if column_count >= 9 else 14.0
        header_height = max(34.0, line_step * header_lines + 10.0) / 841.9
        row_heights = [max(30.0, line_step * line_count + 10.0) / 841.9 for line_count in row_lines]
        content_height = header_height + sum(row_heights)
        required = content_height + 10.0 / 841.9
        self._ensure_space(required)
        axis = self.figure.add_axes([self.LEFT_MARGIN, self.cursor_y - required + 7.0 / 841.9, table_width, required])
        axis.axis("off")
        table = axis.table(
            cellText=wrapped_rows,
            colLabels=wrapped_headers,
            cellLoc="left",
            colLoc="left",
            loc="upper left",
            colWidths=col_widths,
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        for (row_index, _col_index), cell in table.get_celld().items():
            cell.set_edgecolor("#CBD5E1")
            cell.set_linewidth(0.6)
            cell.set_facecolor("#F8FAFC" if row_index == 0 else "#FFFFFF")
            cell.set_height((header_height if row_index == 0 else row_heights[row_index - 1]) / content_height)
            cell.PAD = 0.08
            cell_text = cell.get_text()
            cell_text.set_fontproperties(self.heading_font if row_index == 0 else self.body_font)
            cell_text.set_fontsize(header_font_size if row_index == 0 else body_font_size)
            cell_text.set_color("#0F172A")
        self.cursor_y -= required + 7.0 / 841.9

    def add_section(self, title: str) -> None:
        """添加十五磅一级章节标题并在空间不足时分页。"""
        self._ensure_space(42.0 / 841.9)
        self.figure.text(
            self.LEFT_MARGIN + 7.6 / 595.3,
            self.cursor_y,
            str(title),
            fontsize=self.SECTION_FONT_SIZE,
            color="#000000",
            fontproperties=self.heading_font,
        )
        self.cursor_y -= 30.0 / 841.9

    def add_subsection(self, title: str) -> None:
        """添加十四磅二级标题并保持参考报告的标题间距。"""
        self._ensure_space(38.0 / 841.9)
        self.figure.text(
            self.LEFT_MARGIN + 7.0 / 595.3,
            self.cursor_y,
            str(title),
            fontsize=self.SUBSECTION_FONT_SIZE,
            color="#000000",
            fontproperties=self.heading_font,
        )
        self.cursor_y -= 26.5 / 841.9

    def add_paragraph(self, text: str, first_line_indent: bool = True) -> None:
        """按参考正文宽度、十二磅字号和固定行距写入分析段落。"""
        paragraphs = [item.strip() for item in str(text or "").splitlines() if item.strip()]
        for paragraph in paragraphs:
            lines = self._wrap_paragraph(paragraph, first_line_indent=first_line_indent)
            self._ensure_space(len(lines) * self.BODY_LINE_STEP + self.PARAGRAPH_GAP)
            for index, line in enumerate(lines):
                x = self.LEFT_MARGIN
                if index == 0 and first_line_indent:
                    x += 24.0 / 595.3
                self.figure.text(
                    x,
                    self.cursor_y,
                    line,
                    fontsize=self.BODY_FONT_SIZE,
                    color="#000000",
                    fontproperties=self.body_font,
                )
                self.cursor_y -= self.BODY_LINE_STEP
            self.cursor_y -= self.PARAGRAPH_GAP

    def add_numbered_items(self, items: list[str]) -> None:
        """把建议或发现写成连续编号文字而不使用卡片。"""
        for index, item in enumerate(items, start=1):
            self.add_paragraph(f"{index}. {item}", first_line_indent=False)

    def add_figure(
        self,
        path: Path,
        caption: str,
        max_height: float = 0.46,
        figure_number: int | None = None,
        numbered: bool = True,
    ) -> bool:
        """等比例插入图片；数据库原图可使用后端统一编号，统计图可只保留标题。"""
        image_path = Path(path)
        if not self._valid_image(image_path):
            return False
        with Image.open(image_path) as source:
            width_px, height_px = source.size
            image = source.convert("RGB").copy()
        aspect = width_px / max(height_px, 1)
        max_width = self.RIGHT_MARGIN - self.LEFT_MARGIN
        width = max_width
        height = width * self.PAGE_SIZE[0] / aspect / self.PAGE_SIZE[1]
        if height > max_height:
            height = max_height
            width = height * self.PAGE_SIZE[1] * aspect / self.PAGE_SIZE[0]
        caption_lines = self._caption_lines(caption)
        caption_height = len(caption_lines) * 17.0 / 841.9
        gap = 28.0 / 841.9
        required = height + caption_height + gap
        available = self.cursor_y - self.BOTTOM_MARGIN
        if required > available:
            # 当前页还有足够空间时先缩图，避免上一页留下半页空白。
            shrink_height = available - caption_height - gap
            if shrink_height >= 0.24:
                height = shrink_height
                width = height * self.PAGE_SIZE[1] * aspect / self.PAGE_SIZE[0]
                required = height + caption_height + gap
            else:
                self._save_page()
                self._new_page()
                available = self.cursor_y - self.BOTTOM_MARGIN
                height = min(height, available - caption_height - gap)
                width = height * self.PAGE_SIZE[1] * aspect / self.PAGE_SIZE[0]
                required = height + caption_height + gap
        image_bottom = self.cursor_y - height
        axis = self.figure.add_axes([(1 - width) / 2, image_bottom, width, height])
        # PDF 中插入栅格图时使用高质量重采样，避免统计图二次缩放后发糊。
        axis.imshow(image, interpolation="lanczos", resample=True)
        axis.axis("off")
        self.cursor_y = image_bottom - 9.0 / 841.9
        prefix = ""
        if numbered:
            if figure_number is None:
                self.figure_number += 1
                resolved_number = self.figure_number
            else:
                resolved_number = int(figure_number)
                self.figure_number = max(self.figure_number, resolved_number)
            prefix = f"{chr(0x56fe)} {resolved_number} "
        for index, line in enumerate(caption_lines):
            text = prefix + line if index == 0 else line
            self.figure.text(
                0.5,
                self.cursor_y,
                text,
                ha="center",
                fontsize=self.CAPTION_FONT_SIZE,
                color="#000000",
                fontproperties=self.body_font,
            )
            self.cursor_y -= 17.0 / 841.9
        self.cursor_y -= 16.0 / 841.9
        return True

    def finish(self) -> None:
        """保存最后一页并结束连续排版。"""
        if self.figure is not None:
            self._save_page()

    def _new_page(self) -> None:
        """创建新的白底 A4 页面并重置正文游标。"""
        self.page_number += 1
        # 提高页面画布 dpi，避免 PNG 图表进入 PDF 后被低分辨率重采样。
        self.figure = self.plt.figure(figsize=self.PAGE_SIZE, dpi=300, facecolor="#FFFFFF")
        self.cursor_y = self.TOP_MARGIN

    def _save_page(self) -> None:
        """绘制居中页码后保存完整白底页面。"""
        self.figure.text(
            0.5,
            0.032,
            str(self.page_number),
            ha="center",
            fontsize=9,
            color="#000000",
            fontproperties=self.body_font,
        )
        self.pdf.savefig(self.figure, facecolor="#FFFFFF", edgecolor="none", dpi=300)
        self.plt.close(self.figure)
        self.figure = None

    def _ensure_space(self, required: float) -> None:
        """在当前页空间不足时保存页面并新建下一页。"""
        if self.cursor_y - required >= self.BOTTOM_MARGIN:
            return
        if self.figure is not None:
            self._save_page()
        self._new_page()

    def _wrap_text(self, text: str, limit: float) -> list[str]:
        """按指定视觉宽度拆分短文本，用于标题和表格单元格换行。"""
        remaining = str(text or "").strip()
        lines = []
        while remaining:
            line, remaining = self._take_line(remaining, limit)
            lines.append(line.rstrip())
        return lines or [""]

    def _wrap_table_row(self, values: list[str], col_widths: list[float]) -> list[str]:
        """表格内容按列宽换行，防止长灾种或地市文本溢出单元格。"""
        wrapped = []
        for value, width in zip(values, col_widths):
            limit = max(4.0, width * 42.0)
            wrapped.append("\n".join(self._wrap_text(str(value or '—'), limit=limit)))
        return wrapped

    def _wrap_paragraph(self, text: str, first_line_indent: bool) -> list[str]:
        """按中英文混排宽度拆分正文并为首行预留缩进。"""
        remaining = str(text).strip()
        lines = []
        first = True
        while remaining:
            limit = self.TEXT_WIDTH_UNITS - (2.0 if first and first_line_indent else 0.0)
            line, remaining = self._take_line(remaining, limit)
            lines.append(line.rstrip())
            first = False
        return lines or [""]

    def _take_line(self, text: str, limit: float) -> tuple[str, str]:
        """从文本开头截取不超过指定视觉宽度的一行。"""
        width = 0.0
        end = 0
        for index, char in enumerate(text):
            char_width = self._character_width(char)
            if end and width + char_width > limit:
                break
            width += char_width
            end = index + 1
        if end <= 0:
            end = 1
        return text[:end], text[end:].lstrip()

    def _character_width(self, char: str) -> float:
        """估算单个字符在宋体正文中的相对视觉宽度。"""
        if char.isspace():
            return 0.5
        return 1.0 if unicodedata.east_asian_width(char) in {"W", "F", "A"} else 0.55

    def _caption_lines(self, caption: str) -> list[str]:
        """清理旧图号并完整保留图注，避免 PDF 图题以省略号结尾。"""
        figure_char = chr(0x56FE)
        fallback = "原始证据图"
        clean = re.sub(
            r"^" + figure_char + r"\s*\d+(?:\s*[（(][^）)]*[）)])?\s*",
            "",
            str(caption or fallback),
        ).strip()
        return self._wrap_paragraph(clean, first_line_indent=False)

    def _valid_image(self, path: Path) -> bool:
        """验证图片文件存在且能够被 Pillow 正常解码。"""
        try:
            with Image.open(path) as image:
                image.verify()
            return True
        except Exception:
            return False

    def _first_existing_font(self, *paths: Path | None) -> Path | None:
        """按优先级返回第一个存在的中文字体文件。"""
        return next((Path(path) for path in paths if path and Path(path).is_file()), None)

