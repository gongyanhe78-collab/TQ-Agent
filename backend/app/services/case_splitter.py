"""
案例分割器模块
从 PDF 文本中按章节标题分割，识别并提取气象灾害案例
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.models import ExtractedCase


# 匹配中文序号标题的正则表达式（如"一、xxx" "二、xxx"）
# (?m) 表示多行模式，^ 匹配每行开头
TITLE_PATTERN = re.compile(
    r"(?m)^\s*([一二三四五六七八九十]+)、\s*(.+?)\s*$"
)
# 灾害案例关键词（用于识别案例标题）
CASE_KEYWORDS = (
    "天气过程",
    "暴雪",
    "暴雨",
    "雨雪",
    "降雪",
    "降水",
    "沙尘",
    "寒潮",
    "大风",
    "雷暴",
    "强对流",
    "高温",
    "霜冻",
    "冰冻",
)
# 非案例关键词（用于排除非灾害章节）
NON_CASE_KEYWORDS = ("预报服务情况", "服务情况", "月预报", "旬预报")
# 匹配日期范围的正则表达式（支持"1月1～3日" "1-2日"等格式）
DATE_PATTERN = re.compile(
    r"("
    r"\d+\s*月\s*\d+\s*[—\-~～至]\s*\d+\s*日"
    r"|\d+\s*月\s*\d+\s*日"
    r"|(?:\d+月)?\d+\s*[—\-~～至]\s*\d+\s*日"
    r"|\d+\s*-\s*\d+\s*日"
    r")"
)


@dataclass
class Section:
    """
    文档章节数据模型

    Attributes:
        order_label: 中文序号（如"一"、"二"）
        title: 章节标题
        content: 章节正文内容
    """
    order_label: str
    title: str
    content: str


class CaseSplitter:
    """
    案例分割器
    从 PDF 全文中识别章节标题，过滤出灾害案例章节并提取内容
    """

    def split_cases(self, text: str, source_pdf: str) -> list[ExtractedCase]:
        """
        从文本中分割并提取所有灾害案例

        Args:
            text: PDF 提取的完整文本
            source_pdf: 来源 PDF 文件名

        Returns:
            ExtractedCase 对象列表
        """
        # 先按标题分割所有章节
        sections = self._split_sections(text)
        extracted: list[ExtractedCase] = []
        # 去掉文件扩展名作为基础名称
        base_name = source_pdf.rsplit(".", 1)[0]
        # 遍历每个章节，识别案例
        for index, section in enumerate(sections, start=1):
            # 如果不是案例标题则跳过
            if not self._is_case_title(section.title):
                continue
            # 创建案例对象
            extracted.append(
                ExtractedCase(
                    source_pdf=source_pdf,
                    case_id=f"{base_name}-case-{index:02d}",
                    case_no=index,
                    title=self._normalize_title(section.title),
                    date_range=self._extract_date_range(section.title),
                    content=self._normalize_content(section.content),
                )
            )
        return extracted

    def _split_sections(self, text: str) -> list[Section]:
        """
        将文本按标题分割为章节（内部方法）

        Args:
            text: 完整文本

        Returns:
            Section 对象列表
        """
        # 找到所有标题匹配位置
        matches = list(TITLE_PATTERN.finditer(text))
        sections: list[Section] = []
        # 遍历每个标题，截取标题到下一个标题之间的内容作为正文
        for idx, match in enumerate(matches):
            start = match.end()  # 标题结束位置作为正文开始
            # 下一个标题开始位置作为正文结束（最后一章则到文本末尾）
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            sections.append(
                Section(
                    order_label=match.group(1),  # 中文序号
                    title=match.group(2).strip(),  # 标题文本
                    content=text[start:end].strip(),  # 正文内容
                )
            )
        return sections

    def _is_case_title(self, title: str) -> bool:
        """
        判断标题是否为灾害案例标题（内部方法）
        必须同时满足：不包含非案例关键词 + 包含案例关键词 + 包含日期范围

        Args:
            title: 待判断的标题

        Returns:
            是案例标题返回 True
        """
        # 包含非案例关键词则直接排除
        if any(keyword in title for keyword in NON_CASE_KEYWORDS):
            return False
        # 必须包含案例关键词且能提取出日期范围
        return any(keyword in title for keyword in CASE_KEYWORDS) and bool(
            self._extract_date_range(title)
        )

    def _extract_date_range(self, title: str) -> str:
        """
        从标题中提取日期范围（内部方法）

        Args:
            title: 标题文本

        Returns:
            标准化的日期字符串（无空格），提取失败返回空字符串
        """
        match = DATE_PATTERN.search(title)
        # 移除所有空格后返回
        return re.sub(r"\s+", "", match.group(1)) if match else ""

    def _normalize_title(self, title: str) -> str:
        """
        标准化标题（移除所有空白字符）

        Args:
            title: 原始标题

        Returns:
            无空白的标题
        """
        return re.sub(r"\s+", "", title)

    def _normalize_content(self, content: str) -> str:
        """
        标准化正文内容
        将多个换行压缩为单个换行，移除首尾空白

        Args:
            content: 原始正文

        Returns:
            标准化后的正文
        """
        return re.sub(r"\n{2,}", "\n", content).strip()
