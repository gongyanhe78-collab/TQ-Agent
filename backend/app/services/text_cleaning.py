"""
文本清理工具模块
提供 PDF 文本规范化、标题规范化等功能
"""
from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """
    规范化从 PDF 提取的文本
    统一换行符、移除特殊字符、压缩多余空格和空行

    Args:
        text: 原始文本

    Returns:
        规范化后的文本
    """
    # 统一换行符为 \n，移除 BOM 标记
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("﻿", "")
    # 将全角空格替换为半角空格
    cleaned = cleaned.replace("　", " ")
    # 将多个空格或制表符压缩为单个空格
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    # 将 3 个以上连续换行压缩为 2 个（保留段落分隔）
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # 移除首尾空白
    return cleaned.strip()


def normalize_title(title: str) -> str:
    """
    规范化标题文本
    移除所有空白字符，用于标题比较

    Args:
        title: 原始标题

    Returns:
        无空白的标题字符串
    """
    return re.sub(r"\s+", "", title.strip())
