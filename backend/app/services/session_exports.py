"""会话消息导出和公开分享页的纯展示逻辑。"""
from __future__ import annotations

import html
import re


def report_title(question: str, answer: str) -> str:
    """从原问题和可见回答提取稳定标题，不为导出再次调用模型。"""
    years = list(dict.fromkeys(re.findall(r"(20\d{2})年", question)))
    months = [
        int(value)
        for value in re.findall(r"(?<!\d)(\d{1,2})月", question)
        if 1 <= int(value) <= 12
    ]
    if len(years) == 1 and len(months) >= 2:
        return f"{years[0]}年{months[0]}月至{months[-1]}月气象灾害分析报告"
    if len(years) == 1 and len(months) == 1:
        return f"{years[0]}年{months[0]}月气象灾害分析报告"
    if len(years) == 1:
        return f"{years[0]}年气象灾害分析报告"
    heading = re.search(r"^#{1,3}\s+(.+)$", answer, flags=re.MULTILINE)
    if heading:
        value = re.sub(r"[*_`#]", "", heading.group(1)).strip(" ：:。")
        if value:
            return value if value.endswith("报告") else f"{value}报告"
    compact = re.sub(r"[\s，。！？：；,.!?:;]+", "", question)
    return f"{compact[:24]}报告" if compact else "气象灾害分析报告"


def render_shared_message(snapshot: dict) -> str:
    """生成最小只读分享页，并对所有用户文本做 HTML 转义。"""
    title = html.escape(str(snapshot.get("title") or "气象灾害问答"))
    question = html.escape(str(snapshot.get("question") or ""))
    answer = html.escape(str(snapshot.get("answer") or ""))
    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<style>body{margin:0;background:#fbfbfa;color:#292927;"
        "font:15px/1.75 system-ui,sans-serif}"
        "main{max-width:780px;margin:auto;padding:48px 22px}h1{font-size:22px}"
        ".q{background:#eeeeec;border-radius:14px;padding:10px 14px;margin:22px 0;"
        "white-space:pre-wrap}.a{white-space:pre-wrap;overflow-wrap:anywhere}"
        "</style></head><body><main>"
        f"<h1>{title}</h1><div class='q'>{question}</div>"
        f"<div class='a'>{answer}</div></main></body></html>"
    )
