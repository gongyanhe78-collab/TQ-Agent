"""相似个例聊天页面使用的 SSE 事件编码。

SSE（Server-Sent Events）是一种服务器向浏览器单向推送消息的 HTTP 协议。
前端通过 EventSource 接收，每次推送一条事件，浏览器就能实时更新 UI。
本模块只负责把事件数据编码成 SSE 协议要求的文本格式。
"""
from __future__ import annotations

import json
from typing import Any


def encode_sse(event: str, data: dict[str, Any]) -> str:
    """把结构化阶段结果编码为浏览器可逐段读取的 SSE 文本。

    SSE 消息格式规范：
      event: 事件名\n
      data: JSON 数据\n
      \n（空行表示一条消息结束）

    Args:
        event: 事件类型名，如 "stage" / "matched_cases" / "forecast" / "completed" / "error"
        data: 事件的结构化数据字典，会被序列化为 JSON

    Returns:
        符合 SSE 协议的文本字符串，可直接写入 HTTP 响应流

    编码细节：
    - ensure_ascii=False：保留中文，不转义成 \\uXXXX 形式的 Unicode 编码，省流量也便于调试
    - separators=(",", ":")：紧凑 JSON，去掉空格，减小消息体积
    - 末尾两个换行：SSE 协议要求用空行分隔每条消息
    """
    # 把数据字典序列化成紧凑格式的 JSON 字符串
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # 按 SSE 协议格式拼接：事件行 + 数据行 + 结束空行
    return f"event: {event}\ndata: {payload}\n\n"
