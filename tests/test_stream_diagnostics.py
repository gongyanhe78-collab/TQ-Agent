"""统一主 RAG 流式链路诊断脚本。

本脚本只访问已经启动的后端 HTTP 接口，不修改业务代码。
它会记录每个 SSE 事件的到达时间，用于区分模型首字慢、后端缓冲和前端显示问题。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


TEST_DIR = Path(__file__).resolve().parent


def _request_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """发送 JSON 请求并返回响应对象。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _create_session(base_url: str) -> str:
    """创建独立测试会话，避免诊断脚本污染用户已有会话。"""
    result = _request_json(base_url, "/api/sessions", {"title": "流式链路诊断"})
    session_id = str(result.get("session_id") or result.get("id") or "")
    if not session_id:
        raise RuntimeError(f"创建测试会话失败，响应中没有 session_id: {result}")
    return session_id


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="诊断统一主 RAG 的 SSE 流式输出")
    parser.add_argument(
        "--question",
        default="帮我详细分析一下2025年5月2日的灾害情况",
        help="要测试的问题；默认使用一个主 RAG 个例分析问题",
    )
    parser.add_argument(
        "--session-id",
        default="",
        help="已有会话 ID；不提供时自动创建独立测试会话",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("RISK_API_ORIGIN", "http://127.0.0.1:8000"),
        help="后端地址，默认读取 RISK_API_ORIGIN 或 http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--output",
        default="",
        help="输出文件路径；默认写入 tests/stream_diagnostic_时间戳.json",
    )
    return parser.parse_args()


def _event_from_sse(block: str) -> dict[str, Any] | None:
    """解析一个 SSE data 块，忽略 keep-alive 注释和非 JSON 内容。"""
    data_lines = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
    if not data_lines:
        return None
    try:
        value = json.loads("\n".join(data_lines))
    except json.JSONDecodeError:
        return {"type": "invalid_json", "raw": "\n".join(data_lines)}
    return value if isinstance(value, dict) else {"type": "non_object", "value": value}


def run() -> int:
    """执行一次诊断并保存完整原始事件与统计结果。"""
    args = _parse_args()
    started_at = time.perf_counter()
    wall_started = datetime.now().astimezone().isoformat(timespec="milliseconds")
    output_path = Path(args.output) if args.output else TEST_DIR / (
        f"stream_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        session_id = args.session_id or _create_session(args.base_url)
        request_body = {"question": args.question, "session_id": session_id, "top_k": 5, "top_n": 3}
        body = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{args.base_url.rstrip('/')}/api/query/stream",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )

        events: list[dict[str, Any]] = []
        raw_blocks: list[str] = []
        event_buffer = ""
        first_metadata_ms: float | None = None
        first_delta_ms: float | None = None
        delta_count = 0
        delta_chars = 0
        delta_intervals_ms: list[float] = []
        last_delta_at: float | None = None
        done_event: dict[str, Any] | None = None

        with urllib.request.urlopen(request, timeout=180) as response:
            response_started = time.perf_counter()
            while True:
                chunk = response.readline()
                if not chunk:
                    break
                received_at = time.perf_counter()
                text = chunk.decode("utf-8", errors="replace")
                event_buffer += text
                while "\n\n" in event_buffer:
                    block, event_buffer = event_buffer.split("\n\n", 1)
                    raw_blocks.append(block)
                    event = _event_from_sse(block)
                    if event is None:
                        continue
                    elapsed_ms = round((received_at - started_at) * 1000, 2)
                    event_record = {"elapsed_ms": elapsed_ms, "event": event}
                    events.append(event_record)
                    event_type = event.get("type")
                    if event_type == "metadata" and first_metadata_ms is None:
                        first_metadata_ms = elapsed_ms
                    elif event_type == "delta":
                        delta_count += 1
                        text_value = str(event.get("text") or "")
                        delta_chars += len(text_value)
                        if first_delta_ms is None:
                            first_delta_ms = elapsed_ms
                        if last_delta_at is not None:
                            delta_intervals_ms.append(round((received_at - last_delta_at) * 1000, 2))
                        last_delta_at = received_at
                    elif event_type == "done":
                        done_event = event
            response_elapsed_ms = round((time.perf_counter() - response_started) * 1000, 2)

        summary = {
            "started_at": wall_started,
            "base_url": args.base_url,
            "session_id": session_id,
            "question": args.question,
            "event_count": len(events),
            "metadata_count": sum(item["event"].get("type") == "metadata" for item in events),
            "delta_count": delta_count,
            "delta_chars": delta_chars,
            "first_metadata_ms": first_metadata_ms,
            "first_delta_ms": first_delta_ms,
            "delta_intervals_ms": delta_intervals_ms,
            "response_read_ms": response_elapsed_ms,
            "total_client_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "done_performance": (done_event or {}).get("performance", {}),
            "done_event": done_event,
            "diagnosis_hints": [
                "首个 delta 很晚但随后 delta 连续：优先检查云端模型 TTFT。",
                "delta_count 为 1 且内容很长：云端 API 可能实际一次返回大块，或后端未收到增量。",
                "本脚本逐行读取仍集中收到多个事件：检查代理、压缩中间件或旧后端进程。",
                "脚本收到多个 delta 而页面一次显示：检查浏览器 Network、前端构建版本和 onDelta 调用。",
            ],
            "events": events,
            "raw_sse_blocks": raw_blocks,
        }
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: summary[key] for key in (
            "session_id", "event_count", "delta_count", "delta_chars",
            "first_metadata_ms", "first_delta_ms", "response_read_ms", "done_performance",
        )}, ensure_ascii=False, indent=2))
        print(f"诊断完整输出已写入: {output_path}")
        return 0
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {detail}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"诊断失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("请确认后端已启动，并检查 8000 端口是否可访问。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
