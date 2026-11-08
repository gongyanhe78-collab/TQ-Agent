"""本地小规模并发压测脚本：输出 p50/p95、错误率、模型成功率与降级次数。

用途：验证稳定性、超时和降级行为，不宣称生产吞吐量。默认请求公开健康接口，
也可指定登录后的业务接口进行受限压测。压测前后抓取 /metrics 计算模型成功率与
降级次数增量。
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def aggregate_metrics(text: str) -> dict[str, float]:
    """按指标名汇总 Prometheus 文本全部样本，忽略注释与空行。"""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        space = line.rfind(" ")
        if space <= 0:
            continue
        name = line[:space].split("{", 1)[0]
        try:
            value = float(line[space + 1 :])
        except ValueError:
            continue
        totals[name] = totals.get(name, 0.0) + value
    return totals


def label_total(text: str, metric: str, label: str, value: str) -> float:
    """统计指定指标中某个标签取值对应的样本总和。"""
    total = 0.0
    for line in text.splitlines():
        if not line.startswith(metric):
            continue
        if f"{label}=\"{value}\"" not in line:
            continue
        space = line.rfind(" ")
        try:
            total += float(line[space + 1 :])
        except ValueError:
            continue
    return total


def fetch_text(base_url: str, path: str, token: str = "") -> str:
    """GET 请求并返回响应文本，非 2xx 抛出 HTTPError。"""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        headers=headers,
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.read().decode("utf-8")


def send_request(
    base_url: str,
    endpoint: str,
    token: str,
    payload: dict[str, Any] | None,
) -> tuple[float, int]:
    """发送单个请求并返回耗时与状态码；连接失败按 599 记录。"""
    start = time.perf_counter()
    headers: dict[str, str] = {}
    data: bytes | None = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}{endpoint}",
            data=data,
            headers=headers,
            method="POST",
        )
    else:
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}{endpoint}",
            headers=headers,
            method="GET",
        )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except Exception:
        status = 599
    return time.perf_counter() - start, status


def percentile(values: list[float], ratio: float) -> float:
    """返回有序样本的近似分位数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(ratio * len(ordered)) - 1))
    return ordered[index]


def run_load_test(
    base_url: str,
    token: str,
    endpoint: str,
    concurrency: int,
    total: int,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """执行并发请求并汇总 p50/p95、错误率与指标增量。"""
    metrics_before = fetch_text(base_url, "/metrics", token) if not payload else ""
    durations: list[float] = []
    statuses: list[int] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
        futures = [
            pool.submit(send_request, base_url, endpoint, token, payload)
            for _ in range(max(1, int(total)))
        ]
        for future in futures:
            duration, status = future.result()
            durations.append(duration)
            statuses.append(status)
    errors = sum(1 for status in statuses if status >= 400 or status == 599)
    metrics_after = fetch_text(base_url, "/metrics", token) if not payload else ""

    model_success_rate = None
    degradation = None
    if metrics_before and metrics_after:
        before_model = label_total(metrics_before, "model_calls_total", "status", "success")
        before_all = label_total(metrics_before, "model_calls_total", "status", "error") + before_model
        after_model = label_total(metrics_after, "model_calls_total", "status", "success")
        after_all = label_total(metrics_after, "model_calls_total", "status", "error") + after_model
        delta_success = after_model - before_model
        delta_all = after_all - before_all
        model_success_rate = (
            round(delta_success / delta_all, 4) if delta_all > 0 else None
        )
        before_degrade = (
            aggregate_metrics(metrics_before).get("cache_fallbacks_total", 0.0)
            + aggregate_metrics(metrics_before).get("vector_fallbacks_total", 0.0)
        )
        after_degrade = (
            aggregate_metrics(metrics_after).get("cache_fallbacks_total", 0.0)
            + aggregate_metrics(metrics_after).get("vector_fallbacks_total", 0.0)
        )
        degradation = after_degrade - before_degrade

    return {
        "base_url": base_url,
        "endpoint": endpoint,
        "concurrency": max(1, int(concurrency)),
        "total": max(1, int(total)),
        "ok": len(statuses) - errors,
        "errors": errors,
        "error_rate": round(errors / len(statuses), 4) if statuses else 0.0,
        "p50_ms": round(percentile(durations, 0.5) * 1000, 2),
        "p95_ms": round(percentile(durations, 0.95) * 1000, 2),
        "max_ms": round(max(durations) * 1000, 2) if durations else 0.0,
        "model_success_rate": model_success_rate,
        "degradation_count": degradation,
    }


def main() -> int:
    """命令行入口，输出 JSON 报告并给出稳定性结论。"""
    parser = argparse.ArgumentParser(description="本地小规模并发压测")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API 基础地址")
    parser.add_argument("--token", default="", help="可选访问令牌")
    parser.add_argument("--endpoint", default="/api/health", help="压测接口路径")
    parser.add_argument("--concurrency", type=int, default=8, help="并发数")
    parser.add_argument("--total", type=int, default=100, help="请求总数")
    parser.add_argument("--question", default="", help="业务压测时发送的问题文本")
    args = parser.parse_args()

    payload = {"question": args.question} if args.question else None
    report = run_load_test(
        args.base_url,
        args.token,
        args.endpoint,
        args.concurrency,
        args.total,
        payload,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"] > 0:
        print(f"结论：出现 {report['errors']} 个错误，请检查日志与降级路径。")
    elif report["p95_ms"] > 5000:
        print("结论：p95 超过 5 秒，需要关注超时或排队，但仍未发现未捕获异常。")
    else:
        print("结论：压测期间无未捕获异常，错误率为 0，稳定性通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
