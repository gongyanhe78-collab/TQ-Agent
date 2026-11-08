"""从单个 EvaluationReport 生成本地 JSON 与 Markdown 报告。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from backend.evals.schemas.dataset import EvaluationReport


秘密键模式 = re.compile(
    r"(?i)\b(authorization|bearer|api[_-]?key|access[_-]?token|secret|token)\b"
    r"(?:[ \t]*[:=][ \t]*|[ \t]+)"
    r"((?:bearer[ \t]+)?[^\s,;\"'<>]+)"
)
开放密钥模式 = re.compile(r"\bsk-[A-Za-z0-9_\-]{6,}\b")


def 脱敏(文本: str | None) -> str:
    """统一清洗鉴权信息，避免失败样例把秘密写入报告。"""

    if not 文本:
        return ""
    结果 = 开放密钥模式.sub("[已脱敏]", 文本)
    结果 = 秘密键模式.sub(r"\1 [已脱敏]", 结果)
    return 结果


def _脱敏负载(value: object) -> object:
    """递归脱敏 JSON 负载中的所有字符串值。"""

    if isinstance(value, str):
        return 脱敏(value)
    if isinstance(value, list):
        return [_脱敏负载(item) for item in value]
    if isinstance(value, dict):
        return {key: _脱敏负载(item) for key, item in value.items()}
    return value


def write_reports(
    report: EvaluationReport,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """把同一个报告对象分别写为 JSON 与 Markdown。"""

    json_file = Path(json_path)
    markdown_file = Path(markdown_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    markdown_file.parent.mkdir(parents=True, exist_ok=True)

    payload = _脱敏负载(report.model_dump(mode="json"))
    json_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_file.write_text(_渲染Markdown(report), encoding="utf-8")


def _渲染Markdown(report: EvaluationReport) -> str:
    """直接从报告对象读取值渲染 Markdown，不重复计算指标。"""

    lines: list[str] = [
        "# 评测报告",
        "",
        f"- git commit：`{report.git_commit}`",
        f"- 数据集版本：`{report.dataset_version}`",
        f"- 数据集 SHA-256：`{report.dataset_sha256}`",
        f"- Prompt 版本：`{report.prompt_version}`",
        f"- 模型配置：`{脱敏(json.dumps(report.runtime_model_config, ensure_ascii=False, default=str))}`",
        f"- 路由配置：`{脱敏(json.dumps(report.routing_config, ensure_ascii=False, default=str))}`",
        f"- 开始时间：{_时间文本(report.started_at)}",
        f"- 结束时间：{_时间文本(report.ended_at)}",
        f"- 样例数：{report.case_count}",
        f"- 离线固定夹具：{report.offline_fixture}",
        "",
        "## 汇总指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
    ]
    for key, value in report.metrics.items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## 失败明细", ""])
    if report.failures:
        lines.extend(
            [
                "| case_id | 类别 | 期望路由 | 实际路由 | 原因 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for failure in report.failures:
            reasons = "；".join(脱敏(reason) for reason in failure.failure_reasons) or "无"
            lines.append(
                f"| {failure.case_id} | {failure.category} | "
                f"{failure.expected_route} | {failure.actual_route} | {reasons} |"
            )
    else:
        lines.append("无失败样例。")
    return "\n".join(lines) + "\n"


def _时间文本(value: object) -> str:
    """统一输出时间对象文本，兼容已脱敏或空值。"""

    if value is None:
        return ""
    return str(value)
