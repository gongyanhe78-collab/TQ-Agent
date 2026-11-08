"""把真实模型评测报告归档为可追踪的 JSONL 与 Markdown 趋势。"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def 构建趋势记录(payload: dict[str, Any]) -> dict[str, Any]:
    """从评测报告提取稳定的趋势字段。"""

    return {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "ended_at": str(payload.get("ended_at") or ""),
        "git_commit": str(payload.get("git_commit") or ""),
        "dataset_version": str(payload.get("dataset_version") or ""),
        "dataset_sha256": str(payload.get("dataset_sha256") or ""),
        "case_count": int(payload.get("case_count") or 0),
        "model_config": payload.get("model_config") or {},
        "routing_config": payload.get("routing_config") or {},
        "metrics": payload.get("metrics") or {},
    }


def 追加趋势记录(record: dict[str, Any], trend_dir: Path) -> tuple[Path, Path]:
    """追加 JSONL，并重新生成便于阅读的 Markdown 汇总。"""

    trend_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = trend_dir / "real-evaluation-trends.jsonl"
    markdown_path = trend_dir / "real-evaluation-trends.md"
    with jsonl_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metric_names = (
        "route_accuracy",
        "hit_at_1",
        "hit_at_3",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "evidence_id_hit_rate",
        "required_fact_coverage",
        "behavior_satisfaction_rate",
        "citation_support_rate",
        "forbidden_claim_count",
        "abstention_accuracy",
    )
    lines = [
        "# 真实模型评测趋势",
        "",
        "> 本文件由 `scripts/record_evaluation_trend.py` 自动生成，请勿手工修改。",
        "",
        "| 结束时间 | 提交 | 数据集 | 样例数 | " + " | ".join(metric_names) + " |",
        "| --- | --- | --- | ---: | " + " | ".join("---:" for _ in metric_names) + " |",
    ]
    for item in reversed(records):
        metrics = item.get("metrics") or {}
        values = []
        for name in metric_names:
            raw = metrics.get(name, "")
            if isinstance(raw, (int, float)):
                values.append(f"{float(raw):.4f}")
            else:
                values.append(str(raw))
        lines.append(
            "| {ended_at} | `{git_commit}` | `{dataset_version}` | {case_count} | {values} |".format(
                ended_at=item.get("ended_at", ""),
                git_commit=str(item.get("git_commit", ""))[:12],
                dataset_version=item.get("dataset_version", ""),
                case_count=item.get("case_count", 0),
                values=" | ".join(values),
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    """读取报告并写入趋势目录。"""

    parser = argparse.ArgumentParser(description="记录真实模型评测趋势")
    parser.add_argument("--report", required=True, help="真实模型 EvaluationReport JSON")
    parser.add_argument(
        "--trend-dir",
        default="docs/evals/trends",
        help="趋势文件输出目录",
    )
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if payload.get("offline_fixture"):
        raise SystemExit("趋势记录只接受真实模型评测报告")
    record = 构建趋势记录(payload)
    jsonl_path, markdown_path = 追加趋势记录(record, Path(args.trend_dir))
    print(f"趋势 JSONL：{jsonl_path}")
    print(f"趋势 Markdown：{markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
