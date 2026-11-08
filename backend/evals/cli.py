"""离线评测命令行入口。"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from backend.evals.loader import GoldenDataset
from backend.evals.reports import write_reports
from backend.evals.runners.harness import EvaluationHarness, 离线固定执行器
from backend.evals.schemas.dataset import EvaluationReport


def _仓库根目录() -> Path:
    """返回项目根目录，避免依赖当前工作目录。"""

    return Path(__file__).resolve().parents[2]


def _当前提交(repo_root: Path) -> str:
    """读取当前 git commit；失败时返回 unknown，绝不触网。"""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _加载数据集(dataset_path: str | None) -> GoldenDataset:
    """加载指定数据集，并使用仓库内固定权威引用文件。"""

    repo_root = _仓库根目录()
    if dataset_path is None:
        dataset_file = repo_root / "backend" / "evals" / "datasets" / "golden_v1.json"
    else:
        dataset_file = Path(dataset_path)
    return GoldenDataset.load(
        dataset_file,
        repo_root / "data" / "standard_cases1.json",
        repo_root / "data" / "document_index" / "chunks.json",
    )


def main(argv: list[str] | None = None) -> int:
    """执行离线评测命令并返回进程退出码。"""

    repo_root = _仓库根目录()
    parser = argparse.ArgumentParser(description="运行离线确定性评测并生成报告")
    parser.add_argument("--dataset", type=str, default=None, help="黄金数据集 JSON 路径")
    parser.add_argument("--case-id", type=str, default=None, help="只重放指定样例")
    parser.add_argument("--category", type=str, default=None, help="只评测指定类别")
    parser.add_argument("--failed-only", action="store_true", help="只重放基线报告中的失败样例")
    parser.add_argument(
        "--baseline-report",
        type=str,
        default=None,
        help="读取失败样例的上一条 JSON 报告路径",
    )
    parser.add_argument("--max-concurrency", type=int, default=4, help="并发上限")
    parser.add_argument("--output-dir", type=str, default="eval-results", help="输出目录")
    parser.add_argument("--seed", type=int, default=0, help="固定执行器随机种子")
    args = parser.parse_args(argv)

    if args.failed_only and args.baseline_report is None:
        parser.error("--failed-only 必须同时提供 --baseline-report")

    dataset = _加载数据集(args.dataset)
    baseline_report = None
    if args.baseline_report:
        baseline_payload = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
        baseline_report = EvaluationReport.model_validate(baseline_payload)

    harness = EvaluationHarness(
        离线固定执行器,
        git_commit=_当前提交(repo_root),
    )
    report = harness.run(
        dataset,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        case_id=args.case_id,
        category=args.category,
        failed_only=args.failed_only,
        baseline_report=baseline_report,
    )
    output_dir = Path(args.output_dir)
    write_reports(report, output_dir / "report.json", output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
