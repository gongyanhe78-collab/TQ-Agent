"""离线评测命令行入口。"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

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
    parser.add_argument(
        "--real",
        action="store_true",
        help="调用真实云端模型和真实检索链路，不使用离线固定夹具",
    )
    parser.add_argument("--top-k", type=int, default=30, help="真实模式下的文档召回候选数量")
    parser.add_argument("--top-n", type=int, default=5, help="真实模式下重排后保留的证据数量")
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="真实模式下跳过大模型重排序，仅使用向量检索顺序",
    )
    args = parser.parse_args(argv)

    if args.failed_only and args.baseline_report is None:
        parser.error("--failed-only 必须同时提供 --baseline-report")

    dataset = _加载数据集(args.dataset)
    baseline_report = None
    if args.baseline_report:
        baseline_payload = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
        baseline_report = EvaluationReport.model_validate(baseline_payload)

    output_dir = Path(args.output_dir)
    executor = None
    if args.real:
        # 延迟导入，确保离线固定夹具模式不加载任何模型或向量库依赖。
        from backend.evals.runners.real_model import 真实模型执行器

        executor = 真实模型执行器(
            top_k=args.top_k,
            top_n=args.top_n,
            enable_rerank=not args.no_rerank,
            output_dir=output_dir / "runtime",
        )
        harness = EvaluationHarness(
            executor,
            git_commit=_当前提交(repo_root),
            offline_fixture=False,
            routing_config={
                "routes": ["rag", "multidim_search", "similar_case_match", "clarify"],
                "mode": "real_model",
                "top_k": args.top_k,
                "top_n": args.top_n,
                "rerank": not args.no_rerank,
            },
        )
    else:
        harness = EvaluationHarness(
            离线固定执行器,
            git_commit=_当前提交(repo_root),
        )

    try:
        report = harness.run(
            dataset,
            max_concurrency=args.max_concurrency,
            seed=args.seed,
            case_id=args.case_id,
            category=args.category,
            failed_only=args.failed_only,
            baseline_report=baseline_report,
            enforce_case_deadline=not args.real,
        )
    finally:
        if executor is not None:
            executor.close()

    if executor is not None:
        # 真实模式用本次实际调用的模型覆盖数据集清单里的静态模型配置。
        report.runtime_model_config = {
            "provider": "dashscope",
            "chat_model": executor.llm_client.model,
            "intent_model": executor.intent_llm_client.model,
            "embedding_model": executor.embedding_client.model,
            "rerank_model": executor.rerank_client.model if not args.no_rerank else "",
            "temperature": 0.3,
            "seed": args.seed,
        }

    write_reports(report, output_dir / "report.json", output_dir / "report.md")
    _打印摘要(report, output_dir)
    return 0


def _打印摘要(report: Any, output_dir: Path) -> None:
    """在命令行打印关键指标，便于快速查看本次评测结果。"""

    mode = "真实模型" if not report.offline_fixture else "离线固定夹具"
    print(f"评测完成：{mode}，样例数 {report.case_count}")
    for name, value in report.metrics.items():
        if isinstance(value, int):
            print(f"  {name}: {value}")
        else:
            print(f"  {name}: {value:.4f}")
    print(f"报告目录：{output_dir.resolve()}")


if __name__ == "__main__":
    raise SystemExit(main())
