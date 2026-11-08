"""确定性 Harness、报告和 RAGAS 跳过路径测试。"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from backend.evals.cli import main as cli_main
from backend.evals.loader import GoldenDataset
from backend.evals.reports import write_reports
from backend.evals.runners.harness import EvaluationHarness
from backend.evals.runners.ragas_runner import run_ragas
from backend.evals.schemas.dataset import EvaluationCase, EvaluationCaseResult


仓库根目录 = Path(__file__).resolve().parents[4]
黄金数据集路径 = 仓库根目录 / "backend" / "evals" / "datasets" / "golden_v1.json"
标准个例路径 = 仓库根目录 / "data" / "standard_cases1.json"
文档切片路径 = 仓库根目录 / "data" / "document_index" / "chunks.json"


def 加载黄金数据集() -> GoldenDataset:
    """加载仓库内的权威黄金数据集，不触网也不访问生产服务。"""

    return GoldenDataset.load(黄金数据集路径, 标准个例路径, 文档切片路径)


def 固定执行器(case: EvaluationCase, seed: int) -> EvaluationCaseResult:
    """返回确定性结果，其中一条样例固定路由错误用于失败筛选。"""

    if case.case_id == "golden-v1-retrieval-001":
        actual_route = "rag"
    else:
        actual_route = case.expected_route
    expected_no_result = not case.expected_case_ids and not case.expected_chunk_ids
    return EvaluationCaseResult(
        case_id=case.case_id,
        category=case.category,
        actual_route=actual_route,
        actual_case_ids=list(case.expected_case_ids),
        actual_chunk_ids=list(case.expected_chunk_ids),
        answer="；".join(case.required_facts),
        cited_evidence_ids=list(case.expected_chunk_ids),
        no_result=expected_no_result,
        abstained=expected_no_result,
        latency_ms=0,
        offline_fixture=True,
    )


def test_harness_two_runs_are_identical_and_reports_share_one_result(
    tmp_path: Path,
) -> None:
    """相同 seed 连续运行必须一致，JSON 与 Markdown 共享同一报告对象。"""

    dataset = 加载黄金数据集()
    harness = EvaluationHarness(固定执行器, git_commit="testing")
    first = harness.run(dataset, max_concurrency=1, seed=7)
    second = harness.run(dataset, max_concurrency=1, seed=7)

    assert first.metrics == second.metrics

    write_reports(first, tmp_path / "report.json", tmp_path / "report.md")
    payload = json.loads((tmp_path / "report.json").read_text("utf-8"))
    assert payload["metrics"] == first.metrics

    markdown = (tmp_path / "report.md").read_text("utf-8")
    assert "route_accuracy" in markdown


def test_failed_result_redacts_secrets_from_json(tmp_path: Path) -> None:
    """失败样例必须脱敏 API Key、Bearer token 与 Authorization 头。"""

    dataset = 加载黄金数据集()
    case = dataset.cases[0]

    def 泄漏执行器(_case: EvaluationCase, _seed: int) -> dict[str, object]:
        return {
            "status": "failed",
            "error": "Authorization: Bearer sk-auth-token-123 failed",
            "answer": "Bearer topsecret123 api_key=abcdef123456",
        }

    harness = EvaluationHarness(泄漏执行器, git_commit="testing")
    report = harness.run(
        dataset,
        max_concurrency=1,
        seed=4,
        case_id=case.case_id,
    )
    write_reports(report, tmp_path / "report.json", tmp_path / "report.md")

    text = (tmp_path / "report.json").read_text("utf-8")
    for secret in ["sk-auth-token-123", "topsecret123", "abcdef123456"]:
        assert secret not in text
    assert report.results[0].answer == ""


def test_failed_route_secret_redacted_in_both_formats(tmp_path: Path) -> None:
    """actual_route 中的秘密必须在 JSON 与 Markdown 双格式中被清理。"""

    dataset = 加载黄金数据集()
    case = dataset.cases[0]
    secret = "topsecret123"

    def 泄漏路由执行器(_case: EvaluationCase, _seed: int) -> EvaluationCaseResult:
        return EvaluationCaseResult(
            status="failed",
            actual_route=f"Authorization: Bearer {secret}",
            answer=f"Bearer {secret}",
            error=f"api_key={secret}",
        )

    harness = EvaluationHarness(泄漏路由执行器, git_commit="testing")
    report = harness.run(dataset, max_concurrency=1, seed=9, case_id=case.case_id)
    write_reports(report, tmp_path / "report.json", tmp_path / "report.md")

    json_text = (tmp_path / "report.json").read_text("utf-8")
    markdown_text = (tmp_path / "report.md").read_text("utf-8")
    assert secret not in json_text
    assert secret not in markdown_text
    assert report.results[0].actual_route == ""


def test_explicit_runner_status_is_preserved() -> None:
    """runner 明确给出的 failed/skipped 状态不得被指标推导覆盖为 passed。"""

    dataset = 加载黄金数据集()
    case = dataset.cases[0]

    def 跳过执行器(_case: EvaluationCase, _seed: int) -> EvaluationCaseResult:
        return EvaluationCaseResult(
            status="skipped",
            actual_route=_case.expected_route,
            actual_chunk_ids=list(_case.expected_chunk_ids),
            answer="；".join(_case.required_facts),
            cited_evidence_ids=list(_case.expected_chunk_ids),
            no_result=False,
            abstained=False,
        )

    harness = EvaluationHarness(跳过执行器, git_commit="testing")
    report = harness.run(dataset, max_concurrency=1, seed=5, case_id=case.case_id)
    assert report.results[0].status == "skipped"


def test_timeout_returns_before_slow_runner_finishes() -> None:
    """超时墙钟必须显著小于 runner 的阻塞时间，且不等待工作线程结束。"""

    dataset = 加载黄金数据集()
    case = dataset.cases[0].model_copy(update={"max_latency_ms": 50})

    def 慢执行器(_case: EvaluationCase, _seed: int) -> EvaluationCaseResult:
        time.sleep(0.25)
        return EvaluationCaseResult(
            status="passed",
            actual_route=case.expected_route,
            actual_chunk_ids=list(case.expected_chunk_ids),
            answer="；".join(case.required_facts),
            cited_evidence_ids=list(case.expected_chunk_ids),
            no_result=False,
            abstained=False,
        )

    harness = EvaluationHarness(慢执行器, git_commit="testing")
    started = time.perf_counter()
    results = harness._执行([case], seed=6, max_concurrency=1)
    elapsed = time.perf_counter() - started

    assert len(results) == 1
    assert results[0].status == "failed"
    assert elapsed < 0.18


def test_multi_case_timeout_uses_monotonic_deadline() -> None:
    """多案例低并发下的总墙钟不逐案累加，而由提交时单调 deadline 约束。"""

    dataset = 加载黄金数据集()
    cases = [dataset.cases[0].model_copy(update={"max_latency_ms": 50}) for _ in range(3)]

    def 慢执行器(_case: EvaluationCase, _seed: int) -> EvaluationCaseResult:
        time.sleep(0.12)
        return EvaluationCaseResult(
            status="passed",
            actual_route=_case.expected_route,
            actual_chunk_ids=list(_case.expected_chunk_ids),
            answer="；".join(_case.required_facts),
            cited_evidence_ids=list(_case.expected_chunk_ids),
            no_result=False,
            abstained=False,
        )

    harness = EvaluationHarness(慢执行器, git_commit="testing")
    started = time.perf_counter()
    results = harness._执行(cases, seed=10, max_concurrency=1)
    elapsed = time.perf_counter() - started

    assert len(results) == 3
    assert all(item.status == "failed" for item in results)
    assert elapsed < 0.12


def test_harness_filters_by_case_id_and_category() -> None:
    """case_id 和 category 过滤只保留匹配样例。"""

    dataset = 加载黄金数据集()
    harness = EvaluationHarness(固定执行器, git_commit="testing")

    by_case = harness.run(dataset, max_concurrency=1, seed=1, case_id="golden-v1-route-001")
    assert [item.case_id for item in by_case.results] == ["golden-v1-route-001"]

    by_category = harness.run(dataset, max_concurrency=1, seed=1, category="路由")
    expected_ids = {item.case_id for item in dataset.cases if item.category == "路由"}
    assert {item.case_id for item in by_category.results} == expected_ids


def test_failed_only_uses_baseline_report_failures() -> None:
    """failed_only 必须从基线报告精确筛出失败 case_id。"""

    dataset = 加载黄金数据集()
    harness = EvaluationHarness(固定执行器, git_commit="testing")
    first = harness.run(dataset, max_concurrency=1, seed=2)

    failed_ids = {item.case_id for item in first.failures}
    assert failed_ids == {"golden-v1-retrieval-001"}

    second = harness.run(
        dataset,
        max_concurrency=1,
        seed=2,
        failed_only=True,
        baseline_report=first,
    )
    assert {item.case_id for item in second.results} == failed_ids


def test_failed_case_keeps_expected_actual_and_reason() -> None:
    """失败明细应保留问题、期望、实际和可解释原因。"""

    dataset = 加载黄金数据集()
    harness = EvaluationHarness(固定执行器, git_commit="testing")
    report = harness.run(dataset, max_concurrency=1, seed=2)
    failure = report.failures[0]

    assert failure.case_id == "golden-v1-retrieval-001"
    assert failure.expected_route == "multidim_search"
    assert failure.actual_route == "rag"
    assert failure.question
    assert any("路由" in reason for reason in failure.failure_reasons)


def test_ragas_skips_without_api_key_or_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """RAGAS 缺少 Key 或依赖时必须返回可反序列化的 skipped。"""

    samples = [{"question": "问题", "answer": "答案", "contexts": ["上下文"]}] * 10

    no_key = run_ragas(samples, api_key="")
    assert no_key["status"] == "skipped"
    assert "API Key" in no_key["reason"]

    monkeypatch.setitem(sys.modules, "ragas", None)
    no_dependency = run_ragas(samples, api_key="dummy")
    assert no_dependency["status"] == "skipped"
    assert "RAGAS" in no_dependency["reason"] or "依赖" in no_dependency["reason"]


def test_ragas_requires_between_ten_and_twenty_samples() -> None:
    """最终样本数小于 10 时必须跳过，10 条边界可以继续后续校验。"""

    item = {"question": "问题", "answer": "答案", "contexts": ["上下文"]}

    nine = run_ragas([item] * 9, api_key="")
    assert nine["status"] == "skipped"
    assert "不足" in nine["reason"]

    ten = run_ragas([item] * 10, api_key="")
    assert ten["status"] == "skipped"
    assert "API Key" in ten["reason"]


def test_markdown_includes_model_and_routing_config(tmp_path: Path) -> None:
    """Markdown 元数据应展示模型配置与路由配置，且与 JSON 同源。"""

    dataset = 加载黄金数据集()
    harness = EvaluationHarness(固定执行器, git_commit="testing")
    report = harness.run(dataset, max_concurrency=1, seed=8, case_id="golden-v1-route-001")
    write_reports(report, tmp_path / "report.json", tmp_path / "report.md")

    markdown = (tmp_path / "report.md").read_text("utf-8")
    assert "模型配置" in markdown
    assert "路由配置" in markdown


def test_cli_default_offline_writes_reports_and_marks_fixture(tmp_path: Path) -> None:
    """默认 CLI 走离线固定夹具，标记 offline_fixture 且不触网。"""

    output_dir = tmp_path / "out"
    exit_code = cli_main(
        [
            "--dataset",
            str(黄金数据集路径),
            "--max-concurrency",
            "1",
            "--seed",
            "3",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    payload = json.loads((output_dir / "report.json").read_text("utf-8"))
    assert payload["offline_fixture"] is True
    assert (output_dir / "report.md").exists()


def test_cli_failed_only_requires_baseline_report() -> None:
    """未提供基线报告时不得把 failed_only 当作全量失败执行。"""

    with pytest.raises(SystemExit):
        cli_main(
            [
                "--dataset",
                str(黄金数据集路径),
                "--failed-only",
            ]
        )


def test_run_evaluation_script_resolves_backend_from_repo_root() -> None:
    """脚本直接执行时也必须能把项目根目录加入模块搜索路径。"""

    script = 仓库根目录 / "scripts" / "run_evaluation.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "dataset" in completed.stdout
