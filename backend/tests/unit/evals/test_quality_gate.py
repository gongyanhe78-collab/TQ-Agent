"""评测质量门禁与趋势记录测试。"""
from __future__ import annotations

import json
from datetime import datetime

from backend.evals.quality_gate import (
    从负载检查门禁,
    检查离线固定门禁,
    检查真实模型门禁,
)
from backend.evals.schemas.dataset import EvaluationReport
from scripts.record_evaluation_trend import 追加趋势记录, 构建趋势记录

离线指标 = {
    "route_accuracy": 0.0833,
    "hit_at_1": 0.1167,
    "hit_at_3": 0.1167,
    "recall_at_5": 0.1167,
    "recall_at_10": 0.1167,
    "mrr": 0.1167,
    "evidence_id_hit_rate": 0.1167,
    "no_result_accuracy": 0.1167,
    "required_fact_coverage": 0.0,
    "behavior_satisfaction_rate": 0.1667,
    "citation_support_rate": 1.0,
    "forbidden_claim_count": 0,
    "abstention_accuracy": 0.1167,
}


def _报告(*, offline: bool, metrics: dict) -> EvaluationReport:
    """构造最小但结构完整的评测报告。"""

    results = [
        {
            "case_id": f"case-{index}",
            "category": "测试",
            "question": "问题",
            "expected_route": "rag",
            "actual_route": "rag",
            "offline_fixture": offline,
        }
        for index in range(60 if offline else 1)
    ]
    return EvaluationReport.model_validate({
        "git_commit": "abcdef123456",
        "dataset_sha256": "sha",
        "dataset_version": "golden-v1.5",
        "case_count": 60 if offline else 1,
        "model_config": {"chat_model": "qwen3.7-flash"},
        "routing_config": {"mode": "test"},
        "prompt_version": "enterprise-runtime-v1",
        "offline_fixture": offline,
        "started_at": datetime.now().astimezone(),
        "ended_at": datetime.now().astimezone(),
        "metrics": metrics,
        "results": results,
    })


def test_offline_gate_accepts_locked_baseline() -> None:
    """当前固定夹具基线应通过离线门禁。"""

    result = 检查离线固定门禁(_报告(offline=True, metrics=离线指标), expected_dataset_version="golden-v1.5")
    assert result.passed, result.failures


def test_offline_gate_rejects_metric_drift() -> None:
    """指标口径或 Harness 发生漂移时必须失败。"""

    metrics = dict(离线指标, hit_at_1=0.5)
    result = 检查离线固定门禁(_报告(offline=True, metrics=metrics), expected_dataset_version="golden-v1.5")
    assert not result.passed


def test_real_gate_checks_core_thresholds() -> None:
    """真实门禁应同时检查下限、引用支撑和违规断言。"""

    metrics = {
        "route_accuracy": 0.8,
        "hit_at_1": 0.7,
        "hit_at_3": 0.8,
        "recall_at_5": 0.7,
        "mrr": 0.8,
        "evidence_id_hit_rate": 0.9,
        "no_result_accuracy": 0.98,
        "required_fact_coverage": 0.7,
        "behavior_satisfaction_rate": 0.9,
        "citation_support_rate": 1.0,
        "forbidden_claim_count": 0,
        "abstention_accuracy": 0.98,
    }
    assert 检查真实模型门禁(_报告(offline=False, metrics=metrics)).passed
    failed = dict(metrics, hit_at_1=0.2)
    assert not 检查真实模型门禁(_报告(offline=False, metrics=failed)).passed
    fact_failed = dict(metrics, required_fact_coverage=0.5)
    assert not 检查真实模型门禁(_报告(offline=False, metrics=fact_failed)).passed


def test_payload_gate_and_trend_generation(tmp_path) -> None:
    """报告负载可执行门禁并生成机器可读和可阅读的趋势文件。"""

    report = _报告(offline=False, metrics={
        "route_accuracy": 0.8,
        "hit_at_1": 0.7,
        "hit_at_3": 0.8,
        "recall_at_5": 0.7,
        "mrr": 0.8,
        "evidence_id_hit_rate": 0.9,
        "no_result_accuracy": 0.98,
        "required_fact_coverage": 0.7,
        "behavior_satisfaction_rate": 0.9,
        "citation_support_rate": 1.0,
        "forbidden_claim_count": 0,
        "abstention_accuracy": 0.98,
    })
    payload = report.model_dump(mode="json")
    assert 从负载检查门禁(
        payload,
        mode="real",
        expected_dataset_version="golden-v1.5",
    ).passed
    record = 构建趋势记录(payload)
    jsonl_path, markdown_path = 追加趋势记录(record, tmp_path)
    assert json.loads(jsonl_path.read_text(encoding="utf-8"))["dataset_version"] == "golden-v1.5"
    assert "真实模型评测趋势" in markdown_path.read_text(encoding="utf-8")

