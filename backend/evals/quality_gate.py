"""评测质量门禁：离线基线回归与真实模型阈值检查。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.evals.schemas.dataset import EvaluationReport

必需指标 = (
    "route_accuracy",
    "hit_at_1",
    "hit_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "evidence_id_hit_rate",
    "no_result_accuracy",
    "required_fact_coverage",
    "behavior_satisfaction_rate",
    "citation_support_rate",
    "forbidden_claim_count",
    "abstention_accuracy",
)

# 离线固定夹具不是质量模型，只用于锁定 Harness 和指标口径的可重复基线。
离线固定基线 = {
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

# 真实模型门禁只在手动或定时流水线中作为趋势告警，不阻塞日常提交。
真实模型阈值 = {
    "route_accuracy": 0.70,
    "hit_at_1": 0.60,
    "hit_at_3": 0.75,
    "recall_at_5": 0.60,
    "mrr": 0.80,
    "evidence_id_hit_rate": 0.85,
    "no_result_accuracy": 0.95,
    "required_fact_coverage": 0.65,
    "behavior_satisfaction_rate": 0.85,
    "citation_support_rate": 1.0,
    "forbidden_claim_count": 0,
    "abstention_accuracy": 0.95,
}


@dataclass(frozen=True)
class 门禁结果:
    """保存门禁是否通过以及可读的失败原因。"""

    passed: bool
    failures: list[str]


def _指标值(report: EvaluationReport, name: str) -> float:
    """读取数值指标，缺失时视为失败。"""

    value = report.metrics.get(name)
    if isinstance(value, (int, float)):
        return float(value)
    raise KeyError(name)


def 检查离线固定门禁(
    report: EvaluationReport,
    *,
    expected_dataset_version: str,
) -> 门禁结果:
    """校验离线固定夹具的版本、样例规模和确定性基线。"""

    failures: list[str] = []
    if not report.offline_fixture:
        failures.append("离线门禁收到的是真实模型报告")
    if report.case_count != 60 or len(report.results) != 60:
        failures.append(f"样例数必须为 60，实际为 {report.case_count}/{len(report.results)}")
    if report.dataset_version != expected_dataset_version:
        failures.append(
            f"数据集版本不一致：期望 {expected_dataset_version}，实际 {report.dataset_version}"
        )
    for result in report.results:
        if result.error:
            failures.append(f"样例 {result.case_id} 执行异常：{result.error}")
    for name in 必需指标:
        try:
            actual = _指标值(report, name)
        except KeyError:
            failures.append(f"缺少指标 {name}")
            continue
        expected = 离线固定基线[name]
        if abs(actual - expected) > 0.0002:
            failures.append(f"离线基线漂移：{name} 期望 {expected:.4f}，实际 {actual:.4f}")
    return 门禁结果(not failures, failures)


def 检查真实模型门禁(report: EvaluationReport) -> 门禁结果:
    """校验真实模型核心指标是否达到阶段目标。"""

    failures: list[str] = []
    if report.offline_fixture:
        failures.append("真实门禁收到的是离线固定夹具报告")
    if report.case_count <= 0:
        failures.append("真实模型报告样例数必须大于 0")
    for name, threshold in 真实模型阈值.items():
        try:
            actual = _指标值(report, name)
        except KeyError:
            failures.append(f"缺少指标 {name}")
            continue
        if name == "forbidden_claim_count":
            if actual > threshold:
                failures.append(f"{name} 必须为 0，实际为 {actual:.0f}")
        elif actual + 1e-9 < threshold:
            failures.append(f"{name} 低于阈值 {threshold:.2%}，实际 {actual:.2%}")
    return 门禁结果(not failures, failures)


def 从负载检查门禁(
    payload: dict[str, Any],
    *,
    mode: str,
    expected_dataset_version: str,
) -> 门禁结果:
    """把报告 JSON 转为事实对象后执行对应门禁。"""

    report = EvaluationReport.model_validate(payload)
    if mode == "offline":
        return 检查离线固定门禁(report, expected_dataset_version=expected_dataset_version)
    if mode == "real":
        return 检查真实模型门禁(report)
    raise ValueError(f"未知门禁模式：{mode}")

