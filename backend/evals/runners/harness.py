"""确定性评测 Harness 与离线固定夹具。"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from backend.evals.loader import GoldenDataset
from backend.evals.metrics.answer import (
    abstention_accuracy,
    behavior_satisfaction_rate,
    citation_support_rate,
    forbidden_claim_count,
    required_fact_coverage,
)
from backend.evals.metrics.retrieval import (
    evidence_id_hit_rate,
    hit_at_k,
    mrr,
    no_result_accuracy,
    recall_at_k,
    route_accuracy,
)
from backend.evals.reports import 脱敏
from backend.evals.schemas.dataset import EvaluationCase, EvaluationCaseResult, EvaluationReport

默认路由配置 = {
    "routes": ["rag", "multidim_search", "similar_case_match", "clarify"],
}

汇总指标键 = [
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
]

允许路由值 = {"rag", "multidim_search", "similar_case_match", "clarify"}


def _安全路由(value: str) -> str:
    """校验路由为允许枚举；非法值返回空串以避免自由文本泄露。"""

    return value if value in 允许路由值 else ""


def _脱敏列表(items: list[str]) -> list[str]:
    """统一脱敏字符串列表中的每一元素。"""

    return [脱敏(item) for item in items]


def 离线固定执行器(case: EvaluationCase, seed: int = 0) -> EvaluationCaseResult:
    """默认离线夹具，只输出空证据与拒答，不伪造真实模型结果。"""

    return EvaluationCaseResult(
        case_id=case.case_id,
        category=str(case.category),
        question=case.question,
        actual_route="clarify",
        actual_case_ids=[],
        actual_chunk_ids=[],
        answer="离线固定夹具未调用真实模型。",
        cited_evidence_ids=[],
        no_result=True,
        abstained=True,
        latency_ms=0,
        offline_fixture=True,
    )


class _样例结果槽:
    """封存单例执行结果与完成事件，供并发线程回填。"""

    def __init__(self, timeout_seconds: float, *, start_now: bool = True) -> None:
        self.事件 = threading.Event()
        self.result: EvaluationCaseResult | None = None
        self.timeout_seconds = max(float(timeout_seconds), 0.001)
        # 固定夹具模式按提交时刻计时，保持既有确定性语义；
        # 真实模式不设置截止时间，避免排队等待被误判为执行超时。
        self.deadline: float | None = (
            time.monotonic() + self.timeout_seconds if start_now else None
        )


class EvaluationHarness:
    """接收可注入 case_runner，输出确定性的汇总报告。"""

    def __init__(
        self,
        case_runner: Callable[[EvaluationCase, int], EvaluationCaseResult | dict[str, Any]],
        *,
        git_commit: str = "unknown",
        routing_config: dict[str, Any] | None = None,
        offline_fixture: bool | None = None,
    ) -> None:
        self.case_runner = case_runner
        self.git_commit = git_commit
        self.routing_config = routing_config or dict(默认路由配置)
        self.offline_fixture_flag = offline_fixture

    def run(
        self,
        dataset: GoldenDataset,
        *,
        max_concurrency: int = 4,
        seed: int = 0,
        case_id: str | None = None,
        category: str | None = None,
        failed_only: bool = False,
        baseline_report: EvaluationReport | None = None,
        enforce_case_deadline: bool = True,
    ) -> EvaluationReport:
        """执行筛选后的样例并聚合为单个报告对象。

        enforce_case_deadline 为 False 时不按单例时限中断等待，供真实模型
        评测使用；真实模型调用由客户端自身的超时与熔断保护。
        """

        if max_concurrency < 1:
            raise ValueError("max_concurrency 必须大于 0")
        if failed_only and baseline_report is None:
            raise ValueError("failed_only 需要同时提供 baseline_report")

        cases = dataset.cases
        if failed_only:
            failed_ids = {item.case_id for item in baseline_report.failures}
            cases = [item for item in cases if item.case_id in failed_ids]
        if case_id is not None:
            cases = [item for item in cases if item.case_id == case_id]
        if category is not None:
            cases = [item for item in cases if item.category == category]

        started_at = datetime.now().astimezone()
        results = self._执行(
            cases,
            seed=seed,
            max_concurrency=max_concurrency,
            enforce_deadline=enforce_case_deadline,
        )
        ended_at = datetime.now().astimezone()

        return EvaluationReport(
            git_commit=self.git_commit,
            dataset_sha256=dataset.dataset_sha256(),
            dataset_version=dataset.manifest.dataset_version,
            case_count=len(results),
            runtime_model_config=dict(dataset.manifest.model_config),
            routing_config=dict(self.routing_config),
            prompt_version=dataset.manifest.prompt_version,
            seed=seed,
            offline_fixture=(
                self.offline_fixture_flag
                if self.offline_fixture_flag is not None
                else bool(results) and all(item.offline_fixture for item in results)
            ),
            started_at=started_at,
            ended_at=ended_at,
            metrics=self._聚合指标(results, cases),
            results=results,
        )

    def _执行(
        self,
        cases: list[EvaluationCase],
        *,
        seed: int,
        max_concurrency: int,
        enforce_deadline: bool = True,
    ) -> list[EvaluationCaseResult]:
        """按固定顺序返回结果，超时或异常都转换为失败样例。"""

        holders: list[_样例结果槽] = []
        信号量 = threading.Semaphore(max_concurrency)
        for case in cases:
            holder = _样例结果槽(
                case.max_latency_ms / 1000.0,
                start_now=enforce_deadline,
            )
            holders.append(holder)
            thread = threading.Thread(
                target=self._执行单例,
                args=(case, seed, 信号量, holder),
                daemon=True,
            )
            thread.start()

        results: list[EvaluationCaseResult] = []
        for case, holder in zip(cases, holders, strict=False):
            if not enforce_deadline:
                # 真实模型模式由客户端自身超时与熔断控制，避免排队被误判超时。
                holder.事件.wait()
                results.append(holder.result)
                continue
            remaining = max(0.0, (holder.deadline or 0.0) - time.monotonic())
            if holder.事件.wait(remaining):
                results.append(holder.result)
            else:
                raw = EvaluationCaseResult(
                    status="failed",
                    error=f"执行超时：超过 {case.max_latency_ms} 毫秒",
                )
                results.append(self._计算并落库(case, raw))
        return results

    def _执行单例(
        self,
        case: EvaluationCase,
        seed: int,
        信号量: threading.Semaphore,
        holder: _样例结果槽,
    ) -> None:
        """在受信号量约束的后台线程中执行单个样例并回填结果。"""

        try:
            with 信号量:
                result = self._调用并兜底(case, seed)
        except Exception as exc:
            result = EvaluationCaseResult(status="failed", error=f"{type(exc).__name__}")
        holder.result = result
        holder.事件.set()

    def _调用并兜底(self, case: EvaluationCase, seed: int) -> EvaluationCaseResult:
        """调用注入执行器并把返回值规范化为样例结果。"""

        try:
            if inspect.iscoroutinefunction(self.case_runner):
                raw = asyncio.run(self.case_runner(case, seed))
            else:
                raw = self.case_runner(case, seed)
        except Exception as exc:
            raw = EvaluationCaseResult(status="failed", error=f"{type(exc).__name__}")
        return self._计算并落库(case, raw)

    def _计算并落库(
        self,
        case: EvaluationCase,
        raw: EvaluationCaseResult | dict[str, Any],
    ) -> EvaluationCaseResult:
        """补全期望字段，计算单例指标与失败原因并返回结果。"""

        显式状态: str | None = None
        try:
            if isinstance(raw, EvaluationCaseResult):
                显式状态 = raw.status if "status" in raw.model_fields_set else None
                result = raw.model_copy(deep=True)
            elif isinstance(raw, dict):
                原始状态 = raw.get("status")
                显式状态 = 原始状态 if 原始状态 in {"passed", "failed", "skipped"} else None
                result = EvaluationCaseResult.model_validate(raw)
            else:
                raise TypeError(f"case_runner 返回了不支持的类型：{type(raw)}")
        except Exception as exc:
            result = EvaluationCaseResult(
                status="failed",
                error=f"case_runner 结果无效：{type(exc).__name__}",
            )
            显式状态 = "failed"

        result.case_id = case.case_id
        result.category = str(case.category)
        result.question = case.question
        result.expected_route = case.expected_route
        result.expected_case_ids = list(case.expected_case_ids)
        result.expected_chunk_ids = list(case.expected_chunk_ids)

        result.actual_route = _安全路由(result.actual_route)
        result.error = 脱敏(result.error) if result.error else None
        metrics, failure_reasons = self._计算指标(case, result)
        result.metrics = metrics
        result.failure_reasons = failure_reasons
        if result.error:
            result.status = "failed"
            result.failure_reasons.append(f"执行失败：{result.error}")
        elif 显式状态 is not None:
            result.status = 显式状态
        else:
            result.status = "failed" if failure_reasons else "passed"

        return self._投影结果(result)

    @staticmethod
    def _投影结果(result: EvaluationCaseResult) -> EvaluationCaseResult:
        """失败样例只保留白名单字段，其余字段统一脱敏。"""

        if result.status != "failed":
            result.answer = 脱敏(result.answer)
            return result
        return EvaluationCaseResult(
            case_id=result.case_id,
            category=result.category,
            question=result.question,
            expected_route=result.expected_route,
            actual_route=result.actual_route,
            expected_case_ids=list(result.expected_case_ids),
            expected_chunk_ids=list(result.expected_chunk_ids),
            actual_case_ids=_脱敏列表(result.actual_case_ids),
            actual_chunk_ids=_脱敏列表(result.actual_chunk_ids),
            answer="",
            cited_evidence_ids=_脱敏列表(result.cited_evidence_ids),
            no_result=result.no_result,
            abstained=result.abstained,
            latency_ms=result.latency_ms,
            offline_fixture=result.offline_fixture,
            status=result.status,
            error=result.error,
            failure_reasons=[脱敏(item) for item in result.failure_reasons],
            metrics=result.metrics,
        )

    @staticmethod
    def _计算指标(
        case: EvaluationCase,
        result: EvaluationCaseResult,
    ) -> tuple[dict[str, float | int], list[str]]:
        """围绕统一证据顺序计算单例指标，并给出可读失败原因。"""

        actual_evidence = list(result.actual_case_ids) + list(result.actual_chunk_ids)
        expected_evidence = list(case.expected_case_ids) + list(case.expected_chunk_ids)
        expected_no_result = not bool(case.expected_case_ids) and not bool(case.expected_chunk_ids)

        metrics: dict[str, float | int] = {
            "route_accuracy": route_accuracy([result.actual_route], [case.expected_route]),
            "hit_at_1": hit_at_k(actual_evidence, expected_evidence, 1),
            "hit_at_3": hit_at_k(actual_evidence, expected_evidence, 3),
            "recall_at_5": recall_at_k(actual_evidence, expected_evidence, 5),
            "recall_at_10": recall_at_k(actual_evidence, expected_evidence, 10),
            "mrr": mrr(actual_evidence, expected_evidence),
            "evidence_id_hit_rate": evidence_id_hit_rate(actual_evidence, expected_evidence),
            "no_result_accuracy": no_result_accuracy(result.no_result, expected_no_result),
            "required_fact_coverage": required_fact_coverage(result.answer, case.required_facts),
            "behavior_satisfaction_rate": behavior_satisfaction_rate(
                result.answer,
                actual_route=result.actual_route,
                actual_evidence=actual_evidence,
                no_result=result.no_result,
                abstained=result.abstained,
                behaviors=case.expected_behaviors,
            ),
            "citation_support_rate": citation_support_rate(
                result.cited_evidence_ids,
                actual_evidence,
            ),
            "forbidden_claim_count": forbidden_claim_count(result.answer, case.forbidden_claims),
            "abstention_accuracy": abstention_accuracy(result.abstained, expected_no_result),
        }

        reasons: list[str] = []
        if result.actual_route != case.expected_route:
            reasons.append(f"路由错误：期望 {case.expected_route}，实际 {result.actual_route}")
        if expected_no_result:
            if not result.no_result:
                reasons.append("遗漏无结果判断")
        elif metrics["evidence_id_hit_rate"] == 0.0:
            reasons.append("证据 ID 未命中期望")
        if case.required_facts and metrics["required_fact_coverage"] < 1.0:
            reasons.append("必含事实覆盖不足")
        if case.expected_behaviors and metrics["behavior_satisfaction_rate"] < 1.0:
            reasons.append("行为期望未满足")
        if metrics["forbidden_claim_count"]:
            reasons.append("出现禁止结论")
        if metrics["abstention_accuracy"] == 0.0:
            reasons.append("拒答判断错误")
        return metrics, reasons

    @staticmethod
    def _聚合指标(
        results: list[EvaluationCaseResult],
        cases: list[EvaluationCase],
    ) -> dict[str, float | int]:
        """按适用样例聚合指标，事实和行为为空时不进入分母。"""

        if not results:
            empty = {
                key: 0.0 for key in 汇总指标键 if key != "forbidden_claim_count"
            }
            empty["forbidden_claim_count"] = 0
            empty["required_fact_case_count"] = 0
            empty["behavior_case_count"] = 0
            return empty

        case_by_id = {case.case_id: case for case in cases}
        fact_results = [
            item
            for item in results
            if case_by_id.get(item.case_id)
            and case_by_id[item.case_id].required_facts
        ]
        behavior_results = [
            item
            for item in results
            if case_by_id.get(item.case_id)
            and case_by_id[item.case_id].expected_behaviors
        ]

        summary: dict[str, float | int] = {}
        for key in 汇总指标键:
            if key == "forbidden_claim_count":
                summary[key] = sum(int(item.metrics.get(key, 0)) for item in results)
                continue
            if key == "required_fact_coverage":
                applicable = fact_results
            elif key == "behavior_satisfaction_rate":
                applicable = behavior_results
            else:
                applicable = results
            if not applicable:
                summary[key] = 0.0
                continue
            summary[key] = sum(
                float(item.metrics.get(key, 0.0)) for item in applicable
            ) / len(applicable)
        summary["required_fact_case_count"] = len(fact_results)
        summary["behavior_case_count"] = len(behavior_results)
        return summary
