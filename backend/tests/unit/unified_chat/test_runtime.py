"""验证请求级 Agent Runtime 的预算、取消和状态快照契约。"""
from __future__ import annotations

import asyncio
import importlib
from datetime import datetime, timedelta, timezone

import pytest

from backend.app.agents.unified_chat.orchestration_models import (
    AgentBudget,
    BudgetTermination,
    ExecutionContext,
    TerminationReason,
)


def _runtime_class():
    """延迟导入运行时，先让缺失模块表现为测试失败而不是收集错误。"""
    module = importlib.import_module(
        "backend.app.agents.unified_chat.runtime"
    )
    return module.AgentRuntime


def _run(coroutine):
    """无需额外异步插件即可执行单个协程测试。"""
    return asyncio.run(coroutine)


def test_runtime_check_priority_starts_with_client_cancellation() -> None:
    """取消必须优先于截止时间、步骤、Token 和工具白名单。"""

    async def scenario() -> None:
        cancel_event = asyncio.Event()
        cancel_event.set()
        now = datetime(2026, 11, 8, 8, 0, tzinfo=timezone.utc)
        runtime = _runtime_class()(
            AgentBudget(
                max_steps=0,
                max_model_calls=0,
                token_budget=0,
                allowed_tools=frozenset({"search"}),
            ),
            ExecutionContext(
                deadline=now - timedelta(seconds=1),
                cancel_event=cancel_event,
            ),
            clock=lambda: now,
        )

        with pytest.raises(BudgetTermination) as error:
            runtime.check(tool_name="report")

        assert error.value.reason == TerminationReason.CANCELLED_BY_CLIENT
        assert error.value.termination_reason == "cancelled_by_client"

    _run(scenario())


def test_runtime_step_and_model_limits_do_not_increment_on_failure() -> None:
    """步骤和模型调用达到上限时抛错，且失败检查不再增加计数。"""
    runtime_class = _runtime_class()
    step_runtime = runtime_class(
        AgentBudget(max_steps=1, max_model_calls=1)
    )
    step_runtime.begin_subtask("route", "rag")
    step_runtime.count_step()
    step_runtime.count_model_call()
    with pytest.raises(BudgetTermination) as step_error:
        step_runtime.count_step()
    assert step_error.value.reason == TerminationReason.STEP_BUDGET_EXCEEDED
    assert step_runtime.steps == 1

    model_runtime = runtime_class(
        AgentBudget(max_steps=1, max_model_calls=1)
    )
    model_runtime.begin_subtask("route", "rag")
    model_runtime.count_step()
    model_runtime.count_model_call()
    with pytest.raises(BudgetTermination) as model_error:
        model_runtime.count_model_call()
    assert (
        model_error.value.reason
        == TerminationReason.MODEL_CALL_BUDGET_EXCEEDED
    )
    assert model_runtime.model_calls == 1


def test_runtime_checks_deadline_before_step_timeout_and_step_limit() -> None:
    """总截止时间优先于单步超时和步骤上限。"""
    now = datetime(2026, 11, 8, 8, 0, tzinfo=timezone.utc)
    runtime = _runtime_class()(
        AgentBudget(
            max_steps=0,
            step_timeout_seconds=1,
        ),
        ExecutionContext(deadline=now - timedelta(microseconds=1)),
        clock=lambda: now,
    )

    with pytest.raises(BudgetTermination) as error:
        runtime.check()

    assert error.value.reason == TerminationReason.TIMEOUT


def test_runtime_stops_subtask_after_step_timeout() -> None:
    """子任务持续超过单步上限时必须终止，且不增加后续步骤计数。"""
    now = [datetime(2026, 11, 8, 8, 0, tzinfo=timezone.utc)]
    runtime = _runtime_class()(
        AgentBudget(
            max_steps=2,
            step_timeout_seconds=5,
        ),
        clock=lambda: now[0],
    )
    runtime.begin_subtask("search", "rag")
    now[0] += timedelta(seconds=6)

    with pytest.raises(BudgetTermination) as error:
        runtime.check()

    assert error.value.reason == TerminationReason.TIMEOUT
    assert runtime.steps == 0


def test_runtime_token_estimation_uses_existing_cjk_rule() -> None:
    """Token 估算沿用中文逐字、其他字符约四比一的规则。"""
    runtime = _runtime_class()(AgentBudget(token_budget=4))

    consumed = runtime.count_tokens("气象灾害")

    assert consumed == 4
    with pytest.raises(BudgetTermination) as error:
        runtime.count_tokens("abcd")
    assert error.value.reason == TerminationReason.TOKEN_BUDGET_EXCEEDED
    assert runtime.token_usage.total == 4
    assert runtime.token_usage.source == "estimated"


def test_runtime_accepts_actual_model_usage() -> None:
    """模型返回 usage 时应按真实输入输出 Token 计数。"""
    runtime = _runtime_class()(
        AgentBudget(token_budget=5)
    )

    consumed = runtime.count_token_usage(
        input_tokens=2,
        output_tokens=2,
        source="actual",
    )

    assert consumed == 4
    assert runtime.token_usage.input == 2
    assert runtime.token_usage.output == 2
    assert runtime.token_usage.source == "actual"


def test_runtime_rejects_tool_outside_allowlist() -> None:
    """非白名单工具必须在执行前被拒绝。"""
    runtime = _runtime_class()(
        AgentBudget(allowed_tools=frozenset({"search"}))
    )

    with pytest.raises(BudgetTermination) as error:
        runtime.check(tool_name="delete_session")

    assert error.value.reason == TerminationReason.TOOL_NOT_ALLOWED


def test_runtime_snapshot_records_completed_subtask_and_budget() -> None:
    """正常完成时快照应包含请求关联、预算、计数和子任务状态。"""
    runtime = _runtime_class()(
        AgentBudget(
            max_seconds=30,
            max_steps=2,
            max_model_calls=3,
            token_budget=100,
            step_timeout_seconds=10,
        ),
        ExecutionContext(
            request_id="request-runtime-1",
            trace_id="a" * 32,
        ),
    )
    runtime.begin_subtask("route", "rag")
    runtime.count_step()
    runtime.count_model_call()
    runtime.count_tokens("气象")
    runtime.finish_subtask()
    runtime.mark_completed()

    snapshot = runtime.snapshot()

    assert snapshot["request_id"] == "request-runtime-1"
    assert snapshot["trace_id"] == "a" * 32
    assert snapshot["termination_reason"] == "completed"
    assert snapshot["steps"] == 1
    assert snapshot["model_calls"] == 1
    assert snapshot["token_stats"]["total"] == 2
    assert snapshot["token_usage_source"] == "estimated"
    assert snapshot["budget"]["max_steps"] == 2
    assert snapshot["subtask_status"][0]["status"] == "completed"
