"""请求级 Agent Runtime，统一执行预算、取消和子任务状态记录。"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta

from backend.app.agents.unified_chat.orchestration_models import (
    AgentBudget,
    BudgetTermination,
    ExecutionContext,
    SubtaskExecutionState,
    TerminationReason,
)


@dataclass(slots=True)
class TokenUsage:
    """一次请求累计的 Token 用量及来源。"""

    input: int = 0
    output: int = 0
    total: int = 0
    source: str = "estimated"


class AgentRuntime:
    """在一个流式请求内集中执行取消、超时和预算检查。"""

    def __init__(
        self,
        budget: AgentBudget | None = None,
        execution_context: ExecutionContext | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.budget = budget or AgentBudget()
        self._clock = clock or (lambda: datetime.now(UTC))
        context = execution_context or ExecutionContext(
            started_at=self._clock(),
        )
        if context.deadline is None:
            context = replace(
                context,
                deadline=context.started_at
                + timedelta(seconds=max(0.0, self.budget.max_seconds)),
            )
        self.execution_context = context
        self.steps = 0
        self.model_calls = 0
        self.token_usage = TokenUsage()
        self._termination_reason = ""
        self._subtask_status: list[SubtaskExecutionState] = []
        self._current_subtask: SubtaskExecutionState | None = None

    @property
    def termination_reason(self) -> str:
        """返回当前终止原因，未结束时为空字符串。"""
        return self._termination_reason

    def check(self, *, tool_name: str | None = None) -> None:
        """按固定优先级检查取消、截止时间、单步超时和全部预算。"""
        if (
            self.execution_context.cancel_event is not None
            and self.execution_context.cancel_event.is_set()
        ):
            self._fail(TerminationReason.CANCELLED_BY_CLIENT)

        now = self._now()
        deadline = self.execution_context.deadline
        if deadline is not None and now >= deadline:
            self._fail(TerminationReason.TIMEOUT)

        if self._current_subtask is not None:
            elapsed = now - self._current_subtask.started_at
            if elapsed >= timedelta(
                seconds=max(0.0, self.budget.step_timeout_seconds)
            ):
                self._fail(TerminationReason.TIMEOUT)

        if self.steps >= self.budget.max_steps:
            self._fail(TerminationReason.STEP_BUDGET_EXCEEDED)
        if self.model_calls >= self.budget.max_model_calls:
            self._fail(TerminationReason.MODEL_CALL_BUDGET_EXCEEDED)
        if self.token_usage.total >= self.budget.token_budget:
            self._fail(TerminationReason.TOKEN_BUDGET_EXCEEDED)

        allowed_tools = self.budget.allowed_tools
        if (
            tool_name
            and allowed_tools is not None
            and tool_name not in allowed_tools
        ):
            self._fail(TerminationReason.TOOL_NOT_ALLOWED)

    def count_step(self) -> int:
        """检查通过后记录一个编排步骤，失败检查不增加计数。"""
        self._check_time_boundaries()
        if self.steps >= self.budget.max_steps:
            self._fail(TerminationReason.STEP_BUDGET_EXCEEDED)
        self.steps += 1
        return self.steps

    def count_model_call(self) -> int:
        """检查通过后记录一次外部模型调用，失败检查不增加计数。"""
        self._check_time_boundaries()
        if self.model_calls >= self.budget.max_model_calls:
            self._fail(TerminationReason.MODEL_CALL_BUDGET_EXCEEDED)
        self.model_calls += 1
        if self._current_subtask is not None:
            self._current_subtask.model_calls += 1
        return self.model_calls

    def count_tokens(self, text: str) -> int:
        """按项目既有中文字符规则估算并累计 Token。"""
        return self._add_token_usage(
            estimated_tokens=self.estimate_tokens(text),
            source="estimated",
        )

    def count_token_usage(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        source: str = "actual",
    ) -> int:
        """累计模型返回的真实 Token 用量，缺省按实际来源标记。"""
        input_value = max(0, int(input_tokens))
        output_value = max(0, int(output_tokens))
        return self._add_token_usage(
            input_tokens=input_value,
            output_tokens=output_value,
            source=str(source or "actual"),
        )

    def begin_subtask(self, subtask_id: str, route: str) -> None:
        """开始记录一个路由或耗时子任务。"""
        self.check()
        if self._current_subtask is not None:
            self.finish_subtask()
        self._current_subtask = SubtaskExecutionState(
            subtask_id=str(subtask_id),
            route=str(route),
            started_at=self._now(),
        )
        self._subtask_status.append(self._current_subtask)

    def finish_subtask(
        self,
        reason: TerminationReason | str = TerminationReason.COMPLETED,
    ) -> None:
        """结束当前子任务并记录耗时和终止原因。"""
        state = self._current_subtask
        if state is None:
            return
        reason_value = (
            reason.value
            if isinstance(reason, TerminationReason)
            else str(reason)
        )
        state.status = (
            "completed"
            if reason_value == TerminationReason.COMPLETED.value
            else "terminated"
        )
        state.termination_reason = reason_value
        state.elapsed_ms = max(
            0,
            int(
                (
                    self._now() - state.started_at
                ).total_seconds()
                * 1000
            ),
        )
        self._current_subtask = None

    def mark_completed(self) -> None:
        """标记请求正常完成；已经终止的请求保留原始终止原因。"""
        if self._current_subtask is not None:
            self.finish_subtask()
        if not self._termination_reason:
            self._termination_reason = TerminationReason.COMPLETED.value

    def mark_terminated(
        self,
        reason: TerminationReason | str,
    ) -> None:
        """记录不可覆盖的终止原因，供非异常分支复用。"""
        reason_value = (
            reason.value
            if isinstance(reason, TerminationReason)
            else str(reason)
        )
        if self._termination_reason not in {
            "",
            TerminationReason.COMPLETED.value,
        }:
            return
        self._termination_reason = reason_value
        if self._current_subtask is not None:
            self.finish_subtask(reason_value)

    def snapshot(self) -> dict:
        """生成可写入 SSE、日志和 agent_runs.metadata 的 JSON 安全快照。"""
        budget = asdict(self.budget)
        budget["allowed_tools"] = (
            sorted(self.budget.allowed_tools)
            if self.budget.allowed_tools is not None
            else None
        )
        return {
            "request_id": self.execution_context.request_id,
            "trace_id": self.execution_context.trace_id,
            "termination_reason": self.termination_reason,
            "steps": self.steps,
            "model_calls": self.model_calls,
            "token_stats": {
                "input": self.token_usage.input,
                "output": self.token_usage.output,
                "total": self.token_usage.total,
            },
            "token_usage_source": self.token_usage.source,
            "budget": budget,
            "subtask_status": [
                {
                    **asdict(state),
                    "started_at": state.started_at.isoformat(),
                }
                for state in self._subtask_status
            ],
        }

    @staticmethod
    def estimate_tokens(value: str) -> int:
        """沿用会话记忆的中日韩逐字、其他字符约四比一估算规则。"""
        text = str(value or "")
        cjk = len(re.findall(r"[\u3400-\u9fff]", text))
        other = len(re.sub(r"[\s\u3400-\u9fff]", "", text))
        return cjk + (max(1, other // 4) if other else 0)

    def _check_time_boundaries(self) -> None:
        """检查取消、总截止时间和当前子任务超时。"""
        if (
            self.execution_context.cancel_event is not None
            and self.execution_context.cancel_event.is_set()
        ):
            self._fail(TerminationReason.CANCELLED_BY_CLIENT)

        now = self._now()
        deadline = self.execution_context.deadline
        if deadline is not None and now >= deadline:
            self._fail(TerminationReason.TIMEOUT)

        if self._current_subtask is not None:
            elapsed = now - self._current_subtask.started_at
            if elapsed >= timedelta(
                seconds=max(0.0, self.budget.step_timeout_seconds)
            ):
                self._fail(TerminationReason.TIMEOUT)

    def _add_token_usage(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated_tokens: int | None = None,
        source: str,
    ) -> int:
        """在确认新总量未越界后一次性写入 Token 统计。"""
        added_input = max(0, int(input_tokens))
        added_output = max(0, int(output_tokens))
        if estimated_tokens is None:
            added_total = added_input + added_output
        else:
            added_total = max(0, int(estimated_tokens))
            added_output = added_total
        previous_total = self.token_usage.total
        next_total = self.token_usage.total + added_total
        if next_total > self.budget.token_budget:
            self._fail(TerminationReason.TOKEN_BUDGET_EXCEEDED)

        self.token_usage.input += added_input
        self.token_usage.output += added_output
        self.token_usage.total = next_total
        if previous_total == 0:
            self.token_usage.source = source
        elif source == "estimated":
            self.token_usage.source = "estimated"
        elif self.token_usage.source != "estimated":
            self.token_usage.source = source
        return added_total

    def _fail(self, reason: TerminationReason) -> None:
        """写入终止状态并抛出一致的预算异常。"""
        self.mark_terminated(reason)
        raise BudgetTermination(reason)

    def _now(self) -> datetime:
        """读取注入时钟，并兼容带时区与不带时区的历史时间值。"""
        value = self._clock()
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
