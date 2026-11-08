"""黄金数据集与清单的 Pydantic 数据模型。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

评测类别 = Literal[
    "路由",
    "检索",
    "统计",
    "RAG",
    "相似匹配",
    "多轮记忆",
    "故障场景",
]
评测路由 = Literal[
    "rag",
    "multidim_search",
    "similar_case_match",
    "clarify",
]


class _模型配置别名模型(BaseModel):
    """兼容 Pydantic 保留字段名与评测 JSON 字段名的基类。"""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def __getattribute__(self, name: str) -> Any:
        """让实例的 model_config 指向评测数据中的模型配置。"""

        if name == "model_config":
            return super().__getattribute__("runtime_model_config")
        return super().__getattribute__(name)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """默认按 JSON 字段别名导出，避免泄露内部兼容字段名。"""

        kwargs.setdefault("by_alias", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        """默认按 JSON 字段别名导出字符串。"""

        kwargs.setdefault("by_alias", True)
        return super().model_dump_json(**kwargs)


BehaviorCheckKind = Literal[
    "route_equals",
    "answer_contains_any",
    "answer_contains_all",
    "evidence_hit",
    "no_result",
    "abstained",
]


class BehaviorCheck(BaseModel):
    """一条可由评测 Harness 确定性执行的行为检查。"""

    model_config = ConfigDict(extra="forbid")

    kind: BehaviorCheckKind
    values: list[str] = Field(default_factory=list)


class ExpectedBehavior(BaseModel):
    """人工定义的行为期望及其可执行检查集合。"""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    checks: list[BehaviorCheck] = Field(min_length=1)

class EvaluationCase(_模型配置别名模型):
    """单条黄金评测样例。"""

    case_id: str = Field(min_length=1)
    category: 评测类别
    question: str = Field(min_length=1)
    conversation_setup: list[dict[str, Any]]
    expected_route: 评测路由
    expected_case_ids: list[str]
    expected_chunk_ids: list[str]
    required_facts: list[str]
    expected_behaviors: list[ExpectedBehavior] = Field(default_factory=list)
    forbidden_claims: list[str]
    max_latency_ms: int = Field(gt=0)
    tags: list[str]
    knowledge_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    runtime_model_config: dict[str, Any] = Field(
        validation_alias="model_config",
        serialization_alias="model_config",
    )


class EvaluationManifest(_模型配置别名模型):
    """黄金数据集版本与完整性清单。"""

    dataset_version: str = Field(min_length=1)
    created_at: date
    category_distribution: dict[评测类别, int]
    knowledge_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    runtime_model_config: dict[str, Any] = Field(
        validation_alias="model_config",
        serialization_alias="model_config",
    )
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


评测运行状态 = Literal["passed", "failed", "skipped"]


class EvaluationCaseResult(BaseModel):
    """单条评测样例的执行结果和确定性指标。"""

    model_config = ConfigDict(extra="forbid")

    case_id: str = ""
    category: str = ""
    question: str = ""
    expected_route: str = ""
    actual_route: str = ""
    expected_case_ids: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    actual_case_ids: list[str] = Field(default_factory=list)
    actual_chunk_ids: list[str] = Field(default_factory=list)
    answer: str = ""
    cited_evidence_ids: list[str] = Field(default_factory=list)
    no_result: bool = False
    abstained: bool = False
    latency_ms: int = Field(default=0, ge=0)
    offline_fixture: bool = False
    status: 评测运行状态 = "passed"
    error: str | None = None
    failure_reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, float | int] = Field(default_factory=dict)


class EvaluationReport(_模型配置别名模型):
    """一次评测运行的汇总事实对象，供 JSON 与 Markdown 共用。"""

    git_commit: str = ""
    dataset_sha256: str = ""
    dataset_version: str = ""
    case_count: int = Field(default=0, ge=0)
    runtime_model_config: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="model_config",
        serialization_alias="model_config",
    )
    routing_config: dict[str, Any] = Field(default_factory=dict)
    prompt_version: str = ""
    seed: int | None = None
    offline_fixture: bool = False
    started_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())
    ended_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())
    metrics: dict[str, float | int] = Field(default_factory=dict)
    results: list[EvaluationCaseResult] = Field(default_factory=list)

    @property
    def failures(self) -> list[EvaluationCaseResult]:
        """按当前报告对象直接筛出失败样例，不重新计算指标。"""

        return [result for result in self.results if result.status == "failed"]
