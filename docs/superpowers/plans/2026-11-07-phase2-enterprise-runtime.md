# 第二阶段企业化运行时实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变现有气象灾害查询、Agent 编排和 SSE 业务语义的前提下，完成 P2-01 至 P2-07 的评测、运行时预算、可观测性、向量契约、Milvus 混合检索和长期记忆。

**Architecture:** 继续使用模块化单体。API 通过 `AppServices` 组装业务服务；Agent Runtime 在现有编排器上增加请求级预算、取消和终止状态；向量层通过统一契约按 `Milvus -> Chroma -> JSON` 降级；PostgreSQL 始终是长期记忆权威事实源。

**Tech Stack:** Python 3.11、FastAPI、SQLAlchemy、Alembic、PostgreSQL 16.4、Redis 7.2.5、Chroma 1.5.9、Milvus 2.5.18、OpenTelemetry 1.28.2、RAGAS 0.2.15、Vue 3。

**Spec:** `docs/superpowers/specs/2026-11-07-phase2-enterprise-runtime-design.md`

## Global Constraints

- 仅实现 P2-01 至 P2-07，不提前实施第三阶段。
- Milvus 固定为 `milvusdb/milvus:v2.5.18`，Python 客户端固定为 `pymilvus==2.5.18`。
- Jaeger 固定为 `jaegertracing/all-in-one:1.60.0`。
- PostgreSQL 固定为 `public.ecr.aws/docker/library/postgres:16.4-alpine3.20`。
- Redis 固定为 `public.ecr.aws/docker/library/redis:7.2.5-alpine`。
- 禁止使用 `latest`、Milvus v3、`pymilvus[model]`、PyTorch、CUDA 或本地 Embedding 模型。
- PostgreSQL 是业务事实源；Redis 和 Milvus 均不得保存唯一事实。
- 保留 Chroma、JSON 降级和现有 `/api/query`、`/api/agent/query`、`/api/query/stream` 协议。
- 所有文档、代码注释和 docstring 使用中文。
- 所有用户级记忆读取、删除和召回必须绑定 `user_id`。
- 所有新增行为先写失败测试，确认失败原因后再写最小实现。

---

### Task 1: 黄金数据集与 Schema

**Files:**
- Create: `backend/evals/__init__.py`
- Create: `backend/evals/schemas/__init__.py`
- Create: `backend/evals/schemas/dataset.py`
- Create: `backend/evals/datasets/golden_v1.json`
- Create: `backend/evals/datasets/golden_v1_manifest.json`
- Create: `backend/evals/loader.py`
- Test: `backend/tests/unit/evals/test_dataset.py`

**Interfaces:**
- Produces: `EvaluationCase`、`EvaluationManifest`、`GoldenDataset.load(path: Path, cases_path: Path, chunks_path: Path) -> GoldenDataset`
- Produces: `GoldenDataset.reference_errors() -> list[str]` 和 `GoldenDataset.dataset_sha256() -> str`

- [x] **Step

```python
def test_dataset_has_exactly_sixty_unique_cases_and_valid_authority_ids():
    dataset = load_golden_v1()
    assert len(dataset.cases) == 60
    assert len({item.case_id for item in dataset.cases}) == 60
    assert dataset.reference_errors() == []
```

- [x] **Step

Run: `pytest backend/tests/unit/evals/test_dataset.py -q`
Expected: FAIL，原因是评测包和加载器不存在。

- [x] **Step

数据集按 6/12/8/10/8/10/6 分布覆盖路由、检索、统计、RAG、相似匹配、多轮记忆和故障场景；每个期望个例 ID、chunk ID 必须能在 `data/standard_cases1.json` 和 `data/document_index/chunks.json` 中找到。manifest 的 SHA-256 由数据集文件字节计算。

- [x] **Step

Run: `pytest backend/tests/unit/evals/test_dataset.py -q`
Expected: PASS。

### Task 2: 确定性指标、Harness 与报告

**Files:**
- Create: `backend/evals/metrics/retrieval.py`
- Create: `backend/evals/metrics/answer.py`
- Create: `backend/evals/runners/harness.py`
- Create: `backend/evals/runners/ragas_runner.py`
- Create: `backend/evals/reports.py`
- Create: `backend/evals/cli.py`
- Create: `scripts/run_evaluation.py`
- Create: `requirements-eval.txt`
- Test: `backend/tests/unit/evals/test_metrics.py`
- Test: `backend/tests/integration/evals/test_harness.py`

**Interfaces:**
- Consumes: Task 1 的 `EvaluationCase`、`GoldenDataset`
- Produces: `route_accuracy`、`hit_at_k`、`recall_at_k`、`mrr`、`evidence_id_hit_rate`、`no_result_accuracy`
- Produces: `required_fact_coverage`、`citation_support_rate`、`forbidden_claim_count`、`abstention_accuracy`
- Produces: `EvaluationHarness.run(...) -> EvaluationReport` 和 `write_reports(report, json_path, markdown_path)`

- [x] **Step

```python
def test_hit_at_k_and_mrr_use_stable_ranked_evidence():
    assert hit_at_k(["a", "b", "c"], ["c"], 3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["a", "d"], 5) == 0.5
    assert mrr(["a", "b", "c"], ["c"]) == 1 / 3
```

```python
def test_harness_two_runs_are_identical_and_reports_share_one_result(tmp_path):
    harness = EvaluationHarness(fixed_runner)
    first = harness.run(dataset, max_concurrency=1, seed=7)
    second = harness.run(dataset, max_concurrency=1, seed=7)
    assert first.metrics == second.metrics
    write_reports(first, tmp_path / "report.json", tmp_path / "report.md")
    assert json.loads((tmp_path / "report.json").read_text("utf-8"))["metrics"] == first.metrics
```

- [x] **Step

Run: `pytest backend/tests/unit/evals/test_metrics.py backend/tests/integration/evals/test_harness.py -q`
Expected: FAIL，指标与 Harness 尚不存在。

- [x] **Step

Harness 接收可注入 `case_runner`，默认通过 `AppServices.chat_service_for(user_id)` 执行；离线模式使用临时数据目录和测试用户。报告记录 git commit、hash、模型和 Prompt 版本、开始与结束时间、失败明细。JSON 与 Markdown 都从同一 `EvaluationReport` 生成。

- [x] **Step

`ragas_runner.py` 延迟导入 RAGAS 和 LangChain；缺少依赖或 API Key 时返回 `status="skipped"` 及原因。使用 `ragas==0.2.15`、`langchain-openai==0.3.7`，不导入 PyTorch。

- [x] **Step

Run: `pytest backend/tests/unit/evals/test_metrics.py backend/tests/integration/evals/test_harness.py -q`
Expected: PASS。

### Task 3: 请求上下文、Trace、Metrics 和日志关联

**Files:**
- Create: `backend/app/observability/__init__.py`
- Create: `backend/app/observability/context.py`
- Create: `backend/app/observability/logging.py`
- Create: `backend/app/observability/metrics.py`
- Create: `backend/app/observability/tracing.py`
- Create: `backend/app/observability/middleware.py`
- Create: `backend/app/api/routes/observability.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/core/config.py`
- Test: `backend/tests/unit/observability/test_context.py`
- Test: `backend/tests/integration/api/test_observability.py`

**Interfaces:**
- Produces: `current_context()`、`bind_request_context(...)`、`new_request_id()`、`new_trace_id()`
- Produces: `record_request`、`record_route`、`record_retrieval_hit`、`record_model_call`、`record_fallback`、`record_budget_termination`
- Produces: `start_span(name, **attributes)` 和 `/metrics`

- [x] **Step

```python
def test_request_context_is_isolated_and_uses_w3c_trace_id():
    with bind_request_context(user_id="u1"):
        assert len(current_context().trace_id) == 32
        assert current_context().user_id == "u1"
    assert current_context().user_id == ""
```

```python
def test_stream_response_returns_request_and_trace_headers(client):
    response = client.post("/api/query/stream", json={"question": "test"})
    assert response.headers["X-Request-ID"]
    assert response.headers["X-Trace-ID"]
```

- [x] **Step

Run: `pytest backend/tests/unit/observability/test_context.py backend/tests/integration/api/test_observability.py -q`
Expected: FAIL，观测模块和响应头不存在。

- [x] **Step

观测后端不可用时只降级本地指标和日志。标签使用固定枚举，禁止用户输入进入指标标签。默认本地采样率 1.0。

- [x] **Step

中间件生成或接收请求 ID、W3C trace ID，写入响应头，并记录请求数、错误和耗时。

- [x] **Step

Run: `pytest backend/tests/unit/observability/test_context.py backend/tests/integration/api/test_observability.py -q`
Expected: PASS。

### Task 4: Agent Runtime 预算与 SSE 兼容扩展

**Files:**
- Modify: `backend/app/agents/unified_chat/orchestration_models.py`
- Create: `backend/app/agents/unified_chat/runtime.py`
- Modify: `backend/app/services/query/chat_service.py`
- Modify: `backend/app/agents/unified_chat/orchestrator.py`
- Modify: `backend/app/agents/unified_chat/stream_events.py`
- Modify: `backend/app/agents/unified_chat/persistence.py`
- Modify: `backend/app/agents/unified_chat/rag_stream.py`
- Modify: `backend/app/agents/unified_chat/multidim_stream.py`
- Modify: `backend/app/agents/unified_chat/smart_stream.py`
- Modify: `backend/app/agents/unified_chat/report_stream.py`
- Modify: `backend/app/api/routes/agent.py`
- Modify: `backend/app/db/session_repository.py`
- Test: `backend/tests/unit/unified_chat/test_runtime.py`
- Test: `backend/tests/integration/api/test_runtime_stream.py`
- Test: `backend/tests/unit/unified_chat/test_runtime_lifecycle.py`

**Interfaces:**
- Produces: `AgentBudget`、`ExecutionContext`、`SubtaskExecutionState`、`TerminationReason`
- Produces: `AgentRuntime.check(...) -> None`，超限时抛出带 `termination_reason` 的 `BudgetTermination`
- Consumes: Task 3 的请求和 trace 上下文

- [x] **Step 1: 写失败测试**

```python
def test_runtime_stops_at_step_and_model_call_budget():
    runtime = AgentRuntime(AgentBudget(max_steps=1, max_model_calls=1))
    runtime.begin_subtask("route", "rag")
    runtime.count_step()
    runtime.count_model_call()
    with pytest.raises(BudgetTermination) as error:
        runtime.count_step()
    assert error.value.reason == TerminationReason.STEP_BUDGET_EXCEEDED
```

```python
async def test_cancelled_client_does_not_start_new_model_call(orchestrator, client):
    events = [event async for event in orchestrator.stream(payload, is_disconnected=always_true)]
    assert events[-1]["termination_reason"] == "cancelled_by_client"
    assert orchestrator.runtime.model_calls == 0
```

- [x] **Step 2: 运行并确认失败**

Run: `pytest backend/tests/unit/unified_chat/test_runtime.py backend/tests/integration/api/test_runtime_stream.py -q`
Expected: FAIL，运行时预算不存在。

- [x] **Step 3: 实现不可变预算、执行上下文、取消和终止原因**

默认值从 `Settings` 读取：300 秒、50 步、40 次模型调用、150000 Token、单步 90 秒。`ChatService.stream()` 接收可选 `ExecutionContext`，旧调用无需传参。

- [x] **Step 4: 在每个路由和步骤前检查预算，并扩展 SSE 字段**

`metadata`、最终 `metadata` 和 `done` 增加 `request_id`、`trace_id`、`termination_reason`、`subtask_status`；现有事件名和字段不变。

- [x] **Step 5: 把预算快照、终止原因、子任务状态和 Token 统计写入 `agent_runs.metadata`**

- [x] **Step 6: 运行测试**

Run: `pytest backend/tests/unit/unified_chat/test_runtime.py backend/tests/integration/api/test_runtime_stream.py backend/tests/unit/unified_chat/test_orchestrator.py backend/tests/unit/unified_chat/test_runtime_lifecycle.py -q`
Expected: PASS。

**实施结果（2026-11-08）**
- 四个业务流统一通过运行生命周期辅助方法登记和收口，预算终止不再被业务兜底吞掉。
- 普通异常、业务 `error` 事件和结构化回答预算终止都会把当前运行写为 `failed`，并保留 trace、预算快照和终止原因。
- SQLAlchemy 存储在同一运行再次启动时会同步刷新独立 `trace_id` 列。
- 定向回归为 `48 passed`；后端全量回归为 `385 passed, 48 subtests passed`。

### Task 5: 统一向量存储契约与 Chroma/JSON 适配

**Files:**
- Create: `backend/app/services/vector/__init__.py`
- Create: `backend/app/services/vector/contracts.py`
- Create: `backend/app/services/vector/json_document_store.py`
- Create: `backend/app/services/vector/chroma_document_store.py`
- Create: `backend/app/services/vector/fallback.py`
- Create: `backend/app/services/vector/factory.py`
- Modify: `backend/app/services/document_store.py`
- Modify: `backend/app/services/container.py`
- Modify: `backend/app/services/query/document_retrieval.py`
- Modify: `backend/app/core/config.py`
- Test: `backend/tests/unit/services/vector/test_contracts.py`
- Test: `backend/tests/integration/storage/test_vector_contract.py`

**Interfaces:**
- Produces: `DocumentVectorStore`、`MemoryVectorStore`、`VectorQuery`、`VectorHit`、`CollectionInfo`
- Produces: `build_document_vector_store(settings, embedding_client) -> DocumentVectorStore`
- Preserves: `query(embedding, top_k)`、`get_chunk`、`list_chunks`、`list_chunk_ids`、`collection_info`

- [x] **Step

```python
def test_json_and_chroma_adapters_satisfy_the_same_document_contract(tmp_path):
    for store in make_contract_stores(tmp_path):
        store.upsert_chunks([sample_chunk])
        assert store.get_chunk(sample_chunk.chunk_id).content == sample_chunk.content
        assert sample_chunk.chunk_id in store.list_chunk_ids()
        assert store.collection_info()["schema_version"] == "v1"
        store.close()
        store.close()
```

- [x] **Step

Run: `pytest backend/tests/unit/services/vector/test_contracts.py backend/tests/integration/storage/test_vector_contract.py -q`
Expected: FAIL，统一契约不存在。

- [x] **Step

JSON 和 Chroma 都从权威 chunk JSON 读取正文；查询结果带后端、集合、模式和真实得分，不把词面降级伪装成向量命中。

- [x] **Step

按 `VECTOR_BACKEND` 选择起点，失败顺序固定为 `Milvus -> Chroma -> JSON`。每次降级记录真实模式与原因。

- [x] **Step

Run: `pytest backend/tests/unit/services/vector/test_contracts.py backend/tests/integration/storage/test_vector_contract.py backend/tests/unit/unified_chat/test_orchestrator.py -q`
Expected: PASS。

### Task 6: Milvus 混合检索、Compose 和重建脚本

**Files:**
- Create: `backend/app/services/vector/milvus_document_store.py`
- Create: `backend/app/services/vector/milvus_memory_store.py`
- Create: `backend/app/services/vector/rebuild.py`
- Create: `scripts/rebuild_vector_index.py`
- Create: `scripts/compare_vector_backends.py`
- Modify: `backend/app/services/vector/factory.py`
- Modify: `backend/app/core/config.py`
- Modify: `compose.yaml`
- Modify: `.env.example`
- Modify: `requirements.txt`
- Test: `backend/tests/unit/services/vector/test_milvus_expr.py`
- Test: `backend/tests/integration/storage/test_milvus_adapter.py`

**Interfaces:**
- Consumes: Task 5 的 `DocumentVectorStore` 和 `MemoryVectorStore`
- Produces: `MilvusDocumentStore.upsert_chunks/query/rebuild/health/close`
- Produces: 参数化 `case_ids` 过滤和 `RRFRanker(k=60)` 融合

- [x] **Step

```python
def test_target_case_filter_is_parameterized_and_never_falls_back_to_global_scope():
    expression = build_case_filter(["FST2025-3-std-case-001"])
    assert "ARRAY_CONTAINS_ANY" in expression
    assert "FST2025-3-std-case-001" not in expression
```

- [x] **Step

Run: `pytest backend/tests/unit/services/vector/test_milvus_expr.py backend/tests/integration/storage/test_milvus_adapter.py -q`
Expected: FAIL，Milvus 适配器不存在。

- [x] **Step

文档集合使用固定字段、中文分析器、BM25 Function、稠密与稀疏双路 Top-30、RRF 融合。存在目标个例时只在其 `case_ids` 内检索，无命中返回空列表。记忆查询强制 `user_id` 过滤。

- [x] **Step

Milvus 使用嵌入式 etcd、本地存储、`/healthz` 健康检查、`2 CPU / 2 GB` 限制和 `risk-master-milvus-data` 卷。`.env.example` 设置 `COMPOSE_PROFILES=vector`。

- [x] **Step

重建脚本支持 `--dry-run`、`--collection documents|memories|all`，输出数量、ID 差异、耗时和失败明细。对比脚本从同一黄金数据集运行 Chroma 和 Milvus，输出 JSON 与 Markdown。

- [x] **Step

Run: `pytest backend/tests/unit/services/vector/test_milvus_expr.py backend/tests/integration/storage/test_milvus_adapter.py backend/tests/integration/storage/test_vector_contract.py -q`
Expected: PASS。
Run: `pytest backend/tests/unit/services/vector/test_milvus_expr.py backend/tests/integration/storage/test_milvus_adapter.py backend/tests/integration/storage/test_vector_contract.py -q`
Expected: PASS。

**实施结果（2026-11-08）：**
- 完成 `milvus_document_store.py`、`milvus_memory_store.py`、`rebuild.py`、`scripts/rebuild_vector_index.py`（支持 `--dry-run` 与 `documents|memories|all`）和 `scripts/compare_vector_backends.py`。
- `compose.yaml` 新增 `milvus`（`vector` profile）与 `jaeger`（`observability` profile）；`.env.example` 默认 `COMPOSE_PROFILES=vector`、`MILVUS_URI=http://milvus:19530`。
- 容器实测：文档向量重建 317/317 与权威 JSON 完全一致（缺失 0、多余 0）。
- 修复两个真实缺陷并补充单元测试：`list_chunk_ids` 分页（Milvus 单次查询窗口上限 16384）与 `collection_info` 行数改用 `count(*)` 查询（`get_collection_stats.row_count` 不可靠）。


### Task 7: PostgreSQL 长期记忆、缓存和用户接口

**Files:**
- Create: `backend/alembic/versions/0002_phase2_memory_fields.py`
- Create: `backend/app/repositories/__init__.py`
- Create: `backend/app/repositories/memory_repository.py`
- Create: `backend/app/services/memory_service.py`
- Create: `backend/app/api/routes/memories.py`
- Modify: `backend/app/db/models.py`
- Modify: `backend/app/cache/context_cache.py`
- Modify: `backend/app/services/container.py`
- Modify: `backend/app/agents/unified_chat/orchestration_models.py`
- Modify: `backend/app/agents/unified_chat/orchestrator.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/unit/services/test_memory_service.py`
- Test: `backend/tests/integration/storage/test_long_term_memory.py`
- Test: `backend/tests/integration/api/test_memories_api.py`

**Interfaces:**
- Produces: `MemoryRepository.list_active(user_id, ...)`、`MemoryRepository.soft_delete(user_id, memory_id)`、`MemoryRepository.rebuildable_active(user_id)`
- Produces: `MemoryService.context_for_user(user_id)`、`MemoryService.extract_candidates(...)`、`MemoryService.delete(...)`
- Produces: `GET /api/memories` 和 `DELETE /api/memories/{memory_id}`

- [x] **Step

```python
def test_user_b_cannot_read_or_delete_user_a_memory(repository):
    memory = repository.create(user_id="user-a", content="偏好关注暴雨")
    assert repository.get(user_id="user-b", memory_id=memory.memory_id) is None
    assert repository.soft_delete(user_id="user-b", memory_id=memory.memory_id) is False
    assert repository.get(user_id="user-a", memory_id=memory.memory_id).status == "active"
```

```python
def test_delete_marks_postgres_and_removes_milvus_and_redis(memory_service):
    result = memory_service.delete(user_id="user-a", memory_id="memory-001")
    assert result["status"] == "deleted"
    assert memory_service.vector_store.deleted == [("user-a", "memory-001")]
    assert memory_service.cache.invalidated_user_ids == ["user-a"]
```

- [x] **Step

Run: `pytest backend/tests/unit/services/test_memory_service.py backend/tests/integration/storage/test_long_term_memory.py backend/tests/integration/api/test_memories_api.py -q`
Expected: FAIL，长期记忆字段和服务不存在。

- [x] **Step

为 `memory_items` 增加 `source_message_id`、`version`、`metadata`、`deleted_at`。仓储所有 get/list/delete 都接收 `user_id`。

- [x] **Step

只从明确表达偏好、关注区域、常用灾种、分析习惯和可复用结论中提取候选。Milvus 不可用时读取 PostgreSQL active 记忆、画像和摘要，标记 `fallback_structured`。

- [x] **Step

当前用户记忆上下文合并到普通 RAG 回答；召回失败不阻塞主回答。接口只返回当前用户可见字段，不暴露向量。

- [x] **Step

Run: `pytest backend/tests/unit/services/test_memory_service.py backend/tests/integration/storage/test_long_term_memory.py backend/tests/integration/api/test_memories_api.py backend/tests/integration/storage/test_user_isolation.py -q`
Expected: PASS。
Run: `pytest backend/tests/unit/services/test_memory_service.py backend/tests/integration/storage/test_long_term_memory.py backend/tests/integration/api/test_memories_api.py backend/tests/integration/storage/test_user_isolation.py -q`
Expected: PASS。

**实施结果（2026-11-08）：**
- 完成 `0002_phase2_memory_fields` 迁移（`source_message_id`、`version`、`metadata`、`deleted_at`），迁移测试 8 项通过。
- 完成 `MemoryRepository`（create 默认 `active`、软删除保留审计行）、`MemoryService`（候选抽取、结构化降级、删除联动 PostgreSQL/Milvus/Redis）与 `GET/DELETE /api/memories`。
- 容器实测：跨用户读取隔离、跨用户删除返回 404、软删除后 `deleted_at` 已写入且列表不再返回。


### Task 8: 全量回归与容器验证

**Files:**
- Modify: `README.md`
- Modify: `startEnd.md`
- Test: 全部后端测试
- Test: 前端生产构建
- Test: Compose 配置和运行服务

- [x] **Step

Run: `python -m compileall backend/app backend/tests backend/evals`
Expected: exit 0。

- [x] **Step

Run: `pytest backend/tests -q`
Expected: 全部 PASS。

- [x] **Step

Run: `python scripts/run_evaluation.py --dataset backend/evals/datasets/golden_v1.json --max-concurrency 1 --output-dir .tmp/eval-run-1`
Run: `python scripts/run_evaluation.py --dataset backend/evals/datasets/golden_v1.json --max-concurrency 1 --output-dir .tmp/eval-run-2`
Expected: 两次确定性路由与检索指标一致。

- [x] **Step

Run: `npm run build`，工作目录 `frontend`
Expected: exit 0。

- [x] **Step

Run: `docker compose config`
Run: `docker compose up -d --build`
Run: `docker compose ps`
Expected: PostgreSQL、Redis、API、前端和 Milvus 健康；Jaeger 仅在 `observability` profile 时启动。
Expected: PostgreSQL、Redis、API、前端和 Milvus 健康；Jaeger 仅在 `observability` profile 时启动。

**实施结果（2026-11-08）：**
- 后端全量回归 `424 passed / 3 skipped / 48 subtests passed`（3 个 skipped 为真实 Milvus 集成用例，本机 Milvus 未常驻运行时跳过）。
- 黄金数据集两次运行（`.tmp/eval-run-1`、`.tmp/eval-run-2`）路由与检索指标完全一致，`dataset_sha256` 相同。
- 前端 `npm run build` 成功；Compose 构建后 `postgres/redis/milvus/api/frontend` 全部 healthy，`docker compose --env-file .env.example config` 通过。
- 降级链实测：Milvus 停机并重启 API 后自动回退 Chroma（317 键、检索正常），恢复 Milvus 后回到主后端。
- 浏览器冒烟：登录、创建会话、自然语言流式输出（Milvus 检索 65 个证据片段）、刷新后历史恢复均通过；首次加载无 Cookie 时的 `/api/auth/refresh` 401 属正常行为。


- [x] **Step

验证 Milvus 停机后按 `Milvus -> Chroma -> JSON` 降级；重建后 chunk 数量和 ID 集合与权威 JSON 一致。

- [x] **Step

验证登录、创建会话、自然语言流式输出、历史恢复和记忆删除不回归。

## 自检

- [x] 
- [x] 
- [x] 
- [x] 
