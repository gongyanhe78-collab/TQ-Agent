# 气象灾害个例智能分析系统第二阶段企业化运行时设计

## 1. 文档定位

本文是 `企业化过渡实施方案.md` 第 7 章的技术落地规格。业务目标、产出位置和验收门槛仍以实施方案为准；本文只补齐当前仓库中的接口边界、固定版本、默认配置、数据流和测试方法，不重新定义第二阶段范围。

本规格覆盖 P2-01 至 P2-07，不包含第三阶段的备份恢复演练、CI 平台接入、压测、安全加固和 Kubernetes 改造。

## 2. 总体约束

1. PostgreSQL 保存用户、会话、消息、运行记录、画像、长期记忆和摘要等权威事实。
2. Redis 只保存可重建的短期上下文、缓存、锁和任务状态。
3. JSON 文档切片和 PostgreSQL 长期记忆是向量索引的重建来源，Chroma 与 Milvus 都不是唯一事实源。
4. 现有 `/api/query`、`/api/agent/query`、`/api/query/stream` 及前端消费的 SSE 字段保持兼容，新增字段只做向后兼容扩展。
5. 所有 Markdown、代码注释和 docstring 使用中文。
6. 默认方案保持单机、轻量、模块化单体，不引入 Kafka、Kubernetes、独立模型服务或 PyTorch。
7. Milvus 固定为 `milvusdb/milvus:v2.5.18`，Python 客户端固定为 `pymilvus==2.5.18`，禁止使用 v3、`latest` 和 `pymilvus[model]`。
8. PostgreSQL 继续固定为 `public.ecr.aws/docker/library/postgres:16.4-alpine3.20`，Redis 继续固定为 `public.ecr.aws/docker/library/redis:7.2.5-alpine`。

## 3. 目标架构

第二阶段新增四个相互独立但可组合的能力层：

```text
API / SSE
  -> Agent Runtime（预算、取消、终止原因、子任务状态）
  -> Trace / Metrics（request_id、trace_id、span、日志关联）
  -> ChatService / 现有领域 Agent
  -> DocumentVectorStore / MemoryVectorStore
       -> Milvus（默认）
       -> Chroma（兼容）
       -> JSON 或 PostgreSQL 结构化降级

Evaluation Harness
  -> 复用 AppServices 和 ChatService
  -> 注入隔离数据库、固定模型夹具或缓存结果
  -> 输出确定性指标、RAGAS 补充指标和 JSON/Markdown 报告
```

依赖方向保持为 API -> 业务服务 -> Agent -> 存储和外部集成。观测、评测和向量适配器不得反向依赖 API 层。

## 4. P2-01 Evaluation Harness 与黄金数据集

### 4.1 目录与模型

新增以下结构：

```text
backend/evals/
├── schemas/
│   ├── __init__.py
│   └── dataset.py
├── datasets/
│   ├── golden_v1.json
│   └── golden_v1_manifest.json
├── runners/
│   ├── __init__.py
│   ├── harness.py
│   └── ragas_runner.py
├── metrics/
│   ├── __init__.py
│   ├── retrieval.py
│   └── answer.py
└── reports.py
```

`dataset.py` 使用 Pydantic 模型定义评测样例、数据集清单、单例执行结果和汇总报告。字段至少覆盖实施方案列出的 `case_id`、`category`、`question`、`conversation_setup`、`expected_route`、`expected_case_ids`、`expected_chunk_ids`、`required_facts`、`forbidden_claims`、`max_latency_ms` 和 `tags`，另增加 `knowledge_version`、`prompt_version` 和 `model_config`。

### 4.2 数据规模与覆盖

第一版 `golden_v1.json` 固定为 60 条样例，不允许用重复问题凑数：

| 类别 | 数量 | 重点 |
|---|---:|---|
| 路由 | 6 | 主 RAG、多维检索、相似匹配、澄清和复合任务 |
| 检索 | 12 | 时间、空间、灾种、复合条件、文档原文和无结果 |
| 统计 | 8 | 计数、聚合、排序、筛选和指标口径 |
| RAG | 10 | 原文问答、证据引用、必含事实和禁止结论 |
| 相似匹配 | 8 | 相似过程、跨个例对比和边界匹配 |
| 多轮记忆 | 10 | 个例承接、条件修改、报告复用和用户偏好 |
| 故障场景 | 6 | 无知识、模型不可用、向量降级和跨用户隔离 |

所有 `expected_case_ids` 和 `expected_chunk_ids` 必须存在于 `data/standard_cases1.json` 和 `data/document_index/chunks.json`，数据集加载时执行引用完整性检查。无法由当前权威数据验证的期望值不得写入黄金数据集。

`golden_v1_manifest.json` 保存数据集版本、创建日期、类别分布、知识库版本、Prompt 版本、模型配置和数据集文件 SHA-256。每次运行重新计算 hash，hash 不一致时拒绝运行。

### 4.3 Harness 执行边界

1. Harness 通过 `AppServices` 和现有依赖注入构造 `ChatService`，不得复制查询、路由和检索实现。
2. 离线默认模式使用临时 SQLite 或独立测试数据库、临时会话目录和固定模型夹具，不读写生产 PostgreSQL、用户会话、报告目录或 Redis 生产库。
3. 在线模式允许读取显式指定的测试环境变量，但仍使用独立测试用户和会话命名空间。
4. 支持 `--case-id` 单例重放、`--category` 类别筛选、`--failed-only` 和 `--max-concurrency`。默认并发上限为 4，单例超时继承评测预算。
5. 每次运行记录 git commit、数据集 hash、模型配置、路由配置、Prompt 版本、开始时间、结束时间和汇总结果。
6. 失败样例保存问题、期望值、实际路由、证据 ID、生成输出、错误类型和脱敏日志；不得保存 API Key、完整鉴权头或无关用户正文。
7. JSON 报告是机器可读事实，Markdown 报告从同一结果对象生成，禁止两个格式各自计算一遍指标。

## 5. P2-02 Agent Runtime 执行预算

### 5.1 核心模型

在 `orchestration_models.py` 新增以下不可变数据结构：

- `AgentBudget`：最大耗时、最大步数、最大模型调用次数、Token 预算和工具白名单。
- `ExecutionContext`：`request_id`、`trace_id`、`user_id`、`session_id`、开始时间、截止时间和取消事件。
- `SubtaskExecutionState`：子任务 ID、路由、状态、终止原因、模型调用数和耗时。
- `TerminationReason`：明确枚举终止原因，不使用自由文本替代状态。

终止原因至少包括：

```text
completed
timeout
cancelled_by_client
step_budget_exceeded
model_call_budget_exceeded
token_budget_exceeded
tool_not_allowed
tool_failed
invalid_session
internal_error
```

### 5.2 默认预算

预算默认值来自配置，环境变量可覆盖：

| 配置 | 默认值 | 说明 |
|---|---:|---|
| `AGENT_MAX_SECONDS` | 300 | 单次流式运行墙钟上限 |
| `AGENT_MAX_STEPS` | 50 | 编排步数和子任务循环上限 |
| `AGENT_MAX_MODEL_CALLS` | 40 | Embedding、Rerank、意图和 Chat 等外部模型调用总数 |
| `AGENT_TOKEN_BUDGET` | 150000 | Chat 模型输入与输出 Token 总量上限 |
| `AGENT_STEP_TIMEOUT_SECONDS` | 90 | 单个阻塞步骤上限 |

Token 统计优先使用模型响应中的 usage；兼容接口不返回 usage 时使用项目现有字符估算规则，并在运行元数据标记 `token_usage_source=estimated`。

### 5.3 ID 与传播

1. 中间件为每个 HTTP 请求生成 `request_id`。
2. Trace 层生成或接收 W3C 兼容的 `trace_id`；未启用 OpenTelemetry 时生成 32 位十六进制 ID。
3. `ChatService.stream()` 接收 `ExecutionContext`，同一请求内的编排器、领域 Agent、模型、检索、数据库和 Redis 调用共享相同 ID。
4. 响应头返回 `X-Request-ID` 和 `X-Trace-ID`。
5. SSE 首个 `metadata`、最终 `metadata` 和 `done` 事件增加 `request_id`、`trace_id`、`termination_reason` 和 `subtask_status` 字段。

### 5.4 取消与持久化

1. `agent.py` 的 SSE 生成器在 `finally` 中设置取消事件，客户端断开后不再发起新的模型或工具调用。
2. 每个编排步骤、模型调用和耗时子任务开始前检查取消、截止时间和预算。
3. 已启动会话的运行在开始、终止和完成时写入 `agent_runs`；`trace_id` 使用独立列，预算快照、终止原因、子任务状态和 Token 统计写入 `metadata`。
4. 没有 `session_id` 的兼容查询不伪造会话运行记录，但仍返回请求级 ID 和明确终止原因。
5. 现有事件类型、正文增量顺序和字段名不变。超预算通过现有 `error` 事件表达，并附加结构化终止字段。

## 6. P2-03 OpenTelemetry Trace、Metrics 与日志

### 6.1 轻量观测栈

应用内使用 OpenTelemetry SDK 和 OTLP HTTP 导出。本地 `observability` profile 只启动固定版本 Jaeger，不额外引入 Collector、Prometheus 或独立数据库。

依赖在 `requirements.txt` 中锁定为：

```text
opentelemetry-api==1.28.2
opentelemetry-sdk==1.28.2
opentelemetry-exporter-otlp-proto-http==1.28.2
opentelemetry-exporter-prometheus==0.49b2
```

Jaeger 使用 `jaegertracing/all-in-one:1.60.0`，通过 OTLP HTTP 接收 Trace，使用内置内存存储。

### 6.2 Span 规范

统一 span 名称使用点分结构：

```text
http.request
auth.verify
session.load
memory.load
intent.route
document.retrieve
case.multidim
case.smart_match
embedding.call
rerank.call
chat.call
postgres.query
redis.command
vector.query
sse.stream
```

Span 属性至少包括 `request_id`、`trace_id`、`user_id`、`session_id`、模型名、步骤名、候选数量、缓存命中、成功状态、错误类型和 Token 数。禁止记录用户问题正文、回答正文、证据正文、API Key、Cookie 和 Authorization。

数据库、Redis、向量查询和模型调用采用显式包装建立 span，不依赖需要额外原生扩展的重型自动插桩。

### 6.3 Metrics

提供 OpenTelemetry Meter 和 Prometheus 文本导出端点，指标至少包括：

```text
app_requests_total
app_request_errors_total
app_request_duration_seconds
agent_routes_total
retrieval_hits_total
model_calls_total
model_tokens_total
cache_fallbacks_total
vector_fallbacks_total
agent_budget_terminations_total
```

所有标签使用固定枚举值，禁止把用户问题、用户名或任意错误文本作为标签，避免指标基数失控。

### 6.4 日志关联

日志过滤器从 `contextvars` 注入 `request_id`、`trace_id`、`user_id` 和 `run_id`。本地可用文本日志，容器环境输出单行 JSON。异常日志仅记录错误类型、阶段和脱敏摘要。

采样率由 `OTEL_TRACES_SAMPLER_ARG` 控制。本地默认 1.0，生产默认不高于 0.1；观测后端不可用时降级为本地指标和应用日志，不影响主业务。

## 7. P2-04 确定性评测与 RAGAS

### 7.1 确定性指标

`retrieval.py` 实现路由准确率、Hit@1、Hit@3、Recall@5、MRR、证据 ID 命中率和无结果识别率。证据集合为空时只有期望也为空才计为正确。

`answer.py` 实现必含事实覆盖率、引用支持率、禁止结论违规数和拒答准确率。必含事实使用人工审核过的确定性子串或“任选其一”表达式；禁止结论采用完整短语匹配，避免简单关键词造成误判。

### 7.2 基线策略

首次运行只记录基线，不因主观分数修改业务规则。后续回归门禁为路由准确率和 Hit@3 相对冻结基线下降不超过 2 个百分点；无结果识别率不得低于基线。

失败样例必须保留路由、期望证据、实际证据和原因。确定性核心子集可在普通测试流程中重复运行，完整模型评测使用手动或计划任务。

### 7.3 RAGAS 运行边界

RAGAS 放在独立 `requirements-eval.txt`，固定为 `ragas==0.2.15` 和 `langchain-openai==0.3.7`，默认运行镜像和 API 不安装。不得依赖 PyTorch、CUDA、本地 Embedding 模型或模型微调包。

首版只对 10 至 20 条人工挑选的小样本运行，使用 OpenAI 兼容接口调用外部裁判模型，输出 Faithfulness、Answer Relevancy、Context Precision 和 Context Recall。缺少 API Key 或 RAGAS 依赖时必须生成明确的 `skipped` 报告，不得伪造分数。

RAGAS 报告与人工抽查记录并存，结果不单独作为业务门禁。

## 8. P2-05 向量存储统一契约

### 8.1 模块与接口

新增 `backend/app/services/vector/`：

```text
backend/app/services/vector/
├── __init__.py
├── contracts.py
├── factory.py
├── json_document_store.py
├── chroma_document_store.py
├── milvus_document_store.py
└── milvus_memory_store.py
```

`DocumentVectorStore` 协议至少包含：

```text
upsert_chunks
delete_chunks
query
get_chunk
list_chunk_ids
collection_info
rebuild
health
close
```

`MemoryVectorStore` 独立定义，至少包含保存、删除、按用户检索、重建、健康和关闭。

统一结果模型包含：

- 项目稳定 ID。
- 原始得分和融合得分。
- 来源和过滤元数据。
- 后端名称、集合名和检索模式。
- `request_id` 和 `trace_id` 可追踪字段。

### 8.2 兼容与降级

现有 `ChromaDocumentChunkStore` 对外行为通过适配器保留，`DocumentRetriever` 的调用方不直接感知后端差异。JSON 余弦检索保留为最终降级实现。

文档后端失败顺序固定为：

```text
Milvus -> Chroma -> JSON
```

每次降级必须记录原因和实际模式。JSON 词面召回结果不得伪装成 Milvus 或 Chroma 命中；检索无结果时返回空列表，不得为了“有结果”放宽范围。

### 8.3 健康、重建和释放

`collection_info()` 返回后端、集合名、数量、维度、schema 版本和健康状态。`rebuild()` 只从权威 chunk 数据和 PostgreSQL 记忆事实构建，禁止从旧向量库反向恢复正文。

应用关闭时显式关闭 Milvus 客户端和 Chroma 客户端。重复关闭保持空操作。

## 9. P2-06 Milvus 适配器与对比迁移

### 9.1 Compose 部署

在根目录 `compose.yaml` 增加单个 `milvus` standalone 服务，使用嵌入式 etcd 和本地存储，具名数据卷为 `risk-master-milvus-data`。

服务资源限制为 `2 CPU`、`2 GB` 内存，暴露容器内 `19530` 和健康端口 `9091`，使用 `/healthz` 健康检查。API 不等待 Milvus 健康后才启动。

使用 Compose `vector` profile，并在 `.env.example` 与本机未跟踪 `.env` 中设置：

```text
COMPOSE_PROFILES=vector
```

这样普通 `docker compose up -d` 默认同时启动 Milvus；需要纯核心环境时可显式使用 `COMPOSE_PROFILES=core` 或 `docker compose --profile core up -d`。

### 9.2 配置

新增以下环境变量：

```text
VECTOR_BACKEND=milvus
VECTOR_FALLBACK_BACKEND=chroma
MILVUS_URI=http://milvus:19530
MILVUS_TOKEN=
MILVUS_DATABASE=default
MILVUS_TIMEOUT_SECONDS=10
MILVUS_DOCUMENT_COLLECTION=weather_document_chunks_v1
MILVUS_MEMORY_COLLECTION=weather_user_memories_v1
MILVUS_DENSE_TOP_K=30
MILVUS_SPARSE_TOP_K=30
MILVUS_RRF_K=60
```

默认 `VECTOR_BACKEND=milvus`；仍保留 `VECTOR_BACKEND=chroma` 作为低资源回退开关。

### 9.3 文档集合

`weather_document_chunks_v1` 使用单一集合，不按个例创建物理集合。字段固定为：

```text
chunk_id           VARCHAR 主键
text               VARCHAR，中文分词分析器
dense_vector       FLOAT_VECTOR，维度来自 Embedding 配置
sparse_vector      SPARSE_FLOAT_VECTOR，由 BM25 Function 生成
case_ids           ARRAY<VARCHAR>，表示 chunk 关联的个例
source_pdf         VARCHAR
file_path          VARCHAR
chunk_no           INT64
page_no            INT64
section_type       VARCHAR
is_title           BOOL
text_length        INT64
content_hash       VARCHAR
document_version   VARCHAR
embedding_model    VARCHAR
```

`case_ids` 从 `standard_cases1.json` 的 `source_chunk_ids` 反向生成。chunk 正文和来源以 JSON 为准，向量、BM25 和过滤字段可重建。

### 9.4 记忆集合

`weather_user_memories_v1` 字段固定为：

```text
memory_id          VARCHAR 主键
user_id            VARCHAR
session_id         VARCHAR
memory_type        VARCHAR
sensitivity        VARCHAR
status             VARCHAR
summary            VARCHAR，中文分词分析器
dense_vector       FLOAT_VECTOR
sparse_vector      SPARSE_FLOAT_VECTOR，由 BM25 Function 生成
source_message_id  INT64
version            INT64
created_at         INT64
```

所有记忆查询必须带参数化 `user_id == {user_id}` 过滤，应用层读取后再校验一次用户范围。

### 9.5 混合检索

文档查询流程：

1. 使用标准个例数据解析问题中的确定条件。
2. 若存在目标个例，则先构造参数化 `ARRAY_CONTAINS_ANY(case_ids, {target_case_ids})` 过滤表达式。
3. 在过滤后的范围内执行稠密 Top-30 和 BM25 Top-30。
4. 使用 `RRFRanker(k=60)` 融合。
5. 按 chunk ID 从 JSON 权威数据读取正文和来源。
6. 若显式目标个例没有匹配，返回无结果，不退化为全库检索。

问题不包含个例筛选条件时允许在文档集合内全库检索。

### 9.6 对比报告

`backend/scripts/rebuild_vector_index.py` 支持文档和记忆集合的 `--dry-run`、单集合重建和全量重建，输出构建前后数量、ID 差异、耗时和失败明细。

Chroma 与 Milvus 使用同一黄金数据集、同一问题、同一证据标准。报告记录索引构建时间、空闲内存、检索延迟、Recall@5、MRR、Hit@3、失败样本差异和降级行为，并给出是否维持默认后端的明确理由。

## 10. P2-07 跨会话长期记忆

### 10.1 权威模型

在现有 `memory_items` 和 `memory_summaries` 上增加迁移字段：

- `source_message_id`
- `version`
- `metadata`
- `deleted_at`

状态固定为 `candidate`、`active`、`rejected`、`deleted`。用户删除采用软删除并立即失效向量，保留最小审计信息。

新增 `backend/app/repositories/memory_repository.py` 和 `backend/app/services/memory_service.py`。所有查询和删除必须携带 `user_id`，仓储不接受仅凭 `memory_id` 的跨用户读取。

### 10.2 提取与召回

1. 会话完成后，异步从用户明确表达的偏好、区域、灾种、分析习惯和可复用结论生成候选记忆。
2. 一次性事实、模型推测、敏感高风险内容和无来源内容不得自动成为 `active`。
3. 候选记忆记录来源会话、来源消息、置信度、敏感级别和版本。
4. 回答前按当前 `user_id` 召回 active 记忆，与结构化画像和会话摘要合并为有上限的记忆上下文。
5. Milvus 失败时只读取 PostgreSQL 结构化画像和摘要，并在 `memory_trace` 标记 `fallback_structured`。
6. 记忆召回失败不阻塞主问题回答，响应中保留明确降级原因。

### 10.3 缓存和删除

Redis 使用 `user:{user_id}:memory_context` 保存短期记忆上下文缓存，TTL 不超过 30 分钟。用户切换账号不共享任何进程内或 Redis 记忆状态。

用户删除记忆时依次执行：

1. PostgreSQL 事务内标记 `deleted`。
2. 删除 Milvus 中同 `memory_id` 且同 `user_id` 的向量。
3. 清理 `user:{user_id}:memory_context` 及当前会话相关缓存。
4. 记录不含记忆正文的审计日志。

Milvus 暂时不可用时数据库删除必须成功，向量失效进入可重试清单；重建索引时只生成 `active` 记忆。

### 10.4 用户接口

增加当前用户级接口：

```text
GET    /api/memories
DELETE /api/memories/{memory_id}
```

接口只能访问当前登录用户的数据。返回来源、类型、置信度、敏感级别、状态、版本和更新日期，不返回内部向量。

## 11. 配置与依赖变更

`.env.example` 增加 Agent 预算、OpenTelemetry、评测并发、向量后端和 Milvus 配置，只使用安全占位值。

`requirements.txt` 增加固定版本的生产依赖：

- `pymilvus==2.5.18`
- OpenTelemetry SDK、OTLP HTTP exporter 和 Prometheus exporter
- 必要的轻量观测依赖

`requirements-eval.txt` 单独固定 RAGAS 及其兼容依赖。任何新增依赖都必须确认 Python 3.11 兼容、无 PyTorch、无 CUDA，并记录安装后镜像大小变化。

## 12. 测试设计

### 12.1 单元测试

- 数据集 schema、重复 ID、非法路由、hash 和权威 ID 引用校验。
- AgentBudget 的耗时、步数、模型调用、Token、白名单和取消终止原因。
- ID 生成、SSE 字段兼容和子任务状态汇总。
- 路由、Hit@K、Recall、MRR、证据命中、必含事实和禁止结论公式。
- 文档与记忆向量契约的参数、结果模型、健康、重建和幂等关闭。
- 长期记忆提取候选规则、用户隔离和软删除。

### 12.2 集成测试

- Harness 连续两次运行固定夹具，路由和检索指标一致。
- 单条失败样例可重放，JSON 与 Markdown 同源。
- FastAPI 请求头、SSE metadata/done、应用日志和 `agent_runs.trace_id` 可关联。
- 客户端断开后不再新增模型调用，运行状态可查询。
- Chroma、JSON 和 Milvus 通过同一契约测试。
- 停止 Milvus 后按 Milvus -> Chroma -> JSON 降级，返回真实检索结果或明确空结果。
- 删除并重建索引后，chunk ID 集合和数量与权威 JSON 一致。
- 用户 A 无法读取或召回用户 B 的记忆。
- 删除记忆后 PostgreSQL、Redis 和 Milvus 均不再返回该记忆。

### 12.3 回归测试

1. Python 全量编译检查。
2. 后端现有单元、集成和端到端测试全量执行。
3. 黄金数据集确定性核心子集连续运行两次。
4. Chroma 基线、Milvus 混合检索和停库降级对比。
5. 至少 5 个多轮记忆样例纳入回归。
6. 前端生产构建，确认现有 SSE 和消息渲染不受新增字段影响。
7. `docker compose config`、`docker compose up -d --build`、`docker compose ps` 和服务健康检查。
8. 浏览器登录、创建会话、流式问答、历史恢复和长记忆删除冒烟验证。

## 13. 交付顺序

按依赖关系分四批实施，每批先写失败测试再写实现：

1. P2-01、P2-03、P2-05 的可并行基础：数据模型、Trace 上下文、向量契约。
2. P2-02、P2-04：Agent Runtime 接入现有编排器，完成确定性指标和报告。
3. P2-06：Milvus standalone、混合检索、重建脚本和 Chroma 对比。
4. P2-07：长期记忆仓储、服务的用户隔离、删除和降级。

每批完成后运行聚焦测试；全部完成后运行完整回归和容器级验证。未达到前置条件的后续工作不得用临时假实现跳过。

## 14. 非目标

- 不实现第三阶段的完整备份恢复、CI 平台、压测、安全扫描和生产运维自动化。
- 不引入 Kafka、Kubernetes、微服务拆分或独立推理服务。
- 不使用 PyTorch、本地大模型、CUDA 或 `pymilvus[model]`。
- 不删除 Chroma、JSON 或现有兼容接口。
- 不把 Redis 或 Milvus 当业务权威库。
- 不为了提高评测分数擅自修改现有业务规则、Prompt 或知识库内容。

## 15. 验收标准

1. P2-01 至 P2-07 的实施方案验收项全部有自动化测试或可重复命令。
2. 黄金数据集包含 60 条有效样例，具备版本、覆盖说明、hash 和权威 ID 校验。
3. 同一固定夹具连续运行两次，路由和检索指标完全一致。
4. 一次流式问答的 `request_id`、`trace_id`、日志和 `agent_runs` 可完整关联。
5. 超预算和客户端断开都有明确终止原因，取消后不再新增模型调用。
6. Chroma 基线和 Milvus 对比报告由同一评测集生成，默认后端可配置切换。
7. Milvus 停机时系统明确降级，不返回虚假命中。
8. 长期记忆具备 PostgreSQL 权威存储、用户隔离、软删除、向量失效和重建能力。
9. 现有业务测试、前端构建和 Compose 全链路通过，第一阶段功能无回归。
