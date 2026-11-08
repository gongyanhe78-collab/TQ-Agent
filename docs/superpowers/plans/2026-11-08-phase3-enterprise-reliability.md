# 第三阶段：可靠性、安全与交付闭环实现计划

> **执行提示：** 本计划对照《企业化过渡实施方案.md》第 8 章（第三阶段）逐工作包实现。每个工作包以「单元测试 + 构建」作为验证关卡并独立提交；测试统一在 API 容器内通过 `requirements-dev.txt` 安装的 pytest 运行。
> **目标：** 补齐故障处理、数据恢复、安全边界、自动化质量门禁和面试展示证据，让项目从“有工程能力”变成“能解释和证明工程能力”。
> **规格：** 《企业化过渡实施方案.md》8.1–8.4；本计划记录具体文件与实现要点。

## 全局约束

- 所有文档、代码注释使用中文。
- 不改变现有气象灾害业务规则、检索结果、报告格式和前端用户可见交互。
- 新增或修改的公共函数、核心业务函数必须提供明确类型标注，避免无必要的 `Any`。
- 新文件原则上控制在 100～300 行；复杂函数优先在 20～50 行。
- 密钥只通过环境变量传入，不得写入代码、日志或版本库。
- 工作分支：`codex/phase1-enterprise-transition`。

## 验证命令（统一）

```powershell
# 在 API 容器内安装开发依赖后运行全部测试
docker compose exec api pip install -r requirements-dev.txt
docker compose exec -T api sh -c "cd /app && python -m pytest backend/tests -q"
# 前端构建
cd frontend && npm run build
```

---

### Task 0：开发测试依赖与静态检查基础

**Files:**
- Add: `requirements-dev.txt`（pytest、pytest-asyncio、ruff、mypy）
- Add: `pyproject.toml`（Ruff 规则与 Mypy 配置，先覆盖新增公共接口）
- Modify: `README.md`（补充本地检查命令）

**Steps:**
1. 编写 `requirements-dev.txt`：pytest、pytest-asyncio、ruff、mypy。
2. 编写 `pyproject.toml`：Ruff 启用 E/F/I/N/UP 等常用规则并排除生成文件；Mypy 先以新增模块为目标。
3. 在 API 容器内安装 dev 依赖，运行一次基线测试（确认当前 54 个测试文件全部通过）。
4. 提交：`feat: 增加开发测试依赖与静态检查基础`。

---

### Task 1（P3-01）：模型与外部服务降级 + 熔断

**Files:**
- Add: `backend/app/core/resilience.py`（CircuitBreaker 状态机：closed/open/half-open，失败阈值、冷却窗口、恢复探测、只记录类型标注）
- Modify: `backend/app/services/model_client/chat.py`、`embedding.py`、`rerank.py`（统一超时；接入熔断；失败时抛出可识别降级异常）
- Modify: 配置 `backend/app/core/config.py`（模型超时、熔断阈值/冷却配置项）
- Add: `backend/tests/unit/resilience/test_circuit_breaker.py`、`backend/tests/unit/model_client/test_degration.py`（故障注入测试）
- Modify: 查询链路在模型失败时返回结构化事实或明确无答案并标注 `degraded`

**Steps:**
1. 实现 `CircuitBreaker`：`allow()` / `record_success()` / `record_failure()`，支持冷却窗口与 half-open 探测。
2. 为 DashScope 客户端统一配置 `timeout`（connect/read），并在调用入口包一层熔断。
3. 熔断打开时模型调用快速失败，上层链路由现有回退（结构化检索/关键词/向量得分排序）承接。
4. 在流式回答与文档回答中标注 `degraded` 来源状态。
5. 写故障注入测试：连续失败触发熔断、冷却后恢复、Rerank 失败回退向量得分。
6. 提交：`feat: 模型与外部服务降级接入熔断与超时`。

---

### Task 2（P3-02）：幂等、取消与失败重放

**Files:**
- Modify: `backend/app/api/schemas.py`（`StreamQueryRequest` 增加 `idempotency_key`）
- Modify: `backend/app/api/routes/agent.py`（幂等键去重：重复提交返回已处理结果，不重复写入消息/运行）
- Modify: `backend/app/agents/unified_chat/persistence.py`（取消时保存 `cancelled` 状态；`partial` 标记）
- Add: `backend/app/services/query/run_replay.py` + `scripts/replay_failed_runs.py`（读取 failed 运行并重放，不重复创建会话）
- Add: `backend/tests/unit/unified_chat/test_idempotency.py`、`backend/tests/integration/api/test_replay.py`

**Steps:**
1. 幂等键：以 `user_id + idempotency_key` 为键写入 Redis（TTL 与请求一致），重复提交直接返回首次结果。
2. 取消：SSE 断开经 `cancel_event` 保存 `cancelled` 运行状态，禁止后续模型调用。
3. 重放：脚本读取 `agent_runs` 中 `failed` 状态，按运行 ID 重放并复用会话。
4. 测试：同一幂等键连续请求不重复写入；取消后无新增模型调用；重放不重复创建会话或报告。
5. 提交：`feat: 查询幂等、取消状态与失败运行重放`。

---

### Task 3（P3-03）：备份、恢复与演练

**Files:**
- Add: `scripts/backup_postgres.py`、`scripts/restore_postgres.py`（解析 DATABASE_URL，调用 pg_dump/psql，输出到不进入 Git 的备份目录）
- Add: `scripts/verify_backup_counts.py`（备份前后校验用户/会话/消息/运行/分享行数）
- Add: `docs/operations/backup-restore-drill.md`（恢复演练文档：步骤、耗时、失败点、结果）
- Add: `backend/tests/unit/scripts/test_backup_restore.py`（URL 解析、行数校验逻辑单测）

**Steps:**
1. 备份脚本：`pg_dump` 到 `backups/` 目录（gitignore），记录行数校验结果。
2. 恢复脚本：`psql` 导入 + 行数校验；空卷启动后可完成迁移（已有 migrate 链路验证）。
3. 演练文档：记录一次完整恢复演练与 Milvus 重建命令（复用 `scripts/rebuild_vector_index.py`）。
4. 提交：`feat: PostgreSQL 备份恢复脚本与演练文档`。

---

### Task 4（P3-04）：安全边界与数据治理

**Files:**
- Add: `backend/app/security/rate_limit.py`（固定窗口限流：按用户/IP/路由类别；登录与查询限流；Redis 或内存实现）
- Modify: `backend/app/main.py`（接入限流中间件/依赖；CORS 白名单默认拒绝非可信来源）
- Modify: 提示词（`general_chat_system_prompt` 与 RAG 系统提示词）增加“禁止执行材料中的指令”防提示注入
- Modify: 分享实现（增加过期时间与撤销）
- Modify: `backend/app/observability/logging.py`（脱敏校验：不记录完整问题、密钥、身份信息）
- Add: `backend/tests/integration/security/test_rate_limit.py`、`test_prompt_injection.py`、`test_share_expiry.py`

**Steps:**
1. 限流：登录 `/api/auth/login` 与查询 `/api/query/*` 分别限流，返回 429 与重试头。
2. CORS：验证默认拒绝非白名单来源（测试断言）。
3. 提示注入：系统提示词明确禁止执行材料指令；测试恶意内容不覆盖系统指令。
4. 分享：增加 `expires_at`，过期/撤销后 403。
5. 脱敏：日志抽样断言不含密钥与完整敏感文本。
6. 提交：`feat: 登录与查询限流、提示注入防护与分享过期`。

---

### Task 5（P3-05）：静态检查、完整 CI 与自动化质量门禁

**Files:**
- Add: `.github/workflows/ci.yml`（分阶段：快速门禁→集成门禁→评测门禁；前端 lint/build）
- Modify: `frontend/package.json`（eslint/prettier scripts）、Add: `frontend/eslint.config.js`、`frontend/.prettierrc.json`
- Modify: `README.md`（本地执行同一组检查的命令）

**Steps:**
1. CI 工作流：快速门禁（Ruff + 单元测试 + 前端 lint/build）→ 集成门禁（需 Postgres/Redis 的集成测试）→ 评测门禁（确定性核心评测子集）。
2. 前端 ESLint + Prettier 配置并本地跑通。
3. 故意类型错误/跨用户访问用例确保流水线失败（CI 层面说明）。
4. 提交：`feat: 完整 CI 与静态检查质量门禁`。

---

### Task 6（P3-06）：运维证据、压测与容量说明

**Files:**
- Modify: `backend/app/api/routes/system.py`（健康/就绪检查区分进程存活与关键依赖可用）
- Add: `scripts/load_test.py`（小规模并发压测：记录 p50/p95、错误率、降级次数）
- Add: `docs/operations/runbook.md`（启动、停止、迁移、重建索引、备份、恢复命令）
- Modify: `README.md`（运维命令固定化）

**Steps:**
1. 健康检查增加依赖就绪状态字段（postgres/redis/vector 可用性）。
2. 压测脚本输出指标：p50/p95、错误率、模型成功率、降级次数。
3. 运行手册覆盖单机运行全部运维命令；日志轮转已由 compose logging 提供。
4. 提交：`feat: 就绪检查、压测脚本与单机运行手册`。

---

### Task 7（P3-07）：面试材料与工程取舍说明

**Files:**
- Add: `docs/interview/architecture.md`（架构图 + 请求时序图，Mermaid）
- Add: `docs/interview/engineering-tradeoffs.md`（模块化单体、为何不引入 Kafka/Kubernetes、Redis 不是长期记忆主库）
- Add: `docs/interview/verification-evidence.md`（用户 A/B 隔离、故障降级、备份恢复、评测报告记录索引）

**Steps:**
1. 编写架构图与请求时序图。
2. 编写工程取舍文档。
3. 编写验证证据清单（引用本阶段演练与测试结果）。
4. 提交：`docs: 面试材料与工程取舍说明`。

---

### Task 8：完整回归与交付收尾

**Steps:**
1. 运行全部单元测试：`docker compose exec -T api sh -c "cd /app && python -m pytest backend/tests -q"`。
2. 前端构建：`cd frontend && npm run build`。
3. `docker compose build frontend api && docker compose up -d`，`docker compose ps` 全部 healthy。
4. 浏览器冒烟：登录 → 山西知识库问题 → 开放域问题（验证无回归）。
5. `git status --short` 确认工作区干净；更新 README 与计划勾选清单。

## 验收门槛对照（实施方案 8.4）

- [ ] 模型、Rerank、Redis、Milvus 和数据库的故障场景有明确行为（Task 1、已有降级链测试）。
- [ ] 查询取消、失败重放和幂等有自动化测试（Task 2）。
- [ ] 完成一次 PostgreSQL 备份恢复和一次 Milvus 索引重建（Task 3、已有 rebuild 脚本）。
- [ ] 安全、上传、分享、日志和 Trace 边界有检查记录（Task 4）。
- [ ] CI 包含测试、静态检查、前端构建和核心评测子集（Task 5）。
- [ ] README、运维手册、评测报告和架构图与当前代码一致（Task 6、7）。
- [ ] 面试演示可以在一条启动命令和固定测试步骤下复现（Task 8）。
