# 验证证据索引

> 本文档汇总第三阶段可展示的工程证据：数据隔离、故障降级、备份恢复与评测，
> 并给出对应文件位置，方便面试时按图索骥。所有文档与注释使用中文。

## 1. 用户 A/B 数据隔离

- 代码：`backend/app/db/session_repository.py` 的 `SqlAlchemySessionStore` 在
  SQL 层强制绑定 `user_id`。
- 测试：`backend/tests/integration/security/test_route_isolation.py` 覆盖跨用户
  会话、消息、分享与导出全部返回 404/403。
- 结论：用户 A 的令牌无法读取或修改用户 B 的任何会话与消息。

## 2. 故障降级演示

- 熔断：`backend/app/core/resilience.py`，连续失败进入冷却窗口，恢复后自动探测。
- 测试：`backend/tests/unit/resilience/test_circuit_breaker.py`。
- Redis 回退：Redis 停止后短期上下文由 PostgreSQL 回填，历史会话不受影响。
- 向量库回退：Milvus 不可用时回退 Chroma 或结构化检索，响应中标注降级来源。
- 模型不可用：返回已核验证据或明确无答案，不生成伪造证据。
- 观测：`/metrics` 暴露 `cache_fallbacks_total`、`vector_fallbacks_total`、
  `model_calls_total`（按 success/error 标签），压测脚本
  `scripts/load_test.py` 输出错误率、模型成功率与降级次数增量。

## 3. PostgreSQL 备份与恢复

- 脚本：`scripts/backup_postgres.py`、`scripts/restore_postgres.py`、
  `scripts/verify_backup_counts.py`。
- 演练记录：`docs/operations/backup-restore-drill.md`，包含一次完整恢复演练的
  步骤、耗时与最终结果。
- 单机承诺：不承诺高可用，但保证空卷启动可迁移、从备份恢复后旧会话可继续多轮问答。

## 4. 评测证据

- 黄金数据集：`backend/evals/datasets/golden_v1.json`，共 60 条，覆盖
  路由 6、检索 12、统计 8、RAG 10、相似匹配 8、多轮记忆 10、故障场景 6。
- 确定性指标：`backend/tests/unit/evals/test_dataset.py`、
  `backend/tests/unit/evals/test_metrics.py` 与
  `backend/tests/integration/evals/test_harness.py`，CI 评测门禁运行该子集。
- RAGAS：`backend/evals/runners/ragas_runner.py` 在缺少 API Key 或依赖时明确
  返回 skipped，不阻塞确定性门禁；完整 RAGAS 作为手工/计划任务运行。
- Chroma/Milvus 对比：`scripts/compare_vector_backends.py` 记录基线指标，
  迁移以基线对比和回退开关为准。

## 5. 安全边界

- 限流：`backend/app/security/rate_limit.py` 按登录/查询类别、用户与 IP 限流，
  超限返回 429 + Retry-After。
- 提示注入：`backend/app/services/model_client/chat.py` 的系统提示词明确禁止
  执行材料内指令；测试见 `backend/tests/unit/security/test_prompt_injection.py`。
- 分享治理：公开分享快照只保存不可变一问一答，支持过期与撤销，见
  `backend/tests/integration/api/test_share_expiry.py`。
- 日志脱敏：`backend/app/observability/logging.py` 统一脱敏密钥并截断超长正文。
