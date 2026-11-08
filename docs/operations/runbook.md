# 单机运行手册

> 本文档面向把本项目作为单机 Agent 系统运行与维护的开发者，覆盖启动、停止、
> 数据库迁移、向量索引重建、备份恢复、日志与运行产物清理，并明确单机限制。
> 所有文档与注释使用中文。更详细的备份恢复演练记录见
> [backup-restore-drill.md](backup-restore-drill.md)。

## 1. 环境概览

| 服务 | 容器 | 数据 | 故障影响 |
|---|---|---|---|
| API | `api` | 会话/消息/运行/分享写入 PostgreSQL | 业务不可用 |
| PostgreSQL | `postgres` | 权威数据，单机唯一事实源 | 需从备份恢复 |
| Redis | `redis` | 短期会话上下文与限流计数 | 可重建，自动回退 PostgreSQL |
| 向量库 | `milvus`（可选 profile）/ Chroma | 检索索引，非唯一事实 | 可从 PostgreSQL/JSON 重构建 |
| 前端 | `frontend` | 无状态静态资源 | 仅影响界面 |

启动顺序：`postgres` → `redis` → `api`（依赖健康后）→ `frontend`（依赖 api 健康）。

## 2. 启动与停止

```powershell
# 启动全部核心服务（默认不含 Milvus，使用 Chroma 检索）
docker compose up -d

# 需要 Milvus 时使用 profile 启动
docker compose --profile milvus up -d

# 查看服务状态与健康检查
docker compose ps

# 停止服务（保留数据卷）
docker compose down

# 停止并删除数据卷（会清除 PostgreSQL/Redis 数据，谨慎使用）
docker compose down -v
```

健康检查：`docker compose ps` 中 `api` 显示 `healthy` 表示进程存活；
`GET /api/health` 返回 `postgres/redis/vector` 依赖状态，`GET /api/ready`
在任一关键依赖不可用时返回 503。

## 3. 数据库迁移

```powershell
# 本地开发时初始化/升级 PostgreSQL 表结构
python -m alembic upgrade head

# 从旧 SQLite 导入历史会话与用户（dry-run 先行）
python scripts/migrate_sqlite_to_postgres.py --source data/sessions.sqlite3 --dry-run
python scripts/migrate_sqlite_to_postgres.py --source data/sessions.sqlite3
```

## 4. 重建向量索引

```powershell
# 先对账，不写库
python scripts/rebuild_vector_index.py --dry-run --collection all

# 正式重建文档与记忆集合（从 PostgreSQL/JSON 权威数据重构建）
python scripts/rebuild_vector_index.py --collection all
```

## 5. 备份

```powershell
$env:DATABASE_URL = "postgresql+psycopg://risk:risk_local_password@postgres:5432/risk"

# 通过 postgres 容器执行 pg_dump，备份写入 backups/ 并校验核心表行数
python scripts/backup_postgres.py --url $env:DATABASE_URL --container
```

备份产物位于 `backups/`，该目录已加入 `.gitignore`。

## 6. 恢复

```powershell
# 列出可用备份并按行数校验
python scripts/verify_backup_counts.py --url $env:DATABASE_URL

# 从最新备份恢复（拒绝覆盖已有数据的空库检查）
python scripts/restore_postgres.py --url $env:DATABASE_URL --container --backup <备份文件名>

# 恢复后重建向量索引，并重启 api 让 Redis 缓存按需回填
python scripts/rebuild_vector_index.py --collection all
docker compose restart api
```

恢复演练的完整步骤、耗时与结果记录见 `docs/operations/backup-restore-drill.md`。

## 7. 日志与运行产物清理

- 容器日志由 Docker 的 `json-file` 驱动写入，可按文件大小轮转：
  ```powershell
  docker run -d --log-opt max-size=20m --log-opt max-file=3 ...
  ```
  或在 Docker 守护进程配置中设置 `log-opts` 全局生效。
- 报告、PDF 与多维分析产物写入
  `backend/app/agents/case_multidim_search/runtime`（命名卷 `api-runtime`），
  可按业务需要定期删除旧报告目录。
- 向量重建临时产物写入 `.tmp/vector-rebuild`，脚本完成后可清理：
  ```powershell
  Remove-Item -Recurse -Force .tmp/vector-rebuild -ErrorAction SilentlyContinue
  ```

## 8. 压测与容量说明

```powershell
# 对公开健康接口做小规模并发压测，输出 p50/p95、错误率与降级次数
python scripts/load_test.py --base-url http://127.0.0.1:8000 --concurrency 8 --total 100
```

单机限制：本地演示环境不承诺高可用，PostgreSQL 是唯一权威数据源；
Redis 与向量索引均可从权威数据重建。未来拆分方向是在保持模块化单体的前提下，
将模型调用、向量检索与长期记忆分别独立为可水平扩展的组件，不引入多副本集群。

## 9. 常见故障处理

| 现象 | 排查步骤 |
|---|---|
| `api` 一直不健康 | 查看日志 `docker compose logs api`，检查 PostgreSQL/Redis 是否健康 |
| 登录后会话为空 | 确认 `DATABASE_URL` 指向的库已执行迁移 |
| 检索无结果 | 运行 `python scripts/rebuild_vector_index.py --dry-run --collection documents` 对账 |
| Redis 故障导致报错 | Redis 只做缓存，重启后由 PostgreSQL 自动回填，检查 `/api/health` 中 redis 状态 |
| Milvus 不可用 | 使用 Chroma profile 回退，重建索引后恢复 |
