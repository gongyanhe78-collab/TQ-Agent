# PostgreSQL 备份、恢复与演练记录

> 本文档记录第三阶段（P3-03）数据库备份/恢复脚本的用法、一次完整恢复演练的
> 步骤与结果，以及 Milvus 向量索引重建命令。所有文档与注释使用中文。

## 1. 脚本清单

| 脚本 | 作用 |
|---|---|
| `scripts/backup_postgres.py` | 用 `pg_dump` 把 PostgreSQL 导出为 SQL 到 `backups/`，备份前后统计核心表行数 |
| `scripts/restore_postgres.py` | 用 `psql` 导入最新备份并校验行数，拒绝覆盖已有数据 |
| `scripts/verify_backup_counts.py` | 统计 `users/sessions/messages/agent_runs/shared_message_snapshots` 行数 |

`backups/` 目录已加入 `.gitignore`，备份产物不进入版本库。

## 2. 前置条件

- 本机已启动 `docker compose` 环境（postgres 容器健康）。
- `DATABASE_URL` 指向目标 PostgreSQL，例如：
  `postgresql+psycopg://risk:risk_local_password@postgres:5432/risk`
- 使用容器模式时无需在本机安装 psql/pg_dump，脚本通过
  `docker compose exec -T postgres` 调用容器内工具。

## 3. 备份

```powershell
# 通过 postgres 容器执行 pg_dump（推荐）
$env:DATABASE_URL = "postgresql+psycopg://risk:risk_local_password@postgres:5432/risk"
python scripts/backup_postgres.py --url $env:DATABASE_URL --container

# 本机已安装 pg_dump 时也可以直接执行
python scripts/backup_postgres.py --url $env:DATABASE_URL
```

输出示例（JSON 摘要，不含密码与消息正文）：

```json
{
  "file": "risk-20261108-153000.sql",
  "size_bytes": 204800,
  "before": {"users": 3, "sessions": 5, "messages": 42, "agent_runs": 12, "shared_message_snapshots": 2},
  "after": {"users": 3, "sessions": 5, "messages": 42, "agent_runs": 12, "shared_message_snapshots": 2}
}
```

若备份前后行数不一致，脚本返回非零退出码并打印警告，提示存在并发写入。

## 4. 恢复

恢复面向**空数据库**执行：先删除数据卷让 postgres 回到无表状态，
再导入备份（备份文件包含建表语句与数据）。

```powershell
# 1. 停掉依赖 postgres 的服务并删除数据卷（会清空现有数据，谨慎执行）
docker compose stop api migrate data-init
docker compose down postgres
docker volume rm risk-master-postgres-data

# 2. 只启动 postgres，让备份导入前没有其他服务创建表
docker compose up -d postgres
docker compose ps postgres

# 3. 等待 postgres 健康后导入最新备份
$env:DATABASE_URL = "postgresql+psycopg://risk:risk_local_password@postgres:5432/risk"
python scripts/restore_postgres.py --url $env:DATABASE_URL --container

# 4. 启动完整环境，让 migrate/data-init 幂等补齐剩余初始化
docker compose up -d
docker compose ps
```

恢复脚本输出恢复前后行数摘要；任一核心表行数与备份不一致时返回非零退出码。
目标库已有数据时脚本默认拒绝覆盖，需要 `--force` 才继续。

### 4.1 演练结果（2026-11-08）

| 阶段 | 结果 |
|---|---|
| 备份耗时 | 秒级（本机小数据量，备份文件约 5.5MB） |
| 恢复耗时 | 秒级 |
| 行数校验 | 备份前后一致；恢复后 users=8、sessions=7、messages=28、agent_runs=14、shared_message_snapshots=1，与备份完全一致 |
| 失败点 | 目标库非空时被脚本主动拒绝（符合预期）；`CREATE DATABASE` 遇到 template1 collation 版本告警，改用 `TEMPLATE template0` 创建演练库 |
| 恢复后冒烟 | API `/api/health` 正常，登录与查询可用 |

> 演练结论：恢复链路成立；恢复后应用通过 migrate/data-init 可继续启动，
> 无需手工重建业务表。

## 5. Milvus 向量索引重建

数据库恢复后，向量索引不属于 PostgreSQL，需单独重建（复用
`scripts/rebuild_vector_index.py`）：

```powershell
# 文档向量集合
python scripts/rebuild_vector_index.py --collection documents

# 长期记忆向量集合
python scripts/rebuild_vector_index.py --collection memories

# 全部集合并输出对账结果
python scripts/rebuild_vector_index.py --collection all --output-dir backups/vector-rebuild
```

重建脚本按权威数据（`data/standard_cases.json` 等）重建并输出 chunk 数量
与 ID 集合差异对账；`--dry-run` 可先预览不落库。

## 6. 常见失败点与处理

1. **目标库非空被拒绝**：恢复前必须清空（删卷或 `--force`）。
2. **新建演练库报 collation 版本告警**：`CREATE DATABASE` 可能报
   "template1 has a collation version"；用
   `CREATE DATABASE <name> TEMPLATE template0` 创建即可规避，不影响数据恢复。
2. **psycopg 未安装导致行数校验不可用**：脚本降级为只做备份/恢复，不中断，
   但会打印警告；容器内已安装 psycopg。
3. **备份前后行数不一致**：有并发写入；先停写再备份，或接受轻微差异。
4. **Milvus 未启动**：重建脚本会按固定链降级或提示向量后端不可用，业务检索
   回退结构化/关键词，不影响恢复后的正常启动。
