# 项目启动与关闭说明

本文适用于 `D:\QiuZhao\risk-master092416\risk-master` 下的 `risk-master` 项目。

## 一、项目运行方式

当前项目通过 Docker Compose 运行，前后端服务、数据库和缓存均在容器中，不需要在 PyCharm 中额外启动 `uvicorn` 或 `npm run dev`。

| 服务 | 作用 | 是否常驻 |
|---|---|---|
| `frontend` | 提供前端页面，并将 `/api` 请求代理到后端 | 是 |
| `api` | FastAPI 后端，处理登录、会话、检索和聊天流式接口 | 是 |
| `postgres` | PostgreSQL 主数据库，保存账号、会话和消息等持久化数据 | 是 |
| `redis` | Redis 缓存，保存短期上下文和缓存数据 | 是 |
| `migrate` | 执行数据库迁移和可选的管理员初始化，完成后正常退出 | 否 |
| `data-init` | 同步初始资料到共享数据卷，完成后正常退出 | 否 |
| `milvus` | Milvus 向量库，提供稠密 + BM25 稀疏混合检索（`vector` profile） | 是 |
| `jaeger` | Jaeger 链路追踪 Web UI（`observability` profile，可选） | 否 |

`migrate` 和 `data-init` 显示为 `Exited (0)` 是正常状态，不需要手动启动。只要 `api` 正常健康运行，就表示这两个一次性任务已经成功完成。

## 二、启动前检查

1. 打开 Docker Desktop，等待 Docker 引擎完全启动。
2. 确认项目根目录存在 `.env` 文件。
3. 确认 `.env` 中的真实密钥和密码已配置，不要把密钥提交到 Git。
4. 不要在 PyCharm 中同时启动另一套后端或前端，否则容易造成端口冲突或请求发到错误的后端。
5. 确认 `.env` 中存在 `COMPOSE_PROFILES=vector`，否则 Milvus 不会随项目启动。向量检索按 `Milvus -> Chroma -> JSON` 固定降级，Milvus 未启动时 API 仍可正常使用。

如果 `.env` 不存在，可以从示例文件创建：

```powershell
Copy-Item .env.example .env
```

如果 `.env` 已经存在，只修改缺少的配置，不要直接覆盖，否则可能丢失已有的数据库密码和管理员配置。

## 三、启动项目

在 PyCharm 的终端中进入项目根目录：

```powershell
Set-Location D:\QiuZhao\risk-master092416\risk-master
```

首次启动，或代码和 Dockerfile 发生变化后，执行：

```powershell
docker compose up -d --build
```

日常启动，代码和镜像没有变化时执行：

```powershell
docker compose up -d
```

`-d` 表示在后台运行。命令执行完成并返回终端后，项目仍会继续运行。

启动完成后检查状态：

```powershell
docker compose ps -a
```

正常状态应包含以下常驻服务：

```text
frontend  Up (healthy)
api       Up (healthy)
postgres  Up (healthy)
redis     Up (healthy)
milvus    Up (healthy)
```

`milvus` 由 `vector` profile 控制，`COMPOSE_PROFILES=vector` 已配置时随项目一起启动。如需启用 Jaeger 链路追踪，额外执行：

```powershell
docker compose --profile observability up -d jaeger
```

同时会看到两个一次性任务：

```text
migrate    Exited (0)
data-init  Exited (0)
```

浏览器访问：

```text
http://127.0.0.1:8080
```

## 四、查看后端日志

使用后台方式启动时，后端日志不会自动显示在 PyCharm 终端中。实时查看 API 日志：

```powershell
docker compose logs -f api
```

查看最近 100 行日志：

```powershell
docker compose logs --tail=100 api
```

实时查看前端日志：

```powershell
docker compose logs -f frontend
```

在日志跟随模式下按 `Ctrl+C`，只会停止查看日志，不会停止容器。

## 五、PyCharm 与 Ctrl+C 的关系

不同启动方式下，`Ctrl+C` 的作用不同：

| 当前操作 | 按 `Ctrl+C` 的结果 |
|---|---|
| 已执行 `docker compose up -d --build` | 启动命令已经返回，容器继续运行 |
| 正在执行 `docker compose logs -f api` | 只停止日志显示，`api` 容器继续运行 |
| 正在执行没有 `-d` 的 `docker compose up --build` | Compose 会尝试停止前台服务，数据仍在数据卷中 |

关闭 PyCharm 项目本身不会自动停止 Docker 容器。如果要关闭项目，应先在 PyCharm 终端中停止 Compose 服务，再关闭 PyCharm。

推荐始终使用带 `-d` 的启动命令，避免把容器运行状态绑定在 PyCharm 终端上。

## 六、关闭项目

### 方式一：临时关闭，推荐日常使用

在项目根目录执行：

```powershell
docker compose stop
```

该命令会正常停止所有容器，但保留容器、网络和数据卷。下次启动只需执行：

```powershell
docker compose up -d
```

检查是否已经停止：

```powershell
docker compose ps -a
```

正常情况下，各服务会显示为停止或退出状态。

停止容器后，可以关闭 PyCharm，也可以退出 Docker Desktop。

### 方式二：删除容器和网络，保留数据

如果希望清理运行容器和 Compose 网络，执行：

```powershell
docker compose down
```

`docker compose down` 默认不会删除命名数据卷。数据库、会话和应用数据仍会保留。

下次启动时执行：

```powershell
docker compose up -d
```

### 不要执行的关闭命令

以下命令会删除数据卷，除非明确要清空全部数据，否则不要执行：

```powershell
docker compose down -v
```

```powershell
docker compose down --volumes
```

```powershell
docker volume prune
```

```powershell
docker system prune --volumes
```

在 Docker Desktop 中也不要点击容器的“删除卷”选项。删除数据卷后，PostgreSQL 中的账号、会话和消息将无法恢复。

## 七、数据保存位置

项目使用以下命名数据卷保存持久化数据：

| 数据卷 | 保存内容 |
|---|---|
| `risk-master-postgres-data` | PostgreSQL 数据库数据 |
| `risk-master-redis-data` | Redis 持久化文件 |
| `risk-master-api-data` | API 使用的数据和检索资源 |
| `risk-master-api-runtime` | 多维检索运行时文件 |
| `risk-master-milvus-data` | Milvus 本地向量数据（可重建，不属于业务事实源） |

执行 `docker compose stop` 或普通 `docker compose down` 不会删除这些数据卷。

容器可以重新创建，数据卷则会在下一次启动时继续挂载。只有使用 `docker compose down -v`、删除数据卷或执行全局卷清理时，数据才会丢失。

## 八、配置变化后的重启

如果只修改了项目根目录的 `.env`，例如 API Key、数据库密码或认证配置，需要重新创建对应的容器：

```powershell
docker compose up -d --no-deps --force-recreate api
```

只执行 `docker compose restart api` 不会重新读取容器的环境变量。

如果修改了后端代码，重新构建并启动：

```powershell
docker compose up -d --build api frontend
```

如果修改了 PostgreSQL 或 Redis 的密码配置，需要同时检查已有数据卷是否仍与新的密码配置兼容。不要为了修改密码直接删除数据卷。

## 九、推荐操作流程

启动项目：

```powershell
Set-Location D:\QiuZhao\risk-master092416\risk-master
docker compose up -d
docker compose ps -a
```

查看后端日志：

```powershell
docker compose logs -f api
```

关闭项目：

```powershell
docker compose stop
```

确认停止后，再关闭 PyCharm 或 Docker Desktop。

## 十、可选数据库备份

在进行重大修改前，可以把 PostgreSQL 备份到项目的 `backups` 目录：

```powershell
New-Item -ItemType Directory -Force backups
docker compose exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc -f /tmp/risk.dump'
docker compose cp postgres:/tmp/risk.dump ./backups/risk.dump
```

备份完成后确认文件存在：

```powershell
Get-Item .\backups\risk.dump
```

备份文件可能包含账号和业务数据，不要提交到 Git，也不要上传到公共位置。
