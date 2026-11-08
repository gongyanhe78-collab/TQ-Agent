# 🌦️ 气象灾害个例智能分析系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Vue-3-4FC08D?logo=vuedotjs&logoColor=white" alt="Vue 3">
  <img src="https://img.shields.io/badge/ChromaDB-向量检索-F2C94C" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Milvus-2.5.18-00A1EA" alt="Milvus 2.5.18">
  <img src="https://img.shields.io/badge/PostgreSQL-16.4-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL 16.4">
  <img src="https://img.shields.io/badge/Redis-7.2.5-DC382D?logo=redis&logoColor=white" alt="Redis 7.2.5">
  <img src="https://img.shields.io/badge/状态-持续迭代-2E8B57" alt="持续迭代">
</p>

<p align="center">面向气象业务的个例检索、灾害分析、证据追溯与报告生成平台</p>

基于大语言模型、向量检索和结构化个例数据的气象灾害问答与分析平台。系统面向气象业务人员，将 PDF 个例材料整理为可检索、可分析、可追溯的知识库，用户通过一个聊天入口即可完成统计汇总、过程复盘、区域分析、相似个例匹配、图片证据查看和报告导出。

> 当前版本：第二阶段企业化运行时 —— 黄金评测、Agent 预算、可观测性、Milvus 混合检索与长期记忆（2026 年 11 月）

## 🧩 项目简介

系统把气象灾害个例材料转换为三类可协同使用的知识资产：

| 数据形态 | 主要内容 | 主要用途 |
|----------|----------|----------|
| 📄 文档片段 | PDF 按自然段切分后的文本块 | 语义检索、材料问答和证据引用 |
| 📊 标准化个例 | 时间、灾种、区域、强度、影响等字段 | 统计、筛选、对比、分析和报告 |
| 🛰️ 图片证据 | 雷达图、卫星云图、降水图及来源页码 | 图片预览、证据追溯和报告配图 |

前端统一聊天页负责输入问题、展示流式回答和处理进度；后端根据问题意图自动选择文档 RAG、多维个例检索、Smart 相似匹配、澄清或报告处理链路。所有结果都可以保存到会话中，后续问题可承接上一轮个例范围和分析结果继续追问。

### ✨ 核心能力

- 🧠 **统一意图路由**：识别统计、检索、相似匹配、材料问答、复合任务和报告请求。
- ⚡ **流式回答与进度**：持续推送检索、分析、生成、保存等阶段事件和回答增量。
- 📊 **多维个例分析**：支持指标统计、逐例分析、图表、CSV 和 PDF 报告。
- 🔎 **Smart 相似匹配**：综合时间、空间、灾种、文本和图片等维度解释历史过程相似性。
- 🗂️ **会话上下文**：保存消息、个例范围、证据、图片、图表、报告和 Agent 运行信息。
- 🔁 **重生成与版本切换**：可附带本次重生成要求，并保留、切换查看历史回答版本。
- 💬 **消息操作**：支持复制、分享、反馈、问题上报、重新生成和 PDF 导出。
- 🧠 **长期记忆**：按用户持久化偏好与事实，回答前自动注入记忆上下文，支持按用户读取与软删除。
- 🧪 **确定性评测**：60 条黄金用例与固定 Harness，路由与检索指标可重复复现，可选 RAGAS 评测。
- 🔭 **可观测性**：OpenTelemetry 链路追踪、Prometheus 指标与 Jaeger 可视化，请求级 Agent 预算与终止原因。

### 🧠 代码功能简要说明

| 模块 | 主要功能 |
|------|----------|
| ⚙️ `backend/app/main.py` | FastAPI 应用工厂，只负责 CORS、生命周期和路由装配 |
| 🧱 `backend/app/services/container.py` | `AppServices` 组合根，集中构造存储、模型客户端、查询服务和 Agent |
| 🔌 `backend/app/api/routes/agent.py` | 提供三个公开查询入口，统一委托 `ChatService` |
| 🛂 `backend/app/services/query/chat_service.py` | 三个查询入口共享的查询业务门面 |
| 🧭 `backend/app/agents/unified_chat/` | 统一编排、会话记忆、意图路由、进度事件、结果适配和持久化 |
| 🔎 `backend/app/agents/case_multidim_search/` | 按日期、年份、月份、灾种和区域检索个例，生成分析、图表和 PDF/CSV 报告 |
| 🧠 `backend/app/agents/smart_case_match/` | 综合时间、空间、灾种、文本和图片等维度返回相似历史个例及理由 |
| 🧩 `backend/app/agents/query_intent/` | 对复合问题进行意图识别、任务拆分和路由规划 |
| 📐 `backend/app/services/structured_qa.py` | 基于标准化个例回答统计、对比和证据清单等结构化问题 |
| 🏗️ `backend/app/services/knowledge_builder/` | 构建文档段向量、图片索引和标准化个例数据 |
| 🔐 `backend/app/api/routes/auth.py` | 提供注册、登录、刷新令牌、登出和当前用户接口 |
| 💾 `backend/app/db/session_repository.py` | 使用 PostgreSQL 持久化会话、消息、Agent 运行记录、反馈、分享和回答版本 |
| ⚡ `backend/app/cache/context_cache.py` | 使用 Redis 缓存短期会话上下文，未命中或故障时回退 PostgreSQL |
| 🔭 `backend/app/observability/` | OTel 延迟初始化、日志关联与 Prometheus 文本指标导出 |
| 🧠 `backend/app/services/memory_service.py` | 长期记忆候选抽取、结构化降级、Redis 缓存与删除联动 |
| 🗂️ `backend/app/repositories/memory_repository.py` | PostgreSQL 长期记忆权威仓储与软删除 |
| 📐 `backend/app/services/vector/` | Milvus / Chroma / JSON 统一契约与固定降级链 |
| 🧪 `backend/evals/` | 60 条黄金数据集、确定性指标与评测 Harness |
| 🛠️ `scripts/rebuild_vector_index.py` | 向量索引重建与权威 JSON 对账（支持 dry-run） |
| 💬 `frontend/src/components/` | Vue 3 聊天界面、会话列表、处理进度、证据预览、报告卡和消息操作 |

## 🏗️ 系统架构

```mermaid
flowchart TD
    UI["Vue 3 统一聊天界面<br/>会话列表 · 流式回答 · 图表图片 · 报告操作"]

    subgraph API["FastAPI 接口层"]
        ROUTES["API Routes<br/>查询 · 会话 · 专用 Agent · 系统"]
        DEP["ServicesDep<br/>路由依赖注入"]
    end

    subgraph APP["应用与业务层"]
        LIFE["lifespan 生命周期"]
        CONTAINER["AppServices 组合根"]
        CHAT["ChatService<br/>统一查询业务门面"]
        ORCH["UnifiedChatOrchestrator<br/>每个流式请求独立实例"]
    end

    subgraph AGENTS["业务处理链路"]
        INTENT["QueryIntentAgent<br/>意图识别与任务规划"]
        RAG["文档 RAG<br/>召回 · 重排 · 回答"]
        MULTI["多维个例检索<br/>统计 · 分析 · 报告"]
        SMART["Smart 相似匹配<br/>多维排序 · 解释"]
        CONTROL["澄清 / 复用上一轮结果"]
    end

    subgraph DATA["数据与外部服务"]
        MILVUS["Milvus 2.5.18<br/>混合检索（稠密 + BM25 稀疏）"]
        CHROMA["Chroma / JSON<br/>固定降级链"]
        CASES["标准化个例 JSON"]
        IMAGES["图片证据与元数据"]
        SESSIONS[("PostgreSQL 16.4<br/>用户 · 会话 · 消息 · 反馈 · 版本 · 记忆")]
        CACHE[("Redis 7.2.5<br/>短期上下文缓存")]
        MEMORY["长期记忆<br/>按用户召回 · 权威在 PostgreSQL"]
        TRACE["Jaeger / OTel<br/>链路追踪与指标"]
        MODELS["DashScope<br/>Chat · Embedding · Rerank"]
        RUNTIME["runtime/<br/>报告 · 审计 · 临时产物"]
    end

    UI -->|"HTTP + SSE"| ROUTES
    LIFE --> CONTAINER
    ROUTES --> DEP
    DEP --> CONTAINER
    CONTAINER --> CHAT
    CHAT --> ORCH
    ORCH --> INTENT
    INTENT --> RAG
    INTENT --> MULTI
    INTENT --> SMART
    INTENT --> CONTROL
    RAG --> MILVUS
    MILVUS -.->|停机降级| CHROMA
    ORCH --> MEMORY
    MEMORY --> SESSIONS
    ORCH --> TRACE
    RAG --> MODELS
    MULTI --> CASES
    MULTI --> IMAGES
    SMART --> CASES
    SMART --> IMAGES
    ORCH --> SESSIONS
    ORCH --> CACHE
    CACHE -.->|未命中或故障回退| SESSIONS
    MULTI --> RUNTIME
    SMART --> RUNTIME
```

### 🧱 分层与依赖方向

| 层次 | 位置 | 职责 |
|------|------|------|
| 接口层 | `backend/app/api/` | 接收请求、注入服务、返回响应或 SSE，不包含查询业务实现 |
| 应用装配层 | `backend/app/main.py`、`backend/app/core/` | 创建 FastAPI 应用、管理生命周期、加载配置 |
| 业务服务层 | `backend/app/services/query/` | 文档问答、结构化查询、上下文解析、结果适配和统一门面 |
| Agent 层 | `backend/app/agents/` | 意图路由、统一编排、多维分析、相似匹配和领域处理 |
| 基础设施层 | `backend/app/db/`、`backend/app/cache/`、`backend/app/services/` | PostgreSQL 仓储、Redis 缓存、模型客户端、知识库构建与导出 |

依赖方向保持为“接口 → 应用服务 → Agent / 基础设施”。Router 通过 `ServicesDep` 获取 `AppServices`，不会反向导入 `main.py` 或持有应用全局状态。导入应用模块不会自动创建数据库、连接向量库或调用模型，这些动作只在 FastAPI `lifespan` 启动阶段执行。

### 🔄 唯一主业务链路

前端日常问答统一使用 `/api/query/stream`：

```text
Vue 统一聊天页
  → POST /api/query/stream
  → ChatService.stream
  → UnifiedChatOrchestrator.stream
  → QueryIntentAgent.route
  → document_rag | case_multidim_search | smart_case_match | clarify
  → 报告复用或流式结果适配
  → SSE 增量事件
  → 保存会话、证据、图片、图表、报告与版本信息
```

三个公开查询入口共享 `ChatService`：

| 入口 | 当前定位 | 说明 |
|------|----------|------|
| `POST /api/query/stream` | ✅ 唯一主业务链路 | 前端统一聊天入口，返回完整流式事件和会话记忆 |
| `POST /api/query` | 兼容旧客户端 | 复用结构化优先的文档查询服务 |
| `POST /api/agent/query` | 兼容旧智能体协议 | 返回意图、执行计划、证据和审计信息 |

专用 `/api/case-multidim`、`/api/smart-case-match` 和 `/api/query-intent` 接口继续保留，供独立页面、调试和外部系统直接调用，但不替代统一聊天主链路。

一次问答的典型过程：

1. 前端通过 `/api/query/stream` 提交问题，并立即显示处理中状态。
2. 编排器读取当前会话历史、结构化上下文和上一轮可复用结果。
3. 意图 Agent 识别问题并规划文档 RAG、多维检索、Smart 匹配或澄清链路。
4. 后端持续推送检索、分析、生成和保存阶段的进度与回答增量。
5. 结果先写入 PostgreSQL，再刷新或失效 Redis 上下文缓存，保存回答、证据、图片、图表、报告和版本信息。
6. 用户可以继续追问、反馈、重新生成、分享或导出 PDF。

## 📁 项目结构

```text
├── backend/
│   ├── app/
│   │   ├── main.py                       # 应用工厂：CORS、生命周期和路由装配
│   │   ├── api/
│   │   │   ├── dependencies.py           # 认证、用户级仓储和服务依赖注入
│   │   │   ├── schemas.py                # 请求数据模型
│   │   │   └── routes/
│   │   │       ├── auth.py               # 注册、登录、刷新、登出和当前用户
│   │   │       ├── agent.py              # 三个公开查询入口
│   │   │       ├── sessions.py           # 会话、消息、反馈、分享和导出
│   │   │       ├── system.py             # 健康检查与知识库状态
│   │   │       ├── vectors.py            # 知识库重建与图片证据
│   │   │       ├── case_multidim.py      # 多维个例专用接口
│   │   │       ├── smart_case_match.py   # Smart 匹配专用接口
│   │   │       ├── query_intent.py       # 意图路由专用接口
│   │   │       ├── memories.py           # 长期记忆读取与软删除
│   │   │       └── observability.py      # /metrics 文本指标
│   │   ├── cache/
│   │   │   └── context_cache.py          # Redis 优先、PostgreSQL 回退的上下文缓存
│   │   ├── observability/               # OTel 追踪、日志关联与 Prometheus 指标
│   │   ├── core/
│   │   │   ├── config.py                 # 环境变量、目录与模型配置
│   │   │   └── lifecycle.py              # 服务容器生命周期
│   │   ├── db/
│   │   │   ├── models.py                 # 用户、会话、消息、记忆和审计模型
│   │   │   ├── repositories.py           # 认证数据访问
│   │   │   ├── session_repository.py     # PostgreSQL 会话与消息仓储
│   │   │   └── memory_repository.py      # 长期记忆权威仓储与软删除
│   │   ├── security/                     # 密码哈希、令牌和鉴权依赖
│   │   ├── agents/
│   │   │   ├── unified_chat/             # 统一编排、路由、流式处理和持久化
│   │   │   ├── query_intent/             # 意图识别与任务拆分
│   │   │   ├── case_multidim_search/     # 多维检索、分析、图表与报告
│   │   │   └── smart_case_match/         # Smart 相似个例匹配
│   │   └── services/
│   │       ├── container.py              # AppServices 组合根
│   │       ├── query/                    # 统一查询门面及查询业务服务
│   │       ├── model_client/             # Chat、Embedding、Rerank 客户端
│   │       ├── knowledge_builder/        # 文档、图片和标准化个例构建
│   │       ├── document_store.py         # ChromaDB 文档片段存储
│   │       ├── memory_service.py         # 长期记忆服务（候选抽取与删除联动）
│   │       ├── vector/                   # Milvus / Chroma / JSON 契约与降级链
│   │       ├── image_extraction.py       # 图片证据存储
│   │       └── structured_qa.py          # 结构化问题回答
│   ├── alembic/
│   │   └── versions/                     # PostgreSQL 与测试库结构迁移
│   ├── evals/                            # 60 条黄金数据集、确定性指标与 Harness
│   └── tests/                            # 后端单元、集成和架构测试
├── docker/
│   ├── backend.Dockerfile                # 固定 Python 版本的后端镜像
│   ├── frontend.Dockerfile               # 固定 Node 与 Nginx 版本的前端镜像
│   ├── migrate_entrypoint.py             # Alembic、历史迁移和管理员初始化入口
│   └── nginx.conf                        # 静态资源、API 和分享页同源代理
├── scripts/
│   ├── bootstrap_admin.py                # 幂等初始化首个管理员
│   ├── init_api_data.py                  # 把本机知识库幂等同步到 API 数据卷
│   ├── migrate_sqlite_to_postgres.py     # 旧 SQLite 数据一次性导入 PostgreSQL
│   ├── rebuild_vector_index.py           # Milvus 向量重建与权威 JSON 对账
│   ├── compare_vector_backends.py        # Chroma 与 Milvus 检索对比
│   └── run_evaluation.py                 # 黄金数据集确定性评测入口
├── compose.yaml                          # 一条命令启动全部核心服务
├── frontend/
│   ├── src/
│   │   ├── App.vue                       # 页面状态、SSE 和会话协同
│   │   ├── api.js                        # REST、SSE 和文件接口封装
│   │   ├── components/
│   │   │   ├── ChatPanel.vue             # 聊天面板组合入口
│   │   │   ├── chat/                     # 输入、消息、证据、操作和反馈组件
│   │   │   ├── SessionList.vue           # 会话列表
│   │   │   ├── SmartCaseResult.vue       # Smart 匹配结果
│   │   │   └── PdfReportCard.vue         # PDF 报告卡片
│   │   ├── messageVersions.js            # 回答版本分组与切换
│   │   └── styles/                       # 基础样式与回答内容样式
│   └── package.json
├── docs/                                 # 架构与设计说明
├── resource/                             # 待入库 PDF（不提交）
├── data/                                 # 知识库、图片和历史 SQLite 迁移来源（不提交）
├── tests/                                # 场景验证脚本和本地验证产物
├── tmp/                                  # 本地运行截图（不提交）
├── requirements.txt
└── README.md
```

`runtime/`、`logs/`、`tmp/`、数据库、测试输出、截图、`data/numba_cache/` 和 `.codex_tmp/` 均已加入 `.gitignore`。实际运行数据应保留在本机或部署环境中，不进入版本库。

## ✨ 功能演示

以下截图来自本机 `tmp/` 目录。该目录已被 Git 忽略，因此本地 Markdown 预览可见，推送到远端仓库时不会自动包含图片；如需公开展示，应另行将截图上传到受版本管理的文档资源目录并替换链接。

### 🗺️ 区域分布与影响范围分析

用户直接提问，系统从标准化个例中汇总过程数量、影响范围、降温幅度和大风强度，并给出区域对比。

![区域分布与影响范围分析](tmp/img.png)

### 🌧️ 跨月份灾害特征总结

系统可以把多个时间段的个例合并分析，归纳降水强度、空间分布、地形效应和复合灾害特征。

![跨月份灾害特征总结](tmp/img_1.png)

### 🔬 单个过程的分阶段复盘

对于“详细分析某次灾害”这类问题，系统按时间阶段组织降水演变、强度指标、强对流表现和总体特征。

![单个过程分阶段复盘](tmp/img_2.png)

### 🛰️ 图片证据关联

回答会列出图片类型、来源 PDF、页码和图片 ID。前端支持点击图片 ID 预览对应的雷达图、降水图或其他证据图。

![图片证据关联](tmp/img_3.png)

### 🧾 会话追问与消息操作

统计结果可以继续追问；回答保存后支持复制、分享、收藏/反馈、问题上报、重新生成和导出 PDF。

![会话追问与消息操作](tmp/img_4.png)

## 🚀 快速开始

### 🐳 推荐启动方式：Docker Compose

`compose.yaml` 已把前端、API、迁移任务、PostgreSQL、Redis 和 Milvus 放在同一个 Compose 项目内；Jaeger 属于可选 profile。Docker Desktop 启动后，在项目根目录执行一条命令即可构建并启动全部核心服务：

```powershell
docker compose up -d --build
```

首次运行会自动拉取并构建缺失镜像；PostgreSQL 和 Redis 也可以提前单独拉取：

```powershell
docker compose pull postgres redis
```

后续代码或配置未变化时，直接执行下面的命令即可重新启动整个项目：

```powershell
docker compose up -d
```

打开 <http://127.0.0.1:8080>。API、PostgreSQL、Redis 和 Milvus 默认只在 Compose 内网暴露，浏览器通过前端的 Nginx 同源代理访问 `/api` 和 `/share`。

Milvus 由 `vector` profile 控制，`.env.example` 默认通过 `COMPOSE_PROFILES=vector` 启用。如需额外启动 Jaeger 链路追踪，执行：

```powershell
docker compose --profile observability up -d jaeger
```

### 1️⃣ 准备环境变量

首次启动前，从示例文件创建本机 `.env`：

```powershell
Copy-Item .env.example .env
```

至少修改以下配置：

```env
# 本地数据库凭据；部署前必须替换为强密码
POSTGRES_PASSWORD=请替换为强密码

# JWT 签名密钥；请替换为至少 32 字节的随机字符串
AUTH_JWT_SECRET=请替换为随机密钥

# 首个管理员；两项都填写，或都留空跳过
BOOTSTRAP_ADMIN_USERNAME=admin
BOOTSTRAP_ADMIN_PASSWORD=请替换为管理员强密码

# 模型服务密钥；留空时本地结构化能力仍可启动，但模型能力不可用
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

`.env` 已被 `.gitignore` 排除，真实密码和 API Key 不能提交到版本库。`BOOTSTRAP_ADMIN_PASSWORD` 只用于第一次创建管理员，不会写入日志；管理员已存在时，重复执行迁移不会覆盖现有密码。如果同名账号已经以普通用户身份存在，迁移会明确失败而不会自动提权。

### 2️⃣ 固定镜像版本

Compose 和 Dockerfile 已避免使用 `latest`，当前固定组合如下：

| 组件 | 固定版本 | 选择原因 |
|------|----------|----------|
| PostgreSQL | `public.ecr.aws/docker/library/postgres:16.4-alpine3.20` | 与项目 `SQLAlchemy 2.0.36`、`psycopg 3.2.3` 和 Alembic 迁移兼容；使用同补丁版本的轻量 Alpine 层降低下载和磁盘占用。 |
| Redis | `public.ecr.aws/docker/library/redis:7.2.5-alpine` | 与 `redis-py 5.2.1` 使用的协议兼容，7.2 的 AOF 和过期键能力满足短期上下文缓存需求。 |
| Milvus | `milvusdb/milvus:v2.5.18` | 提供稠密 + BM25 稀疏混合检索；Python 客户端固定 `pymilvus==2.5.18`，不使用 Milvus v3 或本地 Embedding 模型。 |
| Jaeger | `jaegertracing/all-in-one:1.60.0` | 可选 `observability` profile，接收 OTLP 链路并在 Web UI 展示。 |
| Python | `public.ecr.aws/docker/library/python:3.12.7-slim-bookworm` | 与当前 FastAPI、SQLAlchemy、ChromaDB 和 PDF 处理依赖的可安装范围匹配，同时固定 Python 补丁版本保证构建可复现。 |
| Node.js | `public.ecr.aws/docker/library/node:22.11.0-alpine3.20` | 满足 Vite 5 和 Vue 3 构建要求，结合 `npm ci` 与锁文件保证前端依赖可重复安装。 |
| Nginx | `public.ecr.aws/docker/library/nginx:1.27.3-alpine` | 使用固定 Alpine 运行时提供静态文件和同源反向代理，配置与本项目 `/api`、`/share` 路由匹配。 |

后端构建默认使用 `mirrors.aliyun.com` 提供 Debian 系统包，避免国内网络访问默认软件源时长时间等待或下载失败。如需在其他网络环境构建，可显式传入兼容的软件源地址：

```powershell
docker compose build --build-arg DEBIAN_MIRROR=deb.debian.org
```

PDF 知识库使用 `pypdf` 读取逐页文本，并使用 `pdfplumber` 分析图片版面和图注，不引入 `unstructured[pdf]`、PyTorch、CUDA、OCR 或大型视觉模型。该路径适合项目当前的文本型气象 PDF，镜像体积和首次构建时间明显更小。

知识库图片、索引和历史 SQLite 不进入 API 镜像。`data-init` 一次性服务只把 `data/` 中非运行态文件同步到 `risk-master-api-data` 卷，排除旧 SQLite 与 Numba 缓存；重复启动只刷新有变化的文件。`resource/` 以只读方式挂载，避免把待入库 PDF 复制进每一层镜像。

升级任何一个固定版本时，都应同步执行 Compose 配置检查、后端全量回归、前端构建和容器端到端验收，不能只修改标签后直接投产。

### 3️⃣ 检查启动与迁移结果

```powershell
docker compose ps
docker compose logs data-init --no-color
docker compose logs migrate --no-color
docker compose logs api --no-color
docker compose logs frontend --no-color
```

预期结果是 `postgres`、`redis`、`milvus`、`api`、`frontend` 为健康状态，`data-init` 与 `migrate` 执行成功并已退出；Jaeger 仅在 `observability` profile 下启动。启动顺序固定为：

1. PostgreSQL 和 Redis 通过健康检查。
2. `data-init` 把本机知识库同步到 API 数据卷。
3. `migrate` 执行 Alembic 升级到 `head`。
4. 按需执行旧 SQLite 数据迁移。
5. 按环境变量幂等初始化管理员。
6. API 启动并通过健康检查。
7. 前端启动并通过健康检查。

向量检索按 `Milvus -> Chroma -> JSON` 固定降级：Milvus 不可用时自动回退 Chroma，Chroma 再不可用时回退权威 JSON 索引。Redis 与 Milvus 均不保存唯一事实，PostgreSQL 始终是长期记忆与业务数据的权威事实源。

容器内健康检查地址：

- 前端：<http://127.0.0.1:8080/healthz>
- API（经同源代理）：<http://127.0.0.1:8080/api/health>
- 知识库状态：<http://127.0.0.1:8080/api/knowledge/status>

### 4️⃣ 数据持久化与重启

Compose 使用以下命名卷：

| 卷 | 内容 |
|----|------|
| `risk-master-postgres-data` | 用户、会话、消息、运行记录、反馈、分享和记忆等权威数据 |
| `risk-master-redis-data` | Redis AOF 缓存数据，可重建，不属于业务事实源 |
| `risk-master-api-data` | 知识库、图片和模型运行数据 |
| `risk-master-api-runtime` | 多维分析报告和临时运行产物 |
| `risk-master-milvus-data` | Milvus 本地存储（etcd、元数据与对象存储），可重建，不属于业务事实源 |

普通停止不会删除数据：

```powershell
docker compose stop
docker compose start
```

重建容器但保留卷：

```powershell
docker compose down
docker compose up -d --build
```

只有明确需要清空本地环境时才使用 `docker compose down -v`，该命令会删除所有命名卷和其中的历史会话。

### 5️⃣ 导入旧 SQLite 数据

PostgreSQL 是权威数据源，旧 SQLite 只作为一次性迁移来源。默认 `RUN_LEGACY_MIGRATION=false`，不会读取或修改旧库。

首次迁移建议先 dry-run：

```env
RUN_LEGACY_MIGRATION=true
LEGACY_MIGRATION_DRY_RUN=true
LEGACY_SQLITE_PATH=/legacy/sessions.sqlite3
```

将旧库放到 `data/sessions.sqlite3` 后执行：

```powershell
docker compose up migrate
docker compose logs migrate --no-color
```

确认日志中的源表、目标表和行数对账无误后，把 `LEGACY_MIGRATION_DRY_RUN` 改为 `false`，再次执行 `docker compose up migrate` 完成事务提交。脚本会保留旧 SQLite 文件、维护主键映射，并对重复执行做幂等处理。

迁移完成后将 `RUN_LEGACY_MIGRATION` 恢复为 `false`，避免后续每次启动重复扫描历史库。

### 6️⃣ 构建知识库

将气象灾害 PDF 放入 `resource/`，确认模型配置可用并登录管理员账号后，可以调用知识库重建接口：

```powershell
$body = @{ username = "admin"; password = "管理员密码" } | ConvertTo-Json
$login = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8080/api/auth/login" `
  -ContentType "application/json" `
  -Body $body

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8080/api/documents/index" `
  -Headers @{ Authorization = "Bearer $($login.access_token)" }
```

该接口只允许管理员调用，会重建自然段文档向量、图片证据索引和标准化个例数据。首次构建的耗时取决于 PDF 数量、图片数量和模型服务速度。

### 7️⃣ 常用运维命令

```powershell
# 查看服务状态和健康检查
docker compose ps

# 持续查看 API 日志
docker compose logs -f --tail=200 api

# 重启单个服务
docker compose restart api

# 停止 Redis 验证 PostgreSQL 回退，随后恢复
docker compose stop redis
docker compose start redis

# 校验 Compose 配置
docker compose config --quiet
docker compose --env-file .env.example config --quiet
```

### 🧪 本地可选的源码开发方式

需要直接调试 Python 或 Vue 源码时，可以继续使用本机 Conda 和 Node 环境：

```powershell
conda create -n riskMe python=3.11 -y
conda activate riskMe
python -m pip install -r requirements.txt
```

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

本地源码开发应通过 `DATABASE_URL` 和 `REDIS_URL` 明确连接可用的 PostgreSQL 与 Redis。未配置这两个变量时，历史测试和旧脚本可能仍使用兼容的 SQLite 路径，但 Docker Compose 才是第一阶段推荐的完整运行方式。

### 🧪 本地质量检查命令

提交前建议先在本机执行与 CI 相同的检查命令，避免推送到仓库后才发现门禁失败：

```powershell
# 后端：静态检查与单元测试（对应 CI 快速门禁）
python -m ruff check backend/app scripts
python -m mypy --follow-imports=skip backend/app/security/rate_limit.py backend/app/observability/logging.py backend/app/core/config.py backend/app/api/routes/sessions.py backend/app/db/session_repository.py backend/app/services/session_store.py backend/app/services/model_client/chat.py
python -m pytest backend/tests/unit -q
```

```powershell
# 后端：集成测试（需要 PostgreSQL 与 Redis，对应 CI 集成门禁）
$env:DATABASE_URL = "postgresql+psycopg://risk:risk@127.0.0.1:5432/risk"
$env:REDIS_URL = "redis://127.0.0.1:6379/0"
python -m pytest backend/tests/integration -q
```

```powershell
# 后端：确定性核心评测子集（对应 CI 评测门禁）
python -m pytest backend/tests/unit/evals backend/tests/integration/evals -q
```

```powershell
# 前端：lint、格式与构建（对应 CI 前端门禁）
cd frontend
npm run lint
npm run format:check
npm run build
```

## 📖 使用说明

### 💬 统一聊天入口

打开前端后，在同一个输入框中直接使用自然语言提问。系统不要求用户先选择功能模块，会自动判断问题类型并选择对应 Agent。

可以尝试：

```text
2025年山西的寒潮主要分布在哪里，对哪些区域影响最大？
总结全年暴雨的主要特点和成因。
详细分析一下7月7日的灾害。
这次灾害关联了哪些图片？
2025年7月份发生了几次灾害？
当前山西北部出现强对流，历史上有哪些相似过程？
```

### 🛣️ 主要处理链路

| 问题类型 | 处理链路 | 典型结果 |
|----------|----------|----------|
| 📊 统计、筛选、对比和过程分析 | 多维个例检索 Agent | 个例列表、统计指标、强度分析、图表和证据 |
| 🧠 历史过程相似性和相似天气判断 | Smart Case Match Agent | 相似个例、匹配维度、相似度和解释 |
| 📄 材料问答、证据补充和开放性问题 | 文档 RAG | 流式文本回答、文档片段和关联图片 |
| 🧾 复用上一轮结果生成报告 | 报告处理链路 | 复用已保存的分析结论并生成 PDF 报告 |

### ⏳ 流式回答与处理中状态

提交问题后，页面会显示当前 Agent、阶段和百分比进度，例如检索证据、整理结构化事实、生成回答、保存结果。回答正文以 SSE 增量方式显示；在模型或部分外部服务不可用时，系统会尽量使用结构化数据和本地证据完成降级回答。

### 🖼️ 图片、图表与报告

- 回答中的图片 ID 可以点击预览原图。
- 统计和分析结果可以展示表格、柱状图等轻量可视化。
- 多维检索支持 CSV 导出和 PDF 报告生成。
- 对上一轮结果提出“整理成报告”或“导出 PDF”时，可复用已保存的分析结果，再进行本地 PDF 排版。

### 💾 会话、上下文和消息操作

- 左侧会话列表支持创建、重命名、切换和删除会话。
- 后续问题可以引用上一轮的个例范围、指标和分析结果。
- 已完成回答支持重新生成，可附带本次重生成要求，并通过版本导航查看旧答案。
- 支持复制、分享、正向/改进反馈、问题上报和 PDF 导出。

## 🔌 API 接口概览

除健康检查和带随机 token 的公开分享页外，用户数据接口都需要在请求头中携带访问令牌：

```http
Authorization: Bearer <access_token>
```

### 🔐 认证接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/auth/register` | 注册普通用户并签发首组令牌 |
| `POST` | `/api/auth/login` | 校验用户名和密码 |
| `POST` | `/api/auth/refresh` | 使用 HttpOnly 刷新 Cookie 轮换令牌 |
| `POST` | `/api/auth/logout` | 撤销刷新令牌族并清除 Cookie |
| `GET` | `/api/auth/me` | 获取当前登录用户信息 |

刷新令牌只保存在路径为 `/api/auth` 的 HttpOnly Cookie 中，访问令牌由前端保存在内存并通过 `Authorization` 头传递。

### 🤖 查询接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/query/stream` | ✅ 统一流式问答主入口，需要登录 |
| `POST` | `/api/query` | 兼容旧客户端的非流式文档查询，需要登录 |
| `POST` | `/api/agent/query` | 兼容旧智能体协议的结构化查询，需要登录 |

流式入口示例：

```powershell
$login = Invoke-RestMethod `
  -Method Post `
  -Uri 'http://127.0.0.1:8080/api/auth/login' `
  -ContentType 'application/json' `
  -Body '{"username":"demo","password":"demo-password"}'

Invoke-RestMethod `
  -Method Post `
  -Uri 'http://127.0.0.1:8080/api/query/stream' `
  -Headers @{ Authorization = "Bearer $($login.access_token)" } `
  -ContentType 'application/json' `
  -Body '{"question":"2025年7月份发生了几次灾害","session_id":"your-session-id"}'
```

### 🩺 系统与知识库接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 匿名健康检查，返回应用、向量库和模型基础状态 |
| `GET` | `/api/knowledge/status` | 个例数量、向量数量、模型和构建清单 |
| `POST` | `/api/documents/index` | 管理员从 `resource/` 重建文档、图片和标准个例索引 |
| `GET` | `/api/image-evidence/{image_id}` | 登录后获取聊天图片证据文件 |
| `GET` | `/metrics` | Prometheus 文本格式指标，无需登录 |

### 💬 会话与消息接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/sessions` | 创建会话 |
| `GET` | `/api/sessions` | 列出会话 |
| `GET` | `/api/sessions/{session_id}` | 获取单个会话 |
| `PATCH` | `/api/sessions/{session_id}` | 更新会话标题 |
| `DELETE` | `/api/sessions/{session_id}` | 删除会话 |
| `GET` | `/api/sessions/{session_id}/messages` | 获取会话消息和版本信息 |
| `GET` | `/api/sessions/{session_id}/agent-runs` | 查看 Agent 运行记录 |
| `POST` | `/api/sessions/{session_id}/messages/{message_id}/feedback` | 保存回答反馈 |
| `POST` | `/api/sessions/{session_id}/messages/{message_id}/report` | 上报回答质量问题 |
| `POST` | `/api/sessions/{session_id}/messages/{message_id}/share` | 创建分享链接 |
| `POST` | `/api/sessions/{session_id}/messages/{message_id}/export-pdf` | 导出单条回答 PDF |
| `GET` | `/api/shared-messages/{token}` | 匿名获取不可变分享快照 |

### 🧠 长期记忆接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/memories` | 返回当前用户可见的长期记忆（不含向量字段） |
| `DELETE` | `/api/memories/{memory_id}` | 软删除当前用户的一条长期记忆，保留审计行 |

长期记忆的读取、删除和召回全部绑定 `user_id`，跨用户访问统一返回 `404`。

所有会话、消息、运行记录、反馈和分享创建操作都会按当前用户校验父资源属主。非所有者访问统一返回 `404`，不会暴露资源是否存在。

### 🤖 专用 Agent 接口

- `/api/case-multidim`：自然语言解析、结构化检索、进度、CSV/PDF 导出、报告和图片资源。
- `/api/smart-case-match`：相似匹配、自然语言流式会话、进度和证据资源。
- `/api/query-intent`：意图路由和知识库日期范围查询。

完整请求体和响应字段以应用 OpenAPI 定义和 `backend/app/api/schemas.py` 为准。根目录的 `API接口文档.md` 属于历史补充资料，如与当前路由不一致，请优先使用 OpenAPI。

## 🧪 开发与测试

### 后端

```powershell
conda activate riskMe
python -m pip install "pytest>=8,<10"
python -m compileall backend/app backend/tests scripts docker
python -m pytest backend/tests -q
```

开发依赖统一声明在根目录 `requirements-dev.txt`（pytest、pytest-asyncio、ruff、mypy），本地与 CI 使用同一组依赖：

```powershell
python -m pip install -r requirements-dev.txt
python -m compileall backend/app backend/tests scripts docker
python -m pytest backend/tests -q
```

静态检查（Ruff 聚焦新增与核心模块，Mypy 先覆盖新增公共接口）：

```powershell
ruff check backend/app/core backend/app/services/model_client backend/app/security backend/app/services/resilience scripts -q
mypy backend/app/core/resilience.py
```

在 Docker 容器内运行同一组测试（镜像不包含测试目录与 dev 依赖，需先同步再安装）：

```powershell
docker cp backend/app risk-master-api-1:/app/backend/app
docker cp backend/tests risk-master-api-1:/app/backend/tests
docker cp .gitignore risk-master-api-1:/app/.gitignore
docker cp requirements-dev.txt risk-master-api-1:/app/requirements-dev.txt
docker compose exec -T -u root api pip install -r /app/requirements-dev.txt -i https://mirrors.aliyun.com/pypi/simple/
docker compose exec -T -e DATABASE_URL= api sh -c "cd /app && python -m pytest backend/tests -q -p no:cacheprovider"
```

> 说明：容器内测试通过清空 `DATABASE_URL` 让 Alembic 迁移测试使用独立 SQLite 临时库，避免容器内 PostgreSQL 环境变量覆盖测试配置。

### 评测与向量索引

```powershell
# 黄金数据集确定性评测（60 条用例，固定 seed）
python scripts/run_evaluation.py --dataset backend/evals/datasets/golden_v1.json --max-concurrency 1 --output-dir .tmp/eval-run-1

# 向量索引重建对账（dry-run 不写库）
python scripts/rebuild_vector_index.py --collection documents --dry-run

# Chroma 与 Milvus 检索结果对比
python scripts/compare_vector_backends.py
```

评测使用固定夹具与固定 seed，两次运行的确定性路由与检索指标应完全一致；向量重建后 chunk 数量与 ID 集合应和权威 JSON 索引一致。

`pytest` 只用于开发测试；如果 `riskMe` 中已经安装，可以跳过安装命令。

重点验证认证、隔离、迁移和 Redis 降级：

```powershell
python -m pytest `
  backend/tests/integration/auth `
  backend/tests/integration/security `
  backend/tests/integration/storage `
  backend/tests/integration/api/test_architecture.py `
  -q
```

这些测试覆盖注册登录、刷新令牌轮换、管理员初始化、跨用户资源隔离、Alembic 迁移、用户级 Agent 工厂和 Redis 故障回退。

### 前端

```powershell
cd frontend
npm run build
```

### Compose 回归

```powershell
docker compose config --quiet
docker compose --env-file .env.example config --quiet
docker compose up -d --build
docker compose ps
docker compose logs data-init --no-color
```

容器端到端验收至少检查：

- 前端健康检查：`http://127.0.0.1:8080/healthz`
- API 健康检查：`http://127.0.0.1:8080/api/health`
- 注册、登录、刷新、登出和 `/api/auth/me`
- 用户 A 与用户 B 的会话、消息、运行、反馈、报告和分享隔离
- 匿名公开分享页可读，其他用户访问原会话返回 `404`
- 停止 Redis 后历史会话仍可从 PostgreSQL 读取
- 重启 API、PostgreSQL 和 Redis 后历史数据仍存在
- 前端登录、会话列表、聊天页面和静态资源可正常加载

根目录 `tests/` 还保留了历史场景验证脚本和本地验证产物，适合按需运行，不计入后端核心回归测试。

## ❓ 常见问题

### 没有配置 API Key 能否启动？

可以启动服务和查看部分本地接口，但涉及意图解析、Embedding、Rerank 或回答生成时可能无法调用云端模型。系统会在可用范围内退回规则、结构化数据或本地文档证据。

### 为什么接口返回 401？

除健康检查和公开分享页外，用户数据接口都需要登录。请确认前端已登录，且请求携带有效的 `Authorization: Bearer <access_token>`；访问令牌默认 15 分钟过期，前端会通过 `/api/auth/refresh` 自动轮换。

### 管理员密码如何修改或重置？

`BOOTSTRAP_ADMIN_*` 只在管理员不存在时创建账号，重启和重复执行不会覆盖已有密码。需要重置时，应在受控运维流程中更新数据库中的密码哈希，不能通过删除用户后重新执行初始化来绕过现有数据归属。

### 停止 Redis 会丢失历史会话吗？

不会。PostgreSQL 是用户、会话、消息、运行、反馈和分享的权威事实源。Redis 只保存可重建的短期上下文缓存；停止 Redis 后缓存读写会记录降级并直接读取 PostgreSQL，恢复后重新填充。

### 如何彻底重建本地环境？

普通重启使用 `docker compose down` 后再次 `docker compose up -d --build` 会保留命名卷。只有确定不再需要历史数据时才执行 `docker compose down -v`，该命令会删除 PostgreSQL、Redis、知识库和报告卷。

### 如何更新知识库？

把新 PDF 放入 `resource/` 后，使用管理员访问令牌重新调用 `POST /api/documents/index`。普通用户只需在聊天页提问，不能重建共享知识库。

### 为什么回答中有图片 ID 而没有图片？

请确认对应图片已经生成到 `data/document_images/`，并检查 `/api/image-evidence/{image_id}` 是否可访问。图片是增强证据，图片加载失败时文字回答仍会保留。

### 多维检索和 Smart 匹配是否还有独立页面？

后端保留 `/api/case-multidim` 和 `/api/smart-case-match` 专用接口，前端日常使用统一聊天入口。用户无需手动选择 Agent，系统会根据问题自动路由。

### 报告生成需要重新调用模型吗？

首次多维分析报告通常会调用逐例分析和图表生成流程；针对上一轮结果的报告请求会优先复用已保存的分析结果，再进行本地 PDF 排版。

### 为什么 README 截图在远端看不到？

演示图链接指向本机 `tmp/`，该目录按运行产物和截图策略被 `.gitignore` 排除。本地预览不受影响；发布仓库前需要把选定截图放入受版本管理的资源目录并更新链接。

### Milvus 未启动时系统能否正常工作？

可以。向量检索按 `Milvus -> Chroma -> JSON` 固定降级：Milvus 不可用时自动回退 Chroma，再不可用时回退权威 JSON 索引；长期记忆等业务数据始终以 PostgreSQL 为权威事实源，Redis 与 Milvus 只保存可重建的缓存数据。

### 旧查询接口还能继续使用吗？

可以。`/api/query` 和 `/api/agent/query` 保留原协议并共享 `ChatService`；新功能和完整流式交互应优先使用 `/api/query/stream`。

## 🔐 许可证与使用范围

本项目当前用于内部研究和业务验证。知识库中的 PDF、图片、模型密钥、会话数据库和报告可能包含内部资料，请按部署环境配置访问控制、网络边界和备份策略。

不要将 `.env`、`resource/`、`data/`、`runtime/`、`logs/` 或 `tmp/` 中的敏感内容提交到公共仓库。

---

**最后更新：2026 年 11 月**
