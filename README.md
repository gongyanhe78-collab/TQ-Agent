# Weather Case RAG System

气象灾害个例检索增强生成系统 —— 基于大语言模型的天气灾害个例智能问答与分析平台。

## 项目简介

本项目将气象灾害天气过程总结 PDF 文档通过向量化、结构化提取和智能体路由等技术，构建了一套可对话、可检索、可对比分析的智能问答系统。用户可以用自然语言提问，系统自动理解意图，选择合适的检索链路（结构化统计、相似个例匹配、文档片段召回），并给出带证据追溯的回答。

### 核心能力

| 能力 | 说明 |
|------|------|
| **自然语言问答** | 支持统计汇总、单过程复盘、对比分析、证据检索、相似个例匹配、服务提示等多种问题类型 |
| **智能意图路由** | 自动识别用户意图，选择最优数据层和检索链路，统计类问题不走 top-k chunk 检索 |
| **多维个例检索** | 按日期、灾种、地区、图片类型、来源 PDF 等字段精确筛选标准化个例 |
| **相似个例匹配** | 综合时空、灾种、图像、文本等多维度相似度排序，附匹配理由和预报提示 |
| **图片证据关联** | PDF 中的雷达图、卫星云图、降水图等自动提取并与回答关联展示 |
| **流式对话** | SSE 流式响应，打字机效果，支持多轮会话持久化存储 |
| **自动降级** | LLM 不可用时自动切换本地兜底回答，不中断服务 |

---

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                      前端 (Vue 3)                         │
│       ┌──────────────┐  ┌────────────────────────┐       │
│       │   聊天记录    │  │       统一聊天页        │       │
│       │ SessionList  │  │      ChatPanel         │       │
│       └──────────────┘  └────────────────────────┘       │
└───────────────────────┬──────────────────────────────────┘
                        │ HTTP / SSE
┌───────────────────────┴──────────────────────────────────┐
│                   后端 (FastAPI)                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │               智能体路由 (Agent)                    │   │
│  │  意图分析 → 工具规划 → 结构化执行 → 答案合成 → 审查  │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ 文档检索  │ │ 标准个例  │ │ 相似匹配  │ │ 图片证据  │   │
│  │ (Chroma) │ │  (JSON)  │ │ (Matcher)│ │ (Images) │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
└───────────────────────┬──────────────────────────────────┘
                        │
┌───────────────────────┴──────────────────────────────────┐
│                    外部服务                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ DashScope    │  │ DMXAPI       │  │ ChromaDB     │   │
│  │ (Embedding,  │  │ (Chat 回退)  │  │ (向量存储)    │   │
│  │  Rerank, LLM)│  │              │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### 数据分层

系统有三层数据结构，从原始到结构化逐级递进：

```
PDF 文档
  │
  ├──→ Document Chunks（文档片段层）
  │      PDF 切分为文本块 → 向量化 → ChromaDB weather_document_chunks
  │      用途：RAG 问答的兜底证据检索
  │
  ├──→ Standard Cases（标准化个例层）
  │      从 chunks 中提取结构化字段，构建标准化个例
  │      字段：标题、日期范围、灾种、影响区域、图片证据、置信度
  │      用途：全量聚合统计、多维度筛选、对比分析
  │
  └──→ Image Evidence（图片证据层）
         PDF 中的图表自动提取，标注类型（雷达/卫星/降水等）
         用途：与回答关联展示，相似匹配的图像维度
```

---

## 项目结构

```
├── backend/                     # Python FastAPI 后端
│   └── app/
│       ├── api/
│       │   ├── routes/
│       │   │   ├── system.py          # 健康检查、知识库状态
│       │   │   ├── vectors.py         # 主知识库重建与图片证据读取
│       │   │   ├── sessions.py        # 会话管理 CRUD
│       │   │   ├── agent.py           # RAG 问答（流式/非流式）、智能体查询
│       │   └── schemas.py             # 请求/响应 Pydantic 模型
│       ├── services/
│       │   ├── agent/                 # 智能体路由子系统
│       │   │   ├── models.py          # 意图类型、工具路由、执行计划等数据模型
│       │   │   ├── analyzers.py       # 意图分析器（规则链路 + 上下文链路）
│       │   │   ├── planner.py         # 执行规划器（意图→工具映射）
│       │   │   ├── orchestrator.py    # 主编排器（协调各子模块）
│       │   │   ├── answer_synthesizer.py  # 回答合成器
│       │   │   └── final_reviewer.py  # 最终审查（质量检查 + 重试）
│       │   ├── model_client/           # 云端 Chat、Embedding、Rerank 客户端
│       │   ├── knowledge_builder/      # 自然段向量、图片和精简标准个例建库
│       │   ├── document_store.py      # 文档 chunk 向量库（ChromaDB）
│       │   ├── standard_case_store.py # 标准化个例 JSON 存储
│       │   ├── similar_case_matcher.py    # 相似个例匹配器
│       │   ├── structured_qa.py       # 结构化问答（统计/对比/证据检索）
│       │   ├── image_extraction.py    # 图片证据提取与元数据管理
│       │   └── session_store.py       # 会话、Agent 状态和运行结果（SQLite）
│       ├── models.py                  # 核心数据模型
│       ├── config.py                  # 全局配置
│       └── main.py                    # 应用入口（服务初始化 + 路由注册）
├── frontend/                    # Vue 3 + Element Plus 前端
│   ├── index.html
│   └── src/
│       ├── App.vue                    # 根组件（聊天记录 + 统一聊天状态管理）
│       ├── api.js                     # API 封装（axios + SSE 流处理）
│       ├── main.js                    # 入口
│       └── components/
│           ├── SessionList.vue        # 聊天记录面板
│           └── ChatPanel.vue          # 统一聊天面板（文字 + 图片 + 图表 + 报告）
├── data/                        # 数据存储目录（gitignore）
│   ├── extracted_cases/              # LLM 提取的个例 TXT
│   ├── extracted_samples/            # 样本个例
│   ├── document_index/               # 文档 chunk ChromaDB 索引
│   ├── document_images/              # PDF 图片证据
│   ├── image_metadata.json           # 图片元数据
│   ├── standard_cases.json           # 标准化个例 JSON
│   ├── index/                        # 个例 ChromaDB 索引 + 构建状态
│   └── sessions.sqlite3              # 会话数据库
├── resource/                    # PDF 源文件（gitignore）
├── scripts/                     # 辅助脚本
├── tests/                       # 测试套件（21 个测试文件）
├── docs/                        # 设计文档
│   ├── agent-six-stage-architecture.md          # 智能体六阶段架构说明
│   └── superpowers/                             # 开发计划和规格文档
├── weather_qa_results.json      # 内置评估问答对（20+ 条标准问题）
├── API接口文档.md                # 完整 API 接口文档
└── requirements.txt             # Python 依赖
```

---

## 智能体路由系统

智能体采用六阶段渐进式架构，目前已实现阶段 0-5 的核心能力：

```
用户问题
    │
    ▼
┌─────────────────┐
│  阶段1: 意图分析  │  识别意图类型 + 提取结构化槽位（月份/灾种/地区/图片类型）
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  阶段2: 执行规划  │  意图 → 工具映射，生成只读执行计划，标注禁用工具
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  阶段3: 结构化执行（多链路并行）                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│  │ 标准个例链路   │ │ 相似匹配链路  │ │ 文档检索链路  │ │
│  │ (统计/对比/   │ │ (时空+灾种+  │ │ (语义向量检索 │ │
│  │  证据检索)    │ │  图像相似度)  │ │  + 重排序)    │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────┐
│  阶段4: 回答合成  │  正面回答 → 结构化事实 → 检索证据 → 分析 → 缺口说明
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  阶段6: 最终审查  │  质量检查 + 问题规范化重试
└────────┬────────┘
         │
         ▼
    用户看到回答（附执行链路审计信息）
```

### 支持的意图类型

| 意图 | 说明 | 示例问题 |
|------|------|---------|
| `statistical_summary` | 统计汇总 | "5-7月有多少个灾害天气过程？" |
| `comparative_analysis` | 对比分析 | "6月和7月的强对流过程有什么差异？" |
| `evidence_search` | 证据检索 | "5月21-22日过程有哪些雷达图？" |
| `case_review` | 个例复盘 | "请分析5月21-22日的暴雨过程" |
| `similar_case` | 相似匹配 | "当前山西北部强对流像哪些历史过程？" |
| `decision_support` | 决策支持 | "今天这个过程要重点关注什么？" |
| `general_rag` | 通用检索 | 兜底：任何不匹配以上类型的问题 |

---

## 快速开始

### 环境要求

| 依赖 | 版本 |
|------|------|
| Python | ≥ 3.10 |
| Node.js | ≥ 18 |
| 阿里云 DashScope API Key | 需自行申请 |

### 1. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
# 阿里云 API 密钥（必需）
ALIYUN_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# DashScope 服务地址（默认公有云，专有云用户替换为专有地址）
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 聊天模型（默认 qwen3.6-max-preview）
DASHSCOPE_CHAT_MODEL=qwen3.6-max-preview

# 嵌入模型（默认 text-embedding-v4）
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4

# 重排序模型（默认 qwen3-rerank）
DASHSCOPE_RERANK_MODEL=qwen3-rerank

# 可选：DMXAPI 作为聊天通道回退
# DMXAPI=sk-xxxxxxxx
# DMXAPI_BASE_URL=https://www.dmxapi.cn/v1
```

### 2. 安装依赖 & 启动后端

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 启动 FastAPI 后端（默认端口 8000）
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

后端启动后可访问：
- **Swagger API 文档**：http://localhost:8000/docs
- **健康检查**：http://localhost:8000/api/health

### 3. 启动前端

```bash
cd frontend

# 安装前端依赖（首次运行）
npm install

# 启动开发服务器
npm run dev
```

浏览器访问控制台输出的地址（默认 http://localhost:5173）。

### 4. 初始化知识库

将气象个例 PDF 文件放入 `resource/` 目录，然后：

```bash
# 文档化索引，同时构建向量库、图片索引和标准个例
curl -X POST http://localhost:8000/api/documents/index
```

---

## API 接口概览

### 系统接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查（向量库统计 + 模型可用性） |
| GET | `/api/knowledge/status` | 知识库完整状态（含构建信息和模型配置） |

### 向量库

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/documents/index` | 重建唯一的自然段向量库、图片索引和标准个例 |
| GET | `/api/image-evidence/{image_id}` | 获取图片证据文件 |

### 问答

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/query` | 非流式问答（一次性返回） |
| POST | `/api/query/stream` | 流式问答（SSE 事件流，打字机效果） |
| POST | `/api/agent/query` | 智能体查询（面向系统集成，返回完整链路信息） |

### 会话管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/sessions` | 创建会话 |
| GET | `/api/sessions` | 列出所有会话 |
| GET | `/api/sessions/{id}` | 获取单个会话 |
| PATCH | `/api/sessions/{id}` | 更新会话标题 |
| DELETE | `/api/sessions/{id}` | 删除会话 |
| GET | `/api/sessions/{id}/messages` | 获取会话消息记录 |

---

## 使用说明

### 问答示例

在 Web 界面输入自然语言问题：

- **统计汇总**：「2025年5月到7月山西省一共有多少次灾害天气过程？」
- **个例复盘**：「请详细分析5月21-22日的暴雨天气过程」
- **对比分析**：「对比一下6月和7月的暴雨过程有什么不同？」
- **证据检索**：「5月21-22日的过程有哪些雷达图和卫星云图？」
- **相似匹配**：「当前山西北部出现强对流，历史上哪些过程比较类似？」
- **决策支持**：「基于历史个例，今天这个过程要重点关注什么？」

### 统一聊天入口

页面不再提供独立的个例检索表单。用户直接在聊天框输入自然语言，统一入口会按意图自动路由：条件检索、统计分析和报告生成进入多维检索 Agent；历史过程相似性和参考经验进入 Smart Agent；其余问题进入主 RAG。三条链路的文字、图片、图表和报告都在同一条会话中展示和保存。

### 回答的结构

系统回答遵循统一结构：
- **结论**：直接回答用户问题
- **结构化事实**：从标准化个例中提取的统计数据
- **检索证据**：相关文档片段和图片
- **分析**：基于证据的推断
- **缺口**：明确说明当前无法判断的内容

---

## 开发说明

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定模块测试
python -m pytest tests/test_agent_intent.py -v
python -m pytest tests/test_agent_orchestrator.py -v
```

### 技术栈

**后端**：
- FastAPI — Web 框架
- ChromaDB — 向量数据库
- OpenAI SDK — LLM 调用兼容层
- PyPDF / pdfplumber / pypdfium2 — PDF 解析
- langchain-text-splitters — 文本切分

**前端**：
- Vue 3 (Composition API)
- Element Plus — UI 组件库
- Axios — HTTP 客户端
- SSE (Server-Sent Events) — 流式响应

**模型服务**：DashScope 云端 API，统一提供 Embedding (`text-embedding-v4`)、Rerank (`qwen3-rerank`) 和 Chat (`qwen-plus`)。

---

## 常见问题

### Q: 如何更新本地知识库？
A: 页面不再提供 PDF 上传和索引管理入口。管理员通过 `POST /api/documents/index` 从 `resource` 目录重建主知识库，普通用户只需在统一聊天页提问。

### Q: 问答时提示 API Key 错误？
A: 检查 `.env` 中的 `ALIYUN_API_KEY` 是否正确。专有云用户请同时检查 `DASHSCOPE_BASE_URL`。

### Q: LLM 不可用时系统如何处理？
A: 系统会自动降级为本地检索模式——优先使用标准化个例库和本地规则回答，或返回匹配的文档片段原文摘要，不会中断服务。

### Q: 当前使用哪个向量库？
A: 只使用 `data/document_index` 中的 `weather_document_chunks`。它按 PDF 文档化后的自然段建库，同时服务普通 RAG、Smart 相似匹配和多维检索；标准个例 JSON 与图片索引使用同一批 chunk ID 建立关联。

### Q: 多维检索和相似匹配功能是否还在？
A: 功能仍然保留，但不再显示独立面板。统一聊天入口会识别问题意图并调用对应 Agent，最终结果仍按当前会话保存。

---

## 许可证

本项目仅供内部研究使用。

---

**最后更新**：2026年7月
