# Weather Case RAG System

气象个例检索增强生成系统 - 基于通义千问大模型的天气个例智能问答平台

## 项目架构

本项目采用前后端分离架构：

```
├── backend/          # Python FastAPI 后端
│   ├── app/
│   │   ├── api/        # API 路由
│   │   ├── services/   # 核心服务（向量化、检索、LLM、提取等）
│   │   ├── models.py   # 数据模型
│   │   ├── schemas.py  # 请求/响应 Schema
│   │   └── config.py   # 配置管理
├── frontend/         # Vue3 + Element Plus 前端
│   ├── src/
│   │   ├── App.vue      # 主应用
│   │   ├── api.js       # API 封装
│   │   └── components/  # 组件
├── resource/          # PDF 源文件目录
├── data/              # 数据存储目录
│   ├── extracted_cases/  # 提取出的个例 TXT
│   └── index/           # Chroma 向量索引
└── weather_qa_results.json  # 评估问答对
```

## 核心功能

### 向量库设计（两套独立向量库）

| 向量库 | 数据来源 | 粒度 | 用途 | 对应接口 |
|--------|---------|------|------|---------|
| **文档 chunk 库** | PDF 直接切分 | 文档片段 | ✅ RAG 问答核心检索 | `/api/documents/*` |
| **个例向量库** | LLM 提取结构化个例 | 完整天气个例 | 个例列表、详情查看 | `/api/cases/*`, `/api/vectors/*` |

> **重要**：当前 RAG 问答功能使用的是 **文档 chunk 库**（`document_store`），不是个例向量库。

### 主要功能

- 📄 **PDF 文档化索引**：PDF 自动切分、向量化、入库
- 🔍 **智能检索**：向量检索 + 重排序混合检索
- 💬 **流式问答**：SSE 流式响应，打字机效果
- 📝 **会话管理**：多会话支持，自动重命名，历史消息持久化
- 📊 **效果评估**：内置标准评估问答对
- 🔄 **增量更新**：支持增量 PDF 入库，不会重复处理

## 环境要求

| 环境 | 版本要求 |
|------|---------|
| Python | ≥ 3.10 |
| Node.js | ≥ 18 |
| 阿里云 DashScope API Key | 需自行申请 |

## 快速开始

### 第一步：配置环境变量

在项目根目录创建 `.env` 文件：

```env
# 阿里云 API 密钥（必需）
ALIBABA_CLOUD_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 可选配置（有默认值）
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_CHAT_MODEL=qwen3.6-plus
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4
```

> 使用阿里云专有云实例时，请将 `DASHSCOPE_BASE_URL` 替换为你的专有云地址。

---

### 第二步：启动后端服务

在 **项目根目录** 打开终端，执行：

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 启动 FastAPI 后端（默认端口 8000）
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

✅ 后端启动成功后，可访问：
- **API 文档**：http://localhost:8000/docs
- **健康检查**：http://localhost:8000/api/health

---

### 第三步：启动前端服务

**新开一个终端窗口**（不要关闭后端），执行：

```bash
cd frontend

# 安装前端依赖（首次运行需要）
npm install

# 启动开发服务器（默认端口 5173/5174）
npm run dev
```

✅ 前端启动成功后，访问：
- **Web 界面**：http://localhost:5173 或控制台显示的地址

---

## API 接口说明

### 🔧 系统接口

| 方法 | 接口 | 功能 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/knowledge/status` | 知识库状态 |

### 📄 文件处理接口

| 方法 | 接口 | 功能 | 说明 |
|------|------|------|------|
| POST | `/api/extract` | 全量提取个例 | 仅提取 TXT，不入向量库 |
| POST | `/api/refresh` | 增量刷新个例库 | 只处理新增/变更的 PDF |
| POST | `/api/materials` | 单文件上传 | 直接传二进制，header 传 `x-filename` |
| POST | `/api/materials/batch` | 批量上传 | multipart/form-data 格式 |

> **注意**：上传接口使用 **文档 chunk 库**，不涉及个例提取。

### 📊 向量库接口

| 方法 | 接口 | 功能 | 向量库 |
|------|------|------|--------|
| POST | `/api/index` | 重建个例向量库 | 个例库 |
| POST | `/api/documents/index` | 重建文档 chunk 库 | 文档库 |
| GET | `/api/cases` | 列出所有个例 ID | 个例库 |
| GET | `/api/cases/{case_id}` | 获取个例详情 | 个例库 |
| GET | `/api/vectors/keys` | 获取个例向量 keys | 个例库 |
| GET | `/api/documents/keys` | 获取文档 chunk keys | 文档库 |
| GET | `/api/documents/chunks/{chunk_id}` | 获取文档片段详情 | 文档库 |
| POST | `/api/vectors/delete` | 批量删除向量 | 个例库 |

### 💬 会话管理接口

| 方法 | 接口 | 功能 |
|------|------|------|
| POST | `/api/sessions` | 创建会话 |
| GET | `/api/sessions` | 列出所有会话 |
| GET | `/api/sessions/{id}` | 获取单个会话 |
| PATCH | `/api/sessions/{id}` | 更新会话标题 |
| DELETE | `/api/sessions/{id}` | 删除会话 |
| GET | `/api/sessions/{id}/messages` | 列出会话消息 |

### 📋 标准化个例接口

| 方法 | 接口 | 功能 | 说明 |
|------|------|------|------|
| POST | `/api/standard-cases/build` | 构建标准化个例层 | 从 document chunks 生成个例 |
| GET | `/api/standard-cases` | 标准化个例检索 | 支持多字段条件检索 |
| GET | `/api/standard-cases/similar` | 相似个例智能匹配 | 综合时空和灾种相似度排序 |
| GET | `/api/standard-cases/{case_id}` | 获取个例详情 | 单条个例完整信息 |

### 🤖 问答接口

| 方法 | 接口 | 功能 | 说明 |
|------|------|------|------|
| POST | `/api/query` | 非流式问答 | 一次性返回完整结果 |
| POST | `/api/query/stream` | 流式问答 | SSE 事件流，前端使用 |
| POST | `/api/agent/query` | 智能体查询 | 面向其他系统集成 |
| GET | `/api/eval/questions` | 评估问答对 | 内置标准测试集 |

---

## 前端实际调用接口统计（当前使用 12 个）

| 序号 | 接口 | 使用位置 | 功能 |
|-----|------|---------|------|
| 1 | `GET /api/health` | App.vue:98 | 系统健康检查 |
| 2 | `GET /api/documents/keys` | App.vue:104 | 文档 chunk 列表 |
| 3 | `GET /api/documents/chunks/{id}` | App.vue:122 | 查看文档片段 |
| 4 | `GET /api/eval/questions` | App.vue:116 | 加载评估问题 |
| 5 | `POST /api/materials/batch` | App.vue:237 | 批量上传 PDF |
| 6 | `GET /api/sessions` | App.vue:144 | 会话列表 |
| 7 | `GET /api/sessions/{id}/messages` | App.vue:149 | 会话消息 |
| 8 | `POST /api/sessions` | App.vue:199 | 创建会话 |
| 9 | `PATCH /api/sessions/{id}` | App.vue:192 | 自动重命名会话 |
| 10 | `POST /api/query/stream` | App.vue:297 | 流式 RAG 问答 |
| 11 | `GET /api/standard-cases` | App.vue:250 | 标准化个例多维检索 |
| 12 | `GET /api/standard-cases/similar` | App.vue:261 | 相似个例智能匹配 |

---

## 使用说明

### 1. 初始化知识库

将气象个例 PDF 文件放入 `resource/` 目录，然后调用接口：

```bash
# 方式一：文档化索引（推荐，用于 RAG 问答）
curl -X POST http://localhost:8000/api/documents/index

# 方式二：个例提取索引（用于结构化展示）
curl -X POST http://localhost:8000/api/refresh
```

### 2. 上传新 PDF

在前端点击「上传材料」按钮，或调用 API：

```bash
curl -X POST http://localhost:8000/api/materials/batch \
  -F "files=@你的文件.pdf"
```

### 3. 开始问答

在 Web 界面输入问题即可，支持：
- 自然语言提问
- 流式打字机效果
- 自动关联相关文档片段
- 自动保存会话历史

## 常见问题

### Q: 为什么上传 PDF 后看不到个例？
A: 批量上传接口使用 **文档 chunk 索引**，不是个例提取。需要个例提取请调用 `/api/refresh`。

### Q: 问答时提示 API Key 错误？
A: 请检查 `.env` 中的 `ALIBABA_CLOUD_API_KEY` 是否正确，专有云用户请同时检查 `DASHSCOPE_BASE_URL`。

### Q: 提示免费额度耗尽？
A: 系统会自动降级为纯检索模式，只返回命中的文档片段原文。

### Q: 两个向量库有什么区别？
A:
- **文档 chunk 库**：保留原文语义，适合问答
- **个例向量库**：结构化提取，适合个例展示和筛选

## 开发说明

### 后端目录结构

```
backend/app/
├── api/
│   ├── routes/
│   │   ├── system.py     # 系统接口
│   │   ├── files.py      # 文件处理
│   │   ├── vectors.py    # 向量库
│   │   ├── sessions.py   # 会话管理
│   │   └── agent.py      # 问答接口
│   └── schemas.py        # 请求/响应模型
├── services/
│   ├── retrieval.py      # 检索服务
│   ├── llm_client.py     # LLM 调用
│   ├── embedding_client.py  # 向量化
│   ├── extraction_pipeline.py  # 个例提取
│   ├── document_store.py   # 文档向量库
│   └── vector_store.py     # 个例向量库
├── models.py            # 数据模型
├── config.py            # 配置
└── main.py              # 应用入口
```

### 前端目录结构

```
frontend/src/
├── App.vue              # 主应用
├── api.js               # API 封装
└── components/
    ├── CaseList.vue     # 案例列表
    ├── ChatPanel.vue    # 聊天面板
    └── SessionList.vue  # 会话列表
```

## 许可证

本项目仅供内部研究使用。

---

**最后更新**：2026年7月
