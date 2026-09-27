# 气象个例 RAG 系统实施计划

> **给智能体执行者：** 必需子技能：使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务逐步执行本计划。步骤使用 checkbox（`- [ ]`）语法跟踪。

**目标：** 构建端到端流程，从月度 PDF 报告中抽取气象灾害个例，将每个个例保存为独立 TXT 文件，在 ChromaDB 中建立索引，并提供 Vue3 + Element Plus RAG 界面，用于回答 `weather_qa_results.json` 中的问题。

**架构：** 使用 Python 后端执行抽取、索引、检索、重排序和 RAG 流程。使用 Vue3 前端浏览个例 ID，并基于已索引的个例库进行问答。所有中间产物都保存在磁盘上，便于审计追溯。

**技术栈：** Python 3.10、FastAPI、ChromaDB、OpenAI-compatible DashScope APIs、Vue3、Element Plus、Vite

---

### 任务 1：项目骨架

**文件：**
- 新建：`requirements.txt`
- 新建：`backend/app/__init__.py`
- 新建：`frontend/package.json`
- 新建：`frontend/vite.config.js`
- 新建：`README.md`

- [ ] **步骤 1：添加后端和前端依赖清单**
- [ ] **步骤 2：添加后端/前端目录骨架**
- [ ] **步骤 3：添加本地运行说明**

### 任务 2：先写抽取测试

**文件：**
- 测试：`tests/test_case_splitter.py`
- 测试：`tests/test_extraction_pipeline.py`

- [ ] **步骤 1：为章节切分和个例 TXT 格式编写失败测试**
- [ ] **步骤 2：运行 `python -m unittest discover -s tests -v` 并确认测试按预期失败**

### 任务 3：抽取实现

**文件：**
- 新建：`backend/app/config.py`
- 新建：`backend/app/models.py`
- 新建：`backend/app/services/text_cleaning.py`
- 新建：`backend/app/services/case_splitter.py`
- 新建：`backend/app/services/pdf_reader.py`
- 新建：`backend/app/services/llm_client.py`
- 新建：`backend/app/services/extraction_pipeline.py`

- [ ] **步骤 1：实现 PDF 文本加载、规则优先切分和 LLM 精炼钩子**
- [ ] **步骤 2：实现 TXT 持久化，包含 `source_pdf`、`case_id`、`title`、`date_range` 和 `content`**
- [ ] **步骤 3：重新运行抽取测试并确认通过**

### 任务 4：先写检索测试

**文件：**
- 测试：`tests/test_vector_store.py`
- 测试：`tests/test_rag_service.py`

- [ ] **步骤 1：为向量往返写入/读取和重排序检索编写失败测试**
- [ ] **步骤 2：运行 `python -m unittest discover -s tests -v` 并确认测试按预期失败**

### 任务 5：检索实现

**文件：**
- 新建：`backend/app/services/embedding_client.py`
- 新建：`backend/app/services/vector_store.py`
- 新建：`backend/app/services/retrieval.py`

- [ ] **步骤 1：实现 embedding client 和基于 Chroma 的 case store**
- [ ] **步骤 2：实现检索和重排序流程**
- [ ] **步骤 3：重新运行测试并确认通过**

### 任务 6：API 和前端

**文件：**
- 新建：`backend/app/main.py`
- 新建：`frontend/index.html`
- 新建：`frontend/src/main.js`
- 新建：`frontend/src/App.vue`
- 新建：`frontend/src/api.js`
- 新建：`frontend/src/components/CaseList.vue`
- 新建：`frontend/src/components/ChatPanel.vue`

- [ ] **步骤 1：暴露抽取、索引、列表、详情和 RAG 查询接口**
- [ ] **步骤 2：构建简单的 Vue3 + Element Plus UI**
- [ ] **步骤 3：验证 UI 能展示已索引个例 ID，并通过后端回答查询**

### 任务 7：验证

**文件：**
- 修改：`README.md`

- [ ] **步骤 1：运行后端单元测试**
- [ ] **步骤 2：如获得批准，安装依赖并运行 API/前端验证命令**
- [ ] **步骤 3：记录实时 PDF 抽取和 DashScope 调用仍需补充的配置**
