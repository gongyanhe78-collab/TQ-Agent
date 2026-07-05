# Weather Case RAG System API 接口文档

## 目录

- [概述](#概述)
- [system.py - 系统接口](#systempy---系统接口)
- [files.py - 文件处理接口](#filespy---文件处理接口)
- [vectors.py - 向量库接口](#vectorspy---向量库接口)
- [sessions.py - 会话管理接口](#sessionspy---会话管理接口)
- [agent.py - 问答接口](#agentpy---问答接口)
- [standard_cases.py - 标准化个例接口](#standard_casespy---标准化个例接口)
- [接口使用统计](#接口使用统计)

---

## 概述

本文档详细说明 Weather Case RAG System 项目中所有后端 API 接口的功能、区别及实际使用情况。

**Base URL**: `http://127.0.0.1:8000/api`

---

## system.py - 系统接口

### 接口列表

| HTTP 方法 | 接口路径 | 功能说明 |
|---------|---------|---------|
| `GET` | `/api/health` | 健康检查接口 |
| `GET` | `/api/knowledge/status` | 知识库状态接口 |

---

### 1. `GET /api/health` - 健康检查

**功能**：返回系统状态、向量库信息和模型可用性。

**返回示例**：
```json
{
  "status": "ok",
  "vector_keys": 100,
  "vector_dimension": 1024,
  "vector_source": "chroma",
  "embedding_available": true,
  "llm_available": true
}
```

---

### 2. `GET /api/knowledge/status` - 知识库状态

**功能**：返回知识库当前索引、模型和构建状态。

**返回示例**：
```json
{
  "status": "ok",
  "case_count": 100,
  "vector_count": 100,
  "vector_dimension": 1024,
  "vector_source": "chroma",
  "embedding_model": "qwen3-embedding",
  "rerank_model": "qwen3-rerank",
  "chat_model": "qwen3-turbo",
  "embedding_available": true,
  "rerank_available": true,
  "llm_available": true,
  "build": { ... }
}
```

---

### 接口区别

| 接口 | 用途 | 返回内容 |
|------|------|---------|
| `/health` | 轻量健康检查 | 向量库统计 + 模型可用性 |
| `/knowledge/status` | 完整状态监控 | 包含所有模型配置 + 构建状态 |

---

### 实际使用情况

| 接口 | 后端定义 | 前端封装 | 实际调用 | 使用位置 |
|------|---------|---------|---------|---------|
| `GET /api/health` | ✅ | ✅ | ✅ | `App.vue:97` - 初始化时检查系统状态 |
| `GET /api/knowledge/status` | ✅ | ❌ | ❌ | - |

**总结**：只有 `/api/health` 被实际使用，`/knowledge/status` 定义了但前端未调用。

---

## files.py - 文件处理接口

### 接口列表

| HTTP 方法 | 接口路径 | 功能说明 |
|---------|---------|---------|
| `POST` | `/api/extract` | 全量提取 PDF 个例 |
| `POST` | `/api/refresh` | 增量刷新案例库 |
| `POST` | `/api/materials` | 单文件上传并处理 |
| `POST` | `/api/materials/batch` | 批量上传并处理 |

---

### 接口详细说明

#### 1. `POST /api/extract` - 全量提取

**功能**：扫描 `resource` 目录下所有 PDF 文件，提取天气个例并保存为 txt 文件。
- **特点**：只做文本提取，不涉及向量化和索引
- **适用场景**：初次批量导入时单独提取

**返回**：
```json
{
  "written_files": ["..."],
  "count": 10
}
```

---

#### 2. `POST /api/refresh` - 增量刷新

**功能**：扫描 resource 目录，只对**新增或内容变更**的 PDF 执行增量提取和索引。
- **特点**：智能增量，不会重复处理已处理过的文件
- **适用场景**：定期刷新案例库

**返回**：处理的 PDF 列表、写入的文件、索引结果

---

#### 3. `POST /api/materials` - 单文件上传

**功能**：上传单个 PDF 文件，并**仅对该文件**执行文档化 chunk 索引（切分+向量化+入库）。
- **特点**：通过 request body 直接传二进制，header 传 `x-filename`
- **Content-Type**: `application/pdf`
- **注意**：使用文档向量库（`document_store`），**不涉及个例提取**

---

#### 4. `POST /api/materials/batch` - 批量上传

**功能**：批量上传多个 PDF 文件，只对本次上传的文件执行文档化 chunk 索引。
- **特点**：使用 FastAPI 标准的 `UploadFile` multipart/form-data 格式
- **Content-Type**: `multipart/form-data`
- **注意**：使用文档向量库（`document_store`），**不涉及个例提取**

---

### 接口区别对比

| 接口 | 处理范围 | 操作内容 | 向量库 | 适用场景 |
|------|---------|---------|-------|---------|
| `/extract` | resource 全部 PDF | 仅提取个例 txt | 个例库 | 初次批量导入时单独提取 |
| `/refresh` | 新增/变更 PDF | 提取个例+向量化+入库 | 个例库 | 定期刷新案例库 |
| `/materials` | 单个上传的 PDF | 文档化切chunk+向量化+入库 | 文档库 | 单文件上传处理 |
| `/materials/batch` | 本次上传的 PDF | 文档化切chunk+向量化+入库 | 文档库 | 批量上传处理 |

---

### 实际使用情况

| 接口 | 后端定义 | 前端封装 | 实际调用 | 使用位置 |
|------|---------|---------|---------|---------|
| `POST /api/extract` | ✅ | ✅ | ❌ | - |
| `POST /api/refresh` | ✅ | ✅ | ❌ | - |
| `POST /api/materials` | ✅ | ✅ | ❌ | - |
| `POST /api/materials/batch` | ✅ | ✅ | ✅ | `App.vue:234` - 批量上传 PDF |

> **注意**：`/api/refresh` 在 `api.js` 中定义了 `refreshLibrary()` 函数，但 `App.vue` 中的 `handleRefreshLibrary()` 只调用了 `loadCases()` 获取向量 keys，**没有实际调用 refresh 接口**。

**总结**：只有批量上传接口 `/api/materials/batch` 被实际使用。

---

## vectors.py - 向量库接口

### 接口列表

| HTTP 方法 | 接口路径 | 功能说明 | 向量库类型 |
|---------|---------|---------|-----------|
| `POST` | `/api/index` | 重建原个例向量库 | 个例向量库 |
| `GET` | `/api/cases` | 列出所有案例 ID | 个例向量库 |
| `GET` | `/api/vectors/keys` | 获取个例向量 keys | 个例向量库 |
| `POST` | `/api/documents/index` | 重建文档 chunk 向量库 | 文档向量库 |
| `GET` | `/api/documents/keys` | 获取文档 chunk keys | 文档向量库 |
| `GET` | `/api/cases/{case_id}` | 获取单个个例详情 | 个例向量库 |
| `GET` | `/api/documents/chunks/{chunk_id}` | 获取单个文档 chunk 详情 | 文档向量库 |
| `GET` | `/api/image-evidence/{image_id}` | 返回图片证据文件 | 图片存储 |
| `POST` | `/api/vectors/delete` | 批量删除向量 | 个例向量库 |

---

### 接口详细说明

#### 1. `POST /api/index` - 重建个例向量库

**功能**：读取 `samples` 目录下所有已提取的 `.txt` 个例文件，生成向量并**全量重建**原个例向量库。
- **特点**：会清空原有集合后重新写入（`replace_cases`）

---

#### 2. `GET /api/cases` - 列出所有案例 ID

**功能**：获取原个例向量库中的所有案例 ID 列表及集合信息。

**返回**：
```json
{
  "case_ids": ["..."],
  "count": 100,
  "dimension": 1024,
  "source": "chroma"
}
```

---

#### 3. `GET /api/vectors/keys` - 获取个例向量 keys

**功能**：只读取原个例向量库中的 key，**不会触发任何 PDF 扫描、抽取或入库操作**。
- **特点**：比 `/api/cases` 更轻量

**返回**：
```json
{
  "case_ids": ["..."],
  "count": 100,
  "dimension": 1024,
  "source": "chroma",
  "vector_count": 100
}
```

---

#### 4. `POST /api/documents/index` - 重建文档 chunk 向量库

**功能**：将 `resource` 目录下 PDF 进行文档化处理、切分 chunk，然后写入**独立的文档向量库**。
- **特点**：与个例向量库是两个独立的 Chroma 集合

---

#### 5. `GET /api/documents/keys` - 获取文档 chunk keys

**功能**：读取文档向量库中的所有 chunk ID。

**返回**：
```json
{
  "chunk_ids": ["..."],
  "count": 100,
  "dimension": 1024,
  "source": "chroma",
  "vector_count": 100
}
```

---

#### 6. `GET /api/cases/{case_id}` - 获取单个个例详情

**功能**：根据 case_id 获取完整的个例内容（标题、时段、正文等）。

**返回**：
```json
{
  "source_pdf": "...",
  "case_id": "...",
  "case_no": 1,
  "title": "...",
  "date_range": "...",
  "content": "...",
  "file_path": "..."
}
```

---

#### 7. `GET /api/documents/chunks/{chunk_id}` - 获取单个文档 chunk 详情

**功能**：根据 chunk_id 获取完整的文档片段内容。

**返回**：
```json
{
  "source_pdf": "...",
  "chunk_id": "...",
  "chunk_no": 1,
  "content": "...",
  "file_path": "...",
  "images": [...]
}
```

---

#### 8. `GET /api/image-evidence/{image_id}` - 返回图片证据文件

**功能**：根据 image_id 返回已登记的图片证据文件（PDF 中提取的图片）。
- **安全机制**：路径验证，防止目录遍历攻击
- **返回**：图片文件二进制流（FileResponse）

---

#### 9. `POST /api/vectors/delete` - 批量删除向量

**功能**：按向量 key 批量删除原个例向量库中的内容。

**请求体**：
```json
{
  "keys": ["key1", "key2", "..."]
}
```

---

### 接口区别对比

#### 个例向量库 vs 文档向量库

| 维度 | 个例向量库 (case_store) | 文档向量库 (document_store) |
|------|-------------------------|----------------------------|
| 数据来源 | TXT 个例文件（LLM 提取结构化结果） | 原始 PDF 直接切分 chunk |
| 内容粒度 | 完整天气个例（结构化） | 文档片段（半结构化） |
| 索引接口 | `/api/index` | `/api/documents/index` |
| 查询接口 | `/api/cases`、`/api/vectors/keys` | `/api/documents/keys` |
| 详情接口 | `/api/cases/{case_id}` | `/api/documents/chunks/{chunk_id}` |

#### `/api/cases` vs `/api/vectors/keys`

| 接口 | 返回内容 | 状态 |
|------|---------|------|
| `/api/cases` | case_ids + count + collection_info | 旧版本，已被取代 |
| `/api/vectors/keys` | case_ids + count + dimension + source + vector_count | 当前使用 |

---

### 实际使用情况

| 接口 | 后端定义 | 前端封装 | 实际调用 | 使用位置 |
|------|---------|---------|---------|---------|
| `POST /api/index` | ✅ | ✅ | ❌ | - |
| `GET /api/cases` | ✅ | ✅ | ❌ | - |
| `GET /api/vectors/keys` | ✅ | ✅ | ❌ | - |
| `POST /api/documents/index` | ✅ | ❌ | ❌ | - |
| `GET /api/documents/keys` | ✅ | ✅ | ✅ | `App.vue:101` - 加载向量节点列表 |
| `GET /api/cases/{case_id}` | ✅ | ✅ | ❌ | - |
| `GET /api/documents/chunks/{chunk_id}` | ✅ | ✅ | ✅ | `App.vue:120` - 查看文档 chunk 详情 |
| `GET /api/image-evidence/{image_id}` | ✅ | ❌ | ✅ | ChatPanel 中展示图片 |
| `POST /api/vectors/delete` | ✅ | ❌ | ❌ | - |

**总结**：实际使用的有 3 个接口：
1. `GET /api/documents/keys` - 加载文档 chunk 节点列表（原文档误写为 `/api/vectors/keys`）
2. `GET /api/documents/chunks/{chunk_id}` - 查看文档 chunk 详情（原文档误写为 `/api/cases/{case_id}`）
3. `GET /api/image-evidence/{image_id}` - 展示图片证据

---

## sessions.py - 会话管理接口

### 接口列表

| HTTP 方法 | 接口路径 | 功能说明 |
|---------|---------|---------|
| `POST` | `/api/sessions` | 创建聊天会话 |
| `GET` | `/api/sessions` | 列出所有会话 |
| `GET` | `/api/sessions/{session_id}` | 获取单个会话 |
| `PATCH` | `/api/sessions/{session_id}` | 更新会话标题 |
| `DELETE` | `/api/sessions/{session_id}` | 删除会话 |
| `GET` | `/api/sessions/{session_id}/messages` | 列出会话消息 |

---

### 接口详细说明

#### 1. `POST /api/sessions` - 创建会话

**功能**：创建一个新的聊天会话，初始标题默认为"新会话"。

**请求体**：
```json
{
  "title": "新会话"
}
```

---

#### 2. `GET /api/sessions` - 列出所有会话

**功能**：获取所有聊天会话列表（不含消息内容）。

**返回**：
```json
{
  "sessions": [
    {
      "session_id": "...",
      "title": "...",
      "created_at": "...",
      "updated_at": "..."
    }
  ]
}
```

---

#### 3. `GET /api/sessions/{session_id}` - 获取单个会话

**功能**：获取指定会话的详情信息。

---

#### 4. `PATCH /api/sessions/{session_id}` - 更新会话标题

**功能**：修改会话标题，用于根据对话内容自动重命名会话。

**请求体**：
```json
{
  "title": "新标题"
}
```

---

#### 5. `DELETE /api/sessions/{session_id}` - 删除会话

**功能**：删除会话及其所有消息记录。

---

#### 6. `GET /api/sessions/{session_id}/messages` - 列出会话消息

**功能**：获取指定会话中的所有聊天消息记录。

**返回**：
```json
{
  "session_id": "...",
  "messages": [
    {
      "message_id": "...",
      "role": "user",
      "content": "...",
      "created_at": "..."
    }
  ]
}
```

---

### 接口区别对比

| 接口 | 操作类型 | 返回内容 | 主要用途 |
|------|---------|---------|---------|
| `POST /sessions` | 写操作 | 单会话对象 | 用户点击"新建会话"按钮 |
| `GET /sessions` | 读操作 | 会话列表（无消息） | 左侧会话列表展示 |
| `GET /sessions/{id}` | 读操作 | 单会话详情 | 获取特定会话元信息 |
| `PATCH /sessions/{id}` | 写操作 | 更新后的会话 | 根据首条问题自动重命名 |
| `DELETE /sessions/{id}` | 写操作 | 删除确认 | 删除不需要的会话 |
| `GET /sessions/{id}/messages` | 读操作 | 消息列表 | 切换会话时加载历史消息 |

---

### 实际使用情况

| 接口 | 后端定义 | 前端封装 | 实际调用 | 使用位置 |
|------|---------|---------|---------|---------|
| `POST /api/sessions` | ✅ | ✅ | ✅ | `App.vue:196` - 新建会话 |
| `GET /api/sessions` | ✅ | ✅ | ✅ | `App.vue:143` - 加载会话列表 |
| `GET /api/sessions/{id}` | ✅ | ❌ | ❌ | - |
| `PATCH /api/sessions/{id}` | ✅ | ✅ | ✅ | `App.vue:188` - 自动重命名会话 |
| `DELETE /api/sessions/{id}` | ✅ | ✅ | ❌ | - |
| `GET /api/sessions/{id}/messages` | ✅ | ✅ | ✅ | `App.vue:148` - 加载历史消息 |

**总结**：实际使用的有 4 个接口：
1. `GET /api/sessions` - 会话列表
2. `GET /api/sessions/{id}/messages` - 会话消息
3. `POST /api/sessions` - 创建会话
4. `PATCH /api/sessions/{id}` - 更新会话标题

---

## agent.py - 问答接口

### 接口列表

| HTTP 方法 | 接口路径 | 功能说明 |
|---------|---------|---------|
| `POST` | `/api/query` | 非流式 RAG 查询 |
| `POST` | `/api/agent/query` | 智能体结构化查询 |
| `POST` | `/api/query/stream` | 流式问答接口 |
| `GET` | `/api/eval/questions` | 评估问答对 |

---

### 接口详细说明

#### 1. `POST /api/query` - 非流式 RAG 查询

**功能**：为前端执行非流式 RAG 查询，一次性返回完整结果。
- **使用向量库**：文档 chunk 向量库（`document_store`）
- **检索逻辑**：`_retrieve_document_chunks()` → 向量检索 → LLM 问答

**请求体**：
```json
{
  "question": "问题内容",
  "top_k": 10,
  "top_n": 3
}
```

**返回**：
```json
{
  "question": "...",
  "answer": "...",
  "retrieval_mode": "hybrid",
  "llm_used": true,
  "llm_status": "called",
  "hit_count": 3,
  "hits": [...]
}
```

---

#### 2. `POST /api/agent/query` - 智能体结构化查询

**功能**：面向其他智能体项目的结构化 RAG 问答接口，支持会话关联和元数据。

**请求体**：
```json
{
  "question": "问题内容",
  "top_k": 10,
  "top_n": 3,
  "session_id": "可选",
  "return_context": true,
  "metadata": {}
}
```

---

#### 3. `POST /api/query/stream` - 流式问答接口

**功能**：流式问答接口，使用 SSE 返回 metadata、delta、done 和 error 事件。

**事件类型**：
- `metadata` - 检索结果元信息
- `delta` - 回答文本片段
- `done` - 流式结束
- `error` - 错误信息

---

#### 4. `GET /api/eval/questions` - 评估问答对

**功能**：从 `weather_qa_results.json` 读取预设问答对，用于效果评估。

**返回**：
```json
{
  "questions": [
    {
      "pdf_filename": "...",
      "question": "...",
      "answer": "..."
    }
  ]
}
```

---

### 接口区别对比

| 接口 | 响应方式 | 会话支持 | 适用场景 |
|------|---------|---------|---------|
| `/api/query` | 非流式 | ❌ | 简单测试、脚本调用 |
| `/api/agent/query` | 非流式 | ✅ | 其他智能体调用 |
| `/api/query/stream` | SSE 流式 | ✅ | 前端用户交互 |
| `/api/eval/questions` | 非流式 | ❌ | 效果评估 |

---

### 实际使用情况

| 接口 | 后端定义 | 前端封装 | 实际调用 | 使用位置 |
|------|---------|---------|---------|---------|
| `POST /api/query` | ✅ | ✅ | ❌ | - |
| `POST /api/agent/query` | ✅ | ❌ | ❌ | - |
| `POST /api/query/stream` | ✅ | ✅ | ✅ | `App.vue:280` - 流式问答 |
| `GET /api/eval/questions` | ✅ | ✅ | ✅ | `App.vue:115` - 加载评估问题 |

**总结**：实际使用的有 2 个接口：
1. `POST /api/query/stream` - 流式问答（主要功能）
2. `GET /api/eval/questions` - 加载评估问题

---

## standard_cases.py - 标准化个例接口

### 接口列表

| HTTP 方法 | 接口路径 | 功能说明 |
|---------|---------|---------|
| `POST` | `/api/standard-cases/build` | 构建标准化个例层 |
| `GET` | `/api/standard-cases` | 列出/检索标准化个例 |
| `GET` | `/api/standard-cases/similar` | 相似个例智能匹配 |
| `GET` | `/api/standard-cases/{case_id}` | 获取单条标准化个例详情 |

---

### 接口详细说明

#### 1. `POST /api/standard-cases/build` - 构建标准化个例

**功能**：从 document_index 的 chunk 构建第一期标准化个例层。
- **查询参数**：`use_llm`（boolean，默认 true）- 是否使用 LLM 辅助构建

**返回**：构建结果统计

---

#### 2. `GET /api/standard-cases` - 列出/检索标准化个例

**功能**：列出标准化个例，支持字段、图片类型和自然语言组合检索。

**查询参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `q` | string | 自然语言查询，会自动解析出灾种、区域、时段等 |
| `date` | string | 日期过滤 |
| `disaster_type` | string | 灾害类型过滤 |
| `area` | string | 区域过滤 |
| `image_type` | string | 图片类型过滤（如：雷达图、卫星云图等） |
| `data_category` | string | 数据分类过滤 |
| `source_pdf` | string | 来源 PDF 过滤 |

**返回**：
```json
{
  "count": 10,
  "cases": [...]
}
```

---

#### 3. `GET /api/standard-cases/similar` - 相似个例智能匹配

**功能**：相似个例智能匹配，综合结构化字段、图像证据、时空和灾种相似度排序。

**查询参数**：与 `/api/standard-cases` 相同，额外增加：
| 参数 | 类型 | 说明 |
|------|------|------|
| `top_n` | int | 返回数量（1-5，默认 5） |

**返回**：
```json
{
  "count": 5,
  "matches": [
    {
      "case": {...},
      "similarity_score": 0.85,
      "match_reasons": [...]
    }
  ]
}
```

---

#### 4. `GET /api/standard-cases/{case_id}` - 获取单条标准化个例详情

**功能**：获取单条标准化个例的完整详情。

**返回**：标准化个例完整对象

---

### 实际使用情况

| 接口 | 后端定义 | 前端封装 | 实际调用 | 使用位置 |
|------|---------|---------|---------|---------|
| `POST /api/standard-cases/build` | ✅ | ❌ | ❌ | - |
| `GET /api/standard-cases` | ✅ | ✅ | ✅ | `App.vue:247` - 多维检索标准化个例 |
| `GET /api/standard-cases/similar` | ✅ | ✅ | ✅ | `App.vue:258` - 相似个例智能匹配 |
| `GET /api/standard-cases/{case_id}` | ✅ | ❌ | ❌ | - |

**总结**：实际使用的有 2 个接口：
1. `GET /api/standard-cases` - 多维检索标准化个例
2. `GET /api/standard-cases/similar` - 相似个例智能匹配

---

## 接口使用统计

### 总体统计

| 接口文件 | 接口总数 | 实际使用数 | 使用率 |
|---------|---------|-----------|-------|
| system.py | 2 | 1 | 50% |
| files.py | 4 | 1 | 25% |
| vectors.py | 9 | 3 | 33% |
| sessions.py | 6 | 4 | 67% |
| agent.py | 4 | 2 | 50% |
| standard_cases.py | 4 | 2 | 50% |
| **总计** | **29** | **13** | **45%** |

---

### ✅ 正在使用的接口（13个）

| 序号 | 接口 | 使用位置 | 功能 |
|-----|------|---------|------|
| 1 | `GET /api/health` | App.vue:97 | 系统健康检查 |
| 2 | `POST /api/materials/batch` | App.vue:234 | 批量上传 PDF |
| 3 | `GET /api/documents/keys` | App.vue:101 | 文档 chunk 节点列表 |
| 4 | `GET /api/documents/chunks/{chunk_id}` | App.vue:120 | 文档 chunk 详情 |
| 5 | `GET /api/image-evidence/{image_id}` | ChatPanel | 图片证据展示 |
| 6 | `POST /api/sessions` | App.vue:196 | 创建会话 |
| 7 | `GET /api/sessions` | App.vue:143 | 会话列表 |
| 8 | `PATCH /api/sessions/{id}` | App.vue:188 | 更新会话标题 |
| 9 | `GET /api/sessions/{id}/messages` | App.vue:148 | 会话消息 |
| 10 | `POST /api/query/stream` | App.vue:280 | 流式问答 |
| 11 | `GET /api/eval/questions` | App.vue:115 | 评估问题 |
| 12 | `GET /api/standard-cases` | App.vue:247 | 多维检索标准化个例 |
| 13 | `GET /api/standard-cases/similar` | App.vue:258 | 相似个例智能匹配 |

---

### ❌ 定义但未使用的接口（16个）

1. `GET /api/knowledge/status` - 知识库状态
2. `POST /api/extract` - 全量提取
3. `POST /api/refresh` - 增量刷新
4. `POST /api/materials` - 单文件上传
5. `POST /api/index` - 重建个例向量库
6. `GET /api/cases` - 案例列表（旧版）
7. `GET /api/vectors/keys` - 个例向量 keys
8. `POST /api/documents/index` - 重建文档向量库
9. `GET /api/cases/{case_id}` - 个例详情
10. `POST /api/vectors/delete` - 批量删除向量
11. `GET /api/sessions/{id}` - 单个会话详情
12. `DELETE /api/sessions/{id}` - 删除会话
13. `POST /api/query` - 非流式查询
14. `POST /api/agent/query` - 智能体查询
15. `POST /api/standard-cases/build` - 构建标准化个例
16. `GET /api/standard-cases/{case_id}` - 单条标准化个例详情

---

## 说明

本文档基于当前代码库（2026年7月）生成，反映项目实际接口定义和使用情况。部分接口虽已定义但前端未使用，可能是为未来功能预留或用于测试/脚本调用。

**重要更正**：原文档中 vectors.py 部分接口标注有误，实际前端调用的是文档 chunk 相关接口（`/api/documents/keys` 和 `/api/documents/chunks/{chunk_id}`），而非个例接口。
