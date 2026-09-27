# 个例多维检索智能体（共享本地知识库模式）

本模块与 `Smart_Case_Match` 共用 `backend/app/services/agent/data/` 中的本地知识库，不调用公司知识库 HTTP 接口。共享资料包括：

- `standard_cases.json`：标准化个例；
- `document_index/chunks.json`：原文分块；
- `image_metadata.json` 与 `document_images/`：图片证据。

## 运行配置

```env
CASE_MULTIDIM_RUNTIME_DIR=backend/app/services/agent/case_multidim_search/runtime
```

`CASE_MULTIDIM_RUNTIME_DIR` 不是知识库目录，只保存报告、图表、分析缓存和自然语言会话数据；不配置时默认使用模块内的 `runtime/` 目录。

## 数据流程

1. 前端提交时间、灾种、影响区域等结构化条件。
2. `router.py` 创建使用共享本地知识库的 `CaseMultidimSearchAgent`。
3. `LocalStandardCaseStore` 读取 `standard_cases.json`，并交由 `StructuredCaseRetriever` 执行确定性筛选。
4. `LocalDocumentChunkStore` 按命中个例的 `source_chunk_ids` 读取 `chunks.json` 中的原文证据。
5. `LocalImageEvidenceStore` 按 `evidence_image_ids` 读取图片元数据并解析本地图片路径。
6. 原有 agent 逻辑继续完成强度提取、rerank、逐例分析、综合概况、图表、CSV 和 PDF 导出。

## 目录结构

```text
agent/
├── data/                            # 两个 Agent 共用的本地知识库
├── case_multidim_search/
│   ├── integrations/local_stores.py # 本地资料适配为多维检索的既有 Store 接口
│   └── runtime/                     # 报告、图表和分析缓存
└── Smart_Case_Match/
```

## 关键约束

- 检索的时间、灾种、区域筛选语义保持不变。
- 强度指标仍从命中个例关联 chunk 的原文中抽取。
- 图片展示仍遵循“模型正文实际引用哪些图，就展示哪些图”的原有对齐逻辑。
- 在线请求只读共享本地知识库，不会改写资料或向量索引。

## 简单验证

```powershell
python -m py_compile backend\app\services\agent\case_multidim_search\integrations\local_stores.py backend\app\services\agent\case_multidim_search\router.py backend\app\services\agent\case_multidim_search\agent.py
python -m unittest backend.app.services.agent.case_multidim_search.tests.test_local_data_source
```
