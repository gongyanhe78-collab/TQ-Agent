# 优化 Agent 说明

本文档说明本项目围绕“气象灾害个例库智能体”已经完成的第 0 期、第一期、第二期、第三期优化。重点解释每一阶段解决什么问题、代码位于哪里、具体怎么做、做完以后有什么作用。

## 总体目标

项目最初是基于 PDF 文本切分后的 RAG 问答系统。后续优化的目标是把它从“检索几个文本片段给大模型回答”，逐步升级成“可沉淀标准化个例、支持图片证据、多维检索、相似历史个例匹配”的气象灾害个例智能体。

整体链路如下：

1. 第 0 期：把 PDF 中的图片证据接入系统。
2. 第一期：在文档 chunk 之上生成标准化个例层。
3. 第二期：让标准化个例支持多维检索，并让图片进入检索条件。
4. 第三期：做相似个例智能匹配，输出相似度、匹配理由和预报提示。

## 第 0 期：图片证据接入

### 优化目标

原系统主要处理 PDF 文本，问答时只能返回文字证据。气象灾害材料里大量关键信息在雷达图、卫星云图、降水实况图、形势场图、探空图里。如果不提取图片，大模型只能看到图注或附近文本，无法把“图像证据”作为可追溯材料返回给用户。

第 0 期的目标是：从 PDF 中提取图片，建立图片元数据索引，并在问答证据和前端界面中展示相关图片。

### 主要位置

- 图片提取服务：`backend/app/services/image_extraction.py`
- 图片索引脚本：`scripts/build_image_index.py`
- 图片文件目录：`data/document_images/`
- 图片元数据：`data/image_metadata.json`
- 图片访问接口：`backend/app/api/routes/vectors.py`
- 问答证据拼接：`backend/app/main.py`
- 前端图片展示：`frontend/src/components/ChatPanel.vue`

### 怎么做的

第 0 期采用“双轨提取”：

1. 嵌入图片提取

   从 PDF 内部抽取真实嵌入图片，例如雷达图、云图、降水图等。

2. 整页快照提取

   把 PDF 每一页渲染成整页图片。这样即使某些图不是标准嵌入图片，也可以通过整页快照保留原始版面证据。

每张图片都会生成一条元数据，包含：

- `image_id`
- `source_pdf`
- `page_no`
- `image_path`
- `extraction_type`
- `caption`
- `nearby_text`
- `related_chunk_ids`
- 图片尺寸

系统会根据 `related_chunk_ids` 把图片和文档 chunk 关联起来。用户问答时，后端检索到相关 chunk 后，会同步查找该 chunk 对应的图片证据，并返回给前端。

同时还做了浏览器可展示格式过滤，只展示 `png/jpg/jpeg/gif/webp`，避免 `.jp2` 等浏览器不能直接显示的图片造成破图。

### 有什么用

- 问答结果可以附带图片证据，不再只有文字。
- 前端证据片段和回答气泡中可以显示图片缩略图。
- 后续第二期、第三期可以把“雷达图、卫星图、降水图”等图片类型纳入检索和相似度计算。
- 用户可以通过图注和原图追溯模型回答依据。

## 第一期：标准化个例层

### 优化目标

文档 chunk 是按固定长度切分出来的文本片段，适合向量检索，但不适合做业务个例库。气象业务需要的是“一个天气过程/灾害个例”的结构化记录，比如：

- 标题
- 时段
- 灾种
- 影响区域
- 天气实况
- 预报关注点
- 来源 PDF
- 关联 chunk
- 关联图片

第一期的目标是：在 document_index 的 chunk 之上，构建标准化个例层。

### 主要位置

- 标准化个例模型：`backend/app/models.py`
- 标准化个例构建：`backend/app/services/standard_case_builder.py`
- 标准化个例存储：`backend/app/services/standard_case_store.py`
- 标准化个例构建脚本：`scripts/build_standard_cases.py`
- 标准化个例接口：`backend/app/api/routes/standard_cases.py`
- 标准化个例数据：`data/standard_cases.json`
- 测试：`tests/test_standard_cases.py`

### 怎么做的

新增了 `StandardCase` 数据结构，字段包括：

- `case_id`
- `title`
- `date_range`
- `disaster_types`
- `affected_areas`
- `source_pdf`
- `source_chunk_ids`
- `summary`
- `weather_facts`
- `forecast_focus`
- `evidence_image_ids`
- `confidence`

构建逻辑会扫描 `document_index` 中的 chunk，根据标题、日期、灾种、段落结构等规则，把属于同一过程的 chunk 聚合成标准个例。构建时也会查询图片索引，把相关图片 ID 写入 `evidence_image_ids`。

后端提供了标准化个例接口：

- `POST /api/standard-cases/build`
- `GET /api/standard-cases`
- `GET /api/standard-cases/{case_id}`

生成结果存储在 `data/standard_cases.json`，这是后续多维检索和相似匹配的基础数据层。

### 有什么用

- 系统不再只理解零散 chunk，而是理解“一个完整天气过程”。
- 后续可以按标准字段检索，比如日期、灾种、区域、来源 PDF。
- 图片证据可以挂到标准个例上，而不是只挂在文本片段上。
- 为“地市灾害个例库”提供更接近业务数据表的基础结构。

## 第二期：多维检索和图片检索

### 优化目标

用户不只会问“帮我分析某个天气过程”，还会查：

> 2025年5月山西北部雷暴大风，有雷达图的个例

这类需求不是单纯向量检索，而是结构化组合检索：

- 日期
- 灾种
- 区域
- 图片类型
- 数据类别
- 来源 PDF

第二期的目标是：让标准化个例库支持多维组合检索，并让图片类型进入检索体系。

### 主要位置

- 多维检索接口：`backend/app/api/routes/standard_cases.py`
- 多维检索逻辑：`backend/app/main.py`
- 图片分类逻辑：`backend/app/services/image_extraction.py`
- 前端检索入口：`frontend/src/components/ChatPanel.vue`
- 前端接口封装：`frontend/src/api.js`

### 怎么做的

1. 标准化个例接口增加查询参数：

   - `q`
   - `date`
   - `disaster_type`
   - `area`
   - `image_type`
   - `data_category`
   - `source_pdf`

2. 自然语言解析

   对类似“2025年5月山西北部雷暴大风，有雷达图的个例”的自然语言查询，后端会轻量解析出：

   - 日期：`5月`
   - 灾种：`雷暴大风`
   - 区域：`山西北部`
   - 图片类型：`radar`

3. 灾种组合匹配

   `雷暴大风` 会拆成 `雷暴 + 大风` 两个条件，而不是要求标准个例里必须存在完整字符串。

4. 区域别名匹配

   `山西北部` 会映射到北部地市和省级线索，比如：

   - 大同
   - 朔州
   - 忻州
   - 山西

5. 图片类型分类

   图片根据图注和附近文本被分类，例如：

   - `radar`：雷达图
   - `satellite`：卫星图
   - `precipitation`：降水图
   - `wind`：大风图
   - `sounding`：探空图
   - `synoptic`：环流形势图

6. 前端展示

   右侧“个例智能检索”面板中加入“多维检索”页签，支持自然语言输入和结构化字段输入。结果卡片展示标准个例、元数据、摘要和图片缩略图。

### 有什么用

- 用户可以像查数据库一样查标准化个例。
- 图片不只是展示材料，也成为检索条件。
- “有雷达图的个例”“有卫星图的个例”这类需求可以直接查。
- 结果更稳定，不完全依赖大模型自由发挥。

## 第三期：相似个例智能匹配

### 优化目标

相似个例匹配和多维检索不是一回事。

多维检索偏“满足条件的记录过滤”，相似匹配偏“新过程出现后，找最值得参考的历史过程”。相似匹配不仅要返回结果，还要解释为什么相似，并给出预报提示。

第三期的目标是：输入一个新的灾害过程描述，系统综合结构化字段、图片证据、时空相似度、灾种相似度和文本线索，返回 3 到 5 个相似历史个例。

### 主要位置

- 相似匹配服务：`backend/app/services/similar_case_matcher.py`
- 相似匹配入口：`backend/app/main.py`
- 相似匹配接口：`backend/app/api/routes/standard_cases.py`
- 前端页签：`frontend/src/components/ChatPanel.vue`
- 前端接口封装：`frontend/src/api.js`
- 测试：`tests/test_standard_cases.py`

### 怎么做的

新增 `SimilarCaseMatcher` 服务，输入 `SimilarCaseQuery`，输出 `SimilarCaseMatch`。

输入字段包括：

- `q`
- `date`
- `disaster_type`
- `area`
- `image_type`
- `data_category`
- `source_pdf`
- `top_n`

系统会对每个历史标准个例计算综合分，分项包括：

- `disaster`：灾种相似度
- `temporal`：时间/季节相似度
- `spatial`：区域相似度
- `image`：图片证据相似度
- `text`：文本线索相似度
- `source`：来源 PDF 匹配度

综合分采用加权计算：

- 灾种：权重较高
- 区域：权重较高
- 时间：体现同月、同日、相邻月份
- 图片：当用户明确要求雷达图、卫星图等时会成为硬条件
- 文本：补充自然语言描述里的关键词相似
- 来源：作为弱加分项

返回结果包括：

- 标准个例信息
- `similarity_score`
- `score_breakdown`
- `match_reasons`
- `forecast_tips`
- `evidence_images`

前端“个例智能检索”面板中加入“相似匹配”页签。结果卡片会展示：

- 总相似分
- 分项分
- 匹配理由
- 预报提示
- 图片证据

### 有什么用

- 新天气过程出现后，可以快速找历史参考个例。
- 不只是返回“看起来相似”的文本，而是告诉用户相似在哪里。
- 图片证据进入相似匹配，雷达图、卫星图等资料可以参与排序。
- 结果能直接服务预报复盘和业务研判。

## 模型响应速度优化

### 优化目标

原先问答体验是图片先显示，文字回答要等待较久。主要原因是检索 metadata 很快返回，但大模型生成首 token 慢。

### 主要位置

- 流式问答接口：`backend/app/api/routes/agent.py`
- 快速本地回答：`backend/app/main.py`
- LLM 客户端：`backend/app/services/llm_client.py`
- 模型配置：`backend/app/config.py`

### 怎么做的

1. 流式接口先返回检索 metadata。
2. 前端收到 metadata 后立即展示相关图片。
3. 后端立刻生成一个本地快速结论 `_fast_document_answer`，先把检索到的 chunk 和图片图注摘要返回。
4. 大模型后续继续流式补充综合分析。
5. 聊天模型配置从 DashScope 切到 DMXAPI 的 OpenAI 兼容接口：

   - `chat_base_url = https://www.dmxapi.cn/v1`
   - `chat_model = gpt-5.4-nano`
   - `chat_api_key` 优先读取 `.env` 中的 `DMXAPI`

6. 嵌入模型仍保留 DashScope `text-embedding-v4`，避免破坏现有 `document_index` 的向量维度。

### 有什么用

- 用户能更快看到文字首屏，不再只看到图片空等。
- 问答模型切到更轻量的 `gpt-5.4-nano` 后，整体生成速度预期更快。
- 向量库不需要重建，因为 embedding 模型没有被切走。

## 当前阶段关系

第 0 期解决“图片证据有没有”的问题。

第一期解决“个例结构化有没有”的问题。

第二期解决“能不能按业务字段和图片类型查”的问题。

第三期解决“新过程来了能不能找历史参考并给出理由”的问题。

这四期组合起来后，系统已经从普通 RAG 升级为一个具备材料入库、证据追溯、结构化检索、图片检索和相似个例匹配能力的气象灾害个例智能体基础版本。
