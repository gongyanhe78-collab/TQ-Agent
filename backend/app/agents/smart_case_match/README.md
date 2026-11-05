# 相似个例智能匹配体（Smart Case Match Agent）

## 功能定位

本模块面向新灾害过程，使用本地标准化个例库和本地 chunk 向量库，智能选取 3～5 个高质量历史相似个例，并输出匹配理由、逐例参考经验、关键差异、历史证据和跨个例综合预报提示。

- 数据源：本目录下 `data/`（本地标准化个例 + 段落向量库 + 文档图片）
- 技术栈：LangGraph + Embedding + Rerank + LLM
- 接入方式：FastAPI 路由（结构化同步接口 + 自然语言流式接口）

---

## 目录结构与各文件职责

```text
Smart_Case_Match/
├── agent.py、router.py、schemas.py  # 稳定业务入口、HTTP 路由和对外协议
├── workflow/                        # LangGraph 图、节点和进度状态
├── matching/                        # 条件标准化、召回、评分、Rerank、选择和证据富化
├── llm/                             # 模型调用、结构化协议和文本质量控制
├── infrastructure/                  # 本地数据读取、审计日志和 SSE 编码
├── pages/                           # 结构化工作台和自然语言页面
├── data/                            # 标准个例、chunk 向量、图片和离线建库脚本
├── tests/                           # Agent、模型链路、质量和评分测试
└── docs/                            # 设计审查、合并记录和问题记录
```

`workflow/graph.py` 负责 8 节点 LangGraph 拓扑，`workflow/nodes.py` 负责节点实现；`matching/` 只处理候选生成与证据准备；`llm/` 集中处理模型调用与输出约束；`infrastructure/` 不包含业务决策，只提供本地资料、审计和流式基础能力。

---

## 业务流程（LangGraph 8 节点）

整体分为两大阶段：**选得准**（召回与排序）→ **用得好**（富化与研判）

```
┌────────────────── 第一阶段：选得准 ──────────────────┐
│  normalize_input  →  structured_recall  →           │
│  输入标准化          结构化召回（落区/灾种/季节）    │
│                                                      │
│  semantic_supplement  →  rank_and_select             │
│  语义补充+Rerank        融合排序+LLM近分重排+选取     │
└──────────────────────────────────────────────────────┘
                        ↓
┌────────────────── 第二阶段：用得好 ──────────────────┐
│  enrich_cases  →  extract_case_references  →        │
│  个例富化          逐例参考提炼                       │
│                                                      │
│  synthesize_forecast_tips  →  assemble_result        │
│  跨个例综合研判              结果组装与校验           │
└──────────────────────────────────────────────────────┘
```

各节点详细说明：

1. **normalize_input**：合并表单字段与自然语言补充，构建统一查询画像（灾种/区域/日期/语义文本）。
2. **structured_recall**：按落区、灾种、时间维度对全库个例计算结构化分，保留前 15～40 个候选。
3. **semantic_supplement**：在结构化候选范围内计算 Embedding 向量分 + 专用 Rerank 精排。
4. **rank_and_select**：结构化分与语义分融合，LLM 对近分候选微调排序，按质量门槛和多样性选出 top_n。
5. **enrich_cases**：为入选个例准备相关正文、证据图片、过程前历史预警。
6. **extract_case_references**：逐个例独立调用 LLM，提炼参考点、相似点、差异和证据引用（双并发）。
7. **synthesize_forecast_tips**：跨个例综合，生成核心结论摘要和 3～6 条预报提示。
8. **assemble_result**：校验证据，组装为 `SmartCaseMatchResponse` 稳定输出契约。

---

## 接口说明

### 基础接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/smart-case-match/health` | 健康检查（本地资料 + 模型客户端就绪状态） |
| GET | `/api/smart-case-match/chat` | 结构化工作台页面 |
| GET | `/api/smart-case-match/conversation` | 自然语言流式聊天页面 |
| GET | `/api/smart-case-match/progress/{progress_id}` | 查询任务进度 |
| GET | `/api/smart-case-match/assets/chunks/{chunk_id}` | 文字证据按需加载 |
| GET | `/api/smart-case-match/assets/images/{image_id}` | 图片证据按需加载 |

### 同步搜索接口

```
POST /api/smart-case-match/search
Content-Type: application/json
```

请求体（`SmartCaseMatchRequest`）：

```json
{
  "process_name": "晋南持续性强降水过程",
  "start_date": "2026-07-10",
  "end_date": "2026-07-12",
  "disaster_types": ["暴雨", "短时强降水"],
  "affected_areas": ["临汾", "运城"],
  "observation_description": "晋南出现持续性降水，局地小时雨强较大",
  "circulation_description": "副高边缘暖湿气流与低层切变共同影响",
  "intensity_description": "",
  "top_n": 3,
  "diversity_mode": "moderate",
  "include_images": true,
  "progress_id": "smart_case_example_001"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `process_name` | string | 否 | 过程名称，最长 120 字 |
| `start_date` / `end_date` | string | 否 | 过程起止日期 |
| `date` | string | 否 | 自然语言日期补充 |
| `disaster_types` | string[] | 否 | 灾害类型列表 |
| `affected_areas` | string[] | 否 | 影响地市或区域（支持晋南/晋北/山西中部/全省等别名） |
| `observation_description` | string | 否 | 实况描述，最长 6000 字 |
| `circulation_description` | string | 否 | 环流形势描述，最长 6000 字 |
| `intensity_description` | string | 否 | 强度描述，最长 3000 字 |
| `raw_query` | string | 否 | 自然语言补充描述 |
| `top_n` | int | 否 | 返回数量，3～5，默认 3 |
| `diversity_mode` | string | 否 | 多样性控制：`off` / `moderate`（默认） |
| `include_images` | bool | 否 | 是否返回证据图片，默认 true |
| `progress_id` | string | 否 | 进度跟踪编号，不传自动生成 |

> 至少需提供灾种、区域或日期中的一项，否则返回 422。

响应体为 `SmartCaseMatchResponse`，包含 `matched_cases`、`forecast_summary`、`forecast_tips`、`warnings`、`audit` 等字段。

### 自然语言流式接口

```
POST /api/smart-case-match/conversation/stream
Content-Type: application/json
Accept: text/event-stream
```

请求体（`NaturalLanguageMatchRequest`）：

```json
{
  "message": "2026年7月15日至16日，太原、晋中可能出现暴雨和短时强降水，低层切变配合西南急流，局地小时雨强较大。",
  "progress_id": "company_chat_123456"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `message` | string | 是 | 自然语言天气过程描述，2～6000 字 |
| `progress_id` | string | 否 | 进度跟踪编号 |

**SSE 事件顺序**：

```text
accepted          任务已接收，开始理解自然语言
  → parsed_query  自然语言解析完成，返回提取的结构化字段
  → stage         各阶段进度（节点名 + 消息 + 百分比）
  → matched_cases (phase=referenced) 匹配个例 + 参考经验 + 证据
  → forecast      综合摘要 + 预报提示
  → completed     最终完整结果
  → error         （失败时）错误信息
```

---

## 启动与配置

### 启动方式

从项目根目录启动 FastAPI 应用：

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

访问工作台：

```text
结构化页面：http://127.0.0.1:8000/api/smart-case-match/chat
聊天页面：  http://127.0.0.1:8000/api/smart-case-match/conversation
```

### 配置说明

本模块通过以下客户端工厂函数获取模型连接，配置在项目全局 config 中：

- `get_embedding_client()` — Embedding 模型（向量计算）
- `get_rerank_client()` — Rerank 模型（语义精排）
- `get_llm_client()` — 聊天模型（解析/重排/提炼/综合）

Agent 支持构造时注入自定义客户端，便于测试和迁移：

```python
agent = SmartCaseMatchAgent(
    data_dir=Path("./custom_data"),
    embedding_client=my_embedding,
    rerank_client=my_rerank,
    llm_client=my_llm,
    audit_log_dir=Path("./custom_logs"),
)
```

### 运行测试

```powershell
python -m unittest backend.app.services.agent.Smart_Case_Match.tests.test_agent -v
```

---

## 评分口径

### 结构化分

```text
structured_score = area_score × 0.50 + disaster_score × 0.30 + temporal_score × 0.20
```

- 缺省维度不参与计算，其余权重自动归一化。
- 落区分 = 0.65 × 查询区域覆盖率 + 0.35 × Jaccard 重合度。
- 灾种分使用灾种族谱处理父子关系，支持强对流/雷暴大风/短时强降水/暴雨等部分匹配。
- 时间分以月份为主，12 月与 1 月按相邻月份处理；持续天数可做最多 15% 修正。
- 历史年份只用于同分解释，不压过季节相似度。

### 语义分与融合分

```text
semantic_score = 向量余弦分 × 0.70 + Rerank 顺序分 × 0.30
（Rerank 不可用时保留前两个向量 chunk 的平均分）
retrieval_score = structured_score × 0.55 + semantic_score × 0.45
```

Embedding 不可用或维度不一致时，直接使用结构化分，不删除有效候选。

### LLM 重排约束

LLM 只允许在融合分差不超过 **0.08** 的连续候选组内微调排序。
结构化分低于 **0.30** 的个例不进入合格集合（0.15～0.30 仅保留 1 个兜底）。

---

## 降级策略

| 故障点 | 降级方式 |
|--------|----------|
| Embedding 失败 | 使用纯结构化排序 |
| Rerank 失败 | 保留向量分排序 |
| LLM 候选重排失败 | 保留融合分顺序 |
| 单个例提炼失败 | 使用规则句子摘要 |
| 综合生成失败 | 按逐例参考生成保守提示，降低置信度 |
| 高质量个例不足 | 返回实际数量 + warnings 告警，不强行补足 |

---

## 模型调用与可观测性

一次完整匹配依次产生以下外部模型调用：

1. **1 次 Embedding** — 生成查询向量
2. **1 次 Rerank** — 精排候选 chunk（最多 100 个）
3. **1 次 LLM 候选重排** — 输入前 15 个候选的正文
4. **N 次 LLM 逐例提炼** — 每个最终个例 1 次（双并发）
5. **1 次 LLM 综合研判** — 跨个例生成摘要和提示

日志标记：`[SmartCaseMatch][Embedding]` / `[Rerank]` / `[LLM]`，只记录任务名、模型名、数量、耗时和错误，不含密钥和完整业务正文。

审计字段记录在 `audit` 中：`embedding_status`、`semantic_rerank_status`、`llm_rerank_status`、`reference_statuses`、`synthesis_status` 等。

---

## 个例选取详细设计

个例选取不是一次向量检索，而是**"结构化约束优先 → 语义信息补充 → 融合排序 → LLM 近分重排 → 质量门槛截断"**的五层流程。当前本地标准化个例库由 `data/standard_cases.json` 提供，候选正文由 `data/document_index/chunks.json` 提供，`data_store.py` 会同时建立 `chunk_id → case_id` 的多值反向索引。

### 1. 输入标准化与查询画像

`normalize_input` 节点调用 `matching.normalize_query`，把页面字段合并为统一查询画像：

- 灾害类型：使用页面勾选值，并从过程名称、实况、环流、强度和补充描述中补充识别灾种关键词。
- 影响区域：保留地市、晋南、晋北、山西中部、全省等输入；区域别名会展开为地市集合。
- 时间：优先使用开始日期，辅以结束日期和自然语言日期；内部保存月份列表，用于季节相似度计算。
- 语义文本：把灾种、区域、日期、实况描述、环流描述和强度描述拼成 `query_text`，供 Embedding、Rerank 和 LLM 使用。
- 业务参数：保留 `top_n`、`diversity_mode`、`include_images`，并在后续节点中继续传递。

这一步只做字段归一化和查询画像构建，不判断个例，也不调用生成模型。

### 2. 阶段一：本地结构化召回

`structured_recall` 对本地全部标准化个例逐一计算结构化分，主导维度是落区、灾种和月份/季节。

#### 2.1 落区分

落区分由"查询区域覆盖率"和"集合 Jaccard 重合度"组成：

```text
area_score = 0.65 × query_coverage + 0.35 × jaccard_overlap
```

代码中的实际规则包括：

- 晋南、晋北、山西中部等区域别名会展开为对应地市后再计算。
- 查询为具体地市时，历史个例为全省型落区会被限制最高为 `0.68`，避免"全省过程"压过落区更贴近的区域过程。
- 非全省型个例存在有效重叠时额外加 `0.08`，体现局地落区匹配的业务价值。
- 查询明确了影响区域，但个例有落区字段且完全不重合时，结构化总分再乘 `0.5`。

因此，落区完全不合适的个例不会因为文字相似而轻易进入前列。

#### 2.2 灾种分

灾种分同时参考历史个例标题和标准化灾种字段，并使用灾种族谱处理父子关系：

- 标题中出现查询灾种，给最高匹配分。
- 标题中出现同一灾种族的近邻类型，给次高分。
- 只有标准化字段命中时，给中等匹配分。
- 对"强对流—雷暴大风—短时强降水—暴雨"等相关类型允许部分匹配。
- 如果标题明确是另一类灾种，而正文标签只是宽泛误抽结果，灾种分上限限制为 `0.35`。
- 查询明确给出灾种且灾种分低于 `0.45` 时，结构化总分乘 `0.35`，防止高温、寒潮等明显不相关过程被语义文本抬高。

#### 2.3 时间分

时间只以月份和季节为主，不把年份作为主要排序因素：

- 同月：`1.0`
- 相邻月份：`0.72`
- 相隔两个月：`0.5`
- 相隔三个月：`0.3`
- 更远月份：`0.12`
- 月份距离按环形计算，因此 12 月和 1 月视为相邻月份。
- 如果新过程和历史个例都能解析出持续天数，再用持续时间相似度对月份分做最多 15% 的修正。
- 历史年份只写入 `historical_year`，用于结果解释和同分理解，不压过季节相似度。

#### 2.4 结构化候选数量

当前数据量下，代码不是固定留下 30～40 个，而是动态保留：

```text
candidate_count = min(40, max(15, ceil(案例总数 × 0.3)))
```

也就是说：候选库较小时至少保留 15 个，候选库扩大后最多保留 40 个，约保留本地库前 30% 的结构化候选。当前候选阶段只负责排除明显不相关个例，不在这里做最终 3～5 个截断。

### 3. 阶段二：候选内语义补充和 Rerank

当前实现与最初设想有一个重要区别：暂时不对全库独立检索前 80～100 个 chunk，而是在阶段一留下的候选个例的 `source_chunk_ids` 范围内计算语义分。这是因为当前数据源是本地库，且代码优先保证落区和灾种约束不被语义检索冲掉。

#### 3.1 Embedding 语义评分

`semantic_supplement` 使用 `query_text` 调用 Embedding 模型，然后只遍历结构化候选关联的正式 chunk：

- 每个候选个例按其 `source_chunk_ids` 找到正式正文。
- 计算查询向量与这些 chunk 向量的余弦相似度。
- 每个个例取最高的两个 chunk 作为代表，计算初始 `semantic_score`。
- 同时保留前四个语义命中 chunk，供专用 Rerank 和候选 LLM 重排使用。
- 语义分不会把结构化候选集合扩大到候选集合之外。

#### 3.2 专用 Rerank 精排

`semantic_rerank.py` 将候选个例的语义命中 chunk 去重后汇总，最多提交 100 个 chunk 给专用 Rerank 模型。每个 Rerank 文档包含：

- `case_id`
- `chunk_id`
- `source_pdf`
- chunk 正文
- 原始余弦相似度

Rerank 返回顺序后，每个个例取其中排名最靠前的两个代表 chunk，并按以下方式更新语义分：

```text
semantic_score = 0.70 × 代表 chunk 平均余弦分
               + 0.30 × Rerank 顺序分
```

因此，Rerank 只负责在结构化候选内部识别更有语义参考价值的正文片段，不改变落区和灾种的候选边界。

### 4. 阶段三：结构化分与语义分融合

`fuse_scores` 对每个候选计算最终检索分：

```text
retrieval_score = 0.55 × structured_score + 0.45 × semantic_score
```

其中结构化分已经包含落区、灾种和时间三个维度，落区和灾种仍然是业务主导因素。

当 Embedding 或 Rerank 不可用时：

- 不删除结构化候选。
- 语义字段补零并记录降级警告。
- 最终使用结构化分排序，避免模型故障导致结果为空。

### 5. 阶段四：LLM 重排、理由生成和最终选取

`rank_and_select` 先把候选的代表 chunk 正文重新挂载为 `semantic_evidence`，再将融合分最高的前 15 个个例交给聊天模型。

LLM 的输入包括：

- 当前过程完整查询画像。
- 候选个例的 case_id、标题、日期、灾种、落区和融合分。
- 结构化匹配理由。
- 代表 chunk 的真实正文、chunk_id 和来源 PDF。

LLM 只做两项工作：

1. 在真实候选 ID 范围内判断实际预报参考价值，给出建议顺序。
2. 为候选生成简短、具体、业务化的 `match_reasons`。

匹配理由会被限制为真实候选 ID，并优先要求说明低层切变位置、水汽输送条件、层结不稳定、急流配置、副高边缘、冷空气路径、落区演变或强度时段等对预报有影响的专业相似点。模型不能新增个例，也不能编造输入正文中没有的历史事实。

#### 5.1 LLM 只微调近分候选

`selection.apply_llm_tie_break` 不允许 LLM 推翻明显的结构化排序：

- 候选按融合分形成连续分组。
- 只有相邻候选融合分差不超过 `0.08` 时，才采用 LLM 返回的相对顺序。
- 分差较大的候选保留原融合分顺序。

这保证 LLM 负责"解释和近分判断"，而不是把落区、灾种和季节主导的排序完全改写。

#### 5.2 质量门槛与 3～5 个结果

`selection.select_cases` 的实际选择规则是：

- 结构化分达到 `0.30` 才进入合格集合。
- 如果没有候选达到 `0.30`，但最高候选达到 `0.15`，保留最高的 1 个作为最低质量兜底。
- 按页面 `top_n` 返回最多 3～5 个，不使用低相关个例强行补足。
- 合格个例不足请求数量时，返回实际数量，并在 `warnings` 中明确说明样本不足。
- 适度多样性模式下，同一来源 PDF 默认最多先选 2 个，随后再按原排序回填，避免结果全部来自同一份文档。

因此，正常高质量过程通常返回 3～5 个；低质量过程允许返回 1～2 个，极端情况下可以返回 0 个并明确告警。

---

## 选出个例后的 Agent 详细设计

当 `selected_cases` 形成后，后半段不再重新做全库召回，而是围绕入选个例进行**"丰富 → 逐例提炼 → 跨例综合 → 结果组装"**。后半段由四个 LangGraph 节点组成，节点之间通过显式 State 字段传递数据。

### 节点 A：`enrich_cases`（Case Enrichment）

职责是准备事实材料，不做总结、不下预报结论。

对每个入选个例，`enrichment.py` 会：

1. 通过 `source_chunk_ids` 从本地 `chunks.json` 读取正式关联正文。
2. 按查询灾种、影响区域、业务章节词和语义命中 chunk 进行轻量打分。
3. 优先保留最多 6 个相关 chunk；语义命中的 chunk 会额外加权，包含数值单位的正文也会优先保留。
4. 保留 `chunk_id`、`chunk_no`、`source_pdf` 和正文，生成 LLM 可用的 `relevant_chunks` 与 `case_context`。
5. 按 `evidence_image_ids` 准备最多 6 张证据图片，只向前端暴露 `image_id` 接口地址，不暴露本地绝对路径。
6. 从该个例全部正式关联 chunk 中抽取"过程开始日期之前"的历史预警或服务句，形成 `historical_warnings`。

历史预警抽取不是读取 `forecast_services.json`，而是必须同时满足预警关键词、可解析月日和发布时间早于历史个例开始日期。

### 节点 B：`extract_case_references`（Per-Case Reference Extraction）

该节点逐个调用 LLM，每个个例独立处理，避免某个低质量个例影响其他个例。默认双并发（`CASE_REFERENCE_MAX_WORKERS = 2`）。

每个个例的模型输入包括：

- 当前新过程查询画像。
- 个例基础字段：标题、日期、灾种、落区。
- 节点 A 选出的 `relevant_chunks`。
- 过程前历史预警 `historical_warnings`。

模型要求输出：

- `reference_points`：2～4 个历史个例本身的规律或事实，例如强度范围、影响系统、落区演变、持续时间、相态变化和冷空气路径。
- `similarities`：当前过程与历史个例在低层切变、水汽输送、层结不稳定、急流、副高边缘、落区演变等方面的专业相似点。
- `differences`：时间、极值、影响范围、环流配置或强度时段等客观差异。
- `warning_references`：仅引用输入中已有的过程前历史预警。

每条 `reference_points` 必须绑定真实 `evidence_chunk_ids`。模型返回未知 chunk_id 时会被过滤；若无法形成有效证据，则使用可溯源的规则句子摘要降级。

这一节点明确不生成"本次过程应该如何预报"的结论。当前过程的行动建议统一留给节点 C。

### 节点 C：`synthesize_forecast_tips`（Cross-Case Synthesis）

该节点将多个个例的独立参考交给 LLM，站在当前新过程视角进行跨个例综合。

模型必须先输出 `forecast_summary`：

- `similarity_assessment`：整体相似度判断。
- `core_features`：最值得借鉴的 1～2 个核心特征。
- `main_risk`：最需要警惕的主要风险。
- `confidence` 和支持个例。

然后输出 3～6 条 `forecast_tips`。每条提示必须：

- 说明关注对象，例如具体时段、落区、回波、小时雨强、累计降水、风场或环流指标。
- 说明可能偏差，例如模式对对流中心强度、落区、发生时段或极值的偏差。
- 给出建议动作，例如加强主观订正、调整落区预报、加强临近监测或补充服务关注。
- 绑定真实 `support_case_ids` 和 `evidence_chunk_ids`。
- 标注"多数个例共同支持""部分个例支持"或"单个例提示"，并给出置信度。

模型提示被限制为 2～3 句话以内，使用气象台内部业务书面语，禁止简单拼接个例原句，也禁止把单个个例现象表述成多个个例共识。

当综合 LLM 失败时，系统不制造跨个例共识，而是按逐例参考生成保守提示，并明确降低置信度。

### 节点 D：`assemble_result`（Result Assembly）

该节点是唯一对外输出节点，负责把内部字段整理为稳定契约：

- `matched_cases`：按最终排序输出 `rank`、`case_id`、标题、日期、灾种、落区、综合分和分项分数。
- `match_reasons`：优先采用候选 LLM 生成的理由，失败时使用结构化理由。
- `key_references`：逐例历史参考点、证据 chunk 和可选证据图片引用。
- `key_differences`、`similarities`、`historical_warnings`：分别保存差异、相似点和过程前历史预警。
- `evidence_images`：图片使用 `image_id` 接口访问，并生成稳定的"图1、图2……"显示编号。
- `forecast_summary`：综合研判页顶部核心结论摘要。
- `forecast_tips`：跨个例综合预报提示。
- `query_summary`：过程名称、日期、灾种、区域和输入摘要。
- `warnings`、`audit`：降级提示、节点耗时、模型调用状态和数量统计。

参考经验和图片之间目前采用基于图片标题关键词的弱关联：只有能够从标题判断可能对应时才显示"可核验图片：图X"，不强行假设 chunk 与图片一一对应。

---

## LangGraph State 数据流

```text
request
  ↓ normalize_input
query + all_cases
  ↓ structured_recall
candidate_cases（结构化候选）
  ↓ semantic_supplement
candidate_cases（语义分、代表 chunk、Rerank 状态）
  ↓ rank_and_select
scored_cases + selected_cases
  ↓ enrich_cases
enriched_cases（正文、图片、过程前历史预警）
  ↓ extract_case_references
case_references（逐例参考点、相似点、差异、证据）
  ↓ synthesize_forecast_tips
forecast_summary + forecast_tips
  ↓ assemble_result
final_output
```

State 中各字段的职责保持分离：

- `selected_cases`：最终入选的基础个例和综合分。
- `enriched_cases`：入选个例的可用正文、图片和历史预警。
- `case_references`：逐个例独立提炼结果。
- `forecast_summary`：跨个例核心结论。
- `forecast_tips`：跨个例行动提示。
- `final_output`：唯一对外稳定结构。

### 选择流程与后半段流程的边界

前半段解决**"选得准"**：落区、灾种和季节先约束候选，向量和 Rerank 补充语义，LLM 只对前 15 个候选做近分重排和理由生成。

后半段解决**"用得好、说得清"**：只围绕已入选个例准备证据，逐个例提炼历史参考，再跨个例生成当前过程的综合判断和行动提示。

当前本地实现不修改向量库，也不依赖公司知识库；后续迁移到公司知识库时，优先替换 `LocalCaseDataStore` 和语义检索适配层，保留上述 State 和节点职责边界。

---

## 2026-08-11 16:59 自然语言聊天页与分阶段流式输出

本次更新新增自然语言聊天入口，不替换原结构化工作台和同步搜索接口：

- 原结构化页面：`GET /api/smart-case-match/chat`
- 新自然语言页面：`GET /api/smart-case-match/conversation`
- 原同步接口：`POST /api/smart-case-match/search`
- 新流式接口：`POST /api/smart-case-match/conversation/stream`

新页面只提交一段自然语言，例如日期、灾种、影响区域、实况、环流和强度可以写在同一段话中。系统先通过关闭思考模式的聊天模型提取 `process_name`、`start_date`、`end_date`、`date`、`disaster_types`、`affected_areas`、`observation_description`、`circulation_description`、`intensity_description` 和 `raw_query`。用户没有提供的字段允许为空，模型不得自行补充事实；解析失败时保留原文，由现有规则继续识别灾种、地区和月份。

解析结果会被转换为现有 `SmartCaseMatchRequest`，后续仍复用同一套 LangGraph、结构化召回、候选内 Embedding、专用 Rerank、融合排序、候选 LLM 重排、个例丰富、双并发逐例提炼、跨个例综合和结果组装逻辑。聊天入口默认处理 3 个个例。

流式接口使用 SSE 分阶段发送完整 JSON，而不是输出无法校验的半截模型文本。主要事件顺序为：

```text
accepted
  → parsed_query
  → stage
  → matched_cases (phase=selected，先展示已确定排序的个例)
  → matched_cases (phase=referenced，补齐参考经验和证据引用)
  → forecast（综合摘要和预报提示）
  → completed
```

聊天页面不平铺证据图片和补充资料图片。每条参考经验继续返回真实 `evidence_chunk_ids` 和 `evidence_image_refs`：点击 chunk 按钮后调用 `/assets/chunks/{chunk_id}` 获取正文；点击"图1、图2"等按钮时使用返回的图片 URL 加载图片，并在弹窗下方显示统一图号和图注。本地文件绝对路径不会返回给浏览器。

真实联调示例中，首个 SSE 事件约 0.32 秒返回，自然语言解析约 4.23 秒完成，3 个匹配个例约 11.67 秒首次展示，参考经验约 26.65 秒补齐，综合研判约 38.00 秒返回，完整流程约 38.06 秒结束。实际耗时会随模型服务负载和输入长度变化。

---

## 公司 AI 主聊天工具接入说明

相似个例智能匹配是公司 AI 聊天工具下的一个子 Agent。公司主聊天工具先完成意图识别；当用户意图属于"相似个例智能匹配"时，主工具无需自行拆分结构化字段，只需把用户原始自然语言转发给本模块的流式接口。

### 1. 子 Agent 调用地址

```text
POST /api/smart-case-match/conversation/stream
```

本地开发环境完整地址为：

```text
http://127.0.0.1:8000/api/smart-case-match/conversation/stream
```

`127.0.0.1` 仅适用于公司主聊天服务和本项目运行在同一台机器的情况。部署到公司环境后，应替换为实际后端域名或内网 IP。

请求头：

```http
Content-Type: application/json
Accept: text/event-stream
```

请求体只需要自然语言描述，`progress_id` 可选：

```json
{
  "message": "2026年7月15日至16日，太原、晋中可能出现暴雨和短时强降水，低层切变配合西南急流，局地小时雨强较大，过程持续约12小时。",
  "progress_id": "company_chat_123456"
}
```

- `message`：必填，用户的完整自然语言天气过程描述，长度为 2～6000 个字符。
- `progress_id`：可选，用于公司主聊天系统关联会话和跟踪任务；不传时由子 Agent 自动生成。

### 2. 自然语言自动提取

接口收到 `message` 后，会先使用关闭思考模式的聊天模型自动提取以下字段：

- 过程名称 `process_name`
- 开始和结束日期 `start_date`、`end_date`
- 不完整日期或自然语言日期 `date`
- 灾害类型 `disaster_types`
- 影响区域 `affected_areas`
- 实况描述 `observation_description`
- 环流形势 `circulation_description`
- 强度与时效 `intensity_description`
- 补充描述 `raw_query`

用户没有提供的字段允许为空，解析模型不得自行补充事实。解析结果随后转换为现有 `SmartCaseMatchRequest`，继续复用结构化召回、候选内 Embedding、专用 Rerank、融合排序、候选 LLM 重排、个例丰富、双并发逐例提炼、跨个例综合和结果组装流程。

### 3. SSE 流式事件

接口以 SSE 方式分阶段返回结果，主要事件顺序如下：

```text
accepted
  → parsed_query
  → stage
  → matched_cases (phase=selected)
  → matched_cases (phase=referenced)
  → forecast
  → completed
```

- `accepted`：请求已接收，开始理解自然语言。
- `parsed_query`：返回自动提取的结构化查询字段。
- `stage`：返回当前节点、进度文字和百分比。
- `matched_cases / selected`：先返回已确定排序的匹配个例，使主聊天页面可以提前展示结果。
- `matched_cases / referenced`：更新同一批个例，补齐参考经验、关键差异和证据引用。
- `forecast`：返回核心结论摘要和综合预报提示。
- `completed`：返回完整最终结果并标记任务结束。
- `error`：执行失败时返回错误信息和任务编号。

公司主聊天工具可以按事件逐步渲染，也可以只读取 `completed` 事件中的完整最终结果。

### 4. 流式接口返回的核心业务数据

`POST /api/smart-case-match/conversation/stream` 返回页面展示所需的核心业务数据，包括：

- 自然语言解析结果
- 匹配个例及各项得分
- 匹配理由
- 参考经验
- 关键差异
- 文字证据 ID
- 图片证据 ID、统一图号、图注和图片 URL
- 核心结论摘要
- 综合预报提示
- 警告信息和运行状态

流式结果已经包含个例与综合研判的展示数据，但不会把所有 chunk 正文和图片二进制内容直接塞入 SSE。证据详情采用按需加载，避免流式响应过大。

### 5. 文字证据按需加载

参考经验中的 `evidence_chunk_ids` 用于生成文字证据按钮。用户点击后调用：

```text
GET /api/smart-case-match/assets/chunks/{chunk_id}
```

返回示例：

```json
{
  "chunk_id": "FST2025-7-chunk-002",
  "chunk_no": 2,
  "source_pdf": "FST2025-7.pdf",
  "content": "具体的 chunk 正文……"
}
```

该接口只返回可公开核验的来源、序号和正文，不返回向量或本地文件路径。

### 6. 图片证据按需加载

流式结果中的图片元数据包含 `image_id`、`display_no`、`caption` 和 `url`。用户点击"图1、图2"等引用按钮后，通过以下接口加载图片：

```text
GET /api/smart-case-match/assets/images/{image_id}
```

该接口直接返回图片文件。图片图号、图注和 URL 已在流式响应中提供，前端只在用户实际查看时加载图片，并在弹窗下方展示统一图号和图注。后端不会向浏览器暴露图片的本地磁盘绝对路径。

### 7. 公司聊天页面的请求关系

```text
GET /api/smart-case-match/conversation
    获取本模块自带的自然语言聊天网页，仅供独立访问或联调

POST /api/smart-case-match/conversation/stream
    公司主聊天工具调用子 Agent，获取自然语言解析、匹配个例和综合研判

用户点击文字证据
    GET /api/smart-case-match/assets/chunks/{chunk_id}

用户点击图片证据
    GET /api/smart-case-match/assets/images/{image_id}
```

新聊天页面和公司主聊天工具不需要调用 `/progress/{progress_id}`。进度节点、文字和百分比已经包含在 SSE 的 `stage` 事件中。

### 8. 后端内部模型调用

一次相似个例匹配过程中，子 Agent 会在服务器内部调用以下模型服务：

- 聊天模型：自然语言解析、候选个例重排、逐例参考提炼、跨个例综合研判。
- Embedding 模型：计算查询文本与候选个例 chunk 的向量相似度。
- Rerank 模型：对候选范围内的语义命中 chunk 进行精排。

这些调用都属于子 Agent 的后端内部实现。公司主聊天工具不需要分别调用聊天模型、Embedding 或 Rerank，也不需要了解本地个例库和 chunk 映射。正常情况下，公司主聊天工具只调用 `/api/smart-case-match/conversation/stream`；只有用户点击证据时，才调用对应的文字或图片资源接口。
