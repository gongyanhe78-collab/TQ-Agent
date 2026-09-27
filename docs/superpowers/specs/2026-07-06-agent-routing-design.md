# 智能体路由设计

## 目标

将气象个例助手从单一路径的 RAG 升级为只读、可解释的智能体路由器，覆盖第 0-3 阶段：意图体系、并行意图分析、工具计划生成和计划执行。

## 范围

本版本处理以下问题：

- 证据检索类问题，例如查询某个过程对应的雷达图。
- 单过程复盘类问题。
- 基于完整标准化个例层的统计汇总类问题。
- 跨月份、地区或灾种的对比分析类问题。
- 相似历史个例类问题。
- 基于当前过程条件和历史个例的服务决策类问题。
- 同时包含多个意图的混合问题。

本版本不新增写入类操作，不重建索引，不删除数据，也不把 LangChain/LangGraph 作为运行时依赖引入。

## 复用

实现时应复用项目里已有的能力：

- `StructuredQuestionAnswerer` 用于统计、证据检索和对比。
- `JsonStandardCaseStore` 用于读取完整标准化个例。
- `ImageEvidenceStore` 用于图片元数据和图片类型分类。
- `SimilarCaseMatcher` 用于相似历史个例匹配。
- 现有文档 chunk 检索辅助函数用于兜底和通用 RAG。

## 架构

新增代码放在 `backend/app/services/agent/`。

- `models.py` 定义带类型的意图、槽位、计划和结果对象。
- `analyzers.py` 从用户问题中提取规则、上下文、工具和安全信号。
- `planner.py` 将 analyzer 输出合并为有序的 `ExecutionPlan`。
- `answer_synthesizer.py` 负责组织回答，让助手先回答用户问题，再展示支撑证据。
- `orchestrator.py` 协调意图分析、计划生成、只读执行和回答合成。

公开接口可以继续使用 `/api/agent/query`。当请求进入智能体路径时，响应应包含 `intent_trace`、`execution_plan`、`evidence_cases`、`evidence_chunks` 和 `evidence_images`。

## 回答质量

回答不能以“根据检索片段”这类说法开头，也不能只是复述 chunks。回答应做到：

- 先回答用户的核心问题。
- 使用清晰的业务表达。
- 统计类问题先说明统计口径，再给数字。
- 直接答案之后再展示证据。
- 证据不足时明确说明缺口，不编造细节。
- 服务决策类输出要标明基于历史个例经验，而不是实时预报结论。

## 验收标准

- 意图测试覆盖六类核心问题和混合问题。
- Planner 测试证明统计类问题使用 `standard_cases_aggregate`，而不是 top-k chunk 检索。
- 证据检索计划包含 `image_metadata_search`。
- Agent 查询响应包含意图链路和执行计划元数据。
- 兜底 RAG 回答避免“检索片段”式自我描述开头。
- 现有标准化个例和流式接口测试继续通过。
