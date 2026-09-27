# 智能体路由实施计划

> **给智能体执行者：** 必需子技能：使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务逐步执行本计划。步骤使用 checkbox（`- [ ]`）语法跟踪。

**目标：** 在复用现有标准化个例、图片、相似匹配和文档检索服务的基础上，构建气象智能体路由的第 0-3 阶段。

**架构：** 新增聚焦的 `backend/app/services/agent/` 包，用于意图分析、计划生成、回答合成和调度编排。执行保持只读，并通过现有 `/api/agent/query` 路径集成。

**技术栈：** Python dataclasses、FastAPI、现有 service 层、unittest/pytest。

---

### 任务 1：意图和计划模型

**文件：**
- 新建：`backend/app/services/agent/models.py`
- 测试：`tests/test_agent_intent.py`

- [ ] 编写多意图识别和工具路线计划的测试。
- [ ] 实现意图、槽位、analyzer 结果、计划步骤、执行计划和 agent 结果的类型模型。
- [ ] 运行 `python -m pytest tests/test_agent_intent.py -q`。

### 任务 2：确定性分析器

**文件：**
- 新建：`backend/app/services/agent/analyzers.py`
- 修改：`tests/test_agent_intent.py`

- [ ] 编写统计、证据检索、个例复盘、对比、相似个例、服务决策和混合查询测试。
- [ ] 实现规则、上下文、工具需求和安全 analyzer。
- [ ] 运行 `python -m pytest tests/test_agent_intent.py -q`。

### 任务 3：计划器

**文件：**
- 新建：`backend/app/services/agent/planner.py`
- 测试：`tests/test_agent_planner.py`

- [ ] 编写测试，证明统计汇总计划使用完整标准化个例聚合。
- [ ] 编写测试，证明雷达图证据计划包含图片元数据检索。
- [ ] 实现 `AgentPlanner`。
- [ ] 运行 `python -m pytest tests/test_agent_planner.py -q`。

### 任务 4：回答合成器

**文件：**
- 新建：`backend/app/services/agent/answer_synthesizer.py`
- 测试：`tests/test_agent_answer_synthesizer.py`

- [ ] 编写测试，证明兜底回答会先回答问题，并避免“根据检索片段”式开头。
- [ ] 实现结构化结果、相似个例结果和兜底 RAG 结果的直接回答格式。
- [ ] 运行 `python -m pytest tests/test_agent_answer_synthesizer.py -q`。

### 任务 5：只读调度器

**文件：**
- 新建：`backend/app/services/agent/orchestrator.py`
- 修改：`backend/app/main.py`
- 修改：`backend/app/api/routes/agent.py`
- 测试：`tests/test_agent_orchestrator.py`

- [ ] 编写 `/api/agent/query` 返回 `intent_trace`、`execution_plan` 和证据字段的测试。
- [ ] 使用现有 store 和 helper 实现编排。
- [ ] 运行 `python -m pytest tests/test_agent_orchestrator.py -q`。

### 任务 6：回归测试

**文件：**
- 只有测试暴露真实回归时才修改。

- [ ] 运行 agent、标准化个例和流式 API 的定向测试。
- [ ] 定向测试通过后运行完整测试套件。
