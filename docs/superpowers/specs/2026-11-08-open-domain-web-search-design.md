# 开放域问题联网搜索设计规格

> 日期：2026-11-08
> 状态：已确认方案 A（新增 general_chat 意图 + RAG 检索不足自动升级联网搜索）
> 关联计划：`docs/superpowers/plans/2026-11-08-open-domain-web-search.md`

## 1. 背景与问题

当前系统是山西省气象知识库问答系统。意图路由只支持四类意图：`rag`（本地知识库检索回答）、`multidim_search`（多维检索/报告）、`similar_case_match`（相似个例）、`clarify`（澄清追问）。因此与本地知识库无关的开放域问题（例如"今天重庆天气怎么样"）会被默认路由到 `rag`，本地检索不到证据后输出"未检索到相关证据片段。"，无法真正回答用户。

用户使用的 DashScope 百炼 OpenAI 兼容接口（qwen3.7-flash）支持联网搜索，但当前 `DashScopeChatClient` 没有启用该能力。

## 2. 目标

1. 意图层区分问题是否需要本地知识库：需要 → 走现有 RAG / 多维 / 相似匹配链路；不需要 → 直接调用普通聊天模型。
2. 普通聊天调用 qwen3.7-flash 并启用联网搜索（`enable_search=True`，默认强制 `forced_search=True`），流式回答。
3. 需要知识库但本地检索为空/不足时，自动升级为联网搜索回答，而不是输出"未检索到相关证据片段"。
4. 保持现有 `/api/query`、`/api/agent/query`、`/api/query/stream` 的 SSE 协议兼容，字段只增不改，前端无需改动即可显示。

## 3. 方案选型

### 方案 A（已确认）：新增 `general_chat` 意图 + RAG 检索不足自动升级

- 意图模型新增 `general_chat` 分支：与本地气象知识库无关的问题直接路由到普通聊天（强制联网搜索）。
- RAG 链路保留升级点：本地检索为空时自动转联网搜索回答。
- 优点：开放域问题一步到位、最快响应；气象问题检索不到也能联网兜底；双保险。
- 缺点：改动面较大（意图识别 + 新流式链路 + RAG 升级点三处）。

### 方案 B（未采用）：仅"检索不足升级"，不新增意图

- 所有问题先走 RAG，检索为空再联网。改动小，但开放域问题先浪费一次检索，且意图语义仍错误。

### 方案 C（未采用）：开放域直接普通聊天但不强制联网

- 实时性信息（天气、新闻）可能答错或过期，与用户诉求不符。

## 4. 详细设计

### 4.1 配置项

新增配置（`backend/app/core/config.py` 的 `Settings` dataclass）：

| 配置字段 | 环境变量 | 默认值 | 说明 |
|---|---|---|---|
| `web_search_enabled` | `WEB_SEARCH_ENABLED` | `true` | 联网搜索总开关 |
| `web_search_forced` | `WEB_SEARCH_FORCED` | `true` | 是否强制联网（`forced_search`） |
| `rag_empty_upgrade_web_search` | `RAG_EMPTY_UPGRADE_WEB_SEARCH` | `true` | RAG 检索为空时是否升级联网 |
| `rag_min_score` | `RAG_MIN_SCORE` | `0` | 低分升级阈值，0 表示仅空结果触发 |

同步更新 `.env`、`.env.example`、`compose.yaml` 的 `x-backend-environment`。

### 4.2 模型客户端（`backend/app/services/model_client/chat.py`）

- `stream_answer_with_context()` 增加关键字参数 `enable_search: bool = False`、`forced_search: bool = False`：
  - `extra_body` 在 `enable_search=True` 时追加 `"enable_search": True`，并在 `forced_search=True` 时追加 `"search_options": {"forced_search": True}`。
  - 默认行为不变（不联网）。
- `answer_with_context()` 同步增加上述两个参数（非流式调用默认不联网）。
- 新增 `stream_general_chat(question, *, system_prompt="", enable_search=True, forced_search=True)`：
  - 使用通用中文 system prompt（不限定气象身份），用户消息只放原始问题，不携带本地知识库上下文。
  - 调用 `chat.completions.create` 时携带 `enable_search` / `search_options`。
- 新增模块级 `general_chat_system_prompt()` 帮助函数，返回通用中文提示词。

### 4.3 意图路由扩展（`backend/app/agents/query_intent/`）

`schemas.py`：

- `IntentName` 增加 `"general_chat"`。
- `TaskRoute` 增加 `"general_chat"`。
- `TaskType` 增加 `"general_chat"`。
- `suggested_intent` 的 Literal 同步增加 `"general_chat"`。

`agent.py`：

- `_model_understanding()` 的意图 prompt：intent 合法值增加 `general_chat`，并说明"与本地气象知识库（山西气象、历史个例、统计、预警）无关的通用闲聊、实时信息（如其他地区天气、新闻、常识）选择 general_chat"。
- 模型返回 JSON 的 intent 校验集合增加 `"general_chat"`。
- `_model_decision()` 的 JSON 指令 prompt 与校验集合增加 `general_chat`。
- `_merge_decisions()` 的 `llm_over_default_rag` 分支允许集合增加 `"general_chat"`（规则默认 rag 且置信度 ≥ 0.85 时，模型可将其改判为 general_chat）。
- `_target()` 增加 `general_chat → ("开放域聊天", "")` 映射。
- 规则层保持"默认 rag"不变，避免误伤气象问题；模型不可用时开放域问题由 RAG 空检索升级链兜底。

### 4.4 普通聊天流式链路（新增 `backend/app/agents/unified_chat/general_chat_stream.py`）

新增 `GeneralChatStreamMixin`，方法：

```python
async def _stream_general_chat(
    self,
    question: str,
    session_id: str,
    *,
    run_id: str = "",
    performance: PerformanceTrace,
) -> AsyncIterator[dict[str, Any]]:
```

- 构造 `common = self._empty_result("general_chat", run_id, question, "")`，设置 `retrieval_mode="agent:general_chat_web_search"`、`web_search=True`。
- 若 `web_search_enabled` 且 `llm_client.is_available()`：调用 `stream_general_chat(question, enable_search=True, forced_search=settings.web_search_forced)`（TypeError 兼容旧替身时退化到 `stream_answer_with_context(question, [])`），经 `_stream_sync_deltas` 转发为 SSE `delta`。
- 失败/不可用：输出中文兜底文案，`llm_status` 记录失败原因。
- 结束：`_persist_complete`（`already_started=False`，独立 start+finish）、`_completion_metadata`、`_done`、`_schedule_memory_refresh`。
- `termination_reason` 沿用既有 `completed`，不新增枚举值（避免破坏协议）。

### 4.5 编排器接入（`backend/app/agents/unified_chat/orchestrator.py`）

- `UnifiedChatOrchestrator` 继承列表增加 `GeneralChatStreamMixin`。
- `route_messages` 增加 `"general_chat": "正在联网搜索最新信息"`。
- 分发分支：在 `clarify` 之后、`_stream_rag` 之前增加：

```python
if decision.intent == "general_chat":
    async for event in self._stream_general_chat(
        decision.resolved_question,
        session_id,
        run_id=f"general_chat_{uuid4().hex}",
        performance=performance,
    ):
        yield event
    self.runtime.finish_subtask()
    return
```

### 4.6 RAG 检索不足自动升级（`backend/app/agents/unified_chat/rag_stream.py`）

- 升级点位于 `_stream_rag` 检索完成、构建 `common` 之前。
- 触发条件：`hits` 为空 且 `task_context_blocks` 为空 且 `rag_empty_upgrade_web_search` 与 `web_search_enabled` 均开启 且 `llm_client.is_available()`。
- 行为：
  1. 先收口已登记的 document_rag 运行记录（`_finish_persisted_run`，try/except 包裹，避免遗留 running 状态）。
  2. `yield _progress_metadata("general_chat", run_id, "web_search", 40, "正在联网搜索最新信息", performance)`。
  3. `async for event in self._stream_general_chat(question, session_id, run_id=f"general_chat_{uuid4().hex}", performance=performance): yield event`。
  4. `return`（不再走 document_rag 的 metadata/done）。
- 开关关闭或模型不可用时保持现有"未检索到相关证据片段。"兜底不变。

### 4.7 SSE 协议与前端

- SSE 仍是 `data: {json}`，`type` 字段不变：`metadata / delta / done / error`。
- 新增字段只增：`metadata.agent_type="general_chat"`、`common["web_search"]=True`（仅新增键，不影响旧客户端）。
- 前端无需改动；`delta` 文本流照常显示。

## 5. 数据流

```
用户输入
  → Orchestrator.stream()
  → 意图路由（规则 + qwen 意图模型）──→ general_chat？
        │ 是 → _stream_general_chat（enable_search=True 强制联网，流式回答）
        │ 否 → _stream_rag
              → 本地检索（Milvus→Chroma→JSON 降级链）
              → hits 非空 → 知识库回答
              → hits 为空 → rag_empty_upgrade_web_search=True
                  → _stream_general_chat（联网搜索回答）
                  → 失败/关闭 → 原兜底"未检索到相关证据片段。"
```

## 6. 错误处理与降级

| 场景 | 处理 |
|---|---|
| 未配置 DashScope API Key | `is_available()` 为 False → 中文兜底文案，不调用联网 |
| 联网搜索 API 异常/超时 | 捕获异常 → 输出中文兜底文案，`llm_status` 记录失败原因 |
| 意图模型不可用 | 规则兜底默认 `rag` → 开放域问题走"空检索→升级联网"链，仍可回答 |
| 预算/取消 | `BudgetTermination` 穿透（现有逻辑不变） |
| 气象问题被误判为 general_chat | 规则硬保护（明确气象/山西/个例关键词优先 rag）+ RAG 升级联网双保险 |

## 7. 测试计划

1. 意图路由单元测试："今天重庆天气怎么样" → `general_chat`；气象问题仍 → `rag`；模型不可用时规则兜底。
2. 模型客户端测试：`stream_answer_with_context` / `answer_with_context` / `stream_general_chat` 的 `extra_body` 含 `enable_search` 与 `search_options`；默认不联网。
3. 普通聊天流式测试：SSE 序列为 `metadata → delta → done`，`agent_type="general_chat"`，答案落库。
4. RAG 空检索升级测试：空 hits 时调用联网流式，而非输出"未检索到相关证据片段"。
5. 降级测试：搜索抛异常 → 中文兜底文案；开关关闭 → 原兜底行为不变。
6. 回归测试：现有全量测试不回归。

## 8. 风险与权衡

- 意图误判：规则硬保护 + 模型高置信门槛（≥ 0.85）+ RAG 升级联网兜底。
- 成本：每个开放域问题多一次联网搜索调用，可用 `WEB_SEARCH_ENABLED` 控制。
- 延迟：联网搜索增加 2-5 秒，先发 `metadata` 反馈"正在联网搜索最新信息"。
- 内容边界：普通聊天 system prompt 说明"实时信息以联网搜索为准，气象历史/统计类问题优先引导回知识库"。
