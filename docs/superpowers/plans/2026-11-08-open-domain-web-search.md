# 开放域问题联网搜索实现计划

> **执行提示：** 本计划按任务逐步实现，使用 TDD（先写失败测试，再实现）。每个任务独立提交。
> **目标：** 新增 `general_chat` 意图分支并启用 DashScope 联网搜索，使开放域问题直接由普通聊天模型回答；本地知识库检索为空时自动升级联网搜索。
> **架构：** 意图模型识别开放域问题 → 编排器分发到新的 `GeneralChatStreamMixin._stream_general_chat`（调用 `DashScopeChatClient.stream_general_chat`，`enable_search=True`）；`_stream_rag` 检索为空时也升级到该流式链路。SSE 协议与前端保持不变。
> **技术栈：** Python 3 / FastAPI / DashScope 百炼 OpenAI 兼容接口 / pytest（unittest）。
> **规格：** `docs/superpowers/specs/2026-11-08-open-domain-web-search-design.md`

## 全局约束

- 所有文档、注释、docstring 必须使用中文。
- SSE 协议字段只增不改（`metadata / delta / done / error`，`data: {json}`）。
- 现有默认行为不改变：`stream_answer_with_context` 默认不联网；规则路由默认保持 `rag`。
- 测试命令统一使用：`D:\miniconda3\envs\riskMe\python.exe -m pytest <路径> -q`。
- 工作分支：`codex/phase1-enterprise-transition`（沿用当前分支，不新建分支）。

---

### Task 1：新增联网搜索与 RAG 升级配置项

**Files:**
- Modify: `backend/app/core/config.py`（在 `dashscope_rerank_endpoint` 字段后新增）
- Modify: `.env.example`
- Modify: `.env`
- Modify: `compose.yaml`（`x-backend-environment`）
- Test: Create `backend/tests/unit/core/test_settings.py`

**Interfaces:**
- Produces:
  - `Settings.web_search_enabled: bool`（环境变量 `WEB_SEARCH_ENABLED`，默认 true）
  - `Settings.web_search_forced: bool`（环境变量 `WEB_SEARCH_FORCED`，默认 true）
  - `Settings.rag_empty_upgrade_web_search: bool`（环境变量 `RAG_EMPTY_UPGRADE_WEB_SEARCH`，默认 true）
  - `Settings.rag_min_score: float`（环境变量 `RAG_MIN_SCORE`，默认 0，本次仅作为预留配置，不参与逻辑）

- [x] **Step 1: 写失败测试**

```python
"""Settings 新增开放域联网搜索配置的默认值测试。"""
from __future__ import annotations

import unittest

from backend.app.core.config import Settings


class SettingsWebSearchTests(unittest.TestCase):
    """验证联网搜索与 RAG 升级相关配置的默认值。"""

    def test_web_search_config_defaults_enabled(self) -> None:
        settings = Settings()
        self.assertTrue(settings.web_search_enabled)
        self.assertTrue(settings.web_search_forced)
        self.assertTrue(settings.rag_empty_upgrade_web_search)
        self.assertEqual(settings.rag_min_score, 0.0)
```

- [x] **Step 2: 运行测试确认失败**

Run: `D:\miniconda3\envs\riskMe\python.exe -m pytest backend/tests/unit/core/test_settings.py -q`
Expected: FAIL（AttributeError: web_search_enabled 不存在）

- [x] **Step 3: 实现配置字段**

`backend/app/core/config.py` 的 `Settings` dataclass 中、`dashscope_rerank_endpoint` 字段之后新增：

```python
    dashscope_rerank_endpoint: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_RERANK_ENDPOINT", "")
    )

    # 开放域联网搜索：普通聊天模型直接回答库外问题并启用联网搜索。
    web_search_enabled: bool = field(
        default_factory=lambda: _env_value("WEB_SEARCH_ENABLED", "true").lower()
        in {"1", "true", "yes", "on"}
    )
    web_search_forced: bool = field(
        default_factory=lambda: _env_value("WEB_SEARCH_FORCED", "true").lower()
        in {"1", "true", "yes", "on"}
    )
    # 本地知识库检索为空时自动升级联网搜索。
    rag_empty_upgrade_web_search: bool = field(
        default_factory=lambda: _env_value("RAG_EMPTY_UPGRADE_WEB_SEARCH", "true").lower()
        in {"1", "true", "yes", "on"}
    )
    # 预留低分升级阈值；0 表示仅空结果触发，本次不参与判断逻辑。
    rag_min_score: float = field(
        default_factory=lambda: float(_env_value("RAG_MIN_SCORE", "0"))
    )
```

- [x] **Step 4: 同步环境文件**

`.env.example` 的"模型服务配置"区块末尾追加：

```dotenv
# 开放域联网搜索：开放域问题直接调用普通聊天模型并启用联网搜索
WEB_SEARCH_ENABLED=true
WEB_SEARCH_FORCED=true
RAG_EMPTY_UPGRADE_WEB_SEARCH=true
RAG_MIN_SCORE=0
```

`.env`（本地未跟踪文件）末尾追加同样 4 行（与上面内容一致）。

`compose.yaml` 的 `x-backend-environment` 中 `DASHSCOPE_RERANK_ENDPOINT` 之后追加：

```yaml
  WEB_SEARCH_ENABLED: ${WEB_SEARCH_ENABLED:-true}
  WEB_SEARCH_FORCED: ${WEB_SEARCH_FORCED:-true}
  RAG_EMPTY_UPGRADE_WEB_SEARCH: ${RAG_EMPTY_UPGRADE_WEB_SEARCH:-true}
  RAG_MIN_SCORE: ${RAG_MIN_SCORE:-0}
```

- [x] **Step 5: 运行测试确认通过**

Run: `D:\miniconda3\envs\riskMe\python.exe -m pytest backend/tests/unit/core/test_settings.py -q`
Expected: PASS

- [x] **Step 6: 提交**

```bash
git add backend/app/core/config.py backend/tests/unit/core/test_settings.py .env.example compose.yaml
git commit -m "feat: 新增开放域联网搜索与RAG升级配置项"
```

---

### Task 2：DashScopeChatClient 联网搜索能力

**Files:**
- Modify: `backend/app/services/model_client/chat.py`
- Test: Create `backend/tests/unit/model_client/test_web_search.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `DashScopeChatClient.stream_answer_with_context(question, context_blocks, *, session_guidance_prompt="", enable_search=False, forced_search=False)`（默认不联网）
  - `DashScopeChatClient.answer_with_context(question, context_blocks, max_tokens=2048, timeout_seconds=None, json_mode=False, enable_search=False, forced_search=False)`
  - `DashScopeChatClient.stream_general_chat(question, *, system_prompt="", enable_search=True, forced_search=True)`
  - `general_chat_system_prompt() -> str`

- [x] **Step 1: 写失败测试**

创建 `backend/tests/unit/model_client/test_web_search.py`：

```python
"""DashScope 聊天客户端联网搜索参数测试。"""
from __future__ import annotations

import unittest

from backend.app.services.model_client.chat import (
    DashScopeChatClient,
    general_chat_system_prompt,
)


class _FakeDelta:
    """模拟流式响应中的增量内容。"""

    def __init__(self, content: str) -> None:
        self.content = content


class _FakeMessage:
    """模拟完整响应中的消息。"""

    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    """模拟单个流式或完整响应选择项。"""

    def __init__(self, content: str) -> None:
        self.delta = _FakeDelta(content)
        self.message = _FakeMessage(content)


class _FakeChunk:
    """模拟一个流式响应分片。"""

    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletion:
    """模拟非流式完整响应。"""

    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """记录请求参数并按 stream 标志返回流式或完整响应。"""

    def __init__(self) -> None:
        self.last_kwargs: dict = {}

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if kwargs.get("stream"):
            return [_FakeChunk("联网回答内容")]
        return _FakeCompletion("联网回答内容")


class _FakeChat:
    """模拟 OpenAI 兼容客户端的 chat 命名空间。"""

    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeOpenAIClient:
    """模拟 OpenAI 兼容客户端顶层对象。"""

    def __init__(self) -> None:
        self.chat = _FakeChat()


class DashScopeWebSearchTests(unittest.TestCase):
    """验证联网搜索参数只在显式开启时写入请求体。"""

    def setUp(self) -> None:
        self.client = DashScopeChatClient(
            api_key="test-key",
            model="qwen3.7-flash",
        )
        self.fake = _FakeOpenAIClient()
        self.client._openai_client = self.fake

    def test_stream_answer_with_context_does_not_search_by_default(self) -> None:
        list(self.client.stream_answer_with_context("山西暴雨", []))
        body = self.fake.chat.completions.last_kwargs.get("extra_body") or {}
        self.assertNotIn("enable_search", body)

    def test_stream_answer_with_context_enables_search(self) -> None:
        list(
            self.client.stream_answer_with_context(
                "今天重庆天气怎么样",
                [],
                enable_search=True,
                forced_search=True,
            )
        )
        body = self.fake.chat.completions.last_kwargs.get("extra_body") or {}
        self.assertEqual(body["enable_search"], True)
        self.assertEqual(body["search_options"], {"forced_search": True})

    def test_stream_general_chat_uses_web_search_without_knowledge_context(self) -> None:
        text = "".join(self.client.stream_general_chat("今天重庆天气怎么样"))
        kwargs = self.fake.chat.completions.last_kwargs
        body = kwargs.get("extra_body") or {}
        self.assertEqual(body["enable_search"], True)
        self.assertEqual(body["search_options"], {"forced_search": True})
        self.assertEqual(text, "联网回答内容")
        messages = kwargs.get("messages") or []
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "今天重庆天气怎么样")
        self.assertNotIn("历史个例材料", messages[0]["content"])

    def test_general_chat_system_prompt_is_chinese(self) -> None:
        prompt = general_chat_system_prompt()
        self.assertTrue(prompt)
        self.assertIsInstance(prompt, str)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: 运行测试确认失败**

Run: `D:\miniconda3\envs\riskMe\python.exe -m pytest backend/tests/unit/model_client/test_web_search.py -q`
Expected: FAIL（AttributeError: stream_general_chat 不存在 / enable_search 未写入）

- [x] **Step 3: 实现客户端能力**

`backend/app/services/model_client/chat.py`：

a) 新增模块级帮助函数（放在类定义之前）：

```python
def general_chat_system_prompt() -> str:
    """开放域普通聊天使用的通用中文提示词。"""
    return (
        "你是气象与通用知识助手。回答开放域问题时优先使用联网搜索获取的实时信息，"
        "并如实说明信息来源；对于山西省气象历史、个例分析、统计和预警类问题，"
        "可以提示用户回到本地知识库提问以获得可追溯的权威资料。不要编造搜索结果。"
    )
```

b) `answer_with_context` 签名增加 `enable_search: bool = False, forced_search: bool = False`，并在 `request_options` 构建中按需追加 `extra_body["enable_search"]` 与 `search_options`（保持 `enable_thinking`、`response_format` 共存）。

c) `stream_answer_with_context` 签名增加 `enable_search: bool = False, forced_search: bool = False`，把 `extra_body` 改为：

```python
        extra_body: dict[str, Any] = {"enable_thinking": False}
        if enable_search:
            extra_body["enable_search"] = True
            if forced_search:
                extra_body["search_options"] = {"forced_search": True}
        stream = self._client().chat.completions.create(
            model=self.model,
            temperature=0.3,
            stream=True,
            extra_body=extra_body,
            messages=self._answer_messages(
                question,
                context_blocks,
                include_closing=True,
                session_guidance_prompt=session_guidance_prompt,
            ),
        )
```

d) 新增 `stream_general_chat`（放在 `stream_answer_with_context` 之后）：

```python
    def stream_general_chat(
        self,
        question: str,
        *,
        system_prompt: str = "",
        enable_search: bool = True,
        forced_search: bool = True,
    ) -> Iterable[str]:
        """开放域问题流式调用，启用联网搜索且不携带本地知识库上下文。"""
        if not self.is_available():
            return iter(())
        extra_body: dict[str, Any] = {"enable_thinking": False}
        if enable_search:
            extra_body["enable_search"] = True
            if forced_search:
                extra_body["search_options"] = {"forced_search": True}
        stream = self._client().chat.completions.create(
            model=self.model,
            temperature=0.3,
            stream=True,
            extra_body=extra_body,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt or general_chat_system_prompt(),
                },
                {"role": "user", "content": str(question)},
            ],
        )
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            text = getattr(delta, "content", None) if delta else None
            if text:
                yield text
```

注意：若实际文件中的 `answer_with_context` 结构与本计划不同，以实际代码为准，保证 `enable_thinking`、`response_format`、`enable_search` 三个键共存且默认行为不变。

- [x] **Step 4: 运行测试确认通过**

Run: `D:\miniconda3\envs\riskMe\python.exe -m pytest backend/tests/unit/model_client/test_web_search.py -q`
Expected: PASS

- [x] **Step 5: 回归相关客户端测试**

Run: `D:\miniconda3\envs\riskMe\python.exe -m pytest backend/tests/unit/model_client/ -q`
Expected: PASS

- [x] **Step 6: 提交**

```bash
git add backend/app/services/model_client/chat.py backend/tests/unit/model_client/test_web_search.py
git commit -m "feat: DashScope聊天客户端支持联网搜索与开放域流式调用"
```

---

### Task 3：意图路由新增 general_chat 分支

**Files:**
- Modify: `backend/app/agents/query_intent/schemas.py`
- Modify: `backend/app/agents/query_intent/agent.py`
- Test: Modify `backend/tests/integration/query_intent/test_intent_agent.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `IntentName` 增加 `"general_chat"`；`TaskRoute` 增加 `"general_chat"`；`TaskType` 增加 `"general_chat"`。
  - 意图 Agent 对开放域问题返回 `intent="general_chat"`（模型高置信 ≥0.85 时覆盖默认 rag）。

- [x] **Step 1: 写失败测试**

在 `backend/tests/integration/query_intent/test_intent_agent.py` 中新增 mock 类：

```python
class GeneralChatLlm:
    """模拟意图模型把与本地知识库无关的问题路由到普通聊天。"""

    def is_available(self) -> bool:
        return True

    def answer_with_context(self, _prompt, _context, max_tokens=0):
        return json.dumps({
            "is_related": False,
            "relation_type": "standalone",
            "reference_type": "none",
            "context_confidence": 1.0,
            "standalone_message": "今天重庆天气怎么样",
            "referenced_case_ids": [],
            "reference_clues": [],
            "intent": "general_chat",
            "intent_confidence": 0.98,
            "intent_reason": "与本地知识库无关的实时天气查询",
        }, ensure_ascii=False)
```

在 `IntentAgentTest` 类内新增测试方法：

```python
    def test_open_domain_question_routes_to_general_chat(self) -> None:
        """开放域实时信息问题由模型语义判断路由到普通聊天。"""
        llm = GeneralChatLlm()
        agent = agent_module.QueryIntentAgent(
            llm_client=llm,
            audit_path=Path(self.temp_dir.name) / "general-chat-audit.jsonl",
            case_provider=self.agent.case_provider,
        )

        result = agent.route(schema_module.IntentRouteRequest(
            message="今天重庆天气怎么样",
        ))

        self.assertEqual(result.intent, "general_chat")
```

- [x] **Step 2: 运行测试确认失败**

Run: `D:\miniconda3\envs\riskMe\python.exe -m pytest backend/tests/integration/query_intent/test_intent_agent.py -q`
Expected: FAIL（模型返回 general_chat 被校验集合拒绝，实际落入默认 rag）

- [x] **Step 3: 实现意图扩展**

`backend/app/agents/query_intent/schemas.py`：

```python
IntentName = Literal["rag", "multidim_search", "similar_case_match", "clarify", "general_chat"]
```

`TaskType` Literal 增加 `"general_chat"`；`TaskRoute = Literal["rag", "multidim_search", "similar_case_match", "clarify", "general_chat"]`；`suggested_intent` 的 Literal 增加 `"general_chat"`。

`backend/app/agents/query_intent/agent.py` 共 6 处：

a) `_model_understanding` 意图 prompt（L971 附近）的 intent 行改为：

```python
            "intent只能是rag、multidim_search、similar_case_match、clarify、general_chat。"
            "与本地气象知识库（山西气象、历史个例、统计、预警）无关的通用闲聊或实时信息，"
            "例如其他地区天气、新闻、常识，选择general_chat；"
```

b) `_model_understanding` 模型返回校验集合（L1110 附近）：

```python
        if intent in {"rag", "multidim_search", "similar_case_match", "clarify", "general_chat"}:
```

c) `_model_decision` 的 JSON 指令 prompt（L1439 附近）的 intent 行改为：

```python
            "只输出JSON，intent只能是rag、multidim_search、similar_case_match、clarify、general_chat。"
            "与本地气象知识库无关的通用闲聊或实时信息选择general_chat；"
```

d) `_model_decision` 校验集合（L1468 附近）：

```python
        if intent not in {"rag", "multidim_search", "similar_case_match", "clarify", "general_chat"}:
            return None
```

e) `_merge_decisions` 的 `llm_over_default_rag` 分支（L1491 附近）：

```python
            and model.intent in {"multidim_search", "similar_case_match", "general_chat"}
```

f) `_target`（L1546 附近）增加：

```python
        if intent == "general_chat":
            return "开放域聊天", ""
```

g) 新增开放域粗判辅助方法（放在 `_rule_decision` 附近），并放宽 `_model_decision` 的跳过条件：

```python
    @staticmethod
    def _looks_open_domain(text: str) -> bool:
        """粗判是否为开放域问题：不含任何本地气象知识库特征词。"""
        knowledge_terms = (
            "山西", "太原", "大同", "朔州", "忻州", "阳泉", "吕梁", "晋中",
            "长治", "晋城", "临汾", "运城", "暴雨", "暴雪", "寒潮", "大风",
            "沙尘", "高温", "干旱", "冻雨", "强对流", "降水", "降雪", "雨雪",
            "降温", "预警", "预报", "个例", "环流", "冷空气", "天气过程",
            "雷达", "云图",
        )
        return not any(term in text for term in knowledge_terms)
```

`_model_decision` 中现有跳过条件：

```python
        if rule.hard or (rule.intent == "rag" and rule.confidence >= 0.9):
            # 简单聚合和普通问答已有稳定规则结论，不再为路由额外等待一次云端模型。
            return None
```

改为：

```python
        if rule.hard or (
            rule.intent == "rag"
            and rule.confidence >= 0.9
            and not self._looks_open_domain(text)
        ):
            # 简单聚合和普通问答已有稳定规则结论，不再为路由额外等待一次云端模型；
            # 疑似开放域问题时仍调用模型，允许其改判为 general_chat。
            return None
```

- [x] **Step 4: 运行测试确认通过**

Run: `D:\miniconda3\envs\riskMe\python.exe -m pytest backend/tests/integration/query_intent/test_intent_agent.py -q`
Expected: PASS（既有用例 + 新增 1 个）

- [x] **Step 5: 提交**

```bash
git add backend/app/agents/query_intent/schemas.py backend/app/agents/query_intent/agent.py backend/tests/integration/query_intent/test_intent_agent.py
git commit -m "feat: 意图路由新增开放域普通聊天分支"
```

---

### Task 4：普通聊天流式链路与编排器接入

**Files:**
- Create: `backend/app/agents/unified_chat/general_chat_stream.py`
- Modify: `backend/app/agents/unified_chat/orchestrator.py`
- Test: Modify `backend/tests/unit/unified_chat/test_orchestrator.py`

**Interfaces:**
- Consumes: `stream_general_chat`（Task 2）、`IntentName="general_chat"`（Task 3）、`Settings.web_search_enabled / web_search_forced`（Task 1）
- Produces:
  - `GeneralChatStreamMixin._stream_general_chat(question, session_id, *, run_id="", performance) -> AsyncIterator[dict]`（SSE：metadata → delta → done）
  - `UnifiedChatOrchestrator` 对 `decision.intent == "general_chat"` 分发到该方法。

- [x] **Step 1: 写失败测试**

在 `backend/tests/unit/unified_chat/test_orchestrator.py` 中新增两个 mock 类：

```python
class _GeneralChatLlm:
    """模拟支持联网搜索的普通聊天客户端。"""

    def __init__(self) -> None:
        self.last_question = ""
        self.last_search = None

    def is_available(self) -> bool:
        return True

    def stream_general_chat(
        self,
        question,
        *,
        system_prompt="",
        enable_search=True,
        forced_search=True,
    ):
        self.last_question = question
        self.last_search = {
            "enable_search": enable_search,
            "forced_search": forced_search,
        }
        yield "重庆今天多云转晴，气温 18℃ 到 26℃。"


class _GeneralChatIntentAgent:
    """模拟意图模型把开放域问题路由到普通聊天。"""

    case_provider = None

    @staticmethod
    def route(_request):
        return SimpleNamespace(
            intent="general_chat",
            rag_strategy="vector",
            normalized_message="今天重庆天气怎么样",
            clarification_question="",
            referenced_case_ids=[],
            conditions={},
            audit_id="intent-general",
            context_related=False,
            relation_type="standalone",
            reference_type="none",
            reference_clues=[],
            context_confidence=1.0,
            context_resolution_source="llm",
            context_message_ids=[],
            routing_source="llm",
            reason="开放域实时天气查询",
        )
```

在 `UnifiedChatOrchestratorTests` 类内新增：

```python
    async def test_general_chat_uses_web_search_stream_and_persists(self) -> None:
        """开放域问题直接走普通聊天联网搜索链路。"""
        self.main.llm_client = _GeneralChatLlm()
        self.dependencies.llm_client = self.main.llm_client
        self.dependencies.intent_agent = _GeneralChatIntentAgent()

        events = await self._collect("今天重庆天气怎么样", "general-chat-session")

        general_metadata = self._agent_metadata(events, "general_chat")
        self.assertEqual(general_metadata["status"], "running")
        answer = "".join(
            event["text"]
            for event in events
            if event.get("type") == "delta"
        )
        self.assertIn("重庆今天多云转晴", answer)
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["status"], "completed")
        self.assertEqual(self.main.llm_client.last_search, {
            "enable_search": True,
            "forced_search": True,
        })
        messages = self.session_store.list_messages("general-chat-session")
        self.assertTrue(any(
            item.get("role") == "assistant"
            and "重庆今天多云转晴" in str(item.get("content") or "")
            for item in messages
        ))
```

- [x] **Step 2: 运行测试确认失败**

Run: `D:\miniconda3\envs\riskMe\python.exe -m pytest backend/tests/unit/unified_chat/test_orchestrator.py -q`
Expected: FAIL（ImportError / AttributeError: _stream_general_chat 不存在）

- [x] **Step 3: 实现普通聊天流式链路**

创建 `backend/app/agents/unified_chat/general_chat_stream.py`：

```python
"""开放域问题直接调用支持联网搜索的普通聊天模型。"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from backend.app.agents.unified_chat.orchestration_models import (
    BudgetTermination,
)
from backend.app.agents.unified_chat.performance import PerformanceTrace
from backend.app.core.config import get_settings


class GeneralChatStreamMixin:
    """把开放域问题流式转发给支持联网搜索的普通聊天模型。"""

    async def _stream_general_chat(
        self,
        question: str,
        session_id: str,
        *,
        run_id: str = "",
        performance: PerformanceTrace,
    ) -> AsyncIterator[dict[str, Any]]:
        """开放域问题直接调用支持联网搜索的普通聊天模型并写回主会话。"""
        settings = get_settings()
        text = str(question or "").strip()
        run_id = run_id or f"general_chat_{uuid4().hex}"
        common = self._empty_result("general_chat", run_id, text, "")
        common["retrieval_mode"] = "agent:general_chat_web_search"
        common["web_search"] = bool(settings.web_search_enabled)
        common["llm_used"] = False
        common["llm_status"] = "skipped_no_api_key"
        common["answer_renderer"] = "fallback"
        llm_client = self.dependencies.llm_client
        fallback = (
            "抱歉，当前无法回答这个问题。您可以询问山西省内的天气历史、"
            "个例分析、统计或预警信息。"
        )
        yield self._metadata(common, status="running")
        if settings.web_search_enabled and llm_client.is_available():
            answer_parts: list[str] = []
            try:
                if hasattr(llm_client, "stream_general_chat"):
                    answer_stream = llm_client.stream_general_chat(
                        text,
                        enable_search=True,
                        forced_search=settings.web_search_forced,
                    )
                else:
                    # 兼容旧测试客户端和外部适配器。
                    try:
                        answer_stream = llm_client.stream_answer_with_context(
                            text,
                            [],
                            enable_search=True,
                            forced_search=settings.web_search_forced,
                        )
                    except TypeError:
                        answer_stream = llm_client.stream_answer_with_context(text, [])
                async for stream_kind, delta in self._stream_sync_deltas(
                    answer_stream
                ):
                    if stream_kind == "heartbeat":
                        yield self._progress_metadata(
                            "general_chat",
                            run_id,
                            "streaming",
                            60,
                            "正在等待联网搜索返回结果",
                            performance,
                        )
                        continue
                    for visible_delta in self._split_stream_delta(delta):
                        answer_parts.append(visible_delta)
                        yield {"type": "delta", "text": visible_delta}
                        await asyncio.sleep(0)
                if not answer_parts:
                    raise RuntimeError("联网搜索模型返回空答案")
                common["answer"] = "".join(answer_parts)
                common["llm_used"] = True
                common["llm_status"] = "called_with_web_search"
                common["answer_renderer"] = "llm"
            except BudgetTermination:
                # 取消、超时和预算异常必须穿透回答兜底。
                raise
            except Exception as exc:
                # 联网搜索失败时回退中文兜底，不阻塞主链路。
                common["llm_status"] = f"web_search_failed: {exc}"
                common["answer"] = fallback
                yield {"type": "delta", "text": fallback}
        else:
            common["answer"] = fallback
            yield {"type": "delta", "text": fallback}
        if session_id:
            self._persist_complete(
                session_id,
                common,
                status="completed",
                performance=performance,
            )
        yield self._completion_metadata(common, status="completed")
        yield self._done(common, status="completed")
        self._schedule_memory_refresh(session_id)
```

`backend/app/agents/unified_chat/orchestrator.py`：

a) import 增加：

```python
from uuid import uuid4
from backend.app.agents.unified_chat.general_chat_stream import (
    GeneralChatStreamMixin,
)
```

b) 类继承列表（L54 附近）在 `ReportStreamMixin` 后增加 `GeneralChatStreamMixin`。

c) `route_messages`（L320 附近）增加：

```python
            "general_chat": "正在联网搜索最新信息",
```

d) 分发分支：在 `if decision.intent == "clarify":` 分支之后、`async for event in self._stream_rag(...)` 之前增加：

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

- [x] **Step 4: 运行测试确认通过**

Run: `D:\miniconda3\envs\riskMe\python.exe -m pytest backend/tests/unit/unified_chat/test_orchestrator.py -q`
Expected: PASS（既有用例 + 新增 1 个）

- [x] **Step 5: 提交**

```bash
git add backend/app/agents/unified_chat/general_chat_stream.py backend/app/agents/unified_chat/orchestrator.py backend/tests/unit/unified_chat/test_orchestrator.py
git commit -m "feat: 编排器接入开放域联网搜索流式链路"
```

---

### Task 5：RAG 检索为空自动升级联网搜索

**Files:**
- Modify: `backend/app/agents/unified_chat/rag_stream.py`
- Test: Modify `backend/tests/unit/unified_chat/test_orchestrator.py`

**Interfaces:**
- Consumes: `_stream_general_chat`（Task 4）、`Settings.rag_empty_upgrade_web_search / web_search_enabled`（Task 1）
- Produces: `RagStreamMixin._should_upgrade_to_web_search() -> bool`

- [x] **Step 1: 写失败测试**

在 `backend/tests/unit/unified_chat/test_orchestrator.py` 的 `UnifiedChatOrchestratorTests` 类内新增：

```python
    async def test_rag_empty_retrieval_upgrades_to_web_search(self) -> None:
        """本地知识库检索为空时自动升级为联网搜索回答。"""
        self.main.llm_client = _GeneralChatLlm()
        self.dependencies.llm_client = self.main.llm_client
        original = self.main._retrieve_document_chunks

        def empty_retrieve(_question, **_kwargs):
            return "document_vector", []

        self.main._retrieve_document_chunks = empty_retrieve
        try:
            events = await self._collect("山西省2025年3月降水总量", "rag-empty-session")
        finally:
            self.main._retrieve_document_chunks = original

        general_metadata = self._agent_metadata(events, "general_chat")
        self.assertEqual(general_metadata["status"], "running")
        answer = "".join(
            event["text"]
            for event in events
            if event.get("type") == "delta"
        )
        self.assertIn("重庆今天多云转晴", answer)
        self.assertNotIn("未检索到相关证据片段", answer)
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["status"], "completed")
```

- [x] **Step 2: 运行测试确认失败**

Run: `D:\miniconda3\envs\riskMe\python.exe -m pytest backend/tests/unit/unified_chat/test_orchestrator.py -q`
Expected: FAIL（空检索仍输出"未检索到相关证据片段"）

- [x] **Step 3: 实现升级逻辑**

`backend/app/agents/unified_chat/rag_stream.py`：

a) import 增加：

```python
import logging
from backend.app.core.config import get_settings
```

并在模块级新增（若无 `LOGGER`）：

```python
LOGGER = logging.getLogger("uvicorn.error")
```

b) 新增辅助方法（放在 `_stream_rag` 之前）：

```python
    def _should_upgrade_to_web_search(self) -> bool:
        """判断本地检索为空时是否允许升级为联网搜索。"""
        settings = get_settings()
        if not (
            settings.web_search_enabled
            and settings.rag_empty_upgrade_web_search
        ):
            return False
        llm_client = self.dependencies.llm_client
        try:
            return bool(llm_client.is_available())
        except Exception:
            return False
```

c) 在 `_stream_rag` 中检索完成、`yield self._progress_metadata(..., f"已找到 {len(hits)} 个相关证据片段", ...)` 之后、`analysis = AgentIntentAnalyzer().analyze(question)` 之前插入：

```python
        if (
            not hits
            and not task_context_blocks
            and self._should_upgrade_to_web_search()
        ):
            # 本地知识库没有证据，升级为联网搜索回答。
            if session_id:
                try:
                    self._finish_persisted_run(
                        run_id,
                        status="completed",
                        answer="",
                        metadata=self._runtime_metadata(),
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "[RAG升级] 检索运行记录收口失败 run_id=%s error=%s",
                        run_id,
                        exc,
                    )
            yield self._progress_metadata(
                "general_chat",
                run_id,
                "web_search",
                40,
                "正在联网搜索最新信息",
                performance,
            )
            async for event in self._stream_general_chat(
                question,
                session_id,
                run_id=f"general_chat_{uuid4().hex}",
                performance=performance,
            ):
                yield event
            return
```

- [x] **Step 4: 运行测试确认通过**

Run: `D:\miniconda3\envs\riskMe\python.exe -m pytest backend/tests/unit/unified_chat/test_orchestrator.py -q`
Expected: PASS

- [x] **Step 5: 提交**

```bash
git add backend/app/agents/unified_chat/rag_stream.py backend/tests/unit/unified_chat/test_orchestrator.py
git commit -m "feat: 本地知识库检索为空时自动升级联网搜索"
```

---

### Task 6：全量回归测试、Docker 更新与收尾

**Files:** 无新增（验证与部署）

- [x] **Step 1: 全量单元与集成测试**

Run: `D:\miniconda3\envs\riskMe\python.exe -m pytest backend/tests -q`
Expected: 全部 PASS（原 424 passed / 3 skipped，新增用例后总数增加且无失败）

- [x] **Step 2: 确认改动清单与 git 状态**

Run: `git status --short`
Expected: 仅包含本次 6 个任务修改的文件，无意外改动。

- [x] **Step 3: 重建并重启 API 容器（Docker 更新）**

```bash
docker compose build api
docker compose up -d api
docker compose ps
```

Expected: `api` 容器重建并 healthy；`docker compose ps` 显示 api healthy。注意 `compose.yaml` 的 api 服务为镜像构建，代码改动必须通过 `docker compose build api` 进入镜像。

- [x] **Step 4: 冒烟验证容器内健康检查**

Run: `docker compose ps`
Expected: api healthy；必要时 `docker logs risk-master-api-1 --tail 30` 确认启动无异常。

- [x] **Step 5: 最终提交（若 Task 1-5 已逐个提交则本任务仅验证，无需额外提交）**

如果仍有未提交改动：`git status` 检查后统一提交并说明。
