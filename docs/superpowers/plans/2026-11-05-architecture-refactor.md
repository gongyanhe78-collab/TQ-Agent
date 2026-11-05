# 气象灾害个例智能分析系统架构规整实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变现有业务和公开接口的前提下，完成运行产物清理、统一查询业务链路、依赖注入、目录命名规范化和超大模块拆分。

**Architecture:** 采用兼容式重构。FastAPI lifespan 创建 `AppServices`，Router 通过依赖函数获取服务；三个查询 URL 保留原协议并共享 `ChatService`；Router 集中到 API 层，业务包只保留领域逻辑。

**Tech Stack:** Python 3.10+、FastAPI、Pydantic、SQLite、ChromaDB、OpenAI 兼容客户端、Vue 3、Vite、Pytest。

**Spec:** `docs/superpowers/specs/2026-11-05-architecture-refactor-design.md`

## Global Constraints

- 所有 Markdown 文档和代码注释使用中文。
- 不配置 Ruff、Mypy、ESLint、Prettier 或 CI。
- 保留 `/api/query`、`/api/agent/query`、`/api/query/stream` 的 URL、请求字段、响应字段和 SSE 协议。
- `/api/query/stream` 是唯一主业务链路，另外两个接口是兼容适配器。
- Router 不得导入 `backend.app.main`，编排器不得接收 `main` 模块。
- 模块导入时不得连接数据库、加载模型、创建目录或执行任务。
- 新建 Python 文件原则上以 100～300 行为宜，预计超过 400 行时按职责拆分。
- 单函数尽量控制在 20～50 行，超过 80 行时检查是否承担过多职责。
- 不删除本地运行数据；使用 `git rm --cached` 移出跟踪。
- 不将真实密钥写入代码、日志、测试或 Git。
- 不修改现有业务规则、模型提示词、检索算法、排序规则和报告格式。

---

## 文件结构

### 应用装配与基础设施

- `backend/app/main.py`：应用工厂、CORS 和 Router 注册。
- `backend/app/core/config.py`：无文件创建副作用的配置模型。
- `backend/app/core/lifecycle.py`：应用启动时初始化目录和 `AppServices`。
- `backend/app/services/container.py`：应用级服务组合根。
- `backend/app/api/dependencies.py`：Router 使用的 FastAPI 依赖。

### 查询业务

- `backend/app/services/query/chat_service.py`：三个查询入口共享的主业务服务。
- `backend/app/services/query/agent_query_service.py`：旧智能体查询兼容业务。
- `backend/app/services/query/document_qa.py`：文档检索、重排和回答。
- `backend/app/services/query/context_cases.py`：会话上下文个例解析。
- `backend/app/services/query/result_presenters.py`：统一结果和证据响应转换。
- `backend/app/services/query/structured_query.py`：结构化问题回答与指标文档降级。

### API 层

- `backend/app/api/routes/agent.py`
- `backend/app/api/routes/sessions.py`
- `backend/app/api/routes/system.py`
- `backend/app/api/routes/vectors.py`
- `backend/app/api/routes/case_multidim.py`
- `backend/app/api/routes/smart_case_match.py`
- `backend/app/api/routes/query_intent.py`

### Agent 与统一编排

- `backend/app/agents/unified_chat/orchestrator.py`
- `backend/app/agents/unified_chat/stream_context.py`
- `backend/app/agents/unified_chat/structured_context.py`
- `backend/app/agents/unified_chat/persistence.py`
- `backend/app/agents/unified_chat/stream_events.py`
- `backend/app/agents/case_multidim_search/**`
- `backend/app/agents/smart_case_match/**`
- `backend/app/agents/query_intent/**`

### 前端

- `frontend/src/components/chat/ChatComposer.vue`
- `frontend/src/components/chat/ChatMessages.vue`
- `frontend/src/components/chat/EvidencePreview.vue`
- `frontend/src/components/chat/chatRendering.js`
- `frontend/src/components/chat/AssistantActionsToolbar.vue`
- `frontend/src/components/chat/AssistantFeedbackDialog.vue`

### 测试

- `backend/tests/unit/**`
- `backend/tests/integration/**`
- `backend/tests/e2e/**`

---

### Task 1: 建立结构边界测试并清理运行产物

**Files:**
- Create: `backend/tests/unit/test_repository_hygiene.py`
- Modify: `.gitignore`
- Move: 所有固定测试源码到 `backend/tests/`

**Interfaces:**
- Consumes: 现有 Git 跟踪状态。
- Produces: `test_runtime_outputs_are_ignored()` 和 `test_router_modules_do_not_import_main()`，供后续任务持续检查。

- [ ] **Step 1: 编写失败的仓库卫生与边界测试**

```python
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_runtime_outputs_are_ignored() -> None:
    """运行数据、日志、截图和临时目录必须已写入忽略规则。"""
    ignore_text = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    expected_rules = (
        "runtime/",
        "logs/",
        ".codex_tmp/",
        "backend/tests/outputs/",
        "tmp/",
        "data/numba_cache/",
    )
    missing = [rule for rule in expected_rules if rule not in ignore_text]
    assert missing == []


def test_router_modules_do_not_import_main() -> None:
    """Router 只能通过依赖注入访问应用服务。"""
    source = (PROJECT_ROOT / "backend/app/api/routes/agent.py").read_text(encoding="utf-8")
    assert "from backend.app import main" not in source
    assert "import backend.app.main" not in source
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv\Scripts\python.exe -m pytest backend/tests/unit/test_repository_hygiene.py -v`

Expected: `test_runtime_outputs_are_ignored` 因缺少规则失败。

- [ ] **Step 3: 补充 `.gitignore`**

```gitignore
# 运行数据与日志
runtime/
**/runtime/
logs/
**/logs/
.codex_tmp/

# 本地演示截图和测试输出
tmp/
backend/tests/outputs/
backend/tests/**/artifacts/
backend/tests/**/*.png
backend/tests/**/*.pdf

# 数据库及缓存
*.sqlite3
*.sqlite3-*
*.db
*.sqlite
data/numba_cache/

# 构建和临时产物
frontend/dist/
*.tmp
*.bak
```

- [ ] **Step 4: 将运行产物移出 Git 跟踪**

Run:

```powershell
git rm -r --cached -- 'backend/app/services/agent/case_multidim_search/runtime'
git rm -r --cached -- 'backend/app/services/agent/Smart_Case_Match/logs'
git rm -r --cached -- 'backend/app/services/agent/query-intent-agent/runtime'
git rm -r --cached -- '.codex_tmp'
git rm -r --cached -- 'tmp'
git rm -r --cached -- 'data/numba_cache'
```

逐项运行 `git ls-files -- '**/runtime/**' '**/logs/**' '.codex_tmp/**' 'tmp/**' '*.sqlite3'`，Expected: 无输出。

- [ ] **Step 5: 移动固定测试源码到新测试目录**

移动四类内容：

- `backend/app/api/routes/tests/*.py` -> `backend/tests/integration/api/`
- `backend/app/services/agent/unified_chat/tests/*.py` -> `backend/tests/unit/unified_chat/`
- `backend/app/services/agent/Smart_Case_Match/tests/*.py` -> `backend/tests/integration/smart_case_match/`
- `backend/app/services/agent/case_multidim_search/tests/*.py` -> `backend/tests/integration/case_multidim_search/`

测试诊断 JSON、截图和 PDF 不移动，保持本地并退出 Git 跟踪。

- [ ] **Step 6: 运行测试并提交**

Run: `.venv\Scripts\python.exe -m pytest backend/tests/unit/test_repository_hygiene.py -v`

Expected: 两个测试全部通过。

```powershell
git add -- .gitignore backend/tests docs/superpowers/plans/2026-11-05-architecture-refactor.md
git commit -m "chore: 清理运行产物并建立结构边界测试"
```

### Task 2: 建立无副作用配置与应用工厂

**Files:**
- Create: `backend/app/core/__init__.py`
- Create: `backend/app/core/config.py`
- Create: `backend/app/core/lifecycle.py`
- Create: `backend/app/services/container.py`
- Create: `backend/app/api/dependencies.py`
- Create: `backend/tests/unit/test_app_factory.py`
- Modify: `backend/app/main.py`
- Delete: `backend/app/config.py`

**Interfaces:**
- Consumes: 现有 `Settings` 配置字段。
- Produces: `get_settings() -> Settings`、`AppServices`、`build_services() -> AppServices`、`create_app(services_factory: Callable[[], AppServices] | None = None) -> FastAPI`、`get_services(request: Request) -> AppServices`。

- [ ] **Step 1: 编写失败的应用工厂测试**

```python
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_main_module_has_no_global_service_construction() -> None:
    """应用入口只负责装配，不在导入时构造模型和数据库客户端。"""
    source = (PROJECT_ROOT / "backend/app/main.py").read_text(encoding="utf-8")
    forbidden = (
        "ChromaDocumentChunkStore(",
        "SessionStore(",
        "get_llm_client()",
        "get_embedding_client()",
        "settings.ensure_directories()",
    )
    assert [item for item in forbidden if item in source] == []


def test_config_import_does_not_create_directories(tmp_path, monkeypatch) -> None:
    """导入配置模块时不执行数据目录创建。"""
    from backend.app.core.config import Settings

    settings = Settings(data_dir=tmp_path / "runtime")
    assert not settings.data_dir.exists()
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv\Scripts\python.exe -m pytest backend/tests/unit/test_app_factory.py -v`

Expected: `backend.app.core.config` 不存在。

- [ ] **Step 3: 创建配置和生命周期模块**

`Settings` 保持现有字段、默认值和环境变量名。`get_settings()` 缓存配置实例，但只读取配置，不创建目录。`initialize_settings(settings)` 负责创建目录并设置 `NUMBA_CACHE_DIR`。

将全仓 `from backend.app.config import settings` 改为
`from backend.app.core.config import get_settings`，并在需要配置的构造函数中通过参数或
`settings = get_settings()` 获取配置。知识库构建脚本等允许在显式执行入口获取配置，
但模块导入时不得创建目录。

```python
def initialize_settings(settings: Settings) -> None:
    """在应用启动阶段准备配置要求的本地目录。"""
    from functools import lru_cache
    from pathlib import Path

    settings.ensure_directories()
    os.environ.setdefault("NUMBA_CACHE_DIR", str(settings.numba_cache_dir))
```

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """创建并关闭应用级服务容器。"""
    settings = get_settings()
    initialize_settings(settings)
    services = build_services(settings)
    app.state.services = services
    try:
        yield
    finally:
        services.close()
```

- [ ] **Step 4: 创建 `AppServices`**

```python
@dataclass(slots=True)
class AppServices:
    """应用级依赖容器，只负责构造和传递服务。"""

    settings: Settings
    embedding_client: Any
    llm_client: Any
    document_store: ChromaDocumentChunkStore
    image_store: ImageEvidenceStore
    standard_case_store: JsonStandardCaseStore
    session_store: SessionStore
    similar_case_matcher: SimilarCaseMatcher
    structured_answerer: StructuredQuestionAnswerer
    chat_service: ChatService
    multidim_agent: Any
    multidim_store: Any
    smart_agent: Any
    intent_agent: QueryIntentAgent

    def close(self) -> None:
        """关闭支持显式释放的存储与 Agent。"""
```

`build_services()` 按依赖方向构造：配置 -> 客户端 -> 存储 -> 工具 -> 查询服务 -> Agent。构造期间不发起模型请求。

- [ ] **Step 5: 将 `main.py` 改为应用工厂**

```python
def create_app(services_factory=None) -> FastAPI:
    """创建 FastAPI 应用并注册统一 API 路由。"""
    app = FastAPI(title="Weather Case RAG System", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.router.lifespan_context = lifespan
    register_routes(app)
    return app


app = create_app()
```

将 `allow_credentials` 改为 `False`，因为通配来源不能与凭证模式同时启用；现有前端不使用 Cookie 凭证。

- [ ] **Step 6: 运行测试并提交**

Run: `.venv\Scripts\python.exe -m pytest backend/tests/unit/test_app_factory.py -v`

Expected: 全部通过。

```powershell
git add -- backend/app/core backend/app/services/container.py backend/app/api/dependencies.py backend/app/main.py backend/tests/unit/test_app_factory.py
git commit -m "refactor: 引入无副作用配置和应用服务容器"
```

### Task 3: 抽取统一查询业务服务

**Files:**
- Create: `backend/app/services/query/__init__.py`
- Create: `backend/app/services/query/chat_service.py`
- Create: `backend/app/services/query/agent_query_service.py`
- Create: `backend/app/services/query/document_qa.py`
- Create: `backend/app/services/query/context_cases.py`
- Create: `backend/app/services/query/result_presenters.py`
- Create: `backend/app/services/query/structured_query.py`
- Create: `backend/tests/integration/query/test_chat_service.py`
- Modify: `backend/app/services/container.py`
- Modify: `backend/app/main.py`

**Interfaces:**
- Consumes: `AppServices` 中的模型、索引、存储和 Agent。
- Produces: `ChatService.query_document(request: QueryRequest) -> dict`、`ChatService.query_agent(request: AgentQueryRequest) -> dict`、`ChatService.stream(request: StreamQueryRequest, is_disconnected) -> AsyncIterator[dict]`。

- [ ] **Step 1: 编写失败的查询服务测试**

```python
from unittest.mock import Mock

from backend.app.api.schemas import QueryRequest
from backend.app.services.query.chat_service import ChatService


def test_query_document_uses_structured_service_then_document_fallback() -> None:
    """兼容查询入口必须与主链路共享结构化回答服务和文档问答服务。"""
    structured = Mock()
    structured.answer.return_value = None
    document_qa = Mock()
    document_qa.answer.return_value = {
        "question": "7月天气",
        "answer": "回答",
        "retrieval_mode": "vector",
        "llm_used": False,
        "llm_status": "fallback",
        "hits": [],
    }
    service = ChatService(
        session_store=Mock(),
        structured_query=structured,
        document_qa=document_qa,
        orchestrator_factory=Mock(),
    )
    result = service.query_document(QueryRequest(question="7月天气"))
    assert result["answer"] == "回答"
    structured.answer.assert_called_once()
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv\Scripts\python.exe -m pytest backend/tests/integration/query/test_chat_service.py -v`

Expected: `backend.app.services.query.chat_service` 不存在。

- [ ] **Step 3: 按现有函数边界迁移实现**

迁移映射：

- `main._answer_structured_question`、`_structured_result_needs_metric_documents`、`_metric_document_result_from_structured` -> `structured_query.py`
- `main._ask_document_chunks`、`_retrieve_document_chunks`、`_rank_document_chunks_by_question`、文档上下文与 fallback -> `document_qa.py`
- `main._context_case_ids_from_messages` 及其辅助函数 -> `context_cases.py`
- `main._document_hit_to_response`、`_image_evidence_to_response`、`_standard_case_to_response` -> `result_presenters.py`
- `main._answer_agent_question`、`_answer_agent_question_once`、`_apply_final_review` -> `agent_query_service.py`

迁移时保持函数语义，仅把模块级全局变量改为构造函数依赖。保留现有常量和阈值。

- [ ] **Step 4: 实现 `ChatService`**

```python
class ChatService:
    """三个查询入口共享的查询业务服务。"""

    def query_document(self, request: QueryRequest) -> dict[str, Any]:
        """执行非流式结构化优先的文档查询。"""

    def query_agent(self, request: AgentQueryRequest) -> dict[str, Any]:
        """执行兼容旧智能体协议的结构化查询。"""

    async def stream(
        self,
        request: StreamQueryRequest,
        is_disconnected: Callable[[], Awaitable[bool]],
    ) -> AsyncIterator[dict[str, Any]]:
        """执行统一流式查询主链路。"""
```

`stream()` 创建请求级编排器并原样转发事件，不重新实现意图路由。

- [ ] **Step 5: 更新查询接口调用统一服务**

`api/routes/agent.py` 的三个端点分别调用：

```python
return services.chat_service.query_document(payload)
return services.chat_service.query_agent(payload)
async for event in services.chat_service.stream(payload, request.is_disconnected):
    yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
```

- [ ] **Step 6: 运行查询测试与编译检查**

Run:

```powershell
.venv\Scripts\python.exe -m pytest backend/tests/integration/query/test_chat_service.py -v
.venv\Scripts\python.exe -m compileall backend/app
```

Expected: 测试通过，编译退出码为 0。

- [ ] **Step 7: 提交**

```powershell
git add -- backend/app/services/query backend/app/api/routes/agent.py backend/app/services/container.py backend/tests/integration/query
git commit -m "refactor: 统一三个查询入口的业务服务"
```

### Task 4: 将 Router 全部改为依赖注入

**Files:**
- Create: `backend/tests/integration/api/test_dependency_injection.py`
- Modify: `backend/app/api/routes/agent.py`
- Modify: `backend/app/api/routes/sessions.py`
- Modify: `backend/app/api/routes/system.py`
- Modify: `backend/app/api/routes/vectors.py`

**Interfaces:**
- Consumes: `get_services(request) -> AppServices` 和 `ChatService`。
- Produces: 所有 Router 均可通过 `app.dependency_overrides` 与假服务测试。

- [ ] **Step 1: 编写失败的反向依赖测试**

```python
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_api_routes_do_not_resolve_main_module() -> None:
    """API 层不得在运行时读取应用入口模块。"""
    route_dir = PROJECT_ROOT / "backend/app/api/routes"
    offenders = []
    for path in route_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "backend.app import main" in text or "backend.app.main" in text:
            offenders.append(path.name)
    assert offenders == []
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv\Scripts\python.exe -m pytest backend/tests/integration/api/test_dependency_injection.py -v`

Expected: 报告现有 Router 文件名。

- [ ] **Step 3: 为 Router 添加依赖参数**

```python
from backend.app.api.dependencies import ServicesDep


@router.get("/api/health")
def health(services: ServicesDep):
    """返回应用健康状态。"""
    return build_health_response(services)
```

禁止在路由函数中再次导入业务包 Router 的模块级 `_agent`；多维和 Smart Agent 从 `services` 获取。

- [ ] **Step 4: 拆分会话路由辅助逻辑**

将 `sessions.py` 中的 PDF 导出、分享快照、消息元数据和错误响应构造移到 `services/session_exports.py` 与 `api/presenters.py`。路由函数只校验参数、调用服务、返回响应。

- [ ] **Step 5: 运行测试**

Run: `.venv\Scripts\python.exe -m pytest backend/tests/integration/api -v`

Expected: 所有 API 集成测试通过。

- [ ] **Step 6: 提交**

```powershell
git add -- backend/app/api backend/app/services/session_exports.py backend/tests/integration/api
git commit -m "refactor: 使用依赖注入移除路由反向依赖"
```

### Task 5: 统一 Agent 命名并迁移 Router

**Files:**
- Move: `backend/app/services/agent/Smart_Case_Match/**` -> `backend/app/agents/smart_case_match/**`
- Move: `backend/app/services/agent/query-intent-agent/**` -> `backend/app/agents/query_intent/**`
- Move: `backend/app/services/agent/case_multidim_search/**` -> `backend/app/agents/case_multidim_search/**`
- Move: `backend/app/services/agent/unified_chat/**` -> `backend/app/agents/unified_chat/**`
- Move: Agent Router 到 `backend/app/api/routes/`
- Delete: `backend/app/services/agent/query_intent_compat.py`
- Create: `backend/tests/unit/test_package_naming.py`

**Interfaces:**
- Consumes: 现有 Agent 公共类和请求模型。
- Produces: `backend.app.agents.query_intent`、`backend.app.agents.smart_case_match` 等可正常导入的 snake_case 包。

- [ ] **Step 1: 编写失败的包命名测试**

```python
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_agent_directories_use_snake_case() -> None:
    """Agent 包目录必须是可导入的 snake_case。"""
    agent_dir = PROJECT_ROOT / "backend/app/agents"
    names = {path.name for path in agent_dir.iterdir() if path.is_dir()}
    assert "smart_case_match" in names
    assert "query_intent" in names
    assert "Smart_Case_Match" not in names
    assert "query-intent-agent" not in names
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv\Scripts\python.exe -m pytest backend/tests/unit/test_package_naming.py -v`

Expected: `backend/app/agents` 不存在。

- [ ] **Step 3: 移动并重命名 Agent 包**

使用 `Move-Item` 后更新所有 `backend.app.services.agent` 导入为 `backend.app.agents`，更新旧包名引用，删除 `query_intent_compat.py`。

- [ ] **Step 4: 迁移 Router**

- `case_multidim_search/router.py` -> `api/routes/case_multidim.py`
- `smart_case_match/router.py` -> `api/routes/smart_case_match.py`
- `query_intent/router.py` -> `api/routes/query_intent.py`

保留 Router prefix、路径、响应模型和静态资源路径。Agent 生命周期从 Router 单例移到 `AppServices`。

- [ ] **Step 5: 验证导入与旧引用清理**

Run:

```powershell
rg -n "Smart_Case_Match|query-intent-agent|query_intent_compat|services\.agent" backend/app --glob '*.py'
.venv\Scripts\python.exe -m compileall backend/app
```

Expected: `rg` 无输出，编译退出码为 0。

- [ ] **Step 6: 运行测试并提交**

Run: `.venv\Scripts\python.exe -m pytest backend/tests/unit/test_package_naming.py backend/tests/integration -v`

Expected: 全部通过。

```powershell
git add -- backend/app
git commit -m "refactor: 统一 Agent 包命名和路由位置"
```

### Task 6: 拆分统一编排器并移除 `main` 依赖

**Files:**
- Create: `backend/app/agents/unified_chat/stream_events.py`
- Create: `backend/app/agents/unified_chat/stream_context.py`
- Create: `backend/app/agents/unified_chat/structured_context.py`
- Create: `backend/app/agents/unified_chat/persistence.py`
- Create: `backend/tests/unit/unified_chat/test_orchestrator_dependencies.py`
- Modify: `backend/app/agents/unified_chat/orchestrator.py`
- Modify: `backend/app/services/query/chat_service.py`

**Interfaces:**
- Consumes: `ChatService`、会话存储、各 Agent、意图服务和查询 presenter。
- Produces: `UnifiedChatOrchestrator` 构造参数只包含明确依赖；`stream()` 输入输出协议保持不变。

- [ ] **Step 1: 编写失败的依赖测试**

```python
import inspect

from backend.app.agents.unified_chat.orchestrator import UnifiedChatOrchestrator


def test_orchestrator_does_not_accept_main_module() -> None:
    """编排器只接收明确服务，不接收整个应用入口模块。"""
    parameters = inspect.signature(UnifiedChatOrchestrator).parameters
    assert "main_module" not in parameters
    assert "dependencies" in parameters
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv\Scripts\python.exe -m pytest backend/tests/unit/unified_chat/test_orchestrator_dependencies.py -v`

Expected: `main_module` 参数存在。

- [ ] **Step 3: 定义编排依赖**

```python
@dataclass(slots=True)
class OrchestratorDependencies:
    """统一编排器所需的明确应用服务。"""

    session_store: SessionStore
    document_qa: DocumentQA
    structured_context: StructuredContextService
    multidim_agent: Any
    multidim_store: Any
    smart_agent: Any
    intent_agent: QueryIntentAgent
    llm_client: Any
```

构造函数改为：

```python
def __init__(self, dependencies: OrchestratorDependencies):
    """保存编排依赖并创建请求级记忆管理器。"""
    self.dependencies = dependencies
    self.session_store = dependencies.session_store
```

- [ ] **Step 4: 替换 `self.main` 调用**

逐一替换：

- `self.main._retrieve_document_chunks` -> `self.dependencies.document_qa.retrieve`
- `self.main._ask_document_chunks` -> `self.dependencies.document_qa.answer`
- `self.main._answer_structured_question` -> `self.dependencies.structured_context.answer`
- `self.main._standard_case_to_response` -> 查询 presenter
- `self.main.document_store` -> `self.dependencies.document_qa.document_store`
- `self.main.session_store` -> `self.dependencies.session_store`

- [ ] **Step 5: 按职责抽取辅助模块**

- `stream_events.py`：进度、metadata、delta、done 和文本切块。
- `stream_context.py`：重生成上下文、会话记忆、反馈指导与待确认状态。
- `structured_context.py`：任务计划、结构化证据、覆盖说明和上下文补全。
- `persistence.py`：回答保存、消息元数据、记忆刷新和性能信息。

`orchestrator.py` 保留 `stream()` 和三条业务分支协调，不复制辅助模块。

- [ ] **Step 6: 收窄异常处理**

保留必要的领域异常处理，移除会把 `KeyboardInterrupt`、`SystemExit` 和不可预期编程错误统一转成普通回答错误的宽泛捕获。SSE 可预期异常仍转换为现有 `{"type": "error", ...}`。

- [ ] **Step 7: 运行测试并提交**

Run:

```powershell
.venv\Scripts\python.exe -m pytest backend/tests/unit/unified_chat backend/tests/integration/query -v
.venv\Scripts\python.exe -m compileall backend/app
```

Expected: 测试通过，编译退出码为 0。

```powershell
git add -- backend/app/agents/unified_chat backend/app/services/query backend/tests/unit/unified_chat
git commit -m "refactor: 拆分统一编排器并明确依赖边界"
```

### Task 7: 拆分前端大组件

**Files:**
- Create: `frontend/src/components/chat/ChatComposer.vue`
- Create: `frontend/src/components/chat/ChatMessages.vue`
- Create: `frontend/src/components/chat/EvidencePreview.vue`
- Create: `frontend/src/components/chat/chatRendering.js`
- Create: `frontend/src/components/chat/AssistantActionsToolbar.vue`
- Create: `frontend/src/components/chat/AssistantFeedbackDialog.vue`
- Modify: `frontend/src/components/ChatPanel.vue`
- Modify: `frontend/src/components/AssistantMessageActions.vue`

**Interfaces:**
- Consumes: 现有 `ChatPanel` props/emits 和消息结构。
- Produces: `ChatPanel` 保持原 props/emits；子组件只接收展示所需数据。

- [ ] **Step 1: 运行前端基线构建**

Run: `npm run build`

Workdir: `frontend`

Expected: 构建成功，保存当前 chunk 大小作为性能基线。

- [ ] **Step 2: 抽取消息渲染与预览**

`chatRendering.js` 负责 Markdown 安全渲染、图片 ID 识别、资源 URL 和进度百分比。`ChatMessages.vue` 负责消息循环，`EvidencePreview.vue` 负责图片预览。

原 `ChatPanel.vue` 继续负责组合这些子组件和向上抛出 `query`、`regenerate`、`select-version`。

- [ ] **Step 3: 抽取输入区**

`ChatComposer.vue` 接收 `disabled` 和 `loading`，发出 `submit`。`ChatPanel.vue` 只管理输入清空和提交协调。

- [ ] **Step 4: 拆分消息操作组件**

`AssistantActionsToolbar.vue` 负责复制、分享、收藏、更多操作和触发反馈。`AssistantFeedbackDialog.vue` 负责反馈选项和描述提交。

- [ ] **Step 5: 构建并验证**

Run: `npm run build`

Workdir: `frontend`

Expected: 构建成功；组件公开 props/emits 与重构前一致。

- [ ] **Step 6: 提交**

```powershell
git add -- frontend/src/components
git commit -m "refactor: 拆分聊天页面大型组件"
```

### Task 8: 更新 README、补齐回归测试并执行双端口验证

**Files:**
- Modify: `README.md`
- Modify: `backend/tests/e2e/test_query_compatibility.py`
- Modify: `docs/superpowers/specs/2026-11-05-architecture-refactor-design.md`

**Interfaces:**
- Consumes: 重构后的应用工厂、Router 和前端构建产物。
- Produces: 用户可执行的快速开始说明和最终验证记录。

- [ ] **Step 1: 编写查询兼容测试**

```python
from fastapi.testclient import TestClient

from backend.app.main import app


def test_three_query_paths_are_registered() -> None:
    """三个兼容查询 URL 必须继续注册。"""
    paths = {route.path for route in app.routes}
    assert "/api/query" in paths
    assert "/api/agent/query" in paths
    assert "/api/query/stream" in paths
```

- [ ] **Step 2: 运行完整静态检查和测试**

Run:

```powershell
.venv\Scripts\python.exe -m compileall backend/app backend/tests
.venv\Scripts\python.exe -m pytest backend/tests -v
```

Expected: 编译退出码为 0，测试全部通过。外部模型不可用的测试使用 mock 或明确跳过，不使用真实 API Key。

- [ ] **Step 3: 启动后端和前端**

Run:

```powershell
Start-Process -FilePath '.venv\Scripts\uvicorn.exe' -ArgumentList 'backend.app.main:app','--host','127.0.0.1','--port','8000' -WindowStyle Hidden
Start-Process -FilePath 'npm.cmd' -ArgumentList 'run','dev','--','--host','127.0.0.1','--port','5173' -WorkingDirectory 'frontend' -WindowStyle Hidden
```

检查：

- `Invoke-WebRequest http://127.0.0.1:8000/api/health`
- `Invoke-WebRequest http://127.0.0.1:8000/api/knowledge/status`
- `Invoke-WebRequest http://127.0.0.1:5173`
- 向 `/api/query/stream` 发起轻量请求并确认返回 `text/event-stream` 或现有结构化错误事件。

完成后停止本次启动的两个进程，不影响用户已有的其他进程。

- [ ] **Step 4: 按最新代码重写 README**

README 包含项目徽标和小图标、功能简介、系统架构、唯一主查询链路、最新目录结构、快速开始、统一聊天使用说明、兼容接口、知识库构建、测试和本地截图说明。

README 路径、端口、启动命令、环境变量和新目录必须与当前代码一致。

- [ ] **Step 5: 最终检查**

Run:

```powershell
git status --short
git diff --check
rg -n "backend.app import main|self\.main|Smart_Case_Match|query-intent-agent|query_intent_compat" backend/app --glob '*.py'
```

Expected: `git diff --check` 无输出；旧依赖搜索无输出；`git status` 只包含本次重构预期文件和用户原有未跟踪项。

- [ ] **Step 6: 提交**

```powershell
git add -- README.md backend/tests backend/app frontend/src .gitignore
git commit -m "docs: 更新最新版架构与使用说明"
```

## 验收记录

实施完成后在交付说明中列出：

- 后端编译结果。
- 后端测试数量和通过情况。
- 前端构建结果。
- 后端与前端端口健康检查结果。
- SSE 冒烟检查结果。
- 仍未执行的依赖型测试及原因。
- 移出 Git 跟踪的产物范围。
- README 和新目录说明。
