"""相似个例智能匹配体的业务入口。"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable
from uuid import uuid4

from backend.app.services.model_client import get_embedding_client, get_llm_client, get_rerank_client

from .infrastructure.data_store import LocalCaseDataStore
from .infrastructure.audit_log import MatchAuditLogWriter
from .workflow.graph import build_smart_case_graph
from .llm.llm_service import SmartCaseLlmService
from .matching.natural_query import normalize_natural_extraction
from .workflow.nodes import SmartCaseGraphNodes
from .schemas import NaturalLanguageMatchRequest, SmartCaseMatchRequest, SmartCaseMatchResponse
from .workflow import progress


logger = logging.getLogger("uvicorn.error")
SSE_HEARTBEAT_SECONDS = 10.0


STREAM_STAGE_META = {
    "normalize_input": ("正在扫描本地个例库", 8),
    "structured_recall": ("结构化候选召回完成", 22),
    "semantic_supplement": ("语义匹配与 Rerank 完成", 38),
    "rank_and_select": ("相似个例排序完成", 52),
    "enrich_cases": ("已完成个例排序，正在提取匹配理由和参考经验", 65),
    "extract_case_references": ("历史个例参考经验提炼完成", 82),
    "synthesize_forecast_tips": ("综合研判生成完成", 94),
    "assemble_result": ("结果校验完成", 100),
}


class SmartCaseMatchAgent:
    """组装本地数据、模型客户端和 LangGraph，并暴露稳定调用接口。"""

    def __init__(
        self,
        data_dir: Path | None = None,
        embedding_client=None,
        rerank_client=None,
        llm_client=None,
        audit_log_dir: Path | None = None,
    ):
        self.store = LocalCaseDataStore(data_dir)
        self.embedding_client = embedding_client or get_embedding_client()
        self.rerank_client = rerank_client or get_rerank_client()
        self.llm_client = llm_client or get_llm_client()
        self.llm_service = SmartCaseLlmService(self.llm_client)
        self.audit_writer = MatchAuditLogWriter(audit_log_dir)
        self.nodes = SmartCaseGraphNodes(
            self.store,
            self.embedding_client,
            self.rerank_client,
            self.llm_service,
        )
        self.graph = build_smart_case_graph(self.nodes)

    def close(self) -> None:
        """统一释放本 Agent 自己创建的 LLM 连接池和工作线程。"""
        self.llm_service.close()

    def __enter__(self) -> "SmartCaseMatchAgent":
        """支持脚本和测试通过 with 语句安全复用并自动关闭 Agent。"""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """离开上下文时统一关闭资源，异常仍交给原调用方处理。"""
        self.close()

    def run(self, request: SmartCaseMatchRequest) -> SmartCaseMatchResponse:
        """执行一次完整匹配，并用 Pydantic 校验最终输出契约。"""
        client_progress_id = str(request.progress_id or "").strip()
        # 运行号必须由服务端生成，客户端进度号只作查询别名，避免重复值覆盖状态和审计文件。
        run_id = f"smart_case_{uuid4().hex}"
        progress.start(run_id, alias=client_progress_id)
        state = {
            "run_id": run_id,
            "client_progress_id": client_progress_id,
            "request": request.model_dump(),
            "warnings": [],
            "audit": {"data_source": "local", "graph": "langgraph_state_graph", "input_mode": "structured"},
            "deadline_monotonic": None,
        }
        try:
            state = self.graph.invoke(state)
            progress.complete(run_id)
            self.audit_writer.write(state, str((state.get("final_output") or {}).get("status") or "completed"))
            return SmartCaseMatchResponse.model_validate(state["final_output"])
        except Exception as exc:
            progress.fail(run_id, f"匹配失败：{exc}")
            self.audit_writer.write(state, "failed", str(exc))
            raise

    async def stream_natural_match(
        self,
        payload: NaturalLanguageMatchRequest,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[tuple[str, dict]]:
        """解析自然语言并按业务阶段流式返回匹配个例和综合研判。"""
        client_progress_id = str(payload.progress_id or "").strip()
        # SSE 首个 accepted 事件会返回唯一运行号；传入的进度号仅保留为兼容查询别名。
        run_id = f"smart_case_chat_{uuid4().hex}"
        progress.start(run_id, alias=client_progress_id)
        state = {
            "run_id": run_id,
            "client_progress_id": client_progress_id,
            "request": {"raw_query": payload.message},
            "warnings": [],
            "audit": {
                "data_source": "local",
                "graph": "langgraph_state_graph",
                "input_mode": "natural_language_stream",
            },
            "deadline_monotonic": None,
        }
        yield "accepted", {
            "run_id": run_id,
            "stage": "parse_natural_query",
            "message": "正在理解天气过程描述",
            "percent": 2,
        }
        try:
            extracted, parse_status = await asyncio.to_thread(
                self.llm_service.parse_natural_query,
                payload.message,
                None,
            )
            date_resolution_audit = {}
            parsed = normalize_natural_extraction(payload.message, extracted, date_resolution_audit)
            # 模型日期表达只服务于规则解析和审计，不进入稳定的业务请求协议。
            parsed.pop("date_expression", None)
            parsed.pop("date_evidence", None)
            request = SmartCaseMatchRequest(
                **parsed,
                progress_id=client_progress_id,
                top_n=3,
                diversity_mode="moderate",
                include_images=True,
            )
            parse_warnings = []
            if parse_status != "called":
                parse_warnings.append(f"自然语言解析未完成有效 LLM 调用（{parse_status}），已使用原文规则解析。")
            yield "parsed_query", {
                "run_id": run_id,
                "message": "过程信息识别完成，正在检索历史个例",
                "percent": 5,
                "query": request.model_dump(exclude={"progress_id", "include_images", "diversity_mode", "top_n"}),
                "parse_status": parse_status,
            }

            state.update({
                "request": request.model_dump(),
                "warnings": parse_warnings,
                "audit": {
                    "data_source": "local",
                    "graph": "langgraph_state_graph",
                    "input_mode": "natural_language_stream",
                    "natural_query_parse_status": parse_status,
                    "date_resolution": date_resolution_audit,
                },
                # 流式链路同样不设总截止点，综合研判可以等待模型完整返回。
                "deadline_monotonic": None,
            })
            iterator = self.graph.astream(state, stream_mode="updates").__aiter__()
            pending = asyncio.create_task(iterator.__anext__())
            while True:
                done, _ = await asyncio.wait({pending}, timeout=SSE_HEARTBEAT_SECONDS)
                if not done:
                    if is_disconnected and await is_disconnected():
                        pending.cancel()
                        raise asyncio.CancelledError
                    # 心跳既防止代理层断开，也让页面知道后端仍在等待模型结果。
                    progress_snapshot = progress.get(run_id) or {}
                    yield "heartbeat", {
                        "run_id": run_id,
                        "message": "模型仍在处理中",
                        # 透传最近一个已完成阶段的进度，避免流式消费者把心跳误判为卡死。
                        "percent": int(progress_snapshot.get("percent") or 5),
                        "stage": str(progress_snapshot.get("stage") or "processing"),
                    }
                    continue
                try:
                    update = pending.result()
                except StopAsyncIteration:
                    break
                pending = asyncio.create_task(iterator.__anext__())
                if not isinstance(update, dict):
                    continue
                for node_name, node_output in update.items():
                    if not isinstance(node_output, dict):
                        continue
                    state.update(node_output)
                    message, percent = STREAM_STAGE_META.get(node_name, (f"{node_name} 已完成", 50))
                    yield "stage", {
                        "run_id": run_id,
                        "stage": node_name,
                        "message": message,
                        "percent": percent,
                    }
                    if node_name == "extract_case_references":
                        # 等匹配理由和参考经验全部完成后一次性发送，避免页面先展示半成品再刷新。
                        yield "matched_cases", {
                            "run_id": run_id,
                            "phase": "referenced",
                            "query_summary": self.nodes._query_summary(state["query"]),
                            "matched_cases": self.nodes.build_matched_cases(state),
                            "warnings": list(state.get("warnings") or []),
                        }
                    elif node_name == "synthesize_forecast_tips":
                        yield "forecast", {
                            "run_id": run_id,
                            "forecast_summary": dict(state.get("forecast_summary") or {}),
                            "forecast_tips": list(state.get("forecast_tips") or []),
                            "warnings": list(state.get("warnings") or []),
                        }

            final_output = SmartCaseMatchResponse.model_validate(state["final_output"]).model_dump()
            progress.complete(run_id)
            # 审计文件在最终状态完整后一次性原子写入，磁盘异常由写入器隔离，不影响SSE结果。
            await asyncio.to_thread(self.audit_writer.write, state, str(final_output.get("status") or "completed"))
            yield "completed", final_output
        except asyncio.CancelledError:
            progress.fail(run_id, "客户端已断开，任务取消")
            await asyncio.to_thread(self.audit_writer.write, state, "cancelled", "客户端已断开")
            logger.info("[SmartCaseMatch][Stream] 客户端断开，停止后续阶段 run_id=%s", run_id)
            raise
        except Exception as exc:
            logger.exception("[SmartCaseMatch][Stream] 自然语言流式匹配失败 run_id=%s error=%s", run_id, exc)
            progress.fail(run_id, f"匹配失败：{exc}")
            await asyncio.to_thread(self.audit_writer.write, state, "failed", str(exc))
            yield "error", {
                "run_id": run_id,
                "message": f"相似个例匹配失败：{exc}",
            }

    def health(self) -> dict:
        """返回本地资料和当前模型客户端的就绪状态。"""
        status = self.store.health()
        status.update({
            "embedding_available": self._client_available(self.embedding_client),
            "rerank_available": self._client_available(self.rerank_client),
            "llm_available": self.llm_service.available(),
            "models": {
                "embedding": self._client_metadata(self.embedding_client),
                "rerank": self._client_metadata(self.rerank_client),
                "llm": self._client_metadata(self.llm_client),
            },
            "graph": "langgraph_state_graph",
        })
        return status

    def image_path(self, image_id: str) -> Path:
        """按图片 ID 返回本地文件路径，不向页面暴露磁盘绝对路径。"""
        image = self.store.get_image(image_id)
        path = Path(str(image.get("resolved_path") or "")) if image else Path()
        if not image or not path.is_file():
            raise FileNotFoundError(f"未找到图片证据：{image_id}")
        return path

    def chunk_detail(self, chunk_id: str) -> dict:
        """按 chunk ID 返回可核验正文，不暴露向量和本地文件路径。"""
        chunk = self.store.get_chunk(chunk_id)
        if not chunk:
            raise FileNotFoundError(f"未找到文字证据：{chunk_id}")
        return {
            "chunk_id": str(chunk.get("chunk_id") or ""),
            "chunk_no": int(chunk.get("chunk_no") or 0),
            "source_pdf": str(chunk.get("source_pdf") or Path(str(chunk.get("file_path") or "")).name),
            "content": str(chunk.get("content") or ""),
        }

    def _client_available(self, client) -> bool:
        """避免健康检查因第三方客户端异常而失败。"""
        try:
            return bool(client and client.is_available())
        except Exception:
            return False

    def _client_metadata(self, client) -> dict:
        """返回不包含密钥和服务地址的模型展示信息。"""
        return {
            "client": type(client).__name__ if client else "None",
            "model": str(getattr(client, "model", None) or getattr(client, "model_uid", "unknown")),
            "available": self._client_available(client),
        }
