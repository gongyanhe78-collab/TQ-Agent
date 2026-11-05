"""为多维个例检索提供关闭思考模式且有界并发的大模型适配器。"""
from __future__ import annotations

from threading import BoundedSemaphore, Lock, local
from typing import Any


# 多次 HTTP 请求创建的适配器共享同一个信号量，确保整个多维检索模块最多并发四个生成请求。
# 指标抽取、逐例分析和报告增强共用该 API 并发闸门，避免线程池放大外部请求。
_GLOBAL_LLM_SEMAPHORE = BoundedSemaphore(4)

# 多个 HTTP 请求创建的适配器按连接配置共用同一连接池，避免每次模型调用重新进行 TCP/TLS 握手。
_OPENAI_CLIENTS: dict[tuple[str, str], Any] = {}
_OPENAI_CLIENTS_LOCK = Lock()


class NonThinkingLlmClient:
    """在不修改公共模型客户端的前提下，关闭 Qwen3 思考模式并限制并发。"""

    def __init__(self, client: Any):
        """保存原始客户端，其他未覆盖能力继续透明转发。"""
        self._client = client
        # 每个线程独立保存结束原因，避免并发个例分析相互覆盖日志。
        self._state = local()

    def __getattr__(self, name: str) -> Any:
        """把模型名称、地址和可用性检查等属性转发给原客户端。"""
        return getattr(self._client, name)

    def answer_with_context(
        self,
        question: str,
        context_blocks: list[str],
        max_tokens: int = 1536,
    ) -> str:
        """关闭模型思考模板，并通过全局信号量限制生成式 LLM 并发。"""
        with _GLOBAL_LLM_SEMAPHORE:
            if hasattr(self._client, "model") and hasattr(self._client, "api_key"):
                return self._answer_openai_compatible(question, context_blocks, max_tokens)
            # 测试桩或其他客户端无法注入模板参数时仍复用原接口；并发上限依然生效。
            answer = str(
                self._client.answer_with_context(
                    question,
                    context_blocks,
                    max_tokens=int(max_tokens),
                )
                or ""
            )
            self._set_finish_reason(getattr(self._client, "last_finish_reason", "unknown"))
            return answer

    def _set_finish_reason(self, reason: Any) -> None:
        """记录当前线程最近一次生成结束原因。"""
        self._state.finish_reason = str(reason or "unknown")

    def get_last_finish_reason(self) -> str:
        """返回当前线程最近一次生成结束原因。"""
        return str(getattr(self._state, "finish_reason", "unknown") or "unknown")

    def _answer_openai_compatible(
        self,
        question: str,
        context_blocks: list[str],
        max_tokens: int,
    ) -> str:
        """调用 DashScope 的 OpenAI 兼容接口并显式关闭 Qwen3 思考。"""
        from openai import OpenAI

        messages_builder = getattr(self._client, "_answer_messages", None)
        if not callable(messages_builder):
            raise RuntimeError("当前 OpenAI 兼容客户端缺少消息构造接口。")
        completion = self._shared_openai_client(OpenAI).chat.completions.create(
            model=str(getattr(self._client, "model", "") or ""),
            temperature=0.3,
            max_tokens=int(max_tokens),
            messages=messages_builder(question, context_blocks),
            # Qwen3 的 OpenAI 兼容接口要求把关闭思考参数放进模板参数对象。
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        choice = (getattr(completion, "choices", None) or [None])[0]
        reason = choice.get("finish_reason") if isinstance(choice, dict) else getattr(choice, "finish_reason", None)
        self._set_finish_reason(reason)
        return completion.choices[0].message.content or ""

    def _shared_openai_client(self, client_type: Any) -> Any:
        """按密钥和服务地址复用 OpenAI 客户端，隔离不同模型服务配置。"""
        api_key = str(getattr(self._client, "api_key", "") or "EMPTY")
        base_url = str(getattr(self._client, "base_url", "") or "")
        key = (api_key, base_url)
        client = _OPENAI_CLIENTS.get(key)
        if client is not None:
            return client
        with _OPENAI_CLIENTS_LOCK:
            client = _OPENAI_CLIENTS.get(key)
            if client is None:
                client = client_type(api_key=api_key, base_url=base_url)
                _OPENAI_CLIENTS[key] = client
            return client

def ensure_non_thinking_client(client: Any) -> Any:
    """避免重复包装，并允许无模型配置时保持原有降级行为。"""
    if client is None or isinstance(client, NonThinkingLlmClient):
        return client
    return NonThinkingLlmClient(client)
