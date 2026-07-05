"""
通义千问大模型聊天客户端模块
提供案例精炼、问答生成、流式回答等功能
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from backend.app.config import settings
from backend.app.models import ExtractedCase


# LLM 案例提取的系统提示词：告诉模型如何识别和标准化气象灾害案例
SYSTEM_PROMPT = """
你是一个气象灾害个例抽取助手。
你的任务是基于给定章节标题和正文，判断该章节是否属于灾害天气个例，并在属于个例时返回标准化结果。
个例通常具备“日期范围 + 天气过程/灾种”的标题特征，例如“1-3日雨雪天气过程”“14-15日暴雪天气过程”。
像“预报服务情况”“月工作总结”“服务效果”不属于个例。
只输出 JSON，不要输出额外解释。
""".strip()


@dataclass
class LlmRefinementResult:
    """
    LLM 案例精炼结果数据模型

    Attributes:
        is_case: 是否为有效的灾害案例
        title: 标准化后的案例标题
        date_range: 提取出的时间范围
        content: 标准化后的案例内容
        reason: 做出判断的原因说明
    """
    is_case: bool
    title: str
    date_range: str
    content: str
    reason: str = ""


class DashScopeChatClient:
    """
    阿里云 DashScope 通义千问聊天客户端
    通过 OpenAI 兼容 API 调用通义千问模型
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        """
        初始化聊天客户端

        Args:
            api_key: API 密钥，不传则从配置中读取
            base_url: API 基础地址，不传则使用默认配置
            model: 模型名称，不传则使用默认配置
        """
        self.api_key = api_key or settings.chat_api_key
        self.base_url = base_url or settings.chat_base_url
        self.model = model or settings.chat_model

    def is_available(self) -> bool:
        """
        检查客户端是否可用（是否配置了 API 密钥）

        Returns:
            可用返回 True，否则返回 False
        """
        return bool(self.api_key)

    def refine_case(self, candidate: ExtractedCase) -> LlmRefinementResult:
        """
        使用 LLM 精炼候选案例
        1. 判断该章节是否为灾害案例
        2. 标准化标题、时间范围和内容

        Args:
            candidate: 从 PDF 初步提取的候选案例

        Returns:
            LlmRefinementResult 精炼结果
        """
        # 如果没有配置 API Key，直接返回原始候选作为结果（降级处理）
        if not self.is_available():
            return LlmRefinementResult(
                is_case=True,
                title=candidate.title,
                date_range=candidate.date_range,
                content=candidate.content,
                reason="missing_api_key_fallback",
            )

        # 动态导入 OpenAI SDK（避免没有安装时整个程序报错）
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for DashScope chat calls.") from exc

        # 创建 OpenAI 兼容客户端
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        # 构建发给 LLM 的案例信息
        prompt = {
            "source_pdf": candidate.source_pdf,
            "case_id": candidate.case_id,
            "title": candidate.title,
            "date_range": candidate.date_range,
            "content": candidate.content,
        }
        # 调用聊天完成 API
        completion = client.chat.completions.create(
            model=self.model,
            temperature=0.3,  # 较低温度保证输出稳定
            response_format={"type": "json_object"},  # 强制输出 JSON 格式
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "请判断以下章节是否属于灾害个例，并返回 JSON："
                        '{"is_case": bool, "title": str, "date_range": str, "content": str, "reason": str}\n'
                        + json.dumps(prompt, ensure_ascii=False)
                    ),
                },
            ],
        )
        # 解析返回的 JSON 结果
        raw = completion.choices[0].message.content or "{}"
        data = json.loads(raw)
        return LlmRefinementResult(
            is_case=bool(data.get("is_case")),
            title=str(data.get("title", candidate.title)),
            date_range=str(data.get("date_range", candidate.date_range)),
            content=str(data.get("content", candidate.content)),
            reason=str(data.get("reason", "")),
        )

    def answer_with_context(self, question: str, context_blocks: list[str]) -> str:
        """
        基于上下文块生成问答结果（非流式）

        Args:
            question: 用户问题
            context_blocks: 检索到的相关案例文本块列表

        Returns:
            生成的回答字符串
        """
        # 如果没有配置 API Key，返回降级提示
        if not self.is_available():
            return "未配置 DashScope API Key，当前返回的是本地检索结果摘要。"

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for DashScope chat calls.") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        completion = client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            messages=self._answer_messages(question, context_blocks),
        )
        return completion.choices[0].message.content or ""

    def stream_answer_with_context(self, question: str, context_blocks: list[str]) -> Iterable[str]:
        """
        基于上下文块生成流式问答结果
        用于 SSE 流式响应，逐字返回回答内容

        Args:
            question: 用户问题
            context_blocks: 检索到的相关案例文本块列表

        Yields:
            逐段返回回答文本片段
        """
        # 如果没有配置 API Key，直接返回降级提示
        if not self.is_available():
            yield "未配置 DashScope API Key，当前返回的是本地检索结果摘要。"
            return

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for DashScope chat calls.") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        # 启用流式调用
        stream = client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            stream=True,  # 启用流式输出
            messages=self._answer_messages(question, context_blocks),
        )
        # 遍历流中的每个数据块，提取文本内容
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            text = getattr(delta, "content", None) if delta else None
            if text:
                yield text

    def _answer_messages(self, question: str, context_blocks: list[str]) -> list[dict[str, str]]:
        """
        构建问答用的消息格式（内部方法）
        拼接所有上下文块，构建系统提示和用户消息

        Args:
            question: 用户问题
            context_blocks: 上下文块列表

        Returns:
            OpenAI API 格式的消息列表
        """
        # 将所有上下文块用空行拼接
        context = "\n\n".join(context_blocks)
        return [
            {
                "role": "system",
                "content": (
                    "你是气象灾害个例问答助手，回答要像业务同事在帮忙研判：自然、清楚、少套话。"
                    "优先使用给定材料里的事实、时段、灾种、实况、预报提示和环流背景；"
                    "不要把证据外的推测说成事实。"
                    "如果上下文包含“相关图像证据”，可以根据图注在回答中自然提示“可参考图X...”，"
                    "说明这张图支持哪个判断；但不要编造图中没有给出的细节。"
                    "材料不足时，直接说明缺口，再给出基于气象专业知识的判断。"
                ),
            },
            {
                "role": "user",
                "content": f"用户问题：{question}\n\n最相似的历史个例材料：\n{context}",
            },
        ]
