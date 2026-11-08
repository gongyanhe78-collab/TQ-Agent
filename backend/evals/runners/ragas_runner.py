"""RAGAS 可选运行器，只在显式提供依赖时才尝试调用裁判模型。"""

from __future__ import annotations

from typing import Any


def run_ragas(
    samples: list[dict[str, Any]],
    *,
    api_key: str | None = None,
    model_config: dict[str, Any] | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """延迟导入 RAGAS 与 LangChain，缺依赖或缺 Key 时返回 skipped。"""

    sample_list = list(samples)
    sample_limit = max(10, min(20, int(limit)))
    selected = sample_list[:sample_limit]
    if not selected:
        return {"status": "skipped", "reason": "RAGAS 样本为空"}
    if len(selected) < 10:
        return {"status": "skipped", "reason": f"RAGAS 样本不足 10 条：实际 {len(selected)} 条"}
    if len(selected) > 20:
        return {"status": "skipped", "reason": "RAGAS 样本超过 20 条"}
    if not api_key:
        return {"status": "skipped", "reason": "缺少 API Key"}

    try:
        from langchain_openai import ChatOpenAI
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        return {
            "status": "skipped",
            "reason": f"缺少 RAGAS 或 LangChain 依赖：{type(exc).__name__}",
        }

    try:
        return _执行RAGAS(
            selected=selected,
            api_key=api_key,
            model_config=model_config or {},
            chat_openai=ChatOpenAI,
            evaluation_dataset=EvaluationDataset,
            evaluate=evaluate,
            metrics=[answer_relevancy, context_precision, context_recall, faithfulness],
        )
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"RAGAS 执行失败：{type(exc).__name__}: {exc}",
        }


def _执行RAGAS(
    *,
    selected: list[dict[str, Any]],
    api_key: str,
    model_config: dict[str, Any],
    chat_openai: Any,
    evaluation_dataset: Any,
    evaluate: Any,
    metrics: list[Any],
) -> dict[str, Any]:
    """调用 OpenAI 兼容裁判模型；缺少字段或调用失败由上层记录，绝不伪造分数。"""

    model_name = model_config.get("chat_model", "gpt-4o-mini")
    judge = chat_openai(api_key=api_key, model=model_name, temperature=0)
    rows = [
        {
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "contexts": item.get("contexts", []) or [],
        }
        for item in selected
    ]
    dataset = evaluation_dataset.from_list(rows)
    result = evaluate(dataset=dataset, metrics=metrics, llm=judge)
    return {
        "status": "completed",
        "sample_size": len(rows),
        "result": result,
    }
