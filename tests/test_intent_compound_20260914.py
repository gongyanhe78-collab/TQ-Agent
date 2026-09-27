"""使用真实意图模型验收多子任务问题的任务图和依赖关系。"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import settings


def main() -> None:
    # 这些问题覆盖统计、列举、特征、预警、影响区域和比较等组合目标。
    questions = [
        "帮我统计2025年3月发生了几次灾害，分别是哪几次，有什么特征",
        "分析2025年7月28日灾害实况，并查询此前预警，最后给出影响区域",
        "查询2025年5月高温过程数量，列出各过程时间，并比较它们的持续时长",
        "统计2025年全年暴雨过程，列出每次过程并概括其影响区域和主要特征",
    ]
    agent_module = importlib.import_module("backend.app.services.agent.query-intent-agent.agent")
    request_type = importlib.import_module(
        "backend.app.services.agent.query-intent-agent.schemas"
    ).IntentRouteRequest
    agent = agent_module.QueryIntentAgent(data_dir=settings.smart_case_data_dir)
    results = []
    for question in questions:
        response = agent.route(request_type(message=question))
        results.append({
            "question": question,
            "intent": response.intent,
            "confidence": response.confidence,
            "conditions": response.conditions,
            "tasks": [
                {
                    "id": task.id,
                    "type": task.type,
                    "depends_on": task.depends_on,
                    "scope": task.evidence_scope,
                }
                for task in response.tasks
            ],
        })
    output = Path(__file__).with_name("intent_compound_20260914.json")
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
