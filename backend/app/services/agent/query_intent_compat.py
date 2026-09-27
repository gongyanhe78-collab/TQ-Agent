"""为连字符目录下的意图 Agent 提供稳定的 Python 导入入口。"""
from __future__ import annotations

import importlib


_module = importlib.import_module("backend.app.services.agent.query-intent-agent.agent")
_schema = importlib.import_module("backend.app.services.agent.query-intent-agent.schemas")

QueryIntentAgent = _module.QueryIntentAgent
IntentRouteRequest = _schema.IntentRouteRequest

