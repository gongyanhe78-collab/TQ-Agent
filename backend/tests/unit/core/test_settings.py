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
