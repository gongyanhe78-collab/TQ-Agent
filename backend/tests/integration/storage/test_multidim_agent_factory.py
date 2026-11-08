"""多维检索 Agent 的用户级构造与输出隔离测试。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.app.services.container import AppServices


class MultidimAgentFactoryTests(unittest.TestCase):
    """验证每个用户获得稳定且互不共享的 Agent 输出目录。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runtime_dir = Path(self.temp_dir.name) / "runtime"
        self.services = object.__new__(AppServices)
        self.services.settings = SimpleNamespace(
            case_multidim_runtime_dir=self.runtime_dir,
            smart_case_data_dir=Path(self.temp_dir.name) / "data",
        )
        self.services.llm_client = object()
        self.services.multidim_agent = object()
        self.services._multidim_agents = {}

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_same_user_reuses_agent_and_different_users_are_isolated(self) -> None:
        """同一用户重复取用得到同一实例，不同用户使用不同目录。"""

        def build_agent(_settings, _llm_client, output_dir=None):
            return SimpleNamespace(output_dir=Path(output_dir))

        with patch(
            "backend.app.services.container._build_multidim_agent",
            side_effect=build_agent,
        ):
            first = self.services.multidim_agent_for("user-a")
            repeated = self.services.multidim_agent_for("user-a")
            second = self.services.multidim_agent_for("user-b")

        self.assertIs(first, repeated)
        self.assertIsNot(first, second)
        self.assertEqual(
            first.output_dir,
            self.runtime_dir / "users" / "user-a" / "reports",
        )
        self.assertEqual(
            second.output_dir,
            self.runtime_dir / "users" / "user-b" / "reports",
        )

    def test_empty_user_id_is_rejected(self) -> None:
        """用户级 Agent 工厂不能接受空身份。"""
        with self.assertRaises(ValueError):
            self.services.multidim_agent_for("   ")


if __name__ == "__main__":
    unittest.main()
