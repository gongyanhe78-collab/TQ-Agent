import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ApiRouteStructureTests(unittest.TestCase):
    def test_routes_are_split_by_interface_category(self):
        module_names = [
            "backend.app.api.routes.agent",
            "backend.app.api.routes.files",
            "backend.app.api.routes.sessions",
            "backend.app.api.routes.system",
            "backend.app.api.routes.vectors",
        ]

        for module_name in module_names:
            module = importlib.import_module(module_name)
            self.assertTrue(hasattr(module, "router"), module_name)

    def test_main_registers_split_routers_without_inline_route_decorators(self):
        main_path = ROOT / "backend" / "app" / "main.py"
        source = main_path.read_text(encoding="utf-8")

        self.assertIn("app.include_router(system_router)", source)
        self.assertIn("app.include_router(files_router)", source)
        self.assertIn("app.include_router(vectors_router)", source)
        self.assertIn("app.include_router(sessions_router)", source)
        self.assertIn("app.include_router(agent_router)", source)
        self.assertNotIn("@app.get", source)
        self.assertNotIn("@app.post", source)
        self.assertNotIn("@app.patch", source)
        self.assertNotIn("@app.delete", source)


if __name__ == "__main__":
    unittest.main()
