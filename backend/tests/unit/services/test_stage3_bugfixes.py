"""验证第三阶段收尾时修复的两处真实缺陷，防止后续重构再次引入。"""
from __future__ import annotations

from pathlib import Path

from backend.app.services.knowledge_builder.normalize_main_data import _build_manifest
from backend.app.services.structured_qa import StructuredQuestionAnswerer


class _ManifestSettingsStub:
    """最小配置替身，只暴露 _build_manifest 依赖的字段。"""

    def __init__(self, resource_dir: Path) -> None:
        self.resource_dir = resource_dir
        self.document_collection_name = "weather_document_chunks_v1"
        self.dashscope_embedding_model = "embedding-test"


def test_comparison_visuals_returns_real_visuals() -> None:
    """_comparison_visuals 必须返回可视化列表，而不是此前被截断后的 None。"""
    answerer = object.__new__(StructuredQuestionAnswerer)
    profiles = {
        3: {"count": 2, "top_disasters": "暴雨", "top_cities": "太原"},
        4: {"count": 1, "top_disasters": "冰雹", "top_cities": "大同"},
    }
    visuals = answerer._comparison_visuals(profiles, {})
    assert isinstance(visuals, list)
    assert len(visuals) >= 2
    assert visuals[0]["type"] == "bar"
    assert visuals[0]["title"] == "月份过程数量对比"
    assert visuals[1]["type"] == "table"
    assert visuals[1]["title"] == "分月差异摘要"
    assert visuals[1]["rows"] == [
        {"month": "3月", "count": 2, "disasters": "暴雨", "areas": "太原"},
        {"month": "4月", "count": 1, "disasters": "冰雹", "areas": "大同"},
    ]


def test_build_manifest_uses_injected_settings(tmp_path: Path) -> None:
    """_build_manifest 必须接收并读取 settings，而不是引用未定义名称。"""
    stub = _ManifestSettingsStub(tmp_path)
    chunks = [{"embedding": [0.1, 0.2]}]
    cases = [{"case_id": "SX-001"}]
    manifest = _build_manifest(stub, chunks, cases, [], 0)
    assert manifest["collection_name"] == "weather_document_chunks_v1"
    assert manifest["embedding_model"] == "embedding-test"
    assert manifest["chunk_count"] == 1
    assert manifest["standard_case_count"] == 1
    assert manifest["pdfs"] == []
