"""路由契约与黄金数据集版本的一致性测试。"""

from __future__ import annotations

import json
from pathlib import Path

仓库根目录 = Path(__file__).resolve().parents[4]
数据路径 = 仓库根目录 / "backend" / "evals" / "datasets" / "golden_v1.json"
契约路径 = 仓库根目录 / "docs" / "evals" / "routing-contract.md"


def _加载样例() -> dict[str, dict]:
    """按 case_id 读取黄金数据集，便于做精确契约断言。"""

    payload = json.loads(数据路径.read_text(encoding="utf-8"))
    return {item["case_id"]: item for item in payload["cases"]}


def test_routing_contract_document_defines_all_routes() -> None:
    """契约文档必须明确四类路由的边界和示例。"""

    assert 契约路径.exists()
    text = 契约路径.read_text(encoding="utf-8")
    for route in ("rag", "multidim_search", "similar_case_match", "clarify"):
        assert f"`{route}`" in text
    assert "库内" in text
    assert "报告" in text
    assert "库外" in text


def test_known_case_retrieval_and_comparison_use_primary_rag() -> None:
    """库内检索、统计和已知个例比较统一由主 RAG 与结构化问答完成。"""

    cases = _加载样例()
    expected_rag_cases = (
        "golden-v1-route-002",
        "golden-v1-route-003",
        "golden-v1-route-006",
        "golden-v1-retrieval-001",
        "golden-v1-retrieval-002",
        "golden-v1-retrieval-003",
        "golden-v1-similar-001",
        "golden-v1-similar-008",
        "golden-v1-memory-001",
        "golden-v1-memory-003",
        "golden-v1-memory-005",
        "golden-v1-fault-006",
    )
    assert {case_id for case_id in expected_rag_cases if cases[case_id]["expected_route"] == "rag"} == set(
        expected_rag_cases
    )


def test_report_generation_and_out_of_library_matching_keep_heavy_agents() -> None:
    """只有显式报告和库外新过程匹配保留两个重型智能体。"""

    cases = _加载样例()
    assert cases["golden-v1-memory-004"]["expected_route"] == "multidim_search"
    assert cases["golden-v1-fault-006"]["expected_route"] == "rag"
    assert cases["golden-v1-route-004"]["expected_route"] == "clarify"
    assert cases["golden-v1-fault-004"]["expected_route"] == "clarify"


def test_dataset_version_is_bumped_after_route_contract_change() -> None:
    """路由期望发生变化后必须使用新数据集版本，避免报告口径混用。"""

    payload = json.loads(数据路径.read_text(encoding="utf-8"))
    assert payload["dataset_version"] == "golden-v1.5"
