"""评测数据治理规则测试。"""

from __future__ import annotations

import json
from pathlib import Path

仓库根目录 = Path(__file__).resolve().parents[4]
数据路径 = 仓库根目录 / "backend" / "evals" / "datasets" / "golden_v1.json"


def _读取样例() -> list[dict]:
    """读取全部黄金样例，供字段治理断言复用。"""

    payload = json.loads(数据路径.read_text(encoding="utf-8"))
    return payload["cases"]


def test_required_facts_do_not_store_behavior_descriptions() -> None:
    """required_facts 只能保存可逐字核验事实，不能继续混入“应……”行为期望。"""

    invalid = {
        case["case_id"]: fact
        for case in _读取样例()
        for fact in case["required_facts"]
        if fact.strip().startswith(("应", "必须", "需要"))
    }
    assert invalid == {}


def test_required_facts_do_not_include_evaluation_scaffolding() -> None:
    """必含事实应描述业务事实，不能写成“命中的是”“结果为”等评测判语。"""

    invalid = {
        case["case_id"]: fact
        for case in _读取样例()
        for fact in case["required_facts"]
        if fact.strip().startswith(("命中的是", "命中的个例是", "结果为"))
    }
    assert invalid == {}


def test_all_cases_have_structured_expected_behaviors() -> None:
    """每条样例都必须声明 expected_behaviors，避免通过字段缺失绕过行为评测。"""

    cases = _读取样例()
    assert all("expected_behaviors" in case for case in cases)
    assert sum(bool(case["expected_behaviors"]) for case in cases) >= 20


def test_behavior_checks_use_whitelisted_kinds() -> None:
    """行为检查只能使用 Harness 能确定性执行的检查类型。"""

    allowed = {
        "route_equals",
        "answer_contains_any",
        "answer_contains_all",
        "evidence_hit",
        "no_result",
        "abstained",
    }
    for case in _读取样例():
        for behavior in case["expected_behaviors"]:
            assert behavior["description"]
            assert behavior["checks"]
            assert {check["kind"] for check in behavior["checks"]} <= allowed
