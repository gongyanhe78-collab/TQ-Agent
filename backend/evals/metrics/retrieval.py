"""路由与检索类确定性指标。"""

from __future__ import annotations

from collections.abc import Iterable


def _稳定去重(items: Iterable[str]) -> list[str]:
    """按首次出现顺序去重，保证排名不依赖无序集合迭代顺序。"""

    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _空证据修正(actual: list[str], expected: list[str]) -> float | None:
    """处理空证据集合的边界语义；无边界时返回 None。"""

    has_actual = bool(actual)
    has_expected = bool(expected)
    if not has_expected:
        return 1.0 if not has_actual else 0.0
    if not has_actual:
        return 0.0
    return None


def route_accuracy(actual_routes: list[str], expected_routes: list[str]) -> float:
    """计算预测路由与期望路由完全一致的比例。"""

    if not expected_routes:
        return 0.0
    matched = sum(
        1
        for actual, expected in zip(actual_routes, expected_routes)
        if actual == expected
    )
    return matched / len(expected_routes)


def hit_at_k(ranked_ids: list[str], expected_ids: list[str], k: int) -> float:
    """判断稳定排序前 k 项是否命中任一期望证据 ID。"""

    ranked = _稳定去重(ranked_ids)
    expected = set(expected_ids)
    edge = _空证据修正(ranked, list(expected))
    if edge is not None:
        return edge
    return 1.0 if any(item in expected for item in ranked[:k]) else 0.0


def recall_at_k(ranked_ids: list[str], expected_ids: list[str], k: int) -> float:
    """计算稳定排序前 k 项覆盖期望证据的比例。"""

    ranked = _稳定去重(ranked_ids)
    expected = set(expected_ids)
    edge = _空证据修正(ranked, list(expected))
    if edge is not None:
        return edge
    return len(expected & set(ranked[:k])) / len(expected)


def mrr(ranked_ids: list[str], expected_ids: list[str]) -> float:
    """计算首个命中期望证据的倒数排名。"""

    ranked = _稳定去重(ranked_ids)
    expected = set(expected_ids)
    edge = _空证据修正(ranked, list(expected))
    if edge is not None:
        return edge
    for index, item in enumerate(ranked, start=1):
        if item in expected:
            return 1.0 / index
    return 0.0


def evidence_id_hit_rate(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """判断实际证据 ID 是否与期望集合产生交集。"""

    retrieved = _稳定去重(retrieved_ids)
    expected = set(expected_ids)
    edge = _空证据修正(retrieved, list(expected))
    if edge is not None:
        return edge
    return 1.0 if any(item in expected for item in retrieved) else 0.0


def no_result_accuracy(predicted_no_result: bool, expected_no_result: bool) -> float:
    """判断无结果预测与期望是否一致。"""

    return 1.0 if bool(predicted_no_result) == bool(expected_no_result) else 0.0
