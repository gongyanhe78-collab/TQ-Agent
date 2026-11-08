"""动态证据窗口与多案例证据多样化工具。"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import TypeVar

T = TypeVar("T")

# 这些表达通常要求覆盖多个过程或多个切片，不能固定只取三五条证据。
_多证据表达 = (
    "全部",
    "所有",
    "分别",
    "哪些",
    "对比",
    "比较",
    "统计",
    "列出",
    "清单",
    "汇总",
    "几次",
    "几个",
    "多个",
    "两段",
    "两次",
    "三次",
)


def dynamic_evidence_window(
    question: str,
    requested: int = 3,
    matched_case_count: int = 0,
) -> int:
    """根据问题语义和已有个例数量计算证据窗口，范围为 1~10。"""

    requested_window = min(max(int(requested), 1), 10)
    target = requested_window
    if int(matched_case_count) > 1:
        target = max(target, 5, int(matched_case_count) * 2)
    if any(term in str(question or "") for term in _多证据表达):
        target = max(target, 8)
    return min(target, 10)


def diversify_evidence_by_case(
    hits: Sequence[T],
    chunk_to_case: Mapping[str, str],
    *,
    limit: int,
) -> list[T]:
    """按个例轮转选择证据，先覆盖不同个例，再补充同个例的后续切片。"""

    if limit <= 0:
        return []
    groups: OrderedDict[str, list[T]] = OrderedDict()
    for hit in hits:
        chunk_id = str(getattr(getattr(hit, "chunk", None), "chunk_id", "") or "")
        source_pdf = str(getattr(getattr(hit, "chunk", None), "source_pdf", "") or "")
        group_key = chunk_to_case.get(chunk_id) or source_pdf or chunk_id
        groups.setdefault(group_key, []).append(hit)

    selected: list[T] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for group in groups.values():
            if offset < len(group):
                selected.append(group[offset])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        offset += 1
    return selected
