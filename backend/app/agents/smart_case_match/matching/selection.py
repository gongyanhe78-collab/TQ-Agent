"""融合结果的 LLM 微调、质量门槛和多样性选择。"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .dimension_profiles import (
    calculate_dimension_compatibility,
    effective_dimension_compatibility,
    legacy_dimension_scores,
)


# 平局分差阈值：两个个例的融合分差在这个值以内，才允许 LLM 微调它们的先后顺序
TIE_SCORE_GAP = 0.08
# 动态兼容度差距超过该值时不视为平局，避免模型顺序重新抬高机制或强度明显偏弱的候选。
TIE_COMPATIBILITY_GAP = 0.12
# 结构化分最低门槛：低于这个分数的个例质量太差，一般不入选
MIN_STRUCTURED_SCORE = 0.30


def apply_llm_tie_break(
    ranked: list[dict[str, Any]],
    ordered_case_ids: list[str],
    assessment_map: dict[str, dict[str, Any]],
    dimension_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """先按当前灾种的动态指标降权，再在近分候选组内采用 LLM 顺序。

    设计思路：LLM 只负责"打破平局"——分差大的（明显不同层级）不允许 LLM 推翻，
    只有分差接近的（同一梯队内）才让 LLM 微调顺序，保证：
    - 结构化+语义的硬分数是排序基础（不会被 LLM 完全打乱）
    - LLM 只在相近分数的个例之间做精细化排序
    动态指标评估只允许降低原融合分，避免模型用模糊语义抬高明显不一致的个例。
    """
    adjusted = []
    for item in ranked:
        case_id = str(item.get("case_id") or "")
        assessment = assessment_map.get(case_id) or {}
        raw_compatibility, _ = calculate_dimension_compatibility(assessment, dimension_profile)
        compatibility, used_scores, coverage = effective_dimension_compatibility(assessment, dimension_profile)
        updated = dict(item)
        if compatibility is not None:
            # 降权幅度由灾种配置统一控制；证据覆盖不足时即使已有分很高，也不能与完整评估等价。
            profile = dimension_profile or {}
            penalty_floor = max(0.0, min(1.0, float(profile.get("penalty_floor", 0.55))))
            penalty_span = max(0.0, min(1.0 - penalty_floor, float(profile.get("penalty_span", 0.45))))
            penalty_factor = penalty_floor + penalty_span * compatibility
            updated["retrieval_score"] = round(float(item.get("retrieval_score") or 0.0) * penalty_factor, 4)
            breakdown = dict(updated.get("score_breakdown") or {})
            breakdown["llm_compatibility"] = round(compatibility, 4)
            breakdown["llm_raw_compatibility"] = round(float(raw_compatibility or 0.0), 4)
            breakdown["dimension_coverage"] = round(coverage, 4)
            for key, value in used_scores.items():
                breakdown[f"llm_{key}"] = round(value, 4)
                # 保留旧的 score_breakdown 键，避免已有页面和测试因为协议升级失效。
                if key in {"mechanism", "intensity"}:
                    breakdown[key] = round(value, 4)
            updated["score_breakdown"] = breakdown
            updated["dimension_scores"] = dict(used_scores)
            updated["metric_scores"] = dict(assessment.get("metric_scores") or {})
            updated["missing_metrics"] = list(assessment.get("missing_metrics") or [])
            updated["missing_dimensions"] = list(assessment.get("missing_dimensions") or [])
            updated["dimension_compatibility"] = round(compatibility, 4)
            updated["dimension_raw_compatibility"] = round(float(raw_compatibility or 0.0), 4)
            updated["dimension_coverage"] = round(coverage, 4)
            updated["dimension_assessment_source"] = str(assessment.get("assessment_source") or "llm")
            mechanism, intensity = legacy_dimension_scores(used_scores)
            # 旧字段继续透传，旧页面和现有调用方无需同步升级。
            updated["mechanism_score"] = round(mechanism, 4) if mechanism is not None else None
            updated["intensity_score"] = round(intensity, 4) if intensity is not None else None
            updated["candidate_assessment_available"] = True
        elif assessment_map:
            # 同批已有有效评估时，完全缺少动态证据的候选不能保留未经约束的原始高分。
            profile = dimension_profile or {}
            penalty_floor = max(0.0, min(1.0, float(profile.get("penalty_floor", 0.55))))
            updated["retrieval_score"] = round(float(item.get("retrieval_score") or 0.0) * penalty_floor, 4)
            breakdown = dict(updated.get("score_breakdown") or {})
            breakdown["dimension_coverage"] = 0.0
            updated["score_breakdown"] = breakdown
            updated["dimension_coverage"] = 0.0
            updated["candidate_assessment_available"] = False
        adjusted.append(updated)

    # 机制/强度扣分后重新排序，避免原始 93% 仍然压住真正更接近的个例。
    ranked = sorted(adjusted, key=lambda item: (item["retrieval_score"], item.get("case_id", "")), reverse=True)
    # LLM 排好序的 ID → 位置索引映射（用于查表决定谁在前）
    llm_position = {case_id: index for index, case_id in enumerate(ordered_case_ids)}

    # 第一步：按融合分差把候选分组成"连续相近分数组"（同一个梯队）
    # 相邻两个的分差超过 TIE_SCORE_GAP 就算新的一组
    groups: list[list[dict[str, Any]]] = []
    for item in ranked:
        # 没有组，或与组内第一名的分差超过阈值 → 新建一组
        group_head = groups[-1][0] if groups else None
        compatibility_gap = (
            abs(float(group_head.get("dimension_compatibility")) - float(item.get("dimension_compatibility")))
            if group_head is not None
            and group_head.get("dimension_compatibility") is not None
            and item.get("dimension_compatibility") is not None
            else 0.0
        )
        if (
            not groups
            or groups[-1][0]["retrieval_score"] - item["retrieval_score"] > TIE_SCORE_GAP
            or compatibility_gap > TIE_COMPATIBILITY_GAP
        ):
            groups.append([item])
        # 分差在阈值内 → 加入当前组（视为同一梯队）
        else:
            groups[-1].append(item)

    # 第二步：每个组内部按 LLM 给出的顺序重新排列
    reordered = []
    for group in groups:
        # 按 LLM 排序位置从小到大排；LLM 没提到的个例放最后（位置 = 列表总长）
        group.sort(key=lambda item: llm_position.get(str(item.get("case_id")), len(ranked)))
        reordered.extend(group)

    return reordered


def select_cases(
    ranked: list[dict[str, Any]],
    requested_count: int,
    diversity_mode: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """按最低业务相关性选取 1 至 5 个结果，低质量时不强行补足。

    选择流程：
    1. 质量门槛：结构化分 >= 0.30 的才算合格个例
    2. 极差兜底：一个合格的都没有但第一名 >= 0.15，至少给 1 个（总比空着好）
    3. 数量不足警告：合格的不够 3 个（或用户要的数量）就加一条警告
    4. 多样性选择：moderate 模式下做多样性选择，off 模式直接取前 N 个
    5. 恢复原序：按原排好的顺序重新排序（多样性选择可能打乱了原顺序）
    6. 附加信息：给每个入选个例补上排名和历史年份
    """
    warnings: list[str] = []

    # 第一步：筛选出达到结构化分最低门槛的合格个例
    qualified = [item for item in ranked if float(item.get("structured_score") or 0.0) >= MIN_STRUCTURED_SCORE]

    # 第二步：一个合格的都没有，但第一名分数还过得去（>=0.15）→ 至少给 1 个，避免完全空结果
    if not qualified and ranked and float(ranked[0].get("structured_score") or 0.0) >= 0.15:
        qualified = ranked[:1]

    # 第三步：合格的不够多 → 加警告告诉用户"没凑够数，是因为质量不够，不是系统bug"
    if len(qualified) < min(3, requested_count):
        warnings.append(f"仅找到 {len(qualified)} 个达到质量门槛的历史个例，未使用低相关个例补足数量。")

    # 第四步：多样性模式
    # - moderate：调用 _diversified 做多样性选择（同一来源不超过 2 个）
    # - 其他（off）：直接取前 N 个，不做多样性处理
    selected = _diversified(qualified, requested_count) if diversity_mode == "moderate" else qualified[:requested_count]

    # 第五步：按原始排序的位置恢复顺序。
    # 使用稳定 case_id 映射而非 dict 值比较，避免后续节点复制或补充字段后 list.index() 找错位置。
    rank_positions = {str(item.get("case_id") or ""): index for index, item in enumerate(ranked)}
    selected.sort(key=lambda item: rank_positions.get(str(item.get("case_id") or ""), len(ranked)))

    # 第六步：给每个入选个例补上排名和历史年份，然后返回
    return [
        {
            **item,
            "rank": index + 1,                          # 最终排名（从 1 开始）
            "historical_year": _year_from_case(item),   # 个例发生的年份（前端展示用）
        }
        for index, item in enumerate(selected)
    ], warnings


def _diversified(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """相近分数下限制同一来源最多两个，候选不足时再按原序回填。

    多样性策略：同一个 PDF 来源的个例最多选 2 个。
    防止返回的 topN 全是同一份报告里的高度相似个例，结果冗余。

    算法：
    1. 按原排序遍历，同一来源没超过 2 个就入选，超过就先放到 deferred 队列
    2. 如果选够了 limit 个就返回
    3. 没选够的话，再从 deferred 队列里按原顺序回填，直到凑够 limit 个
    """
    selected: list[dict[str, Any]] = []     # 最终入选列表
    deferred: list[dict[str, Any]] = []     # 暂时跳过的（同来源超了 2 个的）候补队列
    sources: Counter[str] = Counter()       # 各来源已入选数量统计

    # 第一轮：原顺序选，同一来源最多 2 个，超了就先放候补
    for item in candidates:
        source = str(item.get("source_pdf") or "unknown")
        if sources[source] >= 2:
            # 这个来源已经有 2 个入选了，先放一放
            deferred.append(item)
            continue
        selected.append(item)
        sources[source] += 1
        if len(selected) >= limit:
            return selected

    # 第二轮：如果还没凑够 limit 个，从候补队列里按原顺序回填
    for item in deferred:
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _year_from_case(case: dict[str, Any]) -> int | None:
    """从日期或来源文件提取年份，供同分时解释新旧程度。

    从 date_range 和 source_pdf 中用正则找 20xx 格式的年份，
    找不到就返回 None。
    """
    text = " ".join([str(case.get("date_range") or ""), str(case.get("source_pdf") or "")])
    match = re.search(r"20\d{2}", text)
    return int(match.group(0)) if match else None
