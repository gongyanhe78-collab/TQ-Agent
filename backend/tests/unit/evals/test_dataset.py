"""黄金数据集、清单与权威引用校验测试。"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.evals.loader import GoldenDataset
from backend.evals.schemas.dataset import EvaluationCase


仓库根目录 = Path(__file__).resolve().parents[4]
黄金数据集路径 = 仓库根目录 / "backend" / "evals" / "datasets" / "golden_v1.json"
黄金清单路径 = 黄金数据集路径.with_name("golden_v1_manifest.json")
标准个例路径 = 仓库根目录 / "data" / "standard_cases1.json"
文档切片路径 = 仓库根目录 / "data" / "document_index" / "chunks.json"

期望类别分布 = {
    "路由": 6,
    "检索": 12,
    "统计": 8,
    "RAG": 10,
    "相似匹配": 8,
    "多轮记忆": 10,
    "故障场景": 6,
}

必需字段 = {
    "case_id",
    "category",
    "question",
    "conversation_setup",
    "expected_route",
    "expected_case_ids",
    "expected_chunk_ids",
    "required_facts",
    "forbidden_claims",
    "max_latency_ms",
    "tags",
    "knowledge_version",
    "prompt_version",
    "model_config",
}


def 加载黄金数据集() -> GoldenDataset:
    """使用仓库权威数据加载黄金数据集。"""

    return GoldenDataset.load(黄金数据集路径, 标准个例路径, 文档切片路径)


def 读取清单() -> dict:
    """读取黄金数据集清单原始对象。"""

    return json.loads(黄金清单路径.read_text(encoding="utf-8"))


def 获取样例(case_id: str) -> EvaluationCase:
    """按 case_id 获取黄金数据集中的唯一评测样例。"""

    dataset = 加载黄金数据集()
    return next(item for item in dataset.cases if item.case_id == case_id)


def 写入数据与清单(
    dataset_path: Path,
    manifest_path: Path,
    dataset_payload: dict,
    manifest_payload: dict,
) -> None:
    """写出修改后的数据集，并按新文件字节同步更新清单哈希。"""

    dataset_bytes = json.dumps(
        dataset_payload,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")
    dataset_path.write_bytes(dataset_bytes)
    manifest_payload["dataset_sha256"] = hashlib.sha256(dataset_bytes).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_dataset_has_exactly_sixty_unique_cases_and_valid_authority_ids() -> None:
    """黄金数据集应包含 60 个唯一且引用有效的样例。"""

    dataset = 加载黄金数据集()
    assert len(dataset.cases) == 60
    assert len({item.case_id for item in dataset.cases}) == 60
    assert dataset.reference_errors() == []


def test_questions_are_unique_and_every_case_has_all_required_fields() -> None:
    """问题不得重复，且每条样例必须覆盖全部评测字段。"""

    dataset = 加载黄金数据集()
    assert len({item.question for item in dataset.cases}) == 60

    for item in dataset.cases:
        序列化结果 = item.model_dump(mode="json")
        assert 必需字段 <= set(序列化结果)
        assert "runtime_model_config" not in 序列化结果


def test_category_distribution_matches_fixed_contract() -> None:
    """类别分布必须与任务规定的固定数量完全一致。"""

    dataset = 加载黄金数据集()
    assert dict(Counter(item.category for item in dataset.cases)) == 期望类别分布
    assert 读取清单()["category_distribution"] == 期望类别分布


def test_manifest_and_dataset_metadata_are_consistent() -> None:
    """清单元数据应覆盖版本、日期、知识、Prompt 与模型配置。"""

    dataset = 加载黄金数据集()
    manifest = 读取清单()

    assert manifest["dataset_version"]
    assert manifest["created_at"]
    assert manifest["knowledge_version"]
    assert manifest["prompt_version"]
    assert isinstance(manifest["model_config"], dict)
    assert manifest["model_config"]

    for item in dataset.cases:
        assert item.knowledge_version == manifest["knowledge_version"]
        assert item.prompt_version == manifest["prompt_version"]
        assert item.model_config == manifest["model_config"]


def test_manifest_sha256_matches_dataset_file_bytes() -> None:
    """清单与加载器的 SHA-256 都应来自数据集文件原始字节。"""

    dataset = 加载黄金数据集()
    expected_hash = hashlib.sha256(黄金数据集路径.read_bytes()).hexdigest()

    assert 读取清单()["dataset_sha256"] == expected_hash
    assert dataset.dataset_sha256() == expected_hash
    assert len(dataset.dataset_sha256()) == 64
    int(dataset.dataset_sha256(), 16)


def test_may_hail_query_references_authoritative_hail_case_and_chunks() -> None:
    """五月冰雹问题应命中包含冰雹事实的权威个例和原文切片。"""

    case = 获取样例("golden-v1-retrieval-011")

    assert case.expected_case_ids == ["FST2025-5-std-case-002"]
    assert case.expected_chunk_ids == [
        "FST2025-5-chunk-007",
        "FST2025-5-chunk-008",
    ]
    required_facts = "；".join(case.required_facts)
    assert "中阳、方山、蒲县" in required_facts
    assert "冰雹" in required_facts


def test_memory_003_preserves_wind_condition_for_all_april_wind_cases() -> None:
    """四月大风日期修改应覆盖权威数据中的第三次大风过程。"""

    case = 获取样例("golden-v1-memory-003")

    assert case.expected_case_ids == [
        "FST2025-4-std-case-001",
        "FST2025-4-std-case-002",
        "FST2025-4-std-case-003",
    ]
    assert case.expected_chunk_ids == [
        "FST2025-4-chunk-003",
        "FST2025-4-chunk-023",
        "FST2025-4-chunk-027",
    ]
    assert "4月20-21日" in "；".join(case.required_facts)


def test_memory_007_preserves_station_speed_and_time_facts() -> None:
    """四月极端大风回答必须保留两个口径的站点、风速和发生时间。"""

    case = 获取样例("golden-v1-memory-007")
    required_facts = "；".join(case.required_facts)

    for expected_fact in (
        "五台山国家基本气象站[53588]",
        "39米/秒",
        "2025年4月13日07:41",
        "神池义井[B5511]",
        "39.7米/秒",
        "2025年4月11日17:05",
    ):
        assert expected_fact in required_facts


def test_similar_008_expects_a_different_non_cold_wave_candidate() -> None:
    """相似匹配应返回候选中的其他个例，不能把原过程当作匹配结果。"""

    case = 获取样例("golden-v1-similar-008")

    assert "候选" in case.question
    assert case.expected_case_ids == ["FST2025-3-std-case-001"]
    assert case.expected_chunk_ids == ["FST2025-3-chunk-002"]


def test_fault_002_routes_existing_evidence_summary_to_rag() -> None:
    """已有证据摘要请求没有相似匹配意图，应使用 RAG 路由。"""

    case = 获取样例("golden-v1-fault-002")

    assert case.expected_route == "rag"


def test_loader_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    """数据集字节变化后，旧清单哈希必须使加载失败。"""

    dataset_path = tmp_path / "golden_v1.json"
    manifest_path = tmp_path / "golden_v1_manifest.json"
    shutil.copyfile(黄金数据集路径, dataset_path)
    shutil.copyfile(黄金清单路径, manifest_path)

    dataset_path.write_bytes(dataset_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="SHA-256"):
        GoldenDataset.load(dataset_path, 标准个例路径, 文档切片路径)


def test_schema_rejects_unknown_route() -> None:
    """Schema 不允许评测数据写出台账之外的路由。"""

    dataset = 加载黄金数据集()
    payload = dataset.cases[0].model_dump(mode="json")
    payload["expected_route"] = "unknown_route"

    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(payload)


def test_loader_rejects_duplicate_case_id_even_with_valid_hash(
    tmp_path: Path,
) -> None:
    """即使同步重算哈希，加载器仍应拒绝重复 case_id。"""

    dataset_path = tmp_path / "golden_v1.json"
    manifest_path = tmp_path / "golden_v1_manifest.json"
    dataset_payload = json.loads(黄金数据集路径.read_text(encoding="utf-8"))
    manifest_payload = 读取清单()
    dataset_payload["cases"][1]["case_id"] = dataset_payload["cases"][0]["case_id"]
    dataset_payload["cases"][1]["question"] = "同步修正后的唯一问题"
    写入数据与清单(dataset_path, manifest_path, dataset_payload, manifest_payload)

    with pytest.raises(ValueError, match="case_id"):
        GoldenDataset.load(dataset_path, 标准个例路径, 文档切片路径)


def test_loader_rejects_duplicate_question_even_with_valid_hash(
    tmp_path: Path,
) -> None:
    """即使同步重算哈希，加载器仍应拒绝重复问题。"""

    dataset_path = tmp_path / "golden_v1.json"
    manifest_path = tmp_path / "golden_v1_manifest.json"
    dataset_payload = json.loads(黄金数据集路径.read_text(encoding="utf-8"))
    manifest_payload = 读取清单()
    dataset_payload["cases"][1]["question"] = dataset_payload["cases"][0]["question"]
    写入数据与清单(dataset_path, manifest_path, dataset_payload, manifest_payload)

    with pytest.raises(ValueError, match="重复问题"):
        GoldenDataset.load(dataset_path, 标准个例路径, 文档切片路径)


def test_loader_rejects_dataset_version_mismatch_even_with_valid_hash(
    tmp_path: Path,
) -> None:
    """数据集顶层版本必须与清单一致，不能靠同步哈希绕过。"""

    dataset_path = tmp_path / "golden_v1.json"
    manifest_path = tmp_path / "golden_v1_manifest.json"
    dataset_payload = json.loads(黄金数据集路径.read_text(encoding="utf-8"))
    manifest_payload = 读取清单()
    dataset_payload["dataset_version"] = "golden-v2"
    写入数据与清单(dataset_path, manifest_path, dataset_payload, manifest_payload)

    with pytest.raises(ValueError, match="dataset_version"):
        GoldenDataset.load(dataset_path, 标准个例路径, 文档切片路径)


def test_loader_rejects_manifest_category_distribution_mismatch(
    tmp_path: Path,
) -> None:
    """样例实际分布固定时，清单单独改分布仍必须被拒绝。"""

    dataset_path = tmp_path / "golden_v1.json"
    manifest_path = tmp_path / "golden_v1_manifest.json"
    dataset_payload = json.loads(黄金数据集路径.read_text(encoding="utf-8"))
    manifest_payload = 读取清单()
    manifest_payload["category_distribution"]["路由"] = 7
    manifest_payload["category_distribution"]["故障场景"] = 5
    写入数据与清单(dataset_path, manifest_path, dataset_payload, manifest_payload)

    with pytest.raises(ValueError, match="类别分布"):
        GoldenDataset.load(dataset_path, 标准个例路径, 文档切片路径)


def test_loader_rejects_non_sixty_cases_even_with_valid_hash(
    tmp_path: Path,
) -> None:
    """黄金第一版固定 60 条，不能靠同步哈希和分布改写绕过。"""

    dataset_path = tmp_path / "golden_v1.json"
    manifest_path = tmp_path / "golden_v1_manifest.json"
    dataset_payload = json.loads(黄金数据集路径.read_text(encoding="utf-8"))
    manifest_payload = 读取清单()
    dataset_payload["cases"].pop()
    manifest_payload["category_distribution"]["故障场景"] = 5
    写入数据与清单(dataset_path, manifest_path, dataset_payload, manifest_payload)

    with pytest.raises(ValueError, match="60"):
        GoldenDataset.load(dataset_path, 标准个例路径, 文档切片路径)


def test_loader_rejects_category_distribution_outside_fixed_contract(
    tmp_path: Path,
) -> None:
    """即使同步重算哈希，类别分布也不能偏离固定契约。"""

    dataset_path = tmp_path / "golden_v1.json"
    manifest_path = tmp_path / "golden_v1_manifest.json"
    dataset_payload = json.loads(黄金数据集路径.read_text(encoding="utf-8"))
    manifest_payload = 读取清单()

    故障样例 = next(
        item
        for item in dataset_payload["cases"]
        if item["category"] == "故障场景"
    )
    故障样例["category"] = "路由"
    manifest_payload["category_distribution"]["路由"] = 7
    manifest_payload["category_distribution"]["故障场景"] = 5
    写入数据与清单(dataset_path, manifest_path, dataset_payload, manifest_payload)

    with pytest.raises(ValueError, match="固定类别分布"):
        GoldenDataset.load(dataset_path, 标准个例路径, 文档切片路径)


@pytest.mark.parametrize(
    ("字段名", "错误值"),
    [
        ("knowledge_version", "standard-cases-v2"),
        ("prompt_version", "enterprise-runtime-v2"),
        (
            "model_config",
            {
                "provider": "other",
                "chat_model": "other",
                "embedding_model": "other",
                "temperature": 0,
                "seed": 42,
            },
        ),
    ],
)
def test_loader_rejects_case_metadata_mismatch_even_with_valid_hash(
    tmp_path: Path,
    字段名: str,
    错误值: object,
) -> None:
    """样例知识、Prompt 或模型配置不一致时不能靠同步哈希绕过。"""

    dataset_path = tmp_path / "golden_v1.json"
    manifest_path = tmp_path / "golden_v1_manifest.json"
    dataset_payload = json.loads(黄金数据集路径.read_text(encoding="utf-8"))
    manifest_payload = 读取清单()
    dataset_payload["cases"][0][字段名] = 错误值
    写入数据与清单(dataset_path, manifest_path, dataset_payload, manifest_payload)

    with pytest.raises(ValueError, match=字段名):
        GoldenDataset.load(dataset_path, 标准个例路径, 文档切片路径)


def test_reference_errors_report_unknown_authority_ids_after_hash_recalculation(
    tmp_path: Path,
) -> None:
    """同步重算哈希后，未知个例和未知切片仍必须被引用校验发现。"""

    dataset_path = tmp_path / "golden_v1.json"
    manifest_path = tmp_path / "golden_v1_manifest.json"
    dataset_payload = json.loads(黄金数据集路径.read_text(encoding="utf-8"))
    manifest_payload = 读取清单()
    未知个例 = "不存在的标准个例"
    未知切片 = "不存在的文档切片"
    dataset_payload["cases"][0]["expected_case_ids"].append(未知个例)
    dataset_payload["cases"][0]["expected_chunk_ids"].append(未知切片)
    写入数据与清单(dataset_path, manifest_path, dataset_payload, manifest_payload)

    dataset = GoldenDataset.load(dataset_path, 标准个例路径, 文档切片路径)
    errors = dataset.reference_errors()

    assert any(未知个例 in error for error in errors)
    assert any(未知切片 in error for error in errors)


def test_schema_rejects_unknown_case_field() -> None:
    """评测样例出现未声明字段时必须拒绝。"""

    dataset = 加载黄金数据集()
    payload = dataset.cases[0].model_dump(mode="json")
    payload["未知字段"] = "不应被接受"

    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(payload)
