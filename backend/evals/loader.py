"""黄金数据集加载、完整性校验与权威引用检查。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from backend.evals.schemas.dataset import EvaluationCase, EvaluationManifest


固定样例数量 = 60
固定类别分布 = {
    "路由": 6,
    "检索": 12,
    "统计": 8,
    "RAG": 10,
    "相似匹配": 8,
    "多轮记忆": 10,
    "故障场景": 6,
}


class GoldenDataset:
    """同时保存评测样例、清单元数据和权威引用集合。"""

    def __init__(
        self,
        *,
        path: Path,
        manifest: EvaluationManifest,
        cases: list[EvaluationCase],
        authority_case_ids: set[str],
        authority_chunk_ids: set[str],
        dataset_sha256: str,
    ) -> None:
        self.path = path
        self.manifest = manifest
        self.cases = cases
        self._authority_case_ids = authority_case_ids
        self._authority_chunk_ids = authority_chunk_ids
        self._dataset_sha256 = dataset_sha256

    @classmethod
    def load(
        cls,
        path: Path,
        cases_path: Path,
        chunks_path: Path,
    ) -> GoldenDataset:
        """加载数据集并校验哈希、唯一性、分布和版本元数据。"""

        dataset_path = Path(path)
        manifest_path = dataset_path.with_name(f"{dataset_path.stem}_manifest.json")
        dataset_bytes = dataset_path.read_bytes()
        actual_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
        manifest_payload = _读取JSON(manifest_path)
        manifest = EvaluationManifest.model_validate(manifest_payload)
        if manifest.dataset_sha256 != actual_sha256:
            raise ValueError(
                "黄金数据集 SHA-256 与清单不一致："
                f"期望 {manifest.dataset_sha256}，实际 {actual_sha256}"
            )

        dataset_payload = json.loads(dataset_bytes.decode("utf-8"))
        if not isinstance(dataset_payload, dict):
            raise ValueError("黄金数据集顶层必须是 JSON 对象")
        if dataset_payload.get("dataset_version") != manifest.dataset_version:
            raise ValueError(
                "黄金数据集 dataset_version 与清单不一致："
                f"期望 {manifest.dataset_version}，"
                f"实际 {dataset_payload.get('dataset_version')}"
            )
        raw_cases = dataset_payload.get("cases")
        if not isinstance(raw_cases, list):
            raise ValueError("黄金数据集必须包含 cases 数组")
        cases = [EvaluationCase.model_validate(item) for item in raw_cases]
        if len(cases) != 固定样例数量:
            raise ValueError(
                "黄金数据集必须恰好包含 60 条样例："
                f"实际 {len(cases)} 条"
            )

        case_ids = [item.case_id for item in cases]
        if len(case_ids) != len(set(case_ids)):
            repeated = sorted(
                case_id
                for case_id, count in Counter(case_ids).items()
                if count > 1
            )
            raise ValueError(f"黄金数据集存在重复 case_id：{', '.join(repeated)}")

        questions = [item.question for item in cases]
        if len(questions) != len(set(questions)):
            repeated_questions = sorted(
                question
                for question, count in Counter(questions).items()
                if count > 1
            )
            raise ValueError(
                "黄金数据集存在重复问题：" + "；".join(repeated_questions)
            )

        actual_distribution = dict(Counter(item.category for item in cases))
        if actual_distribution != 固定类别分布:
            raise ValueError(
                "黄金数据集类别分布必须符合固定类别分布："
                f"期望 {固定类别分布}，实际 {actual_distribution}"
            )
        expected_distribution = {
            str(category): count
            for category, count in manifest.category_distribution.items()
        }
        if actual_distribution != expected_distribution:
            raise ValueError(
                "黄金数据集类别分布与清单不一致："
                f"期望 {expected_distribution}，实际 {actual_distribution}"
            )

        for item in cases:
            if item.knowledge_version != manifest.knowledge_version:
                raise ValueError(
                    f"样例 {item.case_id} 的 knowledge_version 与清单不一致"
                )
            if item.prompt_version != manifest.prompt_version:
                raise ValueError(
                    f"样例 {item.case_id} 的 prompt_version 与清单不一致"
                )
            if item.model_config != manifest.model_config:
                raise ValueError(f"样例 {item.case_id} 的 model_config 与清单不一致")

        authority_case_ids = _读取权威ID(cases_path, "case_id")
        authority_chunk_ids = _读取权威ID(chunks_path, "chunk_id")
        return cls(
            path=dataset_path,
            manifest=manifest,
            cases=cases,
            authority_case_ids=authority_case_ids,
            authority_chunk_ids=authority_chunk_ids,
            dataset_sha256=actual_sha256,
        )

    def reference_errors(self) -> list[str]:
        """返回全部不存在的个例和文档切片引用，顺序稳定。"""

        errors: list[str] = []
        for item in self.cases:
            for case_id in item.expected_case_ids:
                if case_id not in self._authority_case_ids:
                    errors.append(
                        f"样例 {item.case_id} 引用了不存在的标准个例：{case_id}"
                    )
            for chunk_id in item.expected_chunk_ids:
                if chunk_id not in self._authority_chunk_ids:
                    errors.append(
                        f"样例 {item.case_id} 引用了不存在的文档切片：{chunk_id}"
                    )
        return errors

    def dataset_sha256(self) -> str:
        """返回加载时按数据集文件字节计算的 SHA-256。"""

        return self._dataset_sha256


def _读取JSON(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON 对象。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return payload


def _读取权威ID(path: Path, id_field: str) -> set[str]:
    """从权威 JSON 数组提取字符串主键集合。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"权威数据顶层必须是数组：{path}")
    identifiers: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"权威数据元素必须是对象：{path}")
        identifier = item.get(id_field)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"权威数据缺少有效字段 {id_field}：{path}")
        identifiers.add(identifier)
    return identifiers
