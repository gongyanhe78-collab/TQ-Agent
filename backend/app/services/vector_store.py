from __future__ import annotations

import json
import math
from pathlib import Path

from backend.app.models import ExtractedCase, RetrievalHit


class JsonCaseStore:
    """JSON 元数据存储和备用向量存储。"""

    def __init__(self, storage_path: Path):
        """准备 JSON 存储路径。"""
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def upsert_cases(self, cases: list[ExtractedCase]) -> None:
        """根据 case_id 插入或更新案例。"""
        existing = {case.case_id: case for case in self._load_cases()}
        for case in cases:
            existing[case.case_id] = case
        self._save_cases(list(existing.values()))

    def delete_cases(self, case_ids: list[str]) -> dict:
        """从 JSON 元数据存储中按 ID 删除案例。"""
        requested_keys = list(dict.fromkeys(case_ids))
        existing = {case.case_id: case for case in self._load_cases()}
        deleted_keys = [case_id for case_id in requested_keys if case_id in existing]
        missing_keys = [case_id for case_id in requested_keys if case_id not in existing]
        for case_id in deleted_keys:
            existing.pop(case_id, None)
        self._save_cases(list(existing.values()))
        return {
            "requested_keys": requested_keys,
            "deleted_keys": deleted_keys,
            "missing_keys": missing_keys,
            "deleted": len(deleted_keys),
        }

    def list_case_ids(self) -> list[str]:
        """列出所有存储的案例 ID。"""
        return [case.case_id for case in self._load_cases()]

    def get_case(self, case_id: str) -> ExtractedCase | None:
        """根据 ID 返回单个案例。"""
        for case in self._load_cases():
            if case.case_id == case_id:
                return case
        return None

    def list_cases(self) -> list[ExtractedCase]:
        """返回所有存储的案例。"""
        return self._load_cases()

    def query(self, embedding: list[float], top_k: int) -> list[RetrievalHit]:
        """使用余弦相似度查询案例。"""
        hits = []
        for case in self._load_cases():
            if not case.embedding:
                continue
            hits.append(
                RetrievalHit(
                    case=case,
                    score=self._cosine_similarity(embedding, case.embedding),
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def _load_cases(self) -> list[ExtractedCase]:
        """从 JSON 文件加载所有案例。"""
        if not self.storage_path.exists():
            return []
        data = json.loads(self.storage_path.read_text(encoding="utf-8"))
        return [ExtractedCase.from_dict(item) for item in data]

    def _save_cases(self, cases: list[ExtractedCase]) -> None:
        """将案例持久化到 JSON 文件。"""
        data = [case.to_dict() for case in sorted(cases, key=lambda item: item.case_id)]
        self.storage_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        """计算两个向量的余弦相似度。"""
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)


class ChromaCaseStore:
    """基于 ChromaDB 的向量存储，带有 JSON 元数据备用存储。"""

    def __init__(self, persist_dir: Path, collection_name: str):
        """准备 ChromaDB 和 JSON 备用存储路径。"""
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._json_fallback = JsonCaseStore(self.persist_dir / "cases.json")

    def _get_collection(self):
        """返回 ChromaDB 集合，当 ChromaDB 不可用时返回 None。"""
        try:
            import chromadb
        except ImportError:
            return None
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        return client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _reset_collection(self):
        """删除并重新创建 ChromaDB 集合。"""
        try:
            import chromadb
        except ImportError:
            return None
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        try:
            client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        return client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_cases(self, cases: list[ExtractedCase]) -> None:
        """在 JSON 备用存储和 ChromaDB 中插入或更新案例。"""
        collection = self._get_collection()
        self._json_fallback.upsert_cases(cases)
        if collection is None:
            return
        payload = self._build_upsert_payload(cases)
        try:
            collection.upsert(**payload)
        except Exception as exc:
            if "dimension" not in str(exc).lower():
                raise
            collection = self._reset_collection()
            if collection is None:
                return
            collection.upsert(**payload)

    def delete_cases(self, case_ids: list[str]) -> dict:
        """从 JSON 备用存储和 ChromaDB 中按 ID 删除案例。"""
        requested_keys = list(dict.fromkeys(case_ids))
        existing_keys = set(self.list_case_ids())
        deleted_keys = [case_id for case_id in requested_keys if case_id in existing_keys]
        missing_keys = [case_id for case_id in requested_keys if case_id not in existing_keys]

        fallback_result = self._json_fallback.delete_cases(requested_keys)
        collection = self._get_collection()
        if collection is not None and deleted_keys:
            collection.delete(ids=deleted_keys)

        actual_deleted = deleted_keys or fallback_result["deleted_keys"]
        return {
            "requested_keys": requested_keys,
            "deleted_keys": actual_deleted,
            "missing_keys": missing_keys,
            "deleted": len(actual_deleted),
        }

    def replace_cases(self, cases: list[ExtractedCase]) -> None:
        """替换向量存储中的所有案例。"""
        self._json_fallback.replace_cases(cases) if hasattr(self._json_fallback, "replace_cases") else None
        self._json_fallback._save_cases(cases)
        collection = self._reset_collection()
        if collection is None:
            return
        if cases:
            collection.upsert(**self._build_upsert_payload(cases))

    def collection_info(self) -> dict:
        """返回向量集合的数量、维度和存储来源。"""
        collection = self._get_collection()
        if collection is None:
            return {
                "count": len(self._json_fallback.list_case_ids()),
                "dimension": None,
                "source": "json_fallback",
            }
        count = collection.count()
        dimension = None
        if count:
            peek = collection.get(limit=1, include=["embeddings"])
            embeddings = peek.get("embeddings")
            if embeddings is not None and len(embeddings):
                dimension = len(embeddings[0])
        return {
            "count": count,
            "dimension": dimension,
            "source": "chroma",
        }

    def _build_upsert_payload(self, cases: list[ExtractedCase]) -> dict:
        """为提取的案例构建 ChromaDB upsert 负载。"""
        return {
            "ids": [case.case_id for case in cases],
            "documents": [case.content for case in cases],
            "metadatas": [
                {
                    "source_pdf": case.source_pdf,
                    "case_no": case.case_no,
                    "title": case.title,
                    "date_range": case.date_range,
                    "file_path": case.file_path or "",
                }
                for case in cases
            ],
            "embeddings": [case.embedding or [] for case in cases],
        }

    def list_case_ids(self) -> list[str]:
        """列出所有案例 ID，当 ChromaDB 有数据时优先使用。"""
        collection = self._get_collection()
        if collection is None:
            return self._json_fallback.list_case_ids()
        result = collection.get(include=[])
        ids = result.get("ids") or []
        if ids:
            return sorted(ids)
        return self._json_fallback.list_case_ids()

    def get_case(self, case_id: str) -> ExtractedCase | None:
        """从 JSON 备用存储返回完整的案例元数据。"""
        return self._json_fallback.get_case(case_id)

    def list_cases(self) -> list[ExtractedCase]:
        """从 JSON 备用存储返回所有完整的案例元数据。"""
        return self._json_fallback.list_cases()

    def query(self, embedding: list[float], top_k: int) -> list[RetrievalHit]:
        """首先查询 ChromaDB，失败时回退到 JSON 余弦相似度查询。"""
        collection = self._get_collection()
        if collection is None:
            return self._json_fallback.query(embedding, top_k)
        result = collection.query(query_embeddings=[embedding], n_results=top_k)
        hits: list[RetrievalHit] = []
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for case_id, distance in zip(ids, distances):
            case = self.get_case(case_id)
            if case is None:
                continue
            score = 1 - float(distance) if distance is not None else 0.0
            hits.append(RetrievalHit(case=case, score=score))
        if hits:
            return hits
        return self._json_fallback.query(embedding, top_k)
