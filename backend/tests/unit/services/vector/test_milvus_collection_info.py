# -*- coding: utf-8 -*-
"""Milvus 文档适配器 collection_info 行数统计单元测试。"""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


def _load_module(module_name: str):
    """导入待实现模块，并把缺失实现明确报告为测试失败。"""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        pytest.fail(f"待实现模块缺失：{module_name}：{exc}")


def _build_store(module) -> "object":
    """构造不连接真实 Milvus 的文档适配器实例。"""
    return module.MilvusDocumentStore(
        uri="http://127.0.0.1:19530",
        collection_name="weather_document_chunks_v1",
        authority_path="unused",
        dimension=4,
    )


def test_collection_info_count_uses_count_star_query() -> None:
    """集合行数必须通过 count(*) 查询获得，而不是依赖 get_collection_stats。"""
    module = _load_module(
        "backend.app.services.vector.milvus_document_store"
    )
    store = _build_store(module)
    stats_calls: list[str] = []

    def fake_query(**kwargs: object) -> list[dict[str, object]]:
        assert kwargs.get("output_fields") == ["count(*)"]
        return [{"count(*)": 42}]

    def fake_stats(name: str) -> dict[str, int]:
        stats_calls.append(str(name))
        return {"row_count": 0}

    store._client = SimpleNamespace(
        has_collection=lambda name: True,
        query=fake_query,
        get_collection_stats=fake_stats,
    )
    info = store.collection_info()
    assert info.count == 42
    assert stats_calls == [], "不应依赖 get_collection_stats 的 row_count"
    store.close()
