# -*- coding: utf-8 -*-
"""Milvus 文档适配器 list_chunk_ids 分页单元测试。"""
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


def test_list_chunk_ids_paginates_within_milvus_result_window() -> None:
    """列出 ID 必须分页查询，单次 limit 不得超过 Milvus 16384 结果窗口。"""
    module = _load_module(
        "backend.app.services.vector.milvus_document_store"
    )
    store = _build_store(module)
    calls: list[dict] = []
    total_rows = 2500

    def fake_query(**kwargs: object) -> list[dict[str, str]]:
        calls.append(kwargs)
        limit = int(kwargs.get("limit") or 0)
        offset = int(kwargs.get("offset") or 0)
        page_size = min(limit, total_rows - offset)
        return [
            {"chunk_id": f"chunk-{index:04d}"}
            for index in range(offset, offset + page_size)
        ]

    store._client = SimpleNamespace(
        has_collection=lambda name: True,
        query=fake_query,
    )
    chunk_ids = store.list_chunk_ids()
    assert chunk_ids == [f"chunk-{index:04d}" for index in range(total_rows)]
    assert len(calls) >= 2, "必须发生多次查询以完成分页"
    for call in calls:
        offset = int(call.get("offset") or 0)
        limit = int(call.get("limit") or 0)
        assert limit <= 16384, "单次 limit 不得超过 Milvus 结果窗口"
        assert offset + limit <= 16384, "offset+limit 必须落在 Milvus 结果窗口内"
    store.close()
    store.close()


def test_list_chunk_ids_handles_empty_collection() -> None:
    """空集合应返回空列表且只发生一次查询。"""
    module = _load_module(
        "backend.app.services.vector.milvus_document_store"
    )
    store = _build_store(module)
    calls: list[dict] = []

    def fake_query(**kwargs: object) -> list[dict[str, str]]:
        calls.append(kwargs)
        return []

    store._client = SimpleNamespace(
        has_collection=lambda name: True,
        query=fake_query,
    )
    assert store.list_chunk_ids() == []
    assert len(calls) == 1
    store.close()
