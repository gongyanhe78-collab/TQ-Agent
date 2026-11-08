"""Milvus 参数化个例过滤与 RRF 融合表达式单元测试。"""
from __future__ import annotations

import importlib

import pytest


def _load_module(module_name: str):
    """导入待实现模块，并把缺失实现明确报告为测试失败。"""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        pytest.fail(f"待实现模块缺失：{module_name}：{exc}")


def test_target_case_filter_is_parameterized_and_never_falls_back_to_global_scope():
    """目标个例过滤必须参数化，真实个例 ID 不得直接拼进表达式。"""
    module = _load_module(
        "backend.app.services.vector.milvus_document_store"
    )
    expression = module.build_case_filter(["FST2025-3-std-case-001"])
    assert "ARRAY_CONTAINS_ANY" in expression
    assert "FST2025-3-std-case-001" not in expression


def test_build_case_filter_handles_empty_and_single_targets() -> None:
    """空目标必须返回空表达式，单目标也必须保持参数化占位符。"""
    module = _load_module(
        "backend.app.services.vector.milvus_document_store"
    )
    assert module.build_case_filter([]) == ""
    expression = module.build_case_filter(["case-a"])
    assert "ARRAY_CONTAINS_ANY" in expression
    assert "case-a" not in expression


def test_case_filter_expr_params_carries_only_target_ids() -> None:
    """表达式参数应只携带目标个例 ID，避免任意字典注入。"""
    module = _load_module(
        "backend.app.services.vector.milvus_document_store"
    )
    params = module.build_case_filter_params(["case-a", "case-b"])
    assert params == {"target_case_ids": ["case-a", "case-b"]}


def test_milvus_document_store_uses_rrf_k60_and_dense_sparse_top_k() -> None:
    """文档适配器应暴露固定的稠密/稀疏 Top-K 与 RRF 融合参数。"""
    module = _load_module(
        "backend.app.services.vector.milvus_document_store"
    )
    store = module.MilvusDocumentStore(
        uri="http://127.0.0.1:19530",
        collection_name="weather_document_chunks_v1",
        authority_path="unused",
        dimension=4,
        dense_top_k=30,
        sparse_top_k=30,
        rrf_k=60,
    )
    assert store.dense_top_k == 30
    assert store.sparse_top_k == 30
    assert store.rrf_k == 60
    assert store.backend == "milvus"
    store.close()
    store.close()
