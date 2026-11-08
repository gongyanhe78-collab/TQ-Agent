"""CORS 白名单配置的单元测试：默认拒绝通配来源。"""
from __future__ import annotations

from backend.app.main import _cors_origins


def test_cors_defaults_to_trusted_frontend_origins(monkeypatch) -> None:
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    origins = _cors_origins()
    assert "*" not in origins
    assert "http://127.0.0.1:5173" in origins
    assert "http://localhost:5173" in origins


def test_cors_ignores_wildcard_and_keeps_explicit_origins(monkeypatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "*,http://example.com")
    origins = _cors_origins()
    assert "*" not in origins
    assert origins == ["http://example.com"]


def test_cors_empty_config_falls_back_to_whitelist(monkeypatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "")
    origins = _cors_origins()
    assert "*" not in origins
    assert origins
