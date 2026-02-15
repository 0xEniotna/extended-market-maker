from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch


def _load_module():
    path = Path("scripts/tools/fetch_market_info.py")
    spec = importlib.util.spec_from_file_location("fetch_market_info_mod", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_api_base_prefers_cli_override():
    mod = _load_module()
    assert mod._resolve_api_base("https://override.example/api/v1") == "https://override.example/api/v1"


def test_resolve_api_base_prefers_explicit_env_variable():
    mod = _load_module()
    with patch.dict("os.environ", {"EXTENDED_API_BASE": "https://explicit.example/api/v1"}, clear=True):
        assert mod._resolve_api_base(None) == "https://explicit.example/api/v1"


def test_resolve_api_base_uses_environment_fallbacks():
    mod = _load_module()
    with patch.dict("os.environ", {"MM_ENVIRONMENT": "mainnet"}, clear=True):
        assert mod._resolve_api_base(None) == mod.MAINNET_API_BASE
    with patch.dict("os.environ", {"EXTENDED_ENV": "testnet"}, clear=True):
        assert mod._resolve_api_base(None) == mod.TESTNET_API_BASE
