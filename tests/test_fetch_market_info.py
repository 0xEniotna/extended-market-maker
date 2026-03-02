from __future__ import annotations

from unittest.mock import patch

from market_maker.cli.markets import _resolve_api_base
from market_maker.public_markets import MAINNET_API_BASE, TESTNET_API_BASE


def test_resolve_api_base_prefers_cli_override():
    assert _resolve_api_base("https://override.example/api/v1") == "https://override.example/api/v1"


def test_resolve_api_base_prefers_explicit_env_variable():
    with patch.dict("os.environ", {"EXTENDED_API_BASE": "https://explicit.example/api/v1"}, clear=True):
        assert _resolve_api_base(None) == "https://explicit.example/api/v1"


def test_resolve_api_base_uses_environment_fallbacks():
    with patch.dict("os.environ", {"MM_ENVIRONMENT": "mainnet"}, clear=True):
        assert _resolve_api_base(None) == MAINNET_API_BASE
    with patch.dict("os.environ", {"EXTENDED_ENV": "testnet"}, clear=True):
        assert _resolve_api_base(None) == TESTNET_API_BASE
