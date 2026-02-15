from __future__ import annotations

from unittest.mock import MagicMock, patch

from market_maker.public_markets import (
    MAINNET_API_BASE,
    TESTNET_API_BASE,
    PublicMarketsClient,
    resolve_default_api_base,
)


def test_resolve_api_base_prefers_explicit_env():
    with patch.dict("os.environ", {"EXTENDED_API_BASE": "https://custom.example/api/v1"}, clear=True):
        assert resolve_default_api_base() == "https://custom.example/api/v1"


def test_resolve_api_base_uses_mainnet_from_mm_environment():
    with patch.dict("os.environ", {"MM_ENVIRONMENT": "mainnet"}, clear=True):
        assert resolve_default_api_base() == MAINNET_API_BASE


def test_resolve_api_base_defaults_to_testnet():
    with patch.dict("os.environ", {}, clear=True):
        assert resolve_default_api_base() == TESTNET_API_BASE


def test_public_markets_client_parses_ok_payload():
    payload = {
        "status": "ok",
        "data": [{"name": "ETH-USD", "active": True, "status": "ACTIVE"}],
    }

    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None

    with patch("market_maker.public_markets.requests.get", return_value=resp):
        client = PublicMarketsClient(api_base="https://example/api/v1")
        markets = client.fetch_all_markets()
        assert len(markets) == 1
        assert markets[0]["name"] == "ETH-USD"
