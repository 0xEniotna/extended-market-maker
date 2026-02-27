from __future__ import annotations

import importlib.util
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests


def _load_module():
    path = Path("scripts/tools/fetch_rebate_requirements.py")
    spec = importlib.util.spec_from_file_location("fetch_rebate_requirements_mod", path)
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


def test_parse_market_filters_normalizes_and_deduplicates():
    mod = _load_module()
    requested = mod._parse_market_filters(
        " eth-usd,MON-USD, ",
        ["mon-usd", "XPT-USD", "ETH-USD"],
    )
    assert requested == ["ETH-USD", "MON-USD", "XPT-USD"]


def test_parse_retry_after_seconds():
    mod = _load_module()
    assert mod._parse_retry_after_seconds("2.5") == 2.5
    assert mod._parse_retry_after_seconds("bad") is None
    assert mod._parse_retry_after_seconds(None) is None


def test_build_market_tier_requirements_from_market_volume():
    mod = _load_module()
    tiers = mod._build_market_tier_requirements(
        market_30d_volume_usd=Decimal("300000"),
        days=Decimal("30"),
    )
    assert tiers[0]["tier"] == "0.5%"
    assert tiers[0]["required_30d_maker_volume_usd"] == Decimal("1500")
    assert tiers[1]["required_30d_maker_volume_usd"] == Decimal("3000")
    assert tiers[3]["required_30d_maker_volume_usd"] == Decimal("15000")
    assert tiers[0]["required_avg_daily_maker_volume_usd"] == Decimal("50")


def test_build_market_result_uses_daily_volume():
    mod = _load_module()
    result = mod._build_market_result(
        market="ETH-USD",
        daily_volume_usd=Decimal("10000"),
        days=Decimal("30"),
    )
    assert result["market"] == "ETH-USD"
    assert result["daily_volume_usd"] == Decimal("10000")
    assert result["estimated_30d_volume_usd"] == Decimal("300000")
    assert result["tiers"][0]["required_30d_maker_volume_usd"] == Decimal("1500")


def test_fetch_markets_with_retry_retries_429_and_recovers():
    mod = _load_module()
    client = MagicMock()
    response_429 = MagicMock()
    response_429.status_code = 429
    response_429.headers = {"Retry-After": "0"}
    error_429 = requests.exceptions.HTTPError(response=response_429)
    client.fetch_all_markets.side_effect = [error_429, [{"name": "ETH-USD"}]]

    with patch.object(mod.time, "sleep", return_value=None) as sleep_mock:
        rows = mod._fetch_markets_with_retry(
            client,
            max_retries=2,
            initial_backoff_s=1.0,
            max_backoff_s=4.0,
        )

    assert rows == [{"name": "ETH-USD"}]
    assert client.fetch_all_markets.call_count == 2
    sleep_mock.assert_called_once()


def test_request_json_with_retry_retries_429_and_recovers():
    mod = _load_module()

    rate_limited = MagicMock()
    rate_limited.status_code = 429
    rate_limited.headers = {"Retry-After": "0"}
    rate_limited.raise_for_status.side_effect = requests.exceptions.HTTPError(response=rate_limited)

    ok = MagicMock()
    ok.status_code = 200
    ok.headers = {}
    ok.raise_for_status.return_value = None
    ok.json.return_value = {"status": "ok", "data": {"dailyVolume": "123"}}

    with patch.object(mod.requests, "get", side_effect=[rate_limited, ok]) as get_mock:
        with patch.object(mod.time, "sleep", return_value=None) as sleep_mock:
            payload = mod._request_json_with_retry(
                url="https://example/api/v1/info/markets/ETH-USD/stats",
                max_retries=2,
                initial_backoff_s=1.0,
                max_backoff_s=4.0,
            )

    assert payload["status"] == "ok"
    assert get_mock.call_count == 2
    sleep_mock.assert_called_once()


def test_fetch_market_stats_parses_data():
    mod = _load_module()
    with patch.object(
        mod,
        "_request_json_with_retry",
        return_value={"status": "ok", "data": {"dailyVolume": "456"}},
    ):
        data = mod._fetch_market_stats(
            api_base="https://example/api/v1",
            market="eth-usd",
            max_retries=1,
            initial_backoff_s=1.0,
            max_backoff_s=2.0,
        )
    assert data["dailyVolume"] == "456"
