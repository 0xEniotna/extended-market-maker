from __future__ import annotations

import argparse
import importlib.util
import sys
from decimal import Decimal
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_find_markets_build_json_payload_contains_sampling_and_markets():
    mod = _load_module(Path("scripts/tools/find_mm_markets.py"), "find_mm_markets_mod")
    args = argparse.Namespace(
        duration_s=120.0,
        interval_s=2.0,
        min_spread_bps=Decimal("8"),
        max_spread_bps=Decimal("50"),
        min_coverage_pct=Decimal("70"),
        max_vol_pct=Decimal("8"),
        min_daily_volume=Decimal("0"),
        sort_by="open_interest",
    )
    payload = mod._build_json_payload(
        markets=[
            {
                "name": "LIT-USD",
                "asset_name": "LIT",
                "_samples": 10,
                "_coverage_pct": Decimal("90"),
                "_spread_median_bps": Decimal("9"),
                "_spread_p90_bps": Decimal("12"),
                "_spread_mean_bps": Decimal("10"),
                "_vol_pct": Decimal("2"),
                "_open_interest": Decimal("100000"),
                "_daily_volume": Decimal("200000"),
                "_bid": Decimal("100"),
                "_ask": Decimal("101"),
            }
        ],
        sampled_count=40,
        rounds=60,
        elapsed_s=120.5,
        min_samples=20,
        args=args,
    )
    assert payload["sampling"]["rounds"] == 60
    assert payload["matched_markets"] == 1
    assert payload["markets"][0]["name"] == "LIT-USD"


def test_screen_build_json_payload_count():
    mod = _load_module(Path("scripts/screen_mm_markets.py"), "screen_mm_markets_mod")
    args = argparse.Namespace(duration_s=180.0, interval_s=2.0)
    payload = mod._build_json_payload(
        [
            {
                "name": "LIT-USD",
                "score": Decimal("12.3"),
            }
        ],
        args,
        80,
        180.0,
    )
    assert payload["count"] == 1
    assert payload["sampling"]["rounds"] == 80
