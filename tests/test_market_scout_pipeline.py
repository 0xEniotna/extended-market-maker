from __future__ import annotations

import importlib.util
from decimal import Decimal
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_draft_env_scales_caps_and_sets_market(tmp_path: Path):
    mod = _load_module(Path("scripts/tools/market_scout_pipeline.py"), "market_scout_pipeline_mod")
    common = _load_module(Path("scripts/tools/mm_audit_common.py"), "mm_audit_common_mod")

    template = tmp_path / "template.env"
    template.write_text(
        "\n".join(
            [
                "MM_MARKET_NAME=ETH-USD",
                "MM_MAX_POSITION_SIZE=100",
                "MM_MAX_POSITION_NOTIONAL_USD=1000",
                "MM_MAX_ORDER_NOTIONAL_USD=500",
            ]
        )
        + "\n"
    )

    out = tmp_path / ".env.new.candidate"
    mod._prepare_draft_env(
        template_env=template,
        output_path=out,
        market="AMZN_24_5-USD",
        launch_size_multiplier=Decimal("0.25"),
        warmup_hours=6,
    )

    env_map = common.parse_env(common.read_env_lines(out))
    assert env_map["MM_MARKET_NAME"] == "AMZN_24_5-USD"
    assert env_map["MM_MAX_POSITION_SIZE"] == "25"
    assert env_map["MM_MAX_POSITION_NOTIONAL_USD"] == "250"
    assert env_map["MM_MAX_ORDER_NOTIONAL_USD"] == "125"
    assert env_map["MM_SCOUT_CANDIDATE"] == "true"
    assert env_map["MM_SCOUT_WARMUP_HOURS"] == "6"


def test_build_launch_action_contract_has_required_fields(tmp_path: Path):
    mod = _load_module(Path("scripts/tools/market_scout_pipeline.py"), "market_scout_pipeline_mod2")

    action = mod._build_launch_action(
        market_row={
            "market": "LIT-USD",
            "score": Decimal("13.5"),
            "spread_bps": Decimal("7"),
            "spread_p90_bps": Decimal("9"),
            "coverage_pct": Decimal("90"),
            "ticks_in_spread": Decimal("6"),
            "daily_volume": Decimal("200000"),
            "gate_flags": {"score": True},
        },
        created_ts=1_700_000_000,
        expires_ts=1_700_021_600,
        draft_env_path=tmp_path / ".env.lit.candidate",
        repo_root=tmp_path,
        warmup_hours=6,
        launch_size_multiplier=Decimal("0.25"),
    )

    required = {
        "action_id",
        "created_at",
        "expires_at",
        "action_type",
        "market",
        "reason_codes",
        "evidence",
        "risk_profile",
        "commands",
        "expected_evidence",
    }
    assert required.issubset(set(action.keys()))
    assert action["action_type"] == "launch"
    assert action["market"] == "LIT-USD"
    assert isinstance(action["commands"], list)
    assert action["expected_evidence"]["cron_market"] == "LIT-USD"


def test_extract_config_snapshot_includes_allowlisted_and_toxicity_keys(tmp_path: Path):
    mod = _load_module(Path("scripts/tools/market_scout_pipeline.py"), "market_scout_pipeline_mod3")

    env = tmp_path / ".env.eth"
    env.write_text(
        "\n".join(
            [
                "MM_MARKET_NAME=ETH-USD",
                "MM_SPREAD_MULTIPLIER=0.35",
                "MM_REPRICE_TOLERANCE_PERCENT=0.80",
                "MM_ORDER_SIZE_MULTIPLIER=36",
                "MM_INVENTORY_SKEW_FACTOR=0.25",
                "MM_IMBALANCE_PAUSE_THRESHOLD=0.75",
                "MM_MAX_POSITION_NOTIONAL_USD=108000",
                "MM_TOXICITY_MARKOUT_BPS=4.5",
                "MM_TOXICITY_MIN_OBS=25",
                "MM_API_KEY=secret_should_not_show",
            ]
        )
        + "\n"
    )

    snapshot = mod._extract_config_snapshot(str(env))
    assert snapshot["available"] is True
    assert snapshot["error"] is None
    assert snapshot["env_hash"] is not None
    values = snapshot["values"]
    assert values["MM_SPREAD_MULTIPLIER"] == "0.35"
    assert values["MM_REPRICE_TOLERANCE_PERCENT"] == "0.80"
    assert values["MM_ORDER_SIZE_MULTIPLIER"] == "36"
    assert values["MM_INVENTORY_SKEW_FACTOR"] == "0.25"
    assert values["MM_IMBALANCE_PAUSE_THRESHOLD"] == "0.75"
    assert values["MM_MAX_POSITION_NOTIONAL_USD"] == "108000"
    assert values["MM_TOXICITY_MARKOUT_BPS"] == "4.5"
    assert values["MM_TOXICITY_MIN_OBS"] == "25"
    assert "MM_API_KEY" not in values


def test_extract_config_snapshot_supports_alias_keys(tmp_path: Path):
    mod = _load_module(Path("scripts/tools/market_scout_pipeline.py"), "market_scout_pipeline_mod4")

    env = tmp_path / ".env.alt"
    env.write_text(
        "\n".join(
            [
                "SPREAD_PERCENT=1.1",
                "REPRICE_TOLERANCE_PERCENT=0.5",
                "ORDER_SIZE_MULTIPLIER=1.3",
                "INVENTORY_SKEW_FACTOR=0.4",
                "IMBALANCE_PAUSE_THRESHOLD=0.8",
                "MM_MAX_NOTIONAL=50000",
            ]
        )
        + "\n"
    )

    snapshot = mod._extract_config_snapshot(str(env))
    values = snapshot["values"]
    source_keys = snapshot["source_keys"]
    assert values["MM_SPREAD_MULTIPLIER"] == "1.1"
    assert values["MM_REPRICE_TOLERANCE_PERCENT"] == "0.5"
    assert values["MM_ORDER_SIZE_MULTIPLIER"] == "1.3"
    assert values["MM_INVENTORY_SKEW_FACTOR"] == "0.4"
    assert values["MM_IMBALANCE_PAUSE_THRESHOLD"] == "0.8"
    assert values["MM_MAX_ORDER_NOTIONAL_USD"] == "50000"
    assert source_keys["MM_SPREAD_MULTIPLIER"] == "SPREAD_PERCENT"
    assert source_keys["MM_MAX_ORDER_NOTIONAL_USD"] == "MM_MAX_NOTIONAL"


def test_render_markdown_includes_active_config_snapshot_section():
    mod = _load_module(Path("scripts/tools/market_scout_pipeline.py"), "market_scout_pipeline_mod5")

    report = {
        "generated_at": "2026-02-26T10:00:00Z",
        "data_quality": {"ok": True, "issues": [], "default_env": ".env.eth"},
        "rate_limits": {
            "proposed_launches_last24h": 0,
            "max_new_per_day": 2,
            "max_new_per_run": 1,
            "launch_slots_this_run": 1,
        },
        "active_markets": [
            {
                "market": "ETH-USD",
                "underperformance_streak": 0,
                "pnl_24h_usd": "12.5",
                "markout_5s_bps": "-1.1",
                "fill_rate_pct": "32",
                "config_snapshot": {
                    "values": {
                        "MM_SPREAD_MULTIPLIER": "0.35",
                        "MM_REPRICE_TOLERANCE_PERCENT": "0.8",
                        "MM_ORDER_SIZE_MULTIPLIER": "36",
                        "MM_INVENTORY_SKEW_FACTOR": "0.25",
                        "MM_IMBALANCE_PAUSE_THRESHOLD": "0.75",
                        "MM_MAX_POSITION_NOTIONAL_USD": "108000",
                        "MM_TOXICITY_MARKOUT_BPS": "4.5",
                    }
                },
            }
        ],
        "candidate_markets": [],
    }
    md = mod._render_markdown(report, [])
    assert "## Active Config Snapshot" in md
    assert "ETH-USD" in md
    assert "0.35" in md
    assert "MM_TOXICITY_MARKOUT_BPS" in md
