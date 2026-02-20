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
