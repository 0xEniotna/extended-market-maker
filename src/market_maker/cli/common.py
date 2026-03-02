"""Shared CLI helpers — consolidated from duplicated code across 6+ scripts."""

from __future__ import annotations

import asyncio
import json
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional, Tuple

from dotenv import load_dotenv


def find_project_root() -> Path:
    """Walk up from this file to find the repo root (pyproject.toml or .git)."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return current.parents[3]


PROJECT_ROOT = find_project_root()


def resolve_env_file(env_value: str, project_root: Optional[Path] = None) -> Path:
    """Resolve an env argument to a concrete file path.

    Supports:
    - explicit file path (relative or absolute), e.g. `.env.silver`
    - short name, e.g. `silver` -> `.env.silver`
    - env-prefixed, e.g. `env.silver` -> `.env.silver`
    """
    root = project_root or PROJECT_ROOT
    raw = env_value.strip()
    candidates = [raw]
    if raw and not raw.startswith("."):
        candidates.insert(0, f".{raw}")

    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = root / candidate
        if path.exists():
            return path

    # Fall back to first candidate for clear error message.
    path = Path(candidates[0])
    if not path.is_absolute():
        path = root / candidates[0]
    return path


def to_jsonable(value: Any) -> Any:
    """Recursively convert Decimal/Path types for JSON serialization."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def ensure_ok(resp: Any, label: str) -> None:
    """Raise if an x10 API response is not OK."""
    from x10.utils.http import ResponseStatus

    if resp.status != ResponseStatus.OK:
        raise RuntimeError(f"{label} failed: status={resp.status} error={resp.error}")


async def resolve_market_name(client: Any, requested_market: str) -> str:
    """Resolve market name to the exact exchange casing/symbol."""
    market = requested_market.strip()
    if not market:
        raise RuntimeError("Market cannot be empty.")

    markets = await client.markets_info.get_markets_dict()
    if market in markets:
        return market

    lower_map = {name.lower(): name for name in markets.keys()}
    exact_ci = lower_map.get(market.lower())
    if exact_ci:
        return exact_ci

    market_norm = market.replace("-", "").lower()
    suggestions = [
        name
        for name in markets.keys()
        if market_norm in name.replace("-", "").lower()
    ]
    suggestions = sorted(suggestions)[:8]
    if suggestions:
        raise RuntimeError(
            f"Market not found: {requested_market}. Did you mean: {', '.join(suggestions)}"
        )
    raise RuntimeError(f"Market not found: {requested_market}")


def load_env_and_settings(
    env_arg: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> Any:
    """Load env file and return MarketMakerSettings.

    Returns the settings object. Raises on missing credentials.
    """
    from market_maker.config import MarketMakerSettings

    if env_arg:
        env_file = resolve_env_file(env_arg, project_root)
        if not env_file.exists():
            raise RuntimeError(f"Env file not found: {env_file}")
        os.environ["ENV"] = str(env_file)
        load_dotenv(dotenv_path=env_file, override=True)
    else:
        load_dotenv()

    settings = MarketMakerSettings()
    if not settings.is_configured:
        raise RuntimeError(
            "Missing MM credentials. Ensure MM_VAULT_ID/MM_STARK_PRIVATE_KEY/"
            "MM_STARK_PUBLIC_KEY/MM_API_KEY are set."
        )
    return settings


def create_trading_client(settings: Any) -> Any:
    """Create a PerpetualTradingClient from settings."""
    from x10.perpetual.accounts import StarkPerpetualAccount
    from x10.perpetual.trading_client import PerpetualTradingClient

    account = StarkPerpetualAccount(
        vault=int(settings.vault_id),
        private_key=settings.stark_private_key,
        public_key=settings.stark_public_key,
        api_key=settings.api_key,
    )
    return PerpetualTradingClient(settings.endpoint_config, account)


def json_output(
    payload: dict,
    *,
    json_flag: bool = False,
    json_out_path: Optional[str] = None,
) -> None:
    """Standardized JSON output dispatch."""
    if json_out_path:
        out_path = Path(json_out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n")
    if json_flag:
        print(json.dumps(to_jsonable(payload), indent=2))


def run_async(coro: Any) -> Any:
    """Run an async coroutine from sync context."""
    return asyncio.run(coro)


def fmt_decimal(value: Decimal) -> str:
    return f"{value:.6f}"


def fmt_ts(ms: Optional[int]) -> str:
    if not ms:
        return "-"
    import time

    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ms / 1000))


def parse_decimal(raw: Optional[str], *, label: str) -> Optional[Decimal]:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid {label}: {raw}") from exc
