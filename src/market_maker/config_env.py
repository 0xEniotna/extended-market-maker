"""Environment resolution and enums for market maker configuration."""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

# ---------------------------------------------------------------------------
# Env file resolution (mirrors existing pattern in settings.py)
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return current.parents[2]


PROJECT_ROOT = _find_project_root()


def _resolve_env_file() -> Path:
    env_file = os.getenv("ENV", ".env")
    candidates = []
    if env_file:
        if not env_file.startswith("."):
            candidates.append(f".{env_file}")
        candidates.append(env_file)
    else:
        candidates.append(".env")

    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = PROJECT_ROOT / candidate
        if path.exists():
            return path

    return PROJECT_ROOT / ".env"


ENV_FILE = _resolve_env_file()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MMEnvironment(str, Enum):
    TESTNET = "testnet"
    MAINNET = "mainnet"


class OffsetMode(str, Enum):
    """How to compute the per-level price offset.

    FIXED   — use ``price_offset_per_level_percent`` directly (absolute % of price).
    DYNAMIC — compute offset from the live spread:
              offset = max(spread * spread_multiplier, min_offset_bps / 10_000)
              Automatically adapts to any market's liquidity.
    """

    FIXED = "fixed"
    DYNAMIC = "dynamic"


class QuoteAnchor(str, Enum):
    MID = "mid"
    MARK = "mark"
    INDEX = "index"
