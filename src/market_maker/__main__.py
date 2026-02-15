"""Allow ``python -m market_maker`` invocation."""
from __future__ import annotations

import asyncio

from .strategy import MarketMakerStrategy

if __name__ == "__main__":
    asyncio.run(MarketMakerStrategy.run())
