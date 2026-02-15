#!/usr/bin/env python
"""Entry point for the market making strategy."""
from __future__ import annotations

import asyncio

from market_maker.strategy import MarketMakerStrategy

if __name__ == "__main__":
    asyncio.run(MarketMakerStrategy.run())
