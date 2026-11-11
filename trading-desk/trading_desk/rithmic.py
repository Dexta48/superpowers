"""Real-time futures bars from Rithmic — the feed behind TradeSea and most
futures prop-firm platforms. Read-only: it connects to the market-data and
history servers only, never the order server, so it cannot place a trade.

Set these before running with `--source rithmic`:
  RITHMIC_USER, RITHMIC_PASSWORD   the login you use for TradeSea
  RITHMIC_SYSTEM                   system name your firm gives you (shown at login)
  RITHMIC_URL                      gateway host:port your firm gives you
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import Instrument

# timeframe -> (bar type name, period, days of history)
BARS = {"1d": ("DAILY_BAR", 1, 400), "1h": ("MINUTE_BAR", 60, 60), "15m": ("MINUTE_BAR", 15, 5)}
ENV = ("RITHMIC_USER", "RITHMIC_PASSWORD", "RITHMIC_SYSTEM", "RITHMIC_URL")


def to_frame(bars: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Open": [b["open_price"] for b in bars],
            "High": [b["high_price"] for b in bars],
            "Low": [b["low_price"] for b in bars],
            "Close": [b["close_price"] for b in bars],
            "Volume": [b.get("volume", 0) for b in bars],
        },
        index=pd.to_datetime([b["marker"] for b in bars], unit="s", utc=True).tz_convert("America/New_York"),
    )
    return df[~df.index.duplicated(keep="last")].sort_index()


async def _load(insts: list[Instrument]) -> dict:
    from async_rithmic import RithmicClient, SysInfraType, TimeBarType

    missing = [k for k in ENV if not os.environ.get(k)]
    if missing:
        raise RuntimeError("Rithmic source needs these environment variables: " + ", ".join(missing))
    client = RithmicClient(
        user=os.environ["RITHMIC_USER"],
        password=os.environ["RITHMIC_PASSWORD"],
        system_name=os.environ["RITHMIC_SYSTEM"],
        app_name="trading_desk",
        app_version="1.0",
        url=os.environ["RITHMIC_URL"],
    )
    await client.connect(plants=[SysInfraType.TICKER_PLANT, SysInfraType.HISTORY_PLANT])
    out = {}
    try:
        now = datetime.now(timezone.utc)
        for inst in insts:
            try:
                contract = await client.get_front_month_contract(inst.rithmic_root, inst.rithmic_exchange)
                frames = {}
                for tf, (kind, period, days) in BARS.items():
                    bars = await client.get_historical_time_bars(
                        contract, inst.rithmic_exchange, now - timedelta(days=days), now,
                        getattr(TimeBarType, kind), period,
                    )
                    if not bars:
                        raise RuntimeError(f"no {tf} bars for {contract}")
                    frames[tf] = to_frame(bars)
                out[inst.symbol] = (contract, frames)
            except Exception as e:  # keep going so one bad symbol does not sink the rest
                out[inst.symbol] = e
    finally:
        await client.disconnect()
    return out


def fetcher(insts: list[Instrument]):
    """Pull every instrument in one login, then hand back a `fetch(inst, timeframe)`."""
    loaded = asyncio.run(_load(insts))

    def fetch(inst: Instrument, timeframe: str) -> pd.DataFrame:
        got = loaded[inst.symbol]
        if isinstance(got, Exception):
            raise got
        return got[1][timeframe]

    return fetch
