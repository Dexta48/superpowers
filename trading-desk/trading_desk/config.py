"""Instruments and run settings."""
from __future__ import annotations

import os
from dataclasses import dataclass

MODEL = os.environ.get("TRADING_DESK_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("TRADING_DESK_EFFORT", "high")

# (yfinance interval, yfinance period) per timeframe label, highest first.
TIMEFRAMES = {"1d": ("1d", "1y"), "1h": ("1h", "60d"), "15m": ("15m", "5d")}


@dataclass(frozen=True)
class Instrument:
    symbol: str  # name used everywhere in prompts and reports
    yahoo: str  # data ticker
    description: str
    point_value: float  # USD per 1.00 point, per contract
    tick: float
    traded_as: str  # what the dollar-risk maths assumes you trade
    rithmic_root: str = ""  # product code and exchange on the Rithmic feed
    rithmic_exchange: str = ""
    rithmic_note: str = ""  # replaces `description` when prices come from Rithmic


INSTRUMENTS = {
    "MGC": Instrument("MGC", "MGC=F", "Micro Gold futures (COMEX, 10 oz)", 10.0, 0.10, "MGC", "MGC", "COMEX"),
    "MNQ": Instrument("MNQ", "MNQ=F", "Micro E-mini Nasdaq-100 futures (CME)", 2.0, 0.25, "MNQ", "MNQ", "CME"),
    # SPX is a cash index and cannot be traded directly. Levels are analysed on
    # the index; dollar risk is sized as Micro E-mini S&P (MES, $5/pt). MES
    # trades at a basis to the cash index, so shift levels by that basis.
    "SPX": Instrument("SPX", "^GSPC", "S&P 500 cash index", 5.0, 0.25, "MES (proxy)", "MES", "CME",
                      "S&P 500 via Micro E-mini S&P futures — every price here is the MES futures price, "
                      "not the cash index"),
}
