"""Scripted stand-ins for the market feed and the model, so the office can be
watched (and the pipeline tested) with no API key and no network. Everything
it produces is fake."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

from . import agents

LEVELS = {"MGC": 4100.0, "MNQ": 25000.0, "SPX": 6800.0}
FREQ = {"1d": "D", "1h": "h", "15m": "15min"}


def bars(inst, timeframe: str, n: int = 300) -> pd.DataFrame:
    base = LEVELS[inst.symbol]
    scale = base * {"1d": 0.004, "1h": 0.0012, "15m": 0.0005}[timeframe]
    i = np.arange(n)
    close = base - scale * 0.12 * (n - 1 - i) + scale * np.sin(i / 3.0)
    idx = pd.date_range(end="2026-01-09 16:00", periods=n, freq=FREQ[timeframe], tz="America/New_York")
    return pd.DataFrame(
        {"Open": close - scale * 0.1, "High": close + scale * 0.5, "Low": close - scale * 0.5,
         "Close": close, "Volume": 1000},
        index=idx,
    )


BRIEFING = """**DEMO briefing — invented events, not a real calendar.**

- **Thu 08:30 ET — CPI.** Matters to all three: a hot print lifts yields, pressuring MNQ and MGC.
- **Fri 10:00 ET — Consumer sentiment.** Second-tier; mostly MNQ/SPX.

**No-trade windows:** Thu 08:15–09:00 ET."""


def _read(charts):
    out = []
    for c in charts:
        d, h = c["timeframes"]["1d"], c["timeframes"]["1h"]
        gold = c["symbol"] == "MGC"
        out.append(dict(
            symbol=c["symbol"], trend_daily="sideways" if gold else "up", trend_1h="sideways" if gold else "up",
            trend_15m="up", structure="Demo: higher highs and higher lows on the 1h; price in the top third of the 20-bar range.",
            momentum=f"Demo: 1h RSI {h['rsi14']}, MACD histogram positive.", volatility=f"Demo: 1h ATR {h['atr14']}, normal.",
            key_resistance=[d["prior_20bar_high"], d["window_high"]], key_support=[h["ema20"], d["pivots"]["s1"]],
            bias="neutral" if gold else "bullish", bias_strength="weak" if gold else "moderate",
            what_would_change_it="Demo: a 1h close below the EMA50.", data_quality="Demo data: scripted, not a real market."))
    return {"cross_market": "Demo: the two indices are moving together; gold is not confirming.", "analyses": out}


def _signals(charts):
    out = []
    for c in charts:
        last, atr = c["timeframes"]["15m"]["last"], c["timeframes"]["1h"]["atr14"]
        if c["symbol"] == "MGC":
            out.append(dict(symbol="MGC", direction="flat", setup="none", timeframe="intraday",
                            entry=0, stop=0, targets=[], confidence="low",
                            rationale="Demo: mid-range ahead of CPI; nothing to lean on.",
                            invalidation="n/a"))
        else:
            out.append(dict(symbol=c["symbol"], direction="long", setup="pullback to 1h EMA20 in an uptrend",
                            timeframe="intraday", entry=round(last - atr, 2), stop=round(last - 3 * atr, 2),
                            targets=[round(last + 2 * atr, 2), round(last + 5 * atr, 2)], confidence="medium",
                            rationale="Demo: all three timeframes above rising EMAs; buying the first pullback.",
                            invalidation="A 15m close below the 1h EMA50."))
    return {"market_context": "Demo: indices trending up, gold flat ahead of data.", "signals": out}


def _reviews(signals):
    verdicts = {"MNQ": ("take_reduced_size", 6), "SPX": ("reject", 3), "MGC": ("take", 7)}
    return {
        "portfolio_note": "Demo: MNQ and SPX longs are one bet twice — take one, not both.",
        "reviews": [
            dict(symbol=s["symbol"], verdict=verdicts[s["symbol"]][0], score_out_of_10=verdicts[s["symbol"]][1],
                 strengths=["With the higher-timeframe trend"] if s["direction"] != "flat" else ["Standing aside before CPI is right"],
                 weaknesses=["Duplicates the MNQ exposure"] if s["symbol"] == "SPX" else ["Entry is close to a scheduled release"] if s["direction"] != "flat" else [],
                 risk_assessment="Demo: stop sits beyond 1h noise; first target pays more than the risk.",
                 news_risk="CPI on Thursday sits between entry and the second target.",
                 suggested_adjustments="Half size until after the print." if s["direction"] != "flat" else "None — stay flat.")
            for s in signals["signals"]
        ],
    }


class _Stream:
    def __init__(self, events, final, delay):
        self.events, self.final, self.delay = events, final, delay

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        for e in self.events:
            if self.delay:
                time.sleep(self.delay)
            yield e

    def get_final_message(self):
        return self.final


def _block(kind, deltas):
    dtype, field = {"thinking": ("thinking_delta", "thinking"), "text": ("text_delta", "text"),
                    "server_tool_use": ("input_json_delta", "partial_json")}[kind]
    yield NS(type="content_block_start", content_block=NS(type=kind))
    for d in deltas:
        yield NS(type="content_block_delta", delta=NS(type=dtype, **{field: d}))
    yield NS(type="content_block_stop")


def _chunks(text, n=14):
    return [text[i:i + n] for i in range(0, len(text), n)]


class FakeClient:
    """Mimics `client.beta.messages.stream(...)` for the four agents."""

    def __init__(self, delay: float = 0.0, pause_first: bool = False):
        self.delay, self.pause_first, self.calls = delay, pause_first, []
        self.beta = NS(messages=self)

    def stream(self, **kw):
        self.calls.append(kw)
        prompt = kw["messages"][0]["content"]
        if "tools" in kw:  # news desk
            if self.pause_first:
                self.pause_first = False
                return _Stream([], NS(stop_reason="pause_turn", content=[NS(type="text", text="searching")]), 0)
            events = [
                *_block("thinking", _chunks("Need this week's US data and Fed speakers; confirm the CPI time first. ")),
                *_block("text", ["let me check"]),
                *_block("server_tool_use", ['{"query": "US econo', 'mic calendar this week CPI"}']),
                *_block("server_tool_use", ['{"query": "Fed speakers this week"}']),
                *_block("text", _chunks(BRIEFING)),
            ]
            final = NS(stop_reason="end_turn", content=[
                NS(type="text", text="let me check"), NS(type="web_search_tool_result"),
                NS(type="text", text=BRIEFING)])
            return _Stream(events, final, self.delay)

        charts = json.loads(prompt.split("## Chart data\n", 1)[1])
        if kw["output_format"] is agents.ChartBook:
            data = _read(charts)
            thought = "Daily first: where are the EMAs stacked? Then 1h swings, then the 15m. Mark the levels that are in the data. "
        elif kw["output_format"] is agents.SignalSet:
            data = _signals(charts)
            thought = "Daily trend first, then 1h structure. Gold is stuck mid-range into CPI, so flat there. "
        else:
            sig = json.loads(prompt.split("## Signals under review\n", 1)[1].split("\n\n## Risk sheet", 1)[0])
            data = _reviews(sig)
            thought = "Two index longs are the same trade. Is the stop outside 1h ATR? Does CPI land before target two? "
        events = [*_block("thinking", _chunks(thought)), *_block("text", _chunks(json.dumps(data, indent=1), 40))]
        return _Stream(events, NS(stop_reason="end_turn", parsed_output=kw["output_format"].model_validate(data)), self.delay)
