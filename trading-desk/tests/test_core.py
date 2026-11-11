"""Offline tests: indicator/risk maths on synthetic bars, and the full pipeline
with the model client stubbed."""
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading_desk import __main__ as cli
from trading_desk import agents, data, report, risk
from trading_desk.config import INSTRUMENTS


def bars(n=300, start=100.0, step=0.5, freq="D"):
    close = start + step * np.arange(n) + np.sin(np.arange(n))
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="America/New_York")
    return pd.DataFrame(
        {"Open": close - 0.2, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1000},
        index=idx,
    )


def chart(symbol="MNQ"):
    return {
        "symbol": symbol,
        "timeframes": {tf: data.snapshot(bars(), tf) for tf in ("1d", "1h", "15m")},
    }


def test_snapshot_uptrend():
    s = data.snapshot(bars(), "1d")
    assert s["last"] > s["ema20"] > s["ema50"] > s["ema200"]
    assert s["rsi14"] > 60
    assert 1.5 < s["atr14"] < 3.5
    assert s["prior_20bar_high"] == round(bars()["High"].iloc[-21:-1].max(), 2)  # excludes current bar
    assert len(s["recent_bars_time_o_h_l_c_v"]) == data.RECENT_BARS
    assert s["pivots"]["s1"] < s["pivots"]["p"] < s["pivots"]["r1"]


def test_snapshot_short_history_gives_none():
    s = data.snapshot(bars(n=30), "1h")
    assert s["ema200"] is None and s["ema50"] is None and s["ema20"] is not None


def test_rsi_extremes():
    assert data.rsi(pd.Series(np.arange(50.0))).iloc[-1] == 100
    assert data.rsi(pd.Series(np.arange(50.0, 0, -1))).iloc[-1] == 0


def test_risk_sheet():
    sig = {"direction": "long", "entry": 20000.0, "stop": 19950.0, "targets": [20100.0, 20200.0]}
    r = risk.assess(sig, INSTRUMENTS["MNQ"], chart(), account=10_000, risk_pct=1)
    assert r["risk_points"] == 50 and r["risk_usd_per_contract"] == 100
    assert r["reward_to_risk_per_target"] == [2.0, 4.0]
    assert r["max_contracts_within_budget"] == 1
    assert r["structural_problems"] == []


def test_risk_flags_malformed_and_unaffordable():
    sig = {"direction": "short", "entry": 4000.0, "stop": 3900.0, "targets": [4100.0]}
    r = risk.assess(sig, INSTRUMENTS["MGC"], chart("MGC"), account=500, risk_pct=1)
    assert len(r["structural_problems"]) == 2
    assert r["max_contracts_within_budget"] == 0
    assert risk.assess({"direction": "flat"}, INSTRUMENTS["MGC"], chart("MGC"), 1, 1)["direction"] == "flat"


SIGNALS = agents.SignalSet(
    market_context="Risk-on.",
    signals=[
        agents.Signal(symbol="MNQ", direction="long", setup="pullback", timeframe="intraday",
                      entry=250.0, stop=245.0, targets=[260.0], confidence="medium",
                      rationale="Trend up.", invalidation="15m close under 247"),
        agents.Signal(symbol="MGC", direction="flat", setup="none", timeframe="intraday",
                      entry=0, stop=0, targets=[], confidence="low",
                      rationale="CPI tomorrow.", invalidation="n/a"),
    ],
)
REVIEWS = agents.ReviewSet(
    portfolio_note="Single index bet.",
    reviews=[agents.Review(symbol="MNQ", verdict="take_reduced_size", score_out_of_10=6,
                           strengths=["with trend"], weaknesses=["extended"],
                           risk_assessment="ok", news_risk="CPI", suggested_adjustments="half size")],
)


class FakeMessages:
    def __init__(self):
        self.calls = []
        self.news_turns = [
            SimpleNamespace(stop_reason="pause_turn", content=[SimpleNamespace(type="text", text="searching")]),
            SimpleNamespace(stop_reason="end_turn", content=[
                SimpleNamespace(type="text", text="let me check"),
                SimpleNamespace(type="web_search_tool_result"),
                SimpleNamespace(type="text", text="CPI Thu 08:30 ET")]),
        ]

    def create(self, **kw):
        self.calls.append(("create", kw))
        return self.news_turns.pop(0)

    def parse(self, **kw):
        self.calls.append(("parse", kw))
        out = SIGNALS if kw["output_format"] is agents.SignalSet else REVIEWS
        return SimpleNamespace(stop_reason="end_turn", parsed_output=out)


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "fetch", lambda inst, tf: bars())
    monkeypatch.setattr(cli.news, "upcoming_usd_events", lambda: [{"title": "CPI"}])
    fake = FakeMessages()
    client = SimpleNamespace(beta=SimpleNamespace(messages=fake))
    args = SimpleNamespace(symbols=["MNQ", "MGC"], account=10_000, risk_pct=1.0,
                           data_only=False, out=str(tmp_path), webhook=None)
    md = cli.run_once(args, client)

    kinds = [k for k, _ in fake.calls]
    assert kinds == ["create", "create", "parse", "parse"]  # paused news turn was resumed
    assert fake.calls[1][1]["messages"][-1]["role"] == "assistant"
    signal_prompt = fake.calls[2][1]["messages"][0]["content"]
    assert "CPI Thu 08:30 ET" in signal_prompt and "let me check" not in signal_prompt
    critic_prompt = fake.calls[3][1]["messages"][0]["content"]
    assert '"risk_usd_per_contract": 10.0' in critic_prompt  # 5 pts x $2 on MNQ
    assert "MNQ — LONG → risk desk: **TAKE (reduced size)** (6/10)" in md
    assert "MGC — FLAT" in md and "CPI Thu 08:30 ET" in md
    assert len(list(tmp_path.iterdir())) == 2


def test_refusal_raises():
    with pytest.raises(RuntimeError, match="declined"):
        agents._check(SimpleNamespace(stop_reason="refusal", stop_details="x"))


def test_push_chunks(monkeypatch):
    sent = []
    monkeypatch.setattr(report.requests, "post",
                        lambda url, json, timeout: sent.append(json["content"]) or SimpleNamespace(raise_for_status=lambda: None))
    text = "".join(f"line {i}\n" for i in range(600)) + "x" * 5000
    report.push("http://hook", text)
    assert all(len(c) <= 1900 for c in sent) and "".join(sent) == text
