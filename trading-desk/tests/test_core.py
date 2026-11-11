"""Offline tests: indicator/risk maths on synthetic bars, and the full pipeline
with the model client stubbed."""
import json
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading_desk import __main__ as cli
from trading_desk import agents, data, demo, report, risk
from trading_desk.office import Office
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


def run_demo(tmp_path, monkeypatch, **client_kw):
    monkeypatch.setattr(data, "fetch", demo.bars)
    monkeypatch.setattr(cli.news, "upcoming_usd_events", lambda: [{"title": "CPI"}])
    client = demo.FakeClient(**client_kw)
    args = SimpleNamespace(symbols=["MNQ", "MGC", "SPX"], account=10_000, risk_pct=1.0,
                           data_only=False, demo=False, out=str(tmp_path / 'out'), webhook=None,
                           vault=str(tmp_path / 'vault'), office=8765, source='yahoo')
    office = Office()
    return cli.run_once(args, client, office), client, json.loads(office.json())


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    md, client, state = run_demo(tmp_path, monkeypatch, pause_first=True)

    assert len(client.calls) == 4  # paused news turn resumed, then signal, then critic
    assert client.calls[1]["messages"][-1]["role"] == "assistant"
    signal_prompt = client.calls[2]["messages"][0]["content"]
    assert "No-trade windows" in signal_prompt and "let me check" not in signal_prompt
    assert '"risk_usd_per_contract"' in client.calls[3]["messages"][0]["content"]
    assert "MNQ — LONG → risk desk: **TAKE (reduced size)** (6/10)" in md
    assert "MGC — FLAT" in md and "SPX — LONG → risk desk: **REJECT**" in md
    assert len(list((tmp_path / "out").iterdir())) == 2

    # Obsidian vault: report note with properties, a log row per signal, office note linking both
    desk = tmp_path / "vault" / "Trading Desk"
    (rep,) = (desk / "Reports").iterdir()
    assert rep.read_text().startswith("---\ndate: ") and "MNQ: long / TAKE (reduced size)" in rep.read_text()
    log = (desk / "Signals log.md").read_text()
    assert log.count("\n| 20") == 3 and "| SPX | long |" in log and "| MGC | flat |  |  |" in log
    office_note = (desk / "Office.md").read_text()
    assert 'src="http://127.0.0.1:8765"' in office_note and f"Reports/{rep.stem}|" in office_note

    # the office saw every desk work and finish
    assert state["phase"] == "idle" and state["run"] == 1
    assert all(d["status"] == "done" for d in state["desks"].values())
    assert state["desks"]["news"]["log"] == [
        "searched: US economic calendar this week CPI", "searched: Fed speakers this week"]
    assert "CPI time" in state["desks"]["news"]["thought"]
    assert state["desks"]["risk"]["screen"].lstrip().startswith("{")
    assert [b["symbol"] for b in state["board"]] == ["MNQ", "MGC", "SPX"]
    assert state["board"][0]["verdict"] == "TAKE (reduced size)" and state["board"][0]["rr"]
    assert [t["symbol"] for t in state["tickers"]] == ["MNQ", "MGC", "SPX"]


def test_office_server(tmp_path, monkeypatch):
    office = Office()
    server = office.serve(0)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert "Trading desk" in urllib.request.urlopen(base + "/").read().decode()
        assert json.loads(urllib.request.urlopen(base + "/state").read())["phase"] == "idle"
        urllib.request.urlopen(urllib.request.Request(base + "/run", method="POST"))
        assert office.run_requested.is_set()
        office.run_requested.clear()
        evil = urllib.request.Request(base + "/run", method="POST", headers={"Origin": "http://evil.example"})
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(evil)
        assert not office.run_requested.is_set()
    finally:
        server.shutdown()


def test_office_marks_failure():
    office = Office()
    office.begin_run("now")
    office.start("signal")
    office.fail("boom")
    state = json.loads(office.json())
    assert state["error"] == "boom" and state["desks"]["signal"]["status"] == "error"


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


def test_rithmic_source(monkeypatch):
    """Bars from a (faked) Rithmic session become chart data; only data servers are used."""
    import async_rithmic
    from trading_desk import rithmic

    seen = {}

    class FakeRithmic:
        def __init__(self, **kw):
            seen["init"] = kw

        async def connect(self, plants):
            seen["plants"] = plants

        async def get_front_month_contract(self, root, exchange):
            if root == "MGC":
                raise RuntimeError("no market data permission")
            return root + "Z6"

        async def get_historical_time_bars(self, contract, exchange, start, end, bar_type, period):
            seen.setdefault("req", []).append((contract, exchange, bar_type, period))
            return [dict(marker=1_780_000_000 + i * 900, open_price=100 + i, high_price=101 + i,
                         low_price=99 + i, close_price=100.5 + i, volume=10) for i in (1, 0, 2, 2)]

        async def disconnect(self):
            seen["closed"] = True

    monkeypatch.setattr(async_rithmic, "RithmicClient", FakeRithmic)
    for k in rithmic.ENV:
        monkeypatch.setenv(k, "x")
    fetch = rithmic.fetcher([INSTRUMENTS["MNQ"], INSTRUMENTS["SPX"], INSTRUMENTS["MGC"]])

    df = fetch(INSTRUMENTS["SPX"], "15m")
    assert list(df["Close"]) == [100.5, 101.5, 102.5] and df.index.is_monotonic_increasing  # sorted, de-duplicated
    assert ("MESZ6", "CME", async_rithmic.TimeBarType.MINUTE_BAR, 15) in seen["req"]
    assert ("MNQZ6", "CME", async_rithmic.TimeBarType.DAILY_BAR, 1) in seen["req"]
    assert async_rithmic.SysInfraType.ORDER_PLANT not in seen["plants"] and seen["closed"]
    with pytest.raises(RuntimeError, match="permission"):
        fetch(INSTRUMENTS["MGC"], "1d")
    assert data.chart_pack(INSTRUMENTS["MNQ"], fetch)["timeframes"]["1h"]["last"] == 102.5

    monkeypatch.delenv("RITHMIC_URL")
    with pytest.raises(RuntimeError, match="RITHMIC_URL"):
        rithmic.fetcher([INSTRUMENTS["MNQ"]])


def test_claude_code_backend(monkeypatch):
    """A headless Claude Code run is translated into the stream the agents expect."""
    from trading_desk import claude_code

    def ev(e):
        return json.dumps({"type": "stream_event", "event": e})

    def block(kind, name, deltas):
        cb = {"type": kind, **({"name": name} if name else {})}
        dt = "text_delta" if kind == "text" else "input_json_delta"
        key = "text" if kind == "text" else "partial_json"
        return [ev({"type": "content_block_start", "content_block": cb}),
                *[ev({"type": "content_block_delta", "delta": {"type": dt, key: d}}) for d in deltas],
                ev({"type": "content_block_stop"})]

    review = {"portfolio_note": "ok", "reviews": []}
    scripts = {
        "news": [json.dumps({"type": "system", "subtype": "init"}), "not json",
                 *block("tool_use", "WebSearch", ['{"query": "CPI ', 'date", "mode": "extended"}']),
                 *block("text", None, ["CPI is ", "Thursday."]),
                 json.dumps({"type": "result", "is_error": False, "result": "CPI is Thursday."})],
        "risk": [*block("tool_use", "StructuredOutput", ['{"portfolio_note"', ': "ok", "reviews": []}']),
                 json.dumps({"type": "result", "is_error": False, "result": "", "structured_output": review})],
        "fail": [json.dumps({"type": "result", "is_error": True, "result": "Not logged in"})],
    }
    launched = []

    class FakePopen:
        def __init__(self, cmd, **kw):
            launched.append(cmd)
            sys_prompt = cmd[cmd.index("--system-prompt") + 1]
            self.stdout = iter(l + "\n" for l in scripts[sys_prompt])
            self.stdin = SimpleNamespace(write=lambda p: launched.append(p), close=lambda: None)
            self.stderr = SimpleNamespace(read=lambda: "")

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(claude_code.subprocess, "Popen", FakePopen)
    monkeypatch.delenv("TRADING_DESK_MODEL", raising=False)
    monkeypatch.delenv("TRADING_DESK_EFFORT", raising=False)
    client = claude_code.ClaudeCodeClient()
    seen = []
    emit = lambda kind, text: seen.append((kind, text))

    monkeypatch.setattr(agents, "NEWS_SYSTEM", "news")
    assert agents.news_agent(client, "now", None, emit) == "CPI is Thursday."
    cmd = launched[0]
    assert cmd[:2] == ["claude", "-p"] and cmd[cmd.index("--tools") + 1] == "WebSearch" and "--model" not in cmd
    assert "Current time: now" in launched[1]  # prompt goes in on stdin
    assert ("log", "searched: CPI date") in seen and ("screen", "CPI is ") in seen

    seen.clear()
    out = agents._structured(client, emit, "risk", "review these", agents.ReviewSet)
    assert out.portfolio_note == "ok"
    cmd = launched[2]
    assert cmd[cmd.index("--tools") + 1] == "" and "reviews" in cmd[cmd.index("--json-schema") + 1]
    assert ("screen", '{"portfolio_note"') in seen  # structured output shows on the monitor as it is written

    with pytest.raises(RuntimeError, match="Not logged in"):
        agents._structured(client, emit, "fail", "x", agents.ReviewSet)

    def missing(*a, **k):
        raise FileNotFoundError
    monkeypatch.setattr(claude_code.subprocess, "Popen", missing)
    with pytest.raises(RuntimeError, match="Claude Code was not found"):
        agents._structured(client, emit, "risk", "x", agents.ReviewSet)


def test_office_moves_off_a_busy_port():
    """If another program already answers on the port, the office takes a free one."""
    first = Office().serve(0)
    busy = first.server_address[1]
    try:
        second = Office().serve(busy)
        try:
            port = second.server_address[1]
            assert port != busy
            assert "Trading desk" in urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        finally:
            second.shutdown()
    finally:
        first.shutdown()
