"""python -m trading_desk — run news -> signals -> critic once, or on a loop."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import agents, data, news, report, risk
from .config import INSTRUMENTS
from .office import Office


def _board(signals, reviews, sheet) -> list[dict]:
    rev = {r.symbol: r for r in reviews.reviews}
    rk = {r["symbol"]: r for r in sheet}
    rows = []
    for s in signals.signals:
        r, k = rev.get(s.symbol), rk.get(s.symbol, {})
        rows.append({
            "symbol": s.symbol, "direction": s.direction, "setup": s.setup, "entry": s.entry,
            "stop": s.stop, "targets": s.targets, "confidence": s.confidence,
            "rr": k.get("reward_to_risk_per_target", []),
            "risk_usd": k.get("risk_usd_per_contract"), "contracts": k.get("max_contracts_within_budget"),
            "verdict": report.VERDICT[r.verdict] if r else "", "score": r.score_out_of_10 if r else None,
            "weakness": r.weaknesses[0] if r and r.weaknesses else "",
            "adjust": r.suggested_adjustments if r else "",
        })
    return rows


def run_once(args, client, office: Office) -> str:
    now_dt = datetime.now(ZoneInfo("America/New_York"))
    now = now_dt.strftime("%A %Y-%m-%d %H:%M %Z")
    office.begin_run(now)
    insts = [INSTRUMENTS[s] for s in args.symbols]

    charts, errors = [], []
    for inst in insts:
        try:
            charts.append(data.chart_pack(inst))
        except Exception as e:  # one dead ticker should not sink the run
            errors.append(f"{inst.symbol}: {e}")
    if not charts:
        raise RuntimeError("no market data for any instrument: " + "; ".join(errors))
    office.set(tickers=[
        {"symbol": c["symbol"], "last": c["timeframes"]["15m"]["last"],
         "chg": c["timeframes"]["1d"]["change_pct_1bar"]} for c in charts
    ])

    calendar = news.upcoming_usd_events()
    if args.data_only:
        return json.dumps({"time": now, "errors": errors, "calendar": calendar, "charts": charts}, indent=1)

    print("news desk...", file=sys.stderr)
    office.start("news")
    briefing = agents.news_agent(client, now, calendar, office.emitter("news"))
    office.done("news", "Briefing handed to the signal desk")
    office.set(briefing=briefing)

    print("signal desk...", file=sys.stderr)
    office.start("signal")
    signals = agents.signal_agent(client, now, charts, briefing, office.emitter("signal"))
    live = [s for s in signals.signals if s.direction != "flat"]
    office.done("signal", f"{len(live)} live signal(s), {len(signals.signals) - len(live)} flat — sent to risk")

    by_symbol = {c["symbol"]: c for c in charts}
    sheet = [
        risk.assess(s.model_dump(), INSTRUMENTS[s.symbol], by_symbol[s.symbol], args.account, args.risk_pct)
        for s in signals.signals
        if s.symbol in by_symbol
    ]
    print("risk desk...", file=sys.stderr)
    office.start("risk")
    reviews = agents.critic_agent(client, now, charts, briefing, signals, sheet, office.emitter("risk"))
    office.done("risk", ", ".join(f"{r.symbol}: {report.VERDICT[r.verdict]}" for r in reviews.reviews))
    office.set(board=_board(signals, reviews, sheet), market_context=signals.market_context,
               portfolio_note=reviews.portfolio_note, phase="idle")

    md = report.render(now, briefing, signals, reviews, sheet, errors)
    if args.demo:
        return "> DEMO RUN — scripted output, not real analysis.\n\n" + md
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = now_dt.strftime("%Y%m%d-%H%M")
    (out / f"{stem}.md").write_text(md)
    (out / f"{stem}.json").write_text(
        json.dumps(
            {"time": now, "news": briefing, "signals": signals.model_dump(),
             "risk": sheet, "reviews": reviews.model_dump(), "errors": errors},
            indent=1,
        )
    )
    if args.webhook:
        report.push(args.webhook, md)
    return md


def main() -> None:
    p = argparse.ArgumentParser(prog="trading_desk", description=__doc__)
    p.add_argument("--symbols", nargs="+", default=list(INSTRUMENTS), choices=list(INSTRUMENTS))
    p.add_argument("--account", type=float, default=10_000, help="account size in USD (default 10000)")
    p.add_argument("--risk-pct", type=float, default=1.0, help="max %% of account risked per trade (default 1)")
    p.add_argument("--office", nargs="?", type=int, const=8765, metavar="PORT",
                   help="open the office: a local page where you watch the agents work (default port 8765)")
    p.add_argument("--demo", action="store_true",
                   help="scripted fake run: no API key, no market data, no cost — to try the office")
    p.add_argument("--every", type=float, metavar="MINUTES", help="repeat on this interval")
    p.add_argument("--webhook", help="Discord-style webhook URL to push each report to")
    p.add_argument("--out", default="reports", help="directory for report files")
    p.add_argument("--data-only", action="store_true", help="print chart data + calendar as JSON; no model calls")
    args = p.parse_args()

    client = None
    if args.demo:
        from . import demo

        client = demo.FakeClient(delay=0.12 if args.office else 0)
        data.fetch = demo.bars
        news.upcoming_usd_events = lambda: None
    elif not args.data_only:
        import anthropic

        client = anthropic.Anthropic()

    office = Office(demo=args.demo)
    if args.office:
        office.serve(args.office)
        print(f"office open at http://127.0.0.1:{args.office}  (Ctrl-C to close)", file=sys.stderr)

    keep_going = bool(args.every or args.office)
    while True:
        try:
            print(run_once(args, client, office))
        except Exception as e:
            office.fail(str(e))
            if not keep_going:
                raise SystemExit(f"run failed: {e}")
            print(f"run failed: {e!r}", file=sys.stderr)
        if not keep_going:
            return
        try:
            if args.office:  # wait for the Run button or the interval, whichever first
                office.wait_for_next(args.every)
            else:
                time.sleep(args.every * 60)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
