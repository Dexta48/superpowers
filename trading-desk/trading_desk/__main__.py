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


def run_once(args, client) -> str:
    now_dt = datetime.now(ZoneInfo("America/New_York"))
    now = now_dt.strftime("%A %Y-%m-%d %H:%M %Z")
    insts = [INSTRUMENTS[s] for s in args.symbols]

    charts, errors = [], []
    for inst in insts:
        try:
            charts.append(data.chart_pack(inst))
        except Exception as e:  # one dead ticker should not sink the run
            errors.append(f"{inst.symbol}: {e}")
    if not charts:
        raise SystemExit("no market data for any instrument: " + "; ".join(errors))

    calendar = news.upcoming_usd_events()
    if args.data_only:
        return json.dumps({"time": now, "errors": errors, "calendar": calendar, "charts": charts}, indent=1)

    print("news desk...", file=sys.stderr)
    briefing = agents.news_agent(client, now, calendar)
    print("signal desk...", file=sys.stderr)
    signals = agents.signal_agent(client, now, charts, briefing)
    by_symbol = {c["symbol"]: c for c in charts}
    sheet = [
        risk.assess(s.model_dump(), INSTRUMENTS[s.symbol], by_symbol[s.symbol], args.account, args.risk_pct)
        for s in signals.signals
        if s.symbol in by_symbol
    ]
    print("risk desk...", file=sys.stderr)
    reviews = agents.critic_agent(client, now, charts, briefing, signals, sheet)

    md = report.render(now, briefing, signals, reviews, sheet, errors)
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
    p.add_argument("--every", type=float, metavar="MINUTES", help="repeat on this interval")
    p.add_argument("--webhook", help="Discord-style webhook URL to push each report to")
    p.add_argument("--out", default="reports", help="directory for report files")
    p.add_argument("--data-only", action="store_true", help="print chart data + calendar as JSON; no model calls")
    args = p.parse_args()

    client = None
    if not args.data_only:
        import anthropic

        client = anthropic.Anthropic()

    while True:
        try:
            print(run_once(args, client))
        except SystemExit:
            raise
        except Exception as e:
            if not args.every:
                raise
            print(f"run failed, will retry next interval: {e!r}", file=sys.stderr)
        if not args.every:
            return
        time.sleep(args.every * 60)


if __name__ == "__main__":
    main()
