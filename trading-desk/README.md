# Trading desk

Three Claude agents that analyse **MGC** (micro gold), **MNQ** (micro Nasdaq-100) and
**SPX** (S&P 500) and argue with each other before you see a signal.

| Agent | Job | Output |
|---|---|---|
| **News desk** | Searches the web for scheduled, market-moving events in the next 7 days (CPI, NFP, FOMC, earnings, expiries) and anything that broke in the last 24h | Briefing with dates/times in ET and "no-trade windows" |
| **Signal desk** | Reads daily / 1h / 15m chart data (EMAs, RSI, ATR, MACD, pivots, range levels, recent bars) plus the briefing | One idea per instrument: long / short / **flat**, entry, stop, targets, invalidation |
| **Risk desk** | Independently attacks each signal using the same data and a risk sheet computed in code | Verdict (take / reduced size / wait / reject), score, weaknesses, combined-exposure note |

The risk arithmetic (points and dollars at risk per contract, reward-to-risk, how many
contracts fit your risk budget, stop distance in ATR multiples) is done in Python, not by
the model.

## Run it

```bash
pip install -r requirements.txt
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

```bash
python -m trading_desk --account 10000 --risk-pct 1
```

Each run prints a markdown report and saves it with the raw JSON under `reports/`.

## The office

Watch the three agents work in your browser:

```bash
python -m trading_desk --office
```

Open http://127.0.0.1:8765. Each agent has a desk: the monitor shows what it is writing as
it writes, the speech bubble shows a summary of its reasoning, and the news desk lists each
web search it makes. Work passes left to right; finished signals land on the board below with
the risk desk's verdict stamped on them. **Run now** starts another run (each run is three
paid model calls); add `--every 30` to also re-run on a timer. The page is served on
localhost only.

To see the office without an API key, market data or any cost, play a scripted fake run:

```bash
python -m trading_desk --office --demo
```

| Flag | Meaning |
|---|---|
| `--symbols MGC MNQ` | Subset of instruments |
| `--account`, `--risk-pct` | Account size and max % risked per trade, used for position sizing |
| `--office [PORT]` | Open the office page (default port 8765) |
| `--demo` | Scripted fake run, clearly labelled; nothing real is fetched or called |
| `--every 30` | Re-run every 30 minutes |
| `--webhook URL` | Push each report to a Discord-style webhook |
| `--data-only` | Print the chart data and calendar feed as JSON without calling the model |

`TRADING_DESK_MODEL` (default `claude-opus-5-5`) and `TRADING_DESK_EFFORT` (default `high`)
override the model settings. Each run is three model calls, one of them with web search.

## Things to know

- **Data is Yahoo Finance via `yfinance`** (`MGC=F`, `MNQ=F`, `^GSPC`): free, unofficial and
  delayed about 10–15 minutes for futures. Fine for swing and slower intraday levels, not for
  scalping. Swap `data.fetch` for your broker's feed if you need real time.
- **SPX is a cash index.** Levels are on the index; dollar risk is sized as MES ($5/point).
  MES trades at a basis to the index, so shift levels accordingly. SPX options are not modelled.
- **The agents read numbers, not chart images.** They cannot see drawn trendlines or patterns
  beyond what the bars and indicators show.
- **It never places orders.** Output is analysis from an AI model, not financial advice;
  futures are leveraged and you can lose more than you expect quickly.

## Tests

```bash
python -m pytest
```

Offline: indicator and risk maths on synthetic bars, the full pipeline against the scripted
demo client, and the office server.
