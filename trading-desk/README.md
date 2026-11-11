# Trading desk

Four Claude agents that analyse **MGC** (micro gold), **MNQ** (micro Nasdaq-100) and
**SPX** (S&P 500) and argue with each other before you see a signal.

| Agent | Job | Output |
|---|---|---|
| **News desk** (Nia) | Searches the web for scheduled, market-moving events in the next 7 days (CPI, NFP, FOMC, earnings, expiries) and anything that broke in the last 24h | Briefing with dates/times in ET and "no-trade windows" |
| **Chart desk** (Kai) | Pure technical analysis of every chart on the daily, 1h and 15m: trend, structure, momentum, volatility, support and resistance. No news, no trades | A read per market with levels, a bias and what would change it |
| **Signal desk** (Sol) | Takes Kai's read, the raw chart data and the briefing and builds the trades | One idea per instrument: long / short / **flat**, entry, stop, targets, invalidation |
| **Risk desk** (Rook) | Independently attacks each signal using the same data, Kai's read and a risk sheet computed in code | Verdict (take / reduced size / wait / reject), score, weaknesses, combined-exposure note |

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

Watch the four agents work in your browser:

```bash
python -m trading_desk --office
```

Your browser opens the office automatically (normally http://127.0.0.1:47800; if another
program on your computer already uses that port, the office picks a free one and prints the
link). It is a room with four characters:

- **Nia** (news desk, headset), **Kai** (chart desk, curly hair), **Sol** (signal desk, cap)
  and **Rook** (risk desk, glasses).
- Whoever is working sits at their desk typing, with a bubble saying what they are doing
  (thinking, searching the web, writing).
- When a desk finishes, its character carries the paper on: Nia and Kai to Sol, Sol to
  Rook; Rook pins the
  final verdicts on the signal board on the wall. Between jobs they wander to the coffee
  machine or the window.
- Below the room, each desk has a panel with the live output as it is written, a summary of
  the agent's reasoning, and the news desk's web searches. Finished signals land on the board
  at the bottom with Rook's verdict stamped on them.

**Run now** starts another run (each run is four paid model calls); add `--every 30` to
also re-run on a timer. The page is served on localhost only.

To see the office without an API key, market data or any cost, play a scripted fake run:

```bash
python -m trading_desk --office --demo
```

| Flag | Meaning |
|---|---|
| `--symbols MGC MNQ` | Subset of instruments |
| `--account`, `--risk-pct` | Account size and max % risked per trade, used for position sizing |
| `--backend claude-code` | Run the agents on your Claude subscription through Claude Code (default `api`) |
| `--source rithmic` | Real-time futures data through your Rithmic login (default `yahoo`, delayed) |
| `--office [PORT]` | Open the office page (default port 47800) |
| `--vault PATH` | Obsidian vault to file reports, the signals log and the Office note in |
| `--demo` | Scripted fake run, clearly labelled; nothing real is fetched or called |
| `--every 30` | Re-run every 30 minutes |
| `--webhook URL` | Push each report to a Discord-style webhook |
| `--data-only` | Print the chart data and calendar feed as JSON without calling the model |

`TRADING_DESK_MODEL` (default `claude-opus-5-5`) and `TRADING_DESK_EFFORT` (default `high`)
override the model settings. Each run is four model calls, one of them with web search.

## Run on your Claude subscription (no API key)

If you have Claude Pro or Max, the agents can run through Claude Code instead of the paid API:

1. Install Claude Code from https://claude.com/claude-code.
2. Run `claude` once in a terminal and log in with your Claude account, then exit.
3. Add `--backend claude-code` to any command:

```bash
python -m trading_desk --office --backend claude-code
```

No `ANTHROPIC_API_KEY` is needed. Each run draws on your plan's usage limits rather than
being billed, so a frequent `--every` timer can use up your allowance. The agents use your
Claude Code default model; set `TRADING_DESK_MODEL` to pick one your plan includes.

## Real-time data (Rithmic / TradeSea)

TradeSea has no data API of its own, but it runs on Rithmic and your TradeSea login is a
Rithmic login. `--source rithmic` uses that login to pull real-time futures bars:

```bash
export RITHMIC_USER="your login"
```

```bash
export RITHMIC_PASSWORD="your password"
```

```bash
export RITHMIC_SYSTEM="system name from your firm"
```

```bash
export RITHMIC_URL="gateway host:port from your firm"
```

```bash
python -m trading_desk --source rithmic --data-only
```

That last command is the test: it prints the chart data and makes no model calls. If it
prints prices with a current `last_bar_time`, real-time data works; add `--source rithmic` to
your normal runs. If it fails to log in, your firm has not enabled API access on the account
— ask their support to.

- **Read-only.** It connects to Rithmic's market-data and history servers only, never the
  order server, so it cannot place or touch a trade.
- **System name and gateway** are specific to your prop firm; they are the same values other
  Rithmic platforms (R|Trader, Sierra Chart) ask for.
- **Sessions.** Rithmic limits simultaneous logins. If TradeSea drops its data when this
  connects, your account allows only one market-data session at a time.
- **SPX becomes MES.** Rithmic carries futures, not the cash index, so the S&P desk reads
  the MES front-month contract and the agents are told the prices are futures prices.

## Obsidian

Point it at your vault folder and it files its work there, under `Trading Desk/`:

```bash
python -m trading_desk --office --vault "/path/to/your/vault"
```

- **`Office.md`** embeds the live office, so you can watch the room inside Obsidian while the
  program is running. Open it in reading view.
- **`Reports/`** gets one note per run, with properties (date, each market's direction and
  verdict) you can query.
- **`Signals log.md`** gains one table row per signal per run — a running journal.

Demo runs write only the Office note, never fake reports. Obsidian is a desktop app reading a
local folder, so this works on the computer where both it and the program run.

## Things to know

- **Default data is Yahoo Finance via `yfinance`** (`MGC=F`, `MNQ=F`, `^GSPC`): free, unofficial and
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
