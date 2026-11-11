"""The three agents: news, signal, critic. Each is one Claude call with its
own system prompt; the pipeline in __main__ wires them together."""
from __future__ import annotations

import json
from typing import Literal

import anthropic
from pydantic import BaseModel

from .config import EFFORT, MODEL

# If a safety classifier declines a request, the API re-runs it on a fallback
# model inside the same call instead of failing the run.
FALLBACK = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}


class Signal(BaseModel):
    symbol: str
    direction: Literal["long", "short", "flat"]
    setup: str  # e.g. "pullback to 1h EMA20 in daily uptrend"
    timeframe: Literal["intraday", "swing"]
    entry: float  # 0 when flat
    stop: float  # 0 when flat
    targets: list[float]  # empty when flat
    confidence: Literal["low", "medium", "high"]
    rationale: str
    invalidation: str  # what price action kills the idea before the stop


class SignalSet(BaseModel):
    market_context: str
    signals: list[Signal]


class Review(BaseModel):
    symbol: str
    verdict: Literal["take", "take_reduced_size", "wait_for_confirmation", "reject"]
    score_out_of_10: int
    strengths: list[str]
    weaknesses: list[str]
    risk_assessment: str
    news_risk: str
    suggested_adjustments: str


class ReviewSet(BaseModel):
    reviews: list[Review]
    portfolio_note: str  # correlation / combined exposure across the signals


NEWS_SYSTEM = """You are the news desk for a trader in MGC (micro gold futures), MNQ \
(micro Nasdaq-100 futures) and SPX (S&P 500).

Produce a briefing of scheduled, market-moving events over the next 7 days, plus anything \
that broke in the last 24 hours that these three markets are trading on. Cover: US data \
(CPI, PPI, NFP, jobless claims, PCE, GDP, ISM, retail sales), Fed decisions, minutes and \
speakers, Treasury auctions, mega-cap and index-heavyweight earnings, options expiry / \
futures roll dates, and geopolitical or dollar/real-yield drivers for gold.

Use web search to confirm dates and times; do not rely on memory for the schedule. If a \
calendar feed is provided, treat it as a starting point and verify the high-impact items. \
Give every event with its date and time in US Eastern, the consensus forecast and prior \
when available, and which of MGC / MNQ / SPX it matters to and why. Say plainly when you \
could not confirm something. End with a short "No-trade windows" list: the periods when \
opening a new position is a coin-flip on a number.

Output markdown, most imminent event first. No trade recommendations — that is another \
desk's job."""

SIGNAL_SYSTEM = """You are the signal desk. You read multi-timeframe chart data for MGC, \
MNQ and SPX and issue at most one trade idea per instrument.

The chart data is JSON per instrument: for the daily, 1-hour and 15-minute timeframes you \
get the last price, EMAs, RSI, ATR, MACD histogram, range highs/lows, daily pivots and the \
most recent bars. Work top-down: daily for trend and major levels, 1h for structure, 15m \
for the trigger. Check `last_bar_time` — if the data is stale or the market is closed, say \
so and frame entries as levels for the next session.

A signal needs a specific setup you can name, an entry at a level the data supports, a stop \
beyond the structure that would prove the idea wrong (not an arbitrary distance — sanity \
check it against ATR), and targets at real levels. Favour setups where the timeframes \
agree. `flat` is a first-class answer: when the picture is mixed, mid-range, or a major \
release in the news briefing lands before the trade could play out, return flat and explain \
what you are waiting for. A forced signal is worse than none; a second desk will tear apart \
anything weak.

All prices are in the instrument's own quote. SPX levels are on the cash index. Return one \
entry per instrument, in the order given. `market_context` is two or three sentences on the \
cross-market picture (risk appetite, dollar/yields vs. gold, Nasdaq vs. S&P leadership)."""

CRITIC_SYSTEM = """You are the risk desk. Another analyst has issued trade signals for MGC, \
MNQ and SPX; your job is to find what is wrong with them before money is at risk. You are \
independent of that analyst and gain nothing by agreeing.

You get the same chart data, the news briefing, the signals, and a risk sheet computed in \
code (points and dollars at risk per contract, reward-to-risk per target, contracts that \
fit the account's risk budget, stop distance in ATR multiples, structural problems). Trust \
the risk sheet's arithmetic; re-derive nothing.

For each signal, test it: Does the chart data actually show the claimed setup, or does the \
rationale cite levels and readings that are not in the data? Is the trade with or against \
the higher timeframe? Is the stop inside normal noise (well under 1x the 15m/1h ATR) or so \
wide the reward-to-risk fails? Is the first target worth at least the risk taken? Does a \
scheduled event sit between entry and target? Is the entry chasing an extended move? Does \
`max_contracts_within_budget` come out at zero, meaning the trade cannot be taken at this \
account size at all? What would the opposite trade's best argument be?

Verdicts: `take`, `take_reduced_size`, `wait_for_confirmation` (say exactly what \
confirmation), or `reject`. Be specific and blunt, and do not soften a reject. For a flat \
call, judge whether standing aside is right or whether a clear setup was missed. In \
`portfolio_note`, address combined exposure — MNQ and SPX are highly correlated, so two \
same-direction index trades are one bet at double size."""


def _check(resp):
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"model declined the request: {resp.stop_details}")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("response was cut off at max_tokens")
    return resp


def _structured(client: anthropic.Anthropic, system: str, user: str, schema):
    resp = client.beta.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=schema,
        output_config={"effort": EFFORT},
        **FALLBACK,
    )
    return _check(resp).parsed_output


def news_agent(client: anthropic.Anthropic, now: str, calendar: list[dict] | None) -> str:
    feed = (
        "Calendar feed (unofficial, verify):\n" + json.dumps(calendar, indent=1)
        if calendar
        else "No calendar feed was available; build the schedule from search."
    )
    messages = [{"role": "user", "content": f"Current time: {now}\n\n{feed}"}]
    for _ in range(5):  # a long search turn can pause; send it back to resume
        resp = _check(
            client.beta.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=NEWS_SYSTEM,
                messages=messages,
                tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 8}],
                output_config={"effort": EFFORT},
                **FALLBACK,
            )
        )
        if resp.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": resp.content})
    text = [b.text for b in resp.content if b.type == "text"]
    # Text before the last search is narration ("let me check..."); keep the final answer.
    return text[-1].strip() if text else ""


def signal_agent(client, now: str, charts: list[dict], news: str) -> SignalSet:
    user = (
        f"Current time: {now}\n\n## News briefing\n{news}\n\n"
        f"## Chart data\n{json.dumps(charts)}"
    )
    return _structured(client, SIGNAL_SYSTEM, user, SignalSet)


def critic_agent(
    client, now: str, charts: list[dict], news: str, signals: SignalSet, risk: list[dict]
) -> ReviewSet:
    user = (
        f"Current time: {now}\n\n## Signals under review\n{signals.model_dump_json(indent=1)}\n\n"
        f"## Risk sheet\n{json.dumps(risk, indent=1)}\n\n## News briefing\n{news}\n\n"
        f"## Chart data\n{json.dumps(charts)}"
    )
    return _structured(client, CRITIC_SYSTEM, user, ReviewSet)
