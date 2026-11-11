"""The four agents: news, chart, signal, critic. Each is one Claude call with its
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
    chat: str  # your message in the team group chat
    market_context: str
    signals: list[Signal]


class ChartRead(BaseModel):
    symbol: str
    trend_daily: Literal["up", "down", "sideways"]
    trend_1h: Literal["up", "down", "sideways"]
    trend_15m: Literal["up", "down", "sideways"]
    structure: str  # swing highs/lows, ranges, breaks, where price sits in them
    momentum: str  # RSI, MACD histogram and what they are doing, per timeframe where it matters
    volatility: str  # ATR in context: quiet, normal or expanding; typical bar size
    key_resistance: list[float]  # nearest first
    key_support: list[float]  # nearest first
    bias: Literal["bullish", "bearish", "neutral"]
    bias_strength: Literal["weak", "moderate", "strong"]
    what_would_change_it: str  # the price action that flips or cancels the bias
    data_quality: str  # stale, thin or inconsistent data; "fine" if none


class ChartBook(BaseModel):
    chat: str  # your message in the team group chat
    cross_market: str  # how the three charts relate: leadership, divergence, confirmation
    analyses: list[ChartRead]


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
    chat: str  # your message in the team group chat


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

CHART_SYSTEM = """You are the chart desk: the technical analyst for MGC, MNQ and SPX. Your \
only job is to read the charts thoroughly and say what they show. You do not propose trades \
and you ignore news; other desks do that with your read in hand.

The chart data is JSON per instrument: for the daily, 1-hour and 15-minute timeframes you \
get the last price, EMAs (20/50/200), RSI, ATR, MACD histogram, the prior 20-bar range, the \
window high and low, daily pivots, the previous day's high/low/close, and the most recent \
bars. Read every timeframe of every instrument, top-down.

For each instrument report: the trend on each timeframe and whether they agree; the market \
structure (higher highs and lows or the reverse, ranges, recent breaks or failed breaks, \
where price sits inside the current range); momentum and whether it confirms or diverges \
from price; volatility relative to recent bars; and the support and resistance levels that \
matter, nearest first, each one a number that is actually in the data (a swing, a pivot, an \
EMA, a range edge). Then give one overall bias with its strength, and the specific price \
action that would flip or cancel it.

Be exact and literal. Quote readings from the data and do not describe patterns you cannot \
support from the bars given. When timeframes conflict, say so and call the bias neutral or \
weak instead of forcing a view. Check `last_bar_time` against the current time and report \
stale, thin or inconsistent data in `data_quality`. In `cross_market`, compare the three \
charts: which is leading, and whether they confirm or diverge from each other."""

SIGNAL_SYSTEM = """You are the signal desk. You read multi-timeframe chart data for MGC, \
MNQ and SPX and issue at most one trade idea per instrument.

The chart desk has already analysed every chart; its read comes first in your brief. Start \
from it, check it against the raw data, and build on it. If you trade against its bias, say \
why. Constructing the trade (entry, stop, targets) is your job, not theirs.

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

You get the same chart data, the chart desk's technical read, the news briefing, the \
signals, and a risk sheet computed in \
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

Sometimes you also get a second opinion written by a different AI model. Treat it as one \
more analyst: weigh each objection on its merits against the data, say where it disagrees \
with the signal desk and who is right, and do not defer to it.

Verdicts: `take`, `take_reduced_size`, `wait_for_confirmation` (say exactly what \
confirmation), or `reject`. Be specific and blunt, and do not soften a reject. For a flat \
call, judge whether standing aside is right or whether a clear setup was missed. In \
`portfolio_note`, address combined exposure — MNQ and SPX are highly correlated, so two \
same-direction index trades are one bet at double size."""


# The team shares a group chat. Each agent posts one message per run and can
# see what was said before it, so they can answer each other.
TEAM_CHAT = """

You are {name} on a five-person desk: Nia (news), Kai (charts), Sol (signals), Grok (second \
opinion, a different AI model, not always present) and Rook (risk). The desk has a group \
chat. Post one message to it: one or two short sentences, first person, plain text, the way \
you would type to colleagues. Say the one thing your teammates most need from your work this \
run. If the chat so far (in your brief, when there is any) has something you agree or \
disagree with, answer that teammate by name. {voice} The chat is only a heads-up: it must \
agree with your full analysis and add no claim that is not in it. {where}"""

VOICE = {
    "news": ("Nia", "You are brisk and newsroom-dry."),
    "chart": ("Kai", "You are precise and a little nerdy about levels."),
    "signal": ("Sol", "You are decisive and brief."),
    "risk": ("Rook", "You are blunt and sceptical, never rude."),
}
IN_FIELD = "Put the message in the `chat` field."
LAST_LINE = "Put the message on the final line of your answer, starting with `CHAT:`."


def _system(base: str, desk: str, where: str = IN_FIELD) -> str:
    name, voice = VOICE[desk]
    return base + TEAM_CHAT.format(name=name, voice=voice, where=where)


def _chat_block(chat: str) -> str:
    return f"## Team group chat so far\n{chat}\n\n" if chat else ""


def split_chat(text: str) -> tuple[str, str]:
    """Separate a trailing or leading `CHAT:` line from free-text output."""
    lines = text.strip().splitlines()
    for i in (len(lines) - 1, 0):
        if lines and lines[i].strip().upper().startswith("CHAT:"):
            msg = lines[i].strip()[5:].strip()
            return "\n".join(lines[:i] + lines[i + 1:]).strip(), msg
    return text.strip(), ""


def _check(resp):
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"model declined the request: {resp.stop_details}")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("response was cut off at max_tokens")
    return resp


def _quiet(kind: str, text: str) -> None:
    pass


def _stream(client: anthropic.Anthropic, emit, **kw):
    """One streamed model call. `emit(kind, text)` gets a live feed of what the
    agent is doing, for the office view: "doing" (status label), "thought"
    (reasoning summary), "screen" (output as it is written), "log" (searches)."""
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=32000,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": EFFORT},
        **FALLBACK,
        **kw,
    ) as stream:
        tool_json = None
        for ev in stream:
            if ev.type == "content_block_start":
                kind = ev.content_block.type
                if kind == "thinking":
                    emit("doing", "thinking")
                elif kind == "server_tool_use":
                    tool_json = ""
                    emit("doing", "searching the web")
                elif kind == "text":
                    emit("doing", "writing")
                    emit("screen_reset", "")
            elif ev.type == "content_block_delta":
                d = ev.delta
                if d.type == "thinking_delta":
                    emit("thought", d.thinking)
                elif d.type == "text_delta":
                    emit("screen", d.text)
                elif d.type == "input_json_delta" and tool_json is not None:
                    tool_json += d.partial_json
            elif ev.type == "content_block_stop" and tool_json is not None:
                try:
                    query = json.loads(tool_json).get("query")
                except (ValueError, AttributeError):
                    query = None
                if query:
                    emit("log", f"searched: {query}")
                tool_json = None
        return _check(stream.get_final_message())


def news_agent(client, now: str, calendar: list[dict] | None, emit=_quiet) -> str:
    feed = (
        "Calendar feed (unofficial, verify):\n" + json.dumps(calendar, indent=1)
        if calendar
        else "No calendar feed was available; build the schedule from search."
    )
    messages = [{"role": "user", "content": f"Current time: {now}\n\n{feed}"}]
    for _ in range(5):  # a long search turn can pause; send it back to resume
        resp = _stream(
            client,
            emit,
            system=_system(NEWS_SYSTEM, "news", LAST_LINE),
            messages=messages,
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 8}],
        )
        if resp.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": resp.content})
    text = [b.text for b in resp.content if b.type == "text"]
    # Text before the last search is narration ("let me check..."); keep the final answer.
    return text[-1].strip() if text else ""


def _structured(client, emit, system: str, user: str, schema):
    resp = _stream(
        client, emit, system=system, messages=[{"role": "user", "content": user}], output_format=schema
    )
    return resp.parsed_output


def chart_agent(client, now: str, charts: list[dict], emit=_quiet, chat: str = "") -> ChartBook:
    user = f"Current time: {now}\n\n{_chat_block(chat)}## Chart data\n{json.dumps(charts)}"
    return _structured(client, emit, _system(CHART_SYSTEM, "chart"), user, ChartBook)


def signal_agent(
    client, now: str, charts: list[dict], news: str, read: ChartBook, emit=_quiet, chat: str = ""
) -> SignalSet:
    user = (
        f"Current time: {now}\n\n{_chat_block(chat)}## Chart desk's read\n{read.model_dump_json(indent=1)}\n\n"
        f"## News briefing\n{news}\n\n## Chart data\n{json.dumps(charts)}"
    )
    return _structured(client, emit, _system(SIGNAL_SYSTEM, "signal"), user, SignalSet)


def critic_agent(
    client, now: str, charts: list[dict], news: str, read: ChartBook, signals: SignalSet,
    risk: list[dict], emit=_quiet, second_opinion: str = "", chat: str = "",
) -> ReviewSet:
    other = (
        f"## Second opinion from a different AI model (Grok)\n{second_opinion}\n\n" if second_opinion else ""
    )
    user = (
        f"Current time: {now}\n\n{_chat_block(chat)}## Signals under review\n{signals.model_dump_json(indent=1)}\n\n"
        f"## Risk sheet\n{json.dumps(risk, indent=1)}\n\n{other}"
        f"## Chart desk's read\n{read.model_dump_json(indent=1)}\n\n## News briefing\n{news}\n\n"
        f"## Chart data\n{json.dumps(charts)}"
    )
    return _structured(client, emit, _system(CRITIC_SYSTEM, "risk"), user, ReviewSet)
