"""Grok as a second opinion: a different model, from a different company,
reviewing the signals before the risk desk rules on them.

Runs through xAI's Grok command-line tool in its headless mode (`grok -p`),
which uses your Grok subscription login. Install the tool, run `grok` once to
sign in, then start the trading desk with `--grok`."""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile

MAX_ARG = 28_000  # Windows caps a whole command line near 32,000 characters
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
INSTALL_HINT = (
    "The `grok` command was not found. Install xAI's Grok CLI, run `grok` once to sign in "
    "with your Grok account, then try again."
)

BRIEF = """You are a second-opinion analyst on a small trading desk. A different AI model has \
just proposed the trade signals below for MGC (micro gold futures), MNQ (micro Nasdaq-100 \
futures) and SPX (S&P 500). Before a risk manager rules on them, give an independent view. \
You gain nothing by agreeing.

Answer from this brief alone. Do not run commands, read files or use any tools.

The desk has a group chat: Nia (news), Kai (charts), Sol (signals), Rook (risk) and you, \
Grok. Start your answer with one line beginning `CHAT:` — one or two short sentences to your \
teammates with your headline view, answering Sol or anyone else by name if you disagree with \
what they said in the chat. Then leave a blank line and give the full answer.

For each market, in this order: say whether you agree, partly agree or disagree with the \
signal (including a flat call); give your single strongest objection or the main thing it \
gets right; and name anything it missed. Then finish with one short paragraph on the overall \
picture and what you would do differently. Be specific, refer to the numbers given, and keep \
the whole answer under 350 words in plain text.

Current time: {now}

## Team group chat so far
{chat}

## Chart analyst's technical read
{read}

## Proposed signals
{signals}

## Risk sheet (computed in code)
{risk}

## News briefing
{news}
"""


def build_prompt(now: str, read, signals, risk: list[dict], news: str, chat: str = "") -> str:
    def fill(n):
        return BRIEF.format(now=now, read=read.model_dump_json(), signals=signals.model_dump_json(),
                            risk=json.dumps(risk), news=n, chat=chat or "(nothing yet)")

    prompt = fill(news)
    if len(prompt) > MAX_ARG:  # only the news can be long; cut it and say so in the brief
        keep = max(0, len(news) - (len(prompt) - MAX_ARG) - 80)
        prompt = fill(news[:keep] + "\n[briefing cut short here to fit the command-line limit]")
    return prompt


def ask(prompt: str, timeout: int = 420) -> str:
    """Send one prompt to Grok's headless mode and return its answer."""
    cmd = shlex.split(os.environ.get("TRADING_DESK_GROK_CMD", "grok"))
    exe = shutil.which(cmd[0])
    if not exe:
        raise RuntimeError(INSTALL_HINT)
    with tempfile.TemporaryDirectory() as empty:  # nothing for a coding tool to read or change
        try:
            run = subprocess.run(
                [exe, *cmd[1:], "-p", prompt], cwd=empty, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout, stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Grok did not answer within {timeout} seconds") from None
    out = ANSI.sub("", run.stdout or "").strip()
    if run.returncode != 0 or not out:
        detail = ANSI.sub("", run.stderr or "").strip() or out or "no output"
        raise RuntimeError(f"Grok run failed (exit {run.returncode}): {detail[-600:]}")
    return out


def second_opinion(now: str, read, signals, risk: list[dict], news: str, emit=lambda k, t: None, chat: str = "") -> str:
    emit("doing", "thinking")
    text = ask(build_prompt(now, read, signals, risk, news, chat))
    emit("screen_reset", "")
    emit("screen", text)
    return text
