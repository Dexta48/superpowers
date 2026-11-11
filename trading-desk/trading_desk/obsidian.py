"""Obsidian link: an Obsidian vault is a folder of markdown, so "connecting"
means writing notes into it. Everything goes under `Trading Desk/`."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from . import report

FOLDER = "Trading Desk"
TEAM = """## The team

- **Nia — news desk.** Headset on, always mid-search. Builds the week's event briefing.
- **Kai — chart desk.** Curly hair, eyes on the screens. Reads every chart on every timeframe and marks the levels.
- **Sol — signal desk.** Takes Kai's read and the news and writes one idea per market, or stays flat.
- **Rook — risk desk.** Glasses, red pen. Tries to break every signal before it reaches the board."""
LOG_HEADER = (
    "# Signals log\n\n| Time | Symbol | Direction | Entry | Stop | Targets | R:R | Verdict | Score | Report |\n"
    "|---|---|---|---|---|---|---|---|---|---|\n"
)


def write_office_note(vault: str, port: int | None, latest: str | None = None) -> Path:
    """The note you open in Obsidian: the live office embedded, plus links."""
    root = Path(vault).expanduser() / FOLDER
    root.mkdir(parents=True, exist_ok=True)
    live = (
        f'<iframe src="http://127.0.0.1:{port}" style="width:100%;height:1100px;border:0"></iframe>\n\n'
        f"> The office is live only while `python -m trading_desk --office` is running on this computer. "
        f"If the frame is blank, start it, then reopen this note."
        if port
        else "> Start the program with `--office` to embed the live office here."
    )
    links = [f"- [[{FOLDER}/Signals log|Signals log]]"]
    if latest:
        links.insert(0, f"- [[{FOLDER}/Reports/{latest}|Latest report — {latest}]]")
    note = root / "Office.md"
    note.write_text(f"# Trading desk — the office\n\n{live}\n\n## Latest\n\n" + "\n".join(links) + f"\n\n{TEAM}\n")
    return note


def write_run(vault: str, when: datetime, markdown: str, signals, reviews, sheet: list[dict]) -> str:
    """File one run: a report note with properties, and rows in the signals log.
    Returns the report note's name."""
    root = Path(vault).expanduser() / FOLDER
    (root / "Reports").mkdir(parents=True, exist_ok=True)
    name = when.strftime("%Y-%m-%d %H%M")
    rev = {r.symbol: r for r in reviews.reviews}
    rk = {r["symbol"]: r for r in sheet}

    props = ["---", f"date: {when.strftime('%Y-%m-%dT%H:%M')}", "tags: [trading-desk]"]
    for s in signals.signals:
        r = rev.get(s.symbol)
        props.append(f"{s.symbol}: {s.direction}" + (f" / {report.VERDICT[r.verdict]}" if r else ""))
    (root / "Reports" / f"{name}.md").write_text("\n".join(props) + "\n---\n\n" + markdown + "\n")

    log = root / "Signals log.md"
    rows = []
    for s in signals.signals:
        r, k = rev.get(s.symbol), rk.get(s.symbol, {})
        flat = s.direction == "flat"
        rows.append(
            f"| {when.strftime('%Y-%m-%d %H:%M')} | {s.symbol} | {s.direction} | "
            f"{'' if flat else s.entry} | {'' if flat else s.stop} | {', '.join(map(str, s.targets))} | "
            f"{', '.join(str(x) for x in k.get('reward_to_risk_per_target', []))} | "
            f"{report.VERDICT[r.verdict] if r else ''} | {r.score_out_of_10 if r else ''} | [[{FOLDER}/Reports/{name}\\|report]] |\n"
        )
    with log.open("a") as f:
        if f.tell() == 0:
            f.write(LOG_HEADER)
        f.writelines(rows)
    return name
