"""Best-effort economic calendar feed handed to the news agent as a head start."""
from __future__ import annotations

from datetime import datetime, timezone

import requests

FEED = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"  # unofficial weekly feed


def upcoming_usd_events(now: datetime | None = None) -> list[dict] | None:
    """USD medium/high-impact events still ahead this week, or None if the feed fails."""
    now = now or datetime.now(timezone.utc)
    try:
        resp = requests.get(FEED, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        rows = resp.json()
    except (requests.RequestException, ValueError):
        return None
    out = []
    for e in rows:
        if e.get("country") != "USD" or e.get("impact") not in ("High", "Medium"):
            continue
        try:
            if datetime.fromisoformat(e["date"]) < now:
                continue
        except (KeyError, ValueError):
            pass  # keep events whose date we cannot parse
        out.append({k: e.get(k) for k in ("date", "title", "impact", "forecast", "previous")})
    return out
