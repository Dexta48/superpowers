"""Deterministic risk arithmetic. The critic reasons over these numbers
rather than doing its own maths."""
from __future__ import annotations

import math

from .config import Instrument


def assess(signal: dict, inst: Instrument, chart: dict, account: float, risk_pct: float) -> dict:
    if signal["direction"] == "flat":
        return {"symbol": inst.symbol, "direction": "flat"}

    entry, stop, targets = signal["entry"], signal["stop"], signal["targets"]
    long = signal["direction"] == "long"
    problems = []
    if (stop >= entry) if long else (stop <= entry):
        problems.append("stop is on the wrong side of entry")
    for t in targets:
        if (t <= entry) if long else (t >= entry):
            problems.append(f"target {t} is on the wrong side of entry")
    if not targets:
        problems.append("no targets given")

    risk_pts = abs(entry - stop)
    risk_usd = risk_pts * inst.point_value
    budget = account * risk_pct / 100
    tfs = chart["timeframes"]
    last = tfs["15m"]["last"]

    def atr_mult(tf):
        a = tfs[tf]["atr14"]
        return round(risk_pts / a, 2) if a else None

    return {
        "symbol": inst.symbol,
        "direction": signal["direction"],
        "sized_as": inst.traded_as,
        "point_value_usd": inst.point_value,
        "risk_points": round(risk_pts, 2),
        "risk_usd_per_contract": round(risk_usd, 2),
        "reward_to_risk_per_target": [
            round(abs(t - entry) / risk_pts, 2) if risk_pts else None for t in targets
        ],
        "account_usd": account,
        "risk_budget_usd": round(budget, 2),
        "max_contracts_within_budget": math.floor(budget / risk_usd) if risk_usd else 0,
        "stop_distance_in_atr": {"15m": atr_mult("15m"), "1h": atr_mult("1h"), "1d": atr_mult("1d")},
        "entry_distance_from_last_pct": round((entry / last - 1) * 100, 2) if last else None,
        "structural_problems": problems,
    }
