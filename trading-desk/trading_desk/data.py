"""Price data and the indicator snapshot the agents read as "the chart"."""
from __future__ import annotations

import pandas as pd

from .config import TIMEFRAMES, Instrument

RECENT_BARS = 24


def fetch(inst: Instrument, timeframe: str) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so tests run without network deps

    interval, period = TIMEFRAMES[timeframe]
    df = yf.Ticker(inst.yahoo).history(period=period, interval=interval, auto_adjust=False)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    if df.empty:
        raise RuntimeError(f"no {timeframe} data returned for {inst.symbol} ({inst.yahoo})")
    return df


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift()
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def snapshot(df: pd.DataFrame, timeframe: str) -> dict:
    """Summarise one timeframe. Indicators that lack enough bars are None."""
    close = df["Close"]
    n = len(df)

    def r(x):
        return None if pd.isna(x) else round(float(x), 2)

    def ema_last(span):
        return r(ema(close, span).iloc[-1]) if n >= span else None

    macd = ema(close, 12) - ema(close, 26)
    hist = macd - ema(macd, 9)
    window = df.iloc[-21:-1] if n > 21 else df.iloc[:-1]  # prior 20 bars, excl. current
    out = {
        "timeframe": timeframe,
        "bars": n,
        "last_bar_time": str(df.index[-1]),
        "last": r(close.iloc[-1]),
        "change_pct_1bar": r((close.iloc[-1] / close.iloc[-2] - 1) * 100) if n > 1 else None,
        "change_pct_20bar": r((close.iloc[-1] / close.iloc[-21] - 1) * 100) if n > 21 else None,
        "ema20": ema_last(20),
        "ema50": ema_last(50),
        "ema200": ema_last(200),
        "rsi14": r(rsi(close).iloc[-1]) if n > 14 else None,
        "atr14": r(atr(df).iloc[-1]) if n > 14 else None,
        "macd_hist": r(hist.iloc[-1]) if n > 35 else None,
        "macd_hist_prev": r(hist.iloc[-2]) if n > 35 else None,
        "prior_20bar_high": r(window["High"].max()) if len(window) else None,
        "prior_20bar_low": r(window["Low"].min()) if len(window) else None,
        "window_high": r(df["High"].max()),
        "window_low": r(df["Low"].min()),
        "recent_bars_time_o_h_l_c_v": [
            [str(ts), r(b.Open), r(b.High), r(b.Low), r(b.Close), int(b.Volume or 0)]
            for ts, b in df.tail(RECENT_BARS).iterrows()
        ],
    }
    if timeframe == "1d" and n > 1:
        p = df.iloc[-2]
        pivot = (p.High + p.Low + p.Close) / 3
        out["prev_day"] = {"high": r(p.High), "low": r(p.Low), "close": r(p.Close)}
        out["pivots"] = {
            "p": r(pivot),
            "r1": r(2 * pivot - p.Low),
            "s1": r(2 * pivot - p.High),
            "r2": r(pivot + p.High - p.Low),
            "s2": r(pivot - (p.High - p.Low)),
        }
    return out


def chart_pack(inst: Instrument, fetch_fn=None) -> dict:
    fetch_fn = fetch_fn or fetch
    return {
        "symbol": inst.symbol,
        "description": inst.description,
        "tick": inst.tick,
        "timeframes": {tf: snapshot(fetch_fn(inst, tf), tf) for tf in TIMEFRAMES},
    }
