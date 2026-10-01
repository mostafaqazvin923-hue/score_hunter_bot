
"""
V33 — SPOT -> PERPETUAL LEAD-LAG CONVERGENCE
Stage-0 research / pre-registered mechanism test.

Purpose
-------
Test one specific market-microstructure hypothesis:

    A large move in Binance spot that is NOT fully reflected in the
    Binance USD-M perpetual during the same completed 15-minute block
    may be followed by convergence in the perpetual during the next block.

This is NOT a parameter-optimization script.
All thresholds below are frozen before seeing the results.

Research rules
--------------
- Real Binance public historical data only.
- 5-minute source bars, aggregated causally to 15-minute bars.
- Signal is known only after a 15-minute bar closes.
- Entry is next 15-minute OPEN.
- Fixed RR 1:2.
- Initial SL = 1.0 x 15m ATR(14); TP = 2.0 x ATR.
- No timeout, no BE, no trailing, no pyramiding.
- Same-candle SL+TP = LOSS.
- No same-candle re-entry.
- Max 1 open trade per symbol.
- Different symbols may be open simultaneously.
- Max 10 simultaneous positions (because $1,000 / $100 margin = 10).
- Fixed $100 margin/trade, 50x leverage => $5,000 notional.
- Fee = 0.07% per side, slippage = 0.03% per side.
- A trade that reaches neither SL nor TP before the end of the sample is censored,
  not marked as a win or loss.
- Fixed universe. No future symbol selection.
- Chronological 50/25/25 Discovery/Development/Validation split.
- No threshold tuning after seeing Development/Validation.

Frozen signal
-------------
For each symbol and completed 15m bar t:

spot_ret = spot_close[t] / spot_open[t] - 1
perp_ret = perp_close[t] / perp_open[t] - 1
gap      = spot_ret - perp_ret

spot_z = zscore(spot_ret, trailing 96 bars, shifted by 1 bar)
gap_z  = zscore(gap,      trailing 96 bars, shifted by 1 bar)

LONG:
    spot_z >= +1.5 AND gap_z >= +0.75

SHORT:
    spot_z <= -1.5 AND gap_z <= -0.75

The signal is evaluated at bar close t and executed at bar t+1 open.

Why this structure
------------------
It explicitly tests information transmission from the spot market into the
perpetual, rather than adding another generic trend/RSI/EMA filter.

Important
---------
A positive result is NOT accepted merely because total PnL is positive.
Validation must satisfy the frozen gate in the report:
    PF >= 1.20
    WR >= 40%
    >= 150 closed Validation trades
    max losing streak <= 4
    Validation net R > 0
    no single symbol responsible for > 50% of Validation net R
If the mechanism fails these gates, V33 is REJECTED and should not be tuned.
"""

from __future__ import annotations

import io
import math
import os
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# =========================
# Frozen configuration
# =========================

LOOKBACK_DAYS = 365
WARMUP_DAYS = 30
TIMEFRAME = "5m"
AGG_MINUTES = 15

INITIAL_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN_PER_TRADE * LEVERAGE

FEE_RATE = 0.0007
SLIPPAGE = 0.0003

ATR_PERIOD = 14
ATR_STOP_MULT = 1.0
RR = 2.0

Z_LOOKBACK = 96
SPOT_Z_MIN = 1.5
GAP_Z_MIN = 0.75

MAX_SYMBOL_POSITIONS = 1
MAX_PORTFOLIO_POSITIONS = 10

MIN_COVERAGE = 0.995

DISCOVERY_FRAC = 0.50
DEVELOPMENT_FRAC = 0.25
VALIDATION_FRAC = 0.25

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT", "AVAXUSDT",
    "NEARUSDT", "ADAUSDT", "BNBUSDT", "APTUSDT", "CRVUSDT",
    "ONDOUSDT", "PENDLEUSDT", "ICPUSDT", "WIFUSDT",
]

ROOT = Path("v33_data")
ROOT.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "V33-research/1.0"})


# =========================
# Time / download helpers
# =========================

def utc_now_floor_5m() -> pd.Timestamp:
    now = pd.Timestamp.now(tz="UTC")
    return now.floor("5min")


def requested_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    end = utc_now_floor_5m() - pd.Timedelta(minutes=5)
    start = end - pd.Timedelta(days=LOOKBACK_DAYS + WARMUP_DAYS)
    return start, end


def month_starts(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1, tz="UTC")
    while cur <= end:
        yield cur
        if cur.month == 12:
            cur = pd.Timestamp(cur.year + 1, 1, 1, tz="UTC")
        else:
            cur = pd.Timestamp(cur.year, cur.month + 1, 1, tz="UTC")


def daterange_days(start: pd.Timestamp, end: pd.Timestamp):
    d = start.normalize()
    last = end.normalize()
    while d <= last:
        yield d
        d += pd.Timedelta(days=1)


def fetch_zip_csv(url: str, retries: int = 3) -> pd.DataFrame | None:
    for attempt in range(1, retries + 1):
        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code != 200:
                if attempt == retries:
                    return None
                time.sleep(1.0 * attempt)
                continue

            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not names:
                    return None
                with zf.open(names[0]) as fh:
                    return pd.read_csv(fh, header=None)
        except Exception:
            if attempt == retries:
                return None
            time.sleep(1.0 * attempt)
    return None


KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def clean_kline(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    raw = raw.iloc[:, :12].copy()

    # Binance archives may contain a header row.
    first = str(raw.iloc[0, 0]).strip().lower()
    if first in {"open time", "open_time", "opentime"}:
        raw = raw.iloc[1:].copy()

    raw.columns = KLINE_COLUMNS

    raw["open_time"] = pd.to_numeric(raw["open_time"], errors="coerce")
    raw = raw.dropna(subset=["open_time"])

    for c in ["open", "high", "low", "close", "volume",
              "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw["dt"] = pd.to_datetime(raw["open_time"], unit="ms", utc=True)
    raw = raw.dropna(subset=["dt", "open", "high", "low", "close"])
    raw = raw.sort_values("dt").drop_duplicates("dt")

    return raw[
        ["dt", "open", "high", "low", "close", "volume",
         "quote_volume", "taker_buy_base", "taker_buy_quote"]
    ].reset_index(drop=True)


def load_market(symbol: str, market: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    market:
        spot
        futures
    """
    out = []

    if market == "spot":
        base = "https://data.binance.vision/data/spot"
    elif market == "futures":
        base = "https://data.binance.vision/data/futures/um"
    else:
        raise ValueError(market)

    # Prefer monthly archives; fall back to daily if a month archive is absent.
    for m in month_starts(start, end):
        ym = f"{m.year:04d}-{m.month:02d}"
        monthly = (
            f"{base}/monthly/klines/{symbol}/{symbol}-{TIMEFRAME}-{ym}.zip"
        )
        df = fetch_zip_csv(monthly)
        if df is not None and not df.empty:
            out.append(clean_kline(df))

    combined = (
        pd.concat(out, ignore_index=True)
        if out else pd.DataFrame(columns=["dt"])
    )

    if not combined.empty:
        have = set(combined["dt"])
    else:
        have = set()

    # Fill missing dates with daily archives.
    missing_days = []
    for d in daterange_days(start, end):
        # Need at least one expected 5m timestamp on the date.
        if not any((d + pd.Timedelta(minutes=5 * k)) in have for k in range(288)):
            missing_days.append(d)

    for d in missing_days:
        ds = d.strftime("%Y-%m-%d")
        daily = f"{base}/daily/klines/{symbol}/{symbol}-{TIMEFRAME}-{ds}.zip"
        df = fetch_zip_csv(daily)
        if df is not None and not df.empty:
            out.append(clean_kline(df))

    if not out:
        raise RuntimeError(f"{market} {symbol}: no Binance data downloaded")

    result = pd.concat(out, ignore_index=True)
    result = (
        result.sort_values("dt")
        .drop_duplicates("dt")
        .query("@start <= dt <= @end")
        .reset_index(drop=True)
    )

    expected = int((end - start).total_seconds() // 300) + 1
    coverage = len(result) / max(expected, 1)

    if coverage < MIN_COVERAGE:
        raise RuntimeError(
            f"{market} {symbol}: coverage {coverage:.4%} < {MIN_COVERAGE:.2%}; "
            "refusing to forward-fill or hide missing data"
        )

    # Remove an incomplete final source bar if present.
    cutoff = utc_now_floor_5m()
    result = result[result["dt"] < cutoff].copy()

    result.to_csv(ROOT / f"{market}_{symbol}_5m.csv", index=False)
    return result


# =========================
# Causal 15m aggregation
# =========================

def aggregate_15m(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().set_index("dt")

    # All source 5m bars must be present inside each 15m block.
    grouped = x.groupby(pd.Grouper(freq="15min", label="left", closed="left"))

    rows = []
    for ts, g in grouped:
        if len(g) != 3:
            continue

        rows.append({
            "dt": ts,
            "open": float(g["open"].iloc[0]),
            "high": float(g["high"].max()),
            "low": float(g["low"].min()),
            "close": float(g["close"].iloc[-1]),
            "volume": float(g["volume"].sum()),
            "quote_volume": float(g["quote_volume"].sum()),
            "taker_buy_quote": float(g["taker_buy_quote"].sum()),
        })

    y = pd.DataFrame(rows)
    if y.empty:
        raise RuntimeError("15m aggregation produced no complete bars")

    return y.sort_values("dt").reset_index(drop=True)


def add_causal_features(spot15: pd.DataFrame, perp15: pd.DataFrame) -> pd.DataFrame:
    s = spot15.rename(columns={
        "open": "spot_open", "high": "spot_high", "low": "spot_low",
        "close": "spot_close", "volume": "spot_volume",
        "quote_volume": "spot_quote_volume",
        "taker_buy_quote": "spot_taker_buy_quote",
    })
    p = perp15.rename(columns={
        "open": "perp_open", "high": "perp_high", "low": "perp_low",
        "close": "perp_close", "volume": "perp_volume",
        "quote_volume": "perp_quote_volume",
        "taker_buy_quote": "perp_taker_buy_quote",
    })

    x = pd.merge(s, p, on="dt", how="inner").sort_values("dt").reset_index(drop=True)

    x["spot_ret"] = x["spot_close"] / x["spot_open"] - 1.0
    x["perp_ret"] = x["perp_close"] / x["perp_open"] - 1.0
    x["gap"] = x["spot_ret"] - x["perp_ret"]

    # Causal ATR on the target perpetual.
    prev_close = x["perp_close"].shift(1)
    tr = pd.concat([
        x["perp_high"] - x["perp_low"],
        (x["perp_high"] - prev_close).abs(),
        (x["perp_low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    x["atr"] = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean().shift(1)

    # Shift BEFORE rolling: current signal bar can never enter its own baseline.
    sr = x["spot_ret"].shift(1)
    gr = x["gap"].shift(1)

    x["spot_mean"] = sr.rolling(Z_LOOKBACK, min_periods=Z_LOOKBACK).mean()
    x["spot_std"] = sr.rolling(Z_LOOKBACK, min_periods=Z_LOOKBACK).std(ddof=0)
    x["gap_mean"] = gr.rolling(Z_LOOKBACK, min_periods=Z_LOOKBACK).mean()
    x["gap_std"] = gr.rolling(Z_LOOKBACK, min_periods=Z_LOOKBACK).std(ddof=0)

    x["spot_z"] = (x["spot_ret"] - x["spot_mean"]) / x["spot_std"].replace(0, np.nan)
    x["gap_z"] = (x["gap"] - x["gap_mean"]) / x["gap_std"].replace(0, np.nan)

    x["signal"] = 0
    x.loc[
        (x["spot_z"] >= SPOT_Z_MIN) & (x["gap_z"] >= GAP_Z_MIN),
        "signal"
    ] = 1
    x.loc[
        (x["spot_z"] <= -SPOT_Z_MIN) & (x["gap_z"] <= -GAP_Z_MIN),
        "signal"
    ] = -1

    # Explicitly exclude rows without valid causal inputs.
    x.loc[x["atr"].isna() | x["spot_z"].isna() | x["gap_z"].isna(), "signal"] = 0

    return x


# =========================
# Trade engine
# =========================

@dataclass
class Trade:
    symbol: str
    side: int
    signal_dt: pd.Timestamp
    entry_dt: pd.Timestamp
    entry: float
    sl: float
    tp: float
    exit_dt: pd.Timestamp | None
    exit: float | None
    gross_r: float | None
    net_r: float | None
    outcome: str | None


def simulate_symbol(symbol: str, x: pd.DataFrame) -> list[Trade]:
    trades = []
    i = 0

    while i < len(x) - 1:
        row = x.iloc[i]
        side = int(row["signal"])

        if side == 0:
            i += 1
            continue

        atr = float(row["atr"])
        entry_bar = x.iloc[i + 1]
        entry = float(entry_bar["perp_open"])

        if not np.isfinite(atr) or atr <= 0 or not np.isfinite(entry):
            i += 1
            continue

        # Slippage is explicitly applied to the entry execution.
        if side == 1:
            entry_exec = entry * (1.0 + SLIPPAGE)
            sl = entry_exec - ATR_STOP_MULT * atr
            tp = entry_exec + RR * ATR_STOP_MULT * atr
        else:
            entry_exec = entry * (1.0 - SLIPPAGE)
            sl = entry_exec + ATR_STOP_MULT * atr
            tp = entry_exec - RR * ATR_STOP_MULT * atr

        exit_dt = None
        exit_px = None
        outcome = None
        exit_index = None

        # Start checking from the entry bar.
        for j in range(i + 1, len(x)):
            b = x.iloc[j]
            hi = float(b["perp_high"])
            lo = float(b["perp_low"])

            if side == 1:
                hit_sl = lo <= sl
                hit_tp = hi >= tp
            else:
                hit_sl = hi >= sl
                hit_tp = lo <= tp

            if hit_sl and hit_tp:
                # Locked protocol: same-candle SL+TP = LOSS.
                exit_dt = b["dt"]
                exit_px = sl
                outcome = "LOSS_SAME_BAR"
                exit_index = j
                break

            if hit_sl:
                exit_dt = b["dt"]
                exit_px = sl
                outcome = "LOSS"
                exit_index = j
                break

            if hit_tp:
                exit_dt = b["dt"]
                exit_px = tp
                outcome = "WIN"
                exit_index = j
                break

        # Censor unresolved final-sample trades.
        if exit_dt is None:
            break

        # Apply adverse exit slippage exactly once.
        if side == 1:
            exit_exec = float(exit_px) * (1.0 - SLIPPAGE)
            gross_dollars = (exit_exec - entry_exec) * (NOTIONAL / entry_exec)
        else:
            exit_exec = float(exit_px) * (1.0 + SLIPPAGE)
            gross_dollars = (entry_exec - exit_exec) * (NOTIONAL / entry_exec)

        # Fees only here; entry/exit slippage was already applied to prices.
        fee_dollars = NOTIONAL * (2.0 * FEE_RATE)
        net_dollars = gross_dollars - fee_dollars

        stop_fraction = (ATR_STOP_MULT * atr) / entry_exec
        risk_dollars = NOTIONAL * stop_fraction
        if risk_dollars <= 0:
            i += 1
            continue

        gross_r = gross_dollars / risk_dollars
        net_r = net_dollars / risk_dollars

        trades.append(Trade(
            symbol=symbol,
            side=side,
            signal_dt=row["dt"],
            entry_dt=entry_bar["dt"],
            entry=entry_exec,
            sl=sl,
            tp=tp,
            exit_dt=exit_dt,
            exit=exit_exec,
            gross_r=gross_r,
            net_r=net_r,
            outcome=outcome,
        ))

        # Hard symbol lock: do not evaluate any signal before the previous
        # trade has closed. Also prevents same-candle re-entry.
        i = int(exit_index) + 1

    return trades


# =========================
# Portfolio overlap enforcement
# =========================

def enforce_portfolio_rules(all_trades: list[Trade]) -> list[Trade]:
    """
    The individual symbol simulator already enforces max 1 trade/symbol by
    construction because it advances through the completed trade before
    considering a later signal.

    Here we enforce the portfolio max of 10 open trades chronologically.
    If >10 trades would overlap, later entries are rejected.
    """
    candidates = sorted(all_trades, key=lambda t: (t.entry_dt, t.symbol))
    accepted = []
    open_trades: list[Trade] = []

    for t in candidates:
        # Remove trades closed before this entry.
        open_trades = [
            q for q in open_trades
            if q.exit_dt is not None and q.exit_dt > t.entry_dt
        ]

        if len(open_trades) >= MAX_PORTFOLIO_POSITIONS:
            continue

        # Explicit symbol lock.
        if any(q.symbol == t.symbol and q.exit_dt is not None and q.exit_dt > t.entry_dt
               for q in open_trades):
            continue

        accepted.append(t)
        open_trades.append(t)

    return accepted


# =========================
# Metrics
# =========================

def max_loss_streak(r: pd.Series) -> int:
    streak = best = 0
    for v in r:
        if v < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def metrics(trades: list[Trade]) -> dict:
    if not trades:
        return {
            "trades": 0, "wins": 0, "losses": 0, "wr": 0.0,
            "gross_r": 0.0, "net_r": 0.0, "pf": 0.0,
            "max_loss_streak": 0,
        }

    r = pd.Series([t.net_r for t in trades], dtype=float)
    wins = int((r > 0).sum())
    losses = int((r <= 0).sum())
    gross_profit = float(r[r > 0].sum())
    gross_loss = float(-r[r <= 0].sum())

    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "wr": wins / len(trades),
        "gross_r": float(sum(t.gross_r for t in trades)),
        "net_r": float(r.sum()),
        "pf": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "max_loss_streak": max_loss_streak(r),
    }


def split_dates(all_trades: list[Trade]):
    dates = pd.Series(sorted(t.entry_dt for t in all_trades))
    if dates.empty:
        return None

    d0 = dates.min()
    d1 = dates.max()
    span = d1 - d0

    d_disc = d0 + span * DISCOVERY_FRAC
    d_dev = d_disc + span * DEVELOPMENT_FRAC

    return d_disc, d_dev


def print_section(name: str, trades: list[Trade]):
    m = metrics(trades)
    print(
        f"{name:14s} "
        f"trades={m['trades']:5d} "
        f"WR={m['wr']*100:6.2f}% "
        f"PF={m['pf']:6.3f} "
        f"netR={m['net_r']:9.2f} "
        f"grossR={m['gross_r']:9.2f} "
        f"streak={m['max_loss_streak']:3d}"
    )


def symbol_concentration(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["symbol", "trades", "net_r"])

    df = pd.DataFrame([
        {"symbol": t.symbol, "net_r": t.net_r}
        for t in trades
    ])
    out = df.groupby("symbol").agg(
        trades=("net_r", "size"),
        net_r=("net_r", "sum")
    ).sort_values("net_r", ascending=False)
    return out


# =========================
# Main
# =========================

def main():
    start, end = requested_window()
    print("=" * 100)
    print("V33 — SPOT -> PERPETUAL LEAD-LAG CONVERGENCE")
    print(f"Window: {start} -> {end}")
    print(f"Universe: {len(SYMBOLS)} symbols")
    print(f"5m source -> 15m bars | RR={RR}:1 | ATR={ATR_PERIOD}")
    print("=" * 100)

    all_candidates = []

    for k, symbol in enumerate(SYMBOLS, 1):
        print(f"[{k}/{len(SYMBOLS)}] {symbol}")

        spot = load_market(symbol, "spot", start, end)
        perp = load_market(symbol, "futures", start, end)

        spot15 = aggregate_15m(spot)
        perp15 = aggregate_15m(perp)

        x = add_causal_features(spot15, perp15)
        trades = simulate_symbol(symbol, x)

        print(f"  spot15={len(
