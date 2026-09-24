#!/usr/bin/env python3
"""
HUNTER-V4B-R1 — XT USDT-M Futures / 15m / 365d
HTF Liquidity + Market Structure, strict no-lookahead.

Design:
  4H: regime from completed EMA structure + slope
  1H: liquidity sweep + displacement + optional BOS confirmation
  15m: micro-structure break for precise entry
  Entry: NEXT 15m candle open
  RR: fixed 1:2
  No BE / trailing / timeout exit
  Same-bar SL+TP: conservative SL first
  Max 3 concurrent positions, max 1 per cluster
  Missing candles are never forward-filled/bridged.

Important:
  This is a research/backtest engine, not an execution bot.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# XT COLLECTOR
# ============================================================

BASE = "https://fapi.xt.com"
SYMBOL_LIST_URL = f"{BASE}/future/market/v1/public/symbol/list"
KLINE_URL = f"{BASE}/future/market/v1/public/q/kline"

INTERVAL = "15m"
LIMIT = 1500
TIMEOUT = 20
RETRIES = 5
SLEEP = 0.12

SYMBOLS = [
    "BTC", "ETH", "SOL", "SUI", "AVAX", "NEAR", "ADA",
    "BNB", "APT", "CRV", "ONDO", "PENDLE", "ICP", "WIF",
]

CLUSTERS = {
    "BTC": "MAJOR", "ETH": "MAJOR",
    "SOL": "L1", "SUI": "L1", "AVAX": "L1", "NEAR": "L1",
    "ADA": "L1", "BNB": "L1", "APT": "L1",
    "CRV": "DEFI", "ONDO": "DEFI", "PENDLE": "DEFI",
    "ICP": "OTHER", "WIF": "MEME",
}

DATA_DIR = Path("data/xt_futures_15m")
OUT_DIR = DATA_DIR / "backtest_v4b_r2_trigger_opt"

INITIAL_EQUITY = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
FEE_RATE = 0.0007
SLIPPAGE = 0.0003
RR = 2.0

MAX_POSITIONS = 3
MAX_ONE_PER_CLUSTER = True

# Strategy parameters.
EMA_FAST = 50
EMA_SLOW = 200
ATR_PERIOD = 14
SWEEP_LOOKBACK = 20
BOS_LOOKBACK = 8
DISP_AVG_BODY = 20
DISP_BODY_MULT = 1.20
DISP_RANGE_MULT = 1.10
DISP_RVOL_MIN = 1.05

# 15m trigger.
MICRO_LOOKBACK = 2
TRIGGER_WINDOW_15M = 16      # after the 1H setup; signal remains valid for 4 hours
MIN_STOP_PCT = 0.001
MAX_STOP_PCT = 0.04

# Diagnostics: a setup is NOT rejected merely because optional confirmations
# are absent. They are reported for research.
USE_FVG_BONUS = True


# ============================================================
# GENERIC HELPERS
# ============================================================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def api_json(session, url, params=None):
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, dict):
                raise RuntimeError(f"unexpected response type: {type(payload).__name__}")
            rc = payload.get("returnCode")
            if rc not in (None, 0, "0"):
                raise RuntimeError(
                    f"XT API error: {payload.get('error') or payload.get('msgInfo') or payload}"
                )
            return payload
        except Exception as exc:
            last = exc
            if attempt < RETRIES:
                time.sleep(min(2 * attempt, 5))
    raise RuntimeError(f"API request failed after {RETRIES} attempts: {last}")


def extract_list(payload):
    result = payload.get("result")
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ("items", "list", "data", "symbols"):
            if isinstance(result.get(key), list):
                return result[key]
    raise RuntimeError("Could not find symbol list in XT response")


def discover_symbols(session):
    items = extract_list(api_json(session, SYMBOL_LIST_URL))
    found = {}
    for item in items:
        if isinstance(item, str):
            raw = item
        elif isinstance(item, dict):
            raw = item.get("symbol") or item.get("s") or item.get("name") or item.get("pair")
        else:
            continue
        if raw:
            found[str(raw).strip().upper()] = item
    if not found:
        raise RuntimeError("XT symbol list returned zero usable symbols")
    return found


def resolve_symbol(asset, discovered):
    wanted = asset.upper()
    candidates = [f"{wanted}_USDT", f"{wanted}/USDT", f"{wanted}-USDT", wanted]
    for c in candidates:
        if c.upper() in discovered:
            return c.upper()

    def norm(x):
        return str(x).upper().replace("/", "_").replace("-", "_")

    matches = [x for x in discovered if norm(x) == f"{wanted}_USDT"]
    return matches[0] if len(matches) == 1 else None


def normalize_rows(rows, symbol):
    records = []
    for row in rows:
        if isinstance(row, dict):
            required = ("t", "o", "h", "l", "c", "a")
            if not all(k in row for k in required):
                raise RuntimeError(f"{symbol}: malformed dict kline")
            records.append({
                "Timestamp": int(row["t"]),
                "Open": float(row["o"]),
                "High": float(row["h"]),
                "Low": float(row["l"]),
                "Close": float(row["c"]),
                "Volume": float(row["a"]),
            })
        elif isinstance(row, (list, tuple)) and len(row) >= 6:
            records.append({
                "Timestamp": int(row[0]),
                "Open": float(row[1]),
                "High": float(row[2]),
                "Low": float(row[3]),
                "Close": float(row[4]),
                "Volume": float(row[5]),
            })
        else:
            raise RuntimeError(f"{symbol}: unsupported kline row")

    df = pd.DataFrame(records)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    return df


def fetch_batch(session, symbol, start_ms, end_ms):
    params = {
        "symbol": symbol.strip().lower(),
        "interval": INTERVAL,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": LIMIT,
    }
    payload = api_json(session, KLINE_URL, params)
    result = payload.get("result")
    if not isinstance(result, list):
        raise RuntimeError("XT kline result is not a list")
    return normalize_rows(result, symbol)


def audit(df, symbol):
    if df.empty:
        raise RuntimeError(f"{symbol}: empty dataset")

    bad = (
        (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
        | (df["Volume"] < 0)
        | (df["High"] < df[["Open", "Close", "Low"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close", "High"]].min(axis=1))
    )
    if bad.any():
        raise RuntimeError(f"{symbol}: {int(bad.sum())} invalid OHLCV rows")

    diffs = df["Date"].diff().dropna()
    gaps = diffs[diffs > pd.Timedelta(minutes=15)]
    return {
        "rows": int(len(df)),
        "start": df["Date"].iloc[0].isoformat(),
        "end": df["Date"].iloc[-1].isoformat(),
        "duplicates": int(df["Date"].duplicated().sum()),
        "gaps": int(len(gaps)),
        "max_gap_minutes": int(gaps.max().total_seconds() / 60) if len(gaps) else 15,
    }


def download_symbol(session, symbol, start_dt, end_dt):
    start_ms = to_ms(start_dt)
    cursor_end = to_ms(end_dt)
    batches = []
    calls = 0

    while cursor_end >= start_ms:
        batch = fetch_batch(session, symbol, start_ms, cursor_end)
        calls += 1
        if batch.empty:
            break

        batch = batch.sort_values("Date").reset_index(drop=True)
        first = int(batch["Timestamp"].iloc[0])
        last = int(batch["Timestamp"].iloc[-1])

        if first < start_ms:
            batch = batch[batch["Timestamp"] >= start_ms].copy()
            if batch.empty:
                break
            first = int(batch["Timestamp"].iloc[0])
            last = int(batch["Timestamp"].iloc[-1])

        if last > cursor_end:
            raise RuntimeError(f"{symbol}: API returned candle beyond requested end")

        batches.append(batch)
        print(
            f"  {symbol}: request={calls:02d} rows={len(batch):4d} "
            f"{batch['Date'].iloc[0]} -> {batch['Date'].iloc[-1]}"
        )

        if first <= start_ms:
            break

        next_end = first - 1
        if next_end >= cursor_end:
            raise RuntimeError(f"{symbol}: pagination made no progress")
        cursor_end = next_end

        if calls > 1000:
            raise RuntimeError(f"{symbol}: pagination safety stop")
        time.sleep(SLEEP)

    if not batches:
        raise RuntimeError(f"{symbol}: zero historical candles")

    df = pd.concat(batches, ignore_index=True)
    df = df.drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)
    df = df[(df["Date"] >= pd.Timestamp(start_dt)) & (df["Date"] <= pd.Timestamp(end_dt))].copy()

    current = pd.Timestamp.now(tz="UTC")
    if not df.empty and df["Date"].iloc[-1] + pd.Timedelta(minutes=15) > current:
        df = df.iloc[:-1].copy()

    rep = audit(df, symbol)
    rep["api_requests"] = calls
    return df, rep


# ============================================================
# DATA / INDICATORS
# ============================================================

def load_csv(data_dir: Path, asset: str) -> pd.DataFrame:
    path = data_dir / f"{asset}_USDT_15m.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing XT Futures data file: {path}")

    df = pd.read_csv(path)
    rename = {"Date": "timestamp", "Open": "open", "High": "high",
              "Low": "low", "Close": "close", "Volume": "volume"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"{path}: missing columns {set(required) - set(df.columns)}")

    if np.issubdtype(df["timestamp"].dtype, np.number):
        ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        ts = pd.to_datetime(df["timestamp"], utc=True)

    df = df.assign(timestamp=ts).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")].copy()

    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required[1:])

    if (df.index.minute % 15 != 0).any():
        raise ValueError(f"{asset}: timestamps are not aligned to 15m")

    bad = (
        (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (df["volume"] < 0)
        | (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    )
    if bad.any():
        raise ValueError(f"{asset}: invalid OHLCV rows: {int(bad.sum())}")

    return df


def split_segments(df):
    if len(df) < 2:
        return [df]
    delta = df.index.to_series().diff().dt.total_seconds().div(60)
    cuts = np.flatnonzero(delta.to_numpy() > 15.0001)
    starts = [0] + cuts.tolist()
    ends = cuts.tolist() + [len(df)]
    return [df.iloc[s:e].copy() for s, e in zip(starts, ends) if e - s >= 80]


def resample_complete(df, rule):
    x = df.resample(rule, label="right", closed="right", origin="epoch").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    counts = df["close"].resample(rule, label="right", closed="right", origin="epoch").count()
    expected = 16 if rule == "4h" else 4
    return x[(counts == expected)].dropna()


def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def add_4h_regime(h4):
    x = h4.copy()
    x["ema_fast"] = x.close.ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean()
    x["ema_slow"] = x.close.ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean()
    x["atr"] = atr(x, ATR_PERIOD)

    # State is based only on completed 4H candles.
    x["bull"] = (
        (x.ema_fast > x.ema_slow)
        & (x.ema_fast > x.ema_fast.shift(1))
        & (x.close > x.ema_fast)
    )
    x["bear"] = (
        (x.ema_fast < x.ema_slow)
        & (x.ema_fast < x.ema_fast.shift(1))
        & (x.close < x.ema_fast)
    )
    return x


def add_1h_structure(h1):
    x = h1.copy()
    x["atr"] = atr(x, ATR_PERIOD)
    x["body"] = (x.close - x.open).abs()
    x["range"] = x.high - x.low
    x["avg_body"] = x.body.shift(1).rolling(DISP_AVG_BODY, min_periods=DISP_AVG_BODY).mean()
    x["avg_range"] = x.range.shift(1).rolling(DISP_AVG_BODY, min_periods=DISP_AVG_BODY).mean()
    x["rvol"] = x.volume / x.volume.shift(1).rolling(DISP_AVG_BODY, min_periods=DISP_AVG_BODY).mean()

    # Prior liquidity only. Current bar never participates in its own reference.
    x["prior_high"] = x.high.shift(1).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max()
    x["prior_low"] = x.low.shift(1).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min()
    x["bos_high"] = x.high.shift(1).rolling(BOS_LOOKBACK, min_periods=BOS_LOOKBACK).max()
    x["bos_low"] = x.low.shift(1).rolling(BOS_LOOKBACK, min_periods=BOS_LOOKBACK).min()

    x["sweep_low"] = (x.low < x.prior_low) & (x.close > x.prior_low)
    x["sweep_high"] = (x.high > x.prior_high) & (x.close < x.prior_high)

    x["disp_up"] = (
        (x.close > x.open)
        & (x.body >= DISP_BODY_MULT * x.avg_body)
        & (x.range >= DISP_RANGE_MULT * x.avg_range)
        & (x.rvol >= DISP_RVOL_MIN)
    )
    x["disp_down"] = (
        (x.close < x.open)
        & (x.body >= DISP_BODY_MULT * x.avg_body)
        & (x.range >= DISP_RANGE_MULT * x.avg_range)
        & (x.rvol >= DISP_RVOL_MIN)
    )

    # BOS is a confirmation, not a mandatory second event on the same candle.
    x["bos_up"] = x.close > x.bos_high
    x["bos_down"] = x.close < x.bos_low

    # FVG is a bonus/entry-quality feature, not a mandatory filter.
    x["bull_fvg"] = x.low > x.high.shift(2)
    x["bear_fvg"] = x.high < x.low.shift(2)

    x["long_event"] = x.sweep_low & x.disp_up
    x["short_event"] = x.sweep_high & x.disp_down
    return x


def add_15m_trigger(df):
    x = df.copy()
    x["micro_high"] = x.high.shift(1).rolling(MICRO_LOOKBACK, min_periods=MICRO_LOOKBACK).max()
    x["micro_low"] = x.low.shift(1).rolling(MICRO_LOOKBACK, min_periods=MICRO_LOOKBACK).min()
    x["micro_bull"] = x.close > x.micro_high
    x["micro_bear"] = x.close < x.micro_low
    x["body"] = (x.close - x.open).abs()
    return x


# ============================================================
# SIGNAL ENGINE
# ============================================================

def generate_candidates(asset, df, diagnostics=None):
    candidates = []
    d = defaultdict(int)
    if diagnostics is not None:
        d.update(diagnostics)

    for seg in split_segments(df):
        if len(seg) < 600:
            d["segments_too_short"] += 1
            continue

        h1 = resample_complete(seg, "1h")
        h4 = resample_complete(seg, "4h")
        if len(h1) < 220 or len(h4) < 70:
            d["htf_history_insufficient"] += 1
            continue

        h1 = add_1h_structure(h1)
        h4 = add_4h_regime(h4)
        m15 = add_15m_trigger(seg)

        # Each 1H setup is evaluated using only information available at that
        # completed 1H close. Then the 15m trigger occurs afterward.
        for t, row in h1.iterrows():
            if not bool(row.long_event or row.short_event):
                continue

            h4_before = h4[h4.index <= t]
            if h4_before.empty:
                continue
            regime = h4_before.iloc[-1]

            if not np.isfinite(row.atr) or row.atr <= 0:
                continue

            if bool(row.long_event):
                side = "LONG"
                d["h1_liquidity_displacement_long"] += 1
                sweep_extreme = float(row.low)
                bos_seen = bool(row.bos_up)
                fvg_seen = bool(row.bull_fvg)
            else:
                side = "SHORT"
                d["h1_liquidity_displacement_short"] += 1
                sweep_extreme = float(row.high)
                bos_seen = bool(row.bos_down)
                fvg_seen = bool(row.bear_fvg)

            d["regime_pass"] += 1
            if bos_seen:
                d["h1_bos_confirmation"] += 1
            if fvg_seen:
                d["h1_fvg_seen"] += 1

            # Trigger window: next 8 completed 15m candles after the 1H setup.
            sub = m15[(m15.index > t) & (m15.index <= t + pd.Timedelta(hours=2))]
            if sub.empty:
                continue

            if side == "LONG":
                hits = sub.index[sub.micro_bull.fillna(False)]
            else:
                hits = sub.index[sub.micro_bear.fillna(False)]

            if len(hits) == 0:
                d["no_15m_trigger"] += 1
                continue

            trigger_time = hits[0]
            entry_time = trigger_time + pd.Timedelta(minutes=15)
            if entry_time not in seg.index:
                d["entry_bar_missing"] += 1
                continue

            entry_raw = float(seg.loc[entry_time, "open"])
            entry = entry_raw * (1 + SLIPPAGE) if side == "LONG" else entry_raw * (1 - SLIPPAGE)

            # Structural SL: beyond the actual liquidity sweep extreme.
            buffer = 0.15 * float(row.atr)
            sl = sweep_extreme - buffer if side == "LONG" else sweep_extreme + buffer
            risk = entry - sl if side == "LONG" else sl - entry

            if risk <= 0:
                d["invalid_risk"] += 1
                continue

            stop_pct = risk / entry
            if stop_pct < MIN_STOP_PCT or stop_pct > MAX_STOP_PCT:
                d["stop_size_rejected"] += 1
                continue

            tp = entry + RR * risk if side == "LONG" else entry - RR * risk

            d["final_candidate_before_dedupe"] += 1
            candidates.append({
                "asset": asset,
                "side": side,
                "setup_time": t,
                "signal_time": trigger_time,
                "entry_time": entry_time,
                "entry": float(entry),
                "sl": float(sl),
                "tp": float(tp),
                "risk": float(risk),
                "stop_pct": float(stop_pct),
                "h1_bos": bos_seen,
                "h1_fvg": fvg_seen,
                "cluster": CLUSTERS[asset],
            })

    # One signal per asset/side/entry candle.
    uniq = {}
    for c in candidates:
        key = (c["asset"], c["entry_time"], c["side"])
        if key not in uniq:
            uniq[key] = c

    out = sorted(uniq.values(), key=lambda z: z["entry_time"])
    d["final_candidates"] += len(out)
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(d)
    return out


# ============================================================
# TRADE SIMULATION
# ============================================================

def adverse_exit(price, side):
    return price * (1 - SLIPPAGE) if side == "LONG" else price * (1 + SLIPPAGE)


def simulate(candidate, df):
    entry_time = candidate["entry_time"]
    if entry_time not in df.index:
        return None

    pos = df.index.get_loc(entry_time)
    for j in range(pos, len(df)):
        bar = df.iloc[j]
        hi, lo = float(bar.high), float(bar.low)

        if candidate["side"] == "LONG":
            sl_hit = lo <= candidate["sl"]
            tp_hit = hi >= candidate["tp"]
        else:
            sl_hit = hi >= candidate["sl"]
            tp_hit = lo <= candidate["tp"]

        if sl_hit and tp_hit:
            exit_raw = candidate["sl"]
            outcome = "LOSS"
            reason = "BOTH_SAME_BAR_SL_FIRST"
        elif sl_hit:
            exit_raw = candidate["sl"]
            outcome = "LOSS"
            reason = "SL"
        elif tp_hit:
            exit_raw = candidate["tp"]
            outcome = "WIN"
            reason = "TP"
        else:
            continue

        exit_price = adverse_exit(exit_raw, candidate["side"])
        notional = MARGIN * LEVERAGE

        if candidate["side"] == "LONG":
            gross = (exit_price - candidate["entry"]) / candidate["entry"] * notional
        else:
            gross = (candidate["entry"] - exit_price) / candidate["entry"] * notional

        fees = (notional + notional * exit_price / candidate["entry"]) * FEE_RATE
        pnl = gross - fees

        return {
            **candidate,
            "exit_time": df.index[j],
            "exit": float(exit_price),
            "outcome": outcome,
            "reason": reason,
            "pnl": float(pnl),
            "bars_held": int(j - pos + 1),
        }

    # Still-open positions are intentionally NOT converted to losses.
    return None


def portfolio_filter(trades):
    accepted = []
    active = []

    for t in sorted(trades, key=lambda z: z["entry_time"]):
        active = [a for a in active if a["exit_time"] > t["entry_time"]]

        if len(active) >= MAX_POSITIONS:
            continue

        if MAX_ONE_PER_CLUSTER and any(a["cluster"] == t["cluster"] for a in active):
            continue

        accepted.append(t)
        active.append(t)

    return accepted


def streaks(outcomes):
    out = []
    n = 0
    for o in outcomes:
        if o == "LOSS":
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(data_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    all_candidates = []
    all_closed = []
    reports = {}
    diagnostics = defaultdict(int)
    frames = {}

    for asset in SYMBOLS:
        df = load_csv(data_dir, asset)
        frames[asset] = df

        diffs = df.index.to_series().diff().dt.total_seconds().div(900)
        missing = int((diffs - 1).clip(lower=0).sum())

        reports[asset] = {
            "rows": len(df),
            "start": df.index.min().isoformat(),
            "end": df.index.max().isoformat(),
            "missing_15m_bars": missing,
        }

        cands = generate_candidates(asset, df, diagnostics)
        all_candidates.extend(cands)

    # Simulate each candidate before portfolio filtering.
    for c in all_candidates:
        t = simulate(c, frames[c["asset"]])
        if t is not None:
            all_closed.append(t)

    accepted = portfolio_filter(all_closed)
    trades = pd.DataFrame(accepted)

    if trades.empty:
        trades = pd.DataFrame(columns=[
            "asset", "side", "setup_time", "signal_time", "entry_time",
            "exit_time", "entry", "exit", "sl", "tp", "risk",
            "outcome", "reason", "pnl",
        ])

    outcomes = trades["outcome"].tolist() if len(trades) else []
    wins = sum(x == "WIN" for x in outcomes)
    losses = sum(x == "LOSS" for x in outcomes)

    gross_win = float(trades.loc[trades.outcome == "WIN", "pnl"].sum()) if len(trades) else 0.0
    gross_loss = abs(float(trades.loc[trades.outcome == "LOSS", "pnl"].sum())) if len(trades) else 0.0
    pf = gross_win / gross_loss if gross_loss else math.inf

    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    for p in trades["pnl"].tolist() if len(trades) else []:
        equity += float(p)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    ls = streaks(outcomes)
    starts = [v["start"] for v in reports.values()]
    ends = [v["end"] for v in reports.values()]
    days = (pd.Timestamp(max(ends)) - pd.Timestamp(min(starts))).total_seconds() / 86400

    per_symbol = {}
    for asset in SYMBOLS:
        q = trades[trades.asset == asset] if len(trades) else trades
        n = len(q)
        w = int((q.outcome == "WIN").sum()) if n else 0
        per_symbol[asset] = {
            "trades": n,
            "win_rate": (w / n * 100) if n else 0.0,
            "pnl": float(q.pnl.sum()) if n else 0.0,
        }

    report = {
        "strategy": "HUNTER-V4B-R1 H1 Liquidity + Displacement (4H regime diagnostic removed)",
        "raw_candidates": len(all_candidates),
        "simulated_closed_candidates": len(all_closed),
        "accepted_closed_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(trades) * 100 if len(trades) else 0.0,
        "profit_factor": pf,
        "net_pnl": float(trades.pnl.sum()) if len(trades) else 0.0,
        "final_equity": equity,
        "max_drawdown": max_dd,
        "max_loss_streak": max(ls) if ls else 0,
        "loss_streak_list": ls,
        "trades_per_day": len(trades) / days if days else 0.0,
        "data_audit": reports,
        "diagnostics": dict(diagnostics),
        "per_symbol": per_symbol,
    }

    trades.to_csv(out_dir / "TRADES.csv", index=False)
    (out_dir / "BACKTEST_REPORT.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    print("=" * 90)
    print("HUNTER-V4B-R2 — TRIGGER OPTIMIZATION / NO 4H HARD REGIME / XT USDT-M FUTURES / 15m / 365-DAY BACKTEST")
    print("=" * 90)
    print(f"Raw candidates          : {len(all_candidates)}")
    print(f"Simulated closed        : {len(all_closed)}")
    print(f"Accepted closed trades  : {len(trades)}")
    print(f"Wins / Losses           : {wins} / {losses}")
    print(f"Win Rate                : {report['win_rate']:.2f}%")
    print(f"Profit Factor           : {pf:.3f}")
    print(f"Net PnL                 : ${report['net_pnl']:.2f}")
    print(f"Final equity            : ${equity:.2f}")
    print(f"Max Drawdown            : ${max_dd:.2f}")
    print(f"Max loss streak         : {report['max_loss_streak']}")
    print(f"Trades / day            : {report['trades_per_day']:.2f}")
    print("-" * 90)
    print("SIGNAL PIPELINE DIAGNOSTICS")
    for k in sorted(diagnostics):
        print(f"{k:32}: {diagnostics[k]}")
    print("-" * 90)
    print("Reports written to:", out_dir)

    return report


# ============================================================
# DATA DOWNLOAD WRAPPER
# ============================================================

def ensure_xt_data(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    expected = [data_dir / f"{a}_USDT_15m.csv" for a in SYMBOLS]
    if all(p.exists() for p in expected):
        print("XT data: all 14 CSV files already present; skipping download.")
        return

    print("XT data: missing CSVs detected; downloading 365 days...")
    session = requests.Session()
    session.headers.update({"User-Agent": "HUNTER-V4B-XT-Futures/1.0"})

    discovered = discover_symbols(session)
    resolved = {a: resolve_symbol(a, discovered) for a in SYMBOLS}
    missing = [a for a, v in resolved.items() if v is None]
    if missing:
        raise RuntimeError("Unresolved XT Futures symbols: " + ", ".join(missing))

    end_dt = now_utc()
    start_dt = end_dt - timedelta(days=365)

    for asset in SYMBOLS:
        actual = resolved[asset]
        df, rep = download_symbol(session, actual, start_dt, end_dt)
        if rep["rows"] < 30000:
            raise RuntimeError(f"{asset}: suspicious coverage: {rep['rows']} rows")
        out = data_dir / f"{asset}_USDT_15m.csv"
        df.to_csv(out, index=False)
        print(
            f"{asset}: rows={rep['rows']:,} gaps={rep['gaps']} "
            f"max_gap={rep['max_gap_minutes']}m"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--download", action="store_true",
                   help="Download XT data when CSVs are missing (kept for compatibility).")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    # Always ensure data exists. --download is retained for backwards compatibility.
    ensure_xt_data(data_dir)
    run_backtest(data_dir, out_dir)


if __name__ == "__main__":
    main()
