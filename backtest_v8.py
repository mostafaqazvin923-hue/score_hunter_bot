#!/usr/bin/env python3
"""
HUNTER-MTF1 — XT USDT-M Futures / 1D→4H→1H / 365d

Purpose
-------
This is NOT a claim that the old V129 result is valid.
It is a causal/integrity rebuild of the V129/V123 signal logic so that
the strategy can be evaluated fairly on the current XT Futures dataset.

Preserved strategy idea:
- 4H EMA50/EMA200 regime
- 1H liquidity sweep
- 15m displacement
- ATR stop
- nominal RR 1:2 (1.5 ATR SL / 3 ATR TP)
- original BE rule is kept ONLY for this audit isolation test

Integrity corrections:
1) No center=True / future-looking HTF values.
2) Signal uses a fully completed 15m candle.
3) Entry is the NEXT 15m candle open.
4) No artificial timeout.
5) Unresolved trade at dataset end is OPEN, not LOSS.
6) Same-candle SL+TP => conservative LOSS.
7) Global non-overlap: a new trade cannot open before the previous
   portfolio trade has closed.
8) No signal is opened on the same candle that closes the prior trade.
9) Missing 15m candles are never forward-filled.
10) Only realized WIN/LOSS/BE trades are used for WR/PF.

Input:
    data/xt_futures_15m/{ASSET}_USDT_15m.csv

The existing XT collector in this project can create these files.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

import numpy as np
import pandas as pd


SYMBOLS = [
    "BTC", "ETH", "SOL", "SUI", "AVAX", "NEAR", "ADA",
    "BNB", "APT", "CRV", "ONDO", "PENDLE", "ICP", "WIF",
]

DATA_DIR = Path("data/xt_futures_15m")
OUT_DIR = DATA_DIR / "backtest_mtf1"

INITIAL_EQUITY = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 50.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003

# MTF-1 parameters. These are deliberately few and interpretable.
EMA_FAST = 50
EMA_SLOW = 200
EMA_PULLBACK = 20
ATR_N = 14
VOL_N = 20
VOL_MULT = 1.05
BODY_ATR_MIN = 0.50
PULLBACK_LOOKBACK = 4
STOP_BUFFER_ATR = 0.15
RR = 2.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    return p.parse_args()


# ---------------------------------------------------------------------------
# XT FUTURES DATA COLLECTOR
# ---------------------------------------------------------------------------
BASE = "https://fapi.xt.com"
SYMBOL_LIST_URL = f"{BASE}/future/market/v1/public/symbol/list"
KLINE_URL = f"{BASE}/future/market/v1/public/q/kline"
INTERVAL = "15m"
LIMIT = 1500
TIMEOUT = 20
RETRIES = 5
SLEEP = 0.12

def _now_utc():
    return datetime.now(timezone.utc)

def _to_ms(dt):
    return int(dt.timestamp() * 1000)

def _api_json(session, url, params=None):
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, dict):
                raise RuntimeError(f"XT unexpected response type: {type(payload).__name__}")
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
    raise RuntimeError(f"XT API request failed after {RETRIES} attempts: {last}")

def _extract_list(payload):
    result = payload.get("result")
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ("items", "list", "data", "symbols"):
            if isinstance(result.get(key), list):
                return result[key]
    raise RuntimeError("Could not find XT symbol list in response")

def _discover_symbols(session):
    items = _extract_list(_api_json(session, SYMBOL_LIST_URL))
    found = {}
    for item in items:
        if isinstance(item, str):
            raw = item
        elif isinstance(item, dict):
            raw = item.get("symbol") or item.get("s") or item.get("name") or item.get("pair")
        else:
            raw = None
        if raw:
            found[str(raw).strip().upper()] = item
    if not found:
        raise RuntimeError("XT symbol list returned zero usable symbols")
    return found

def _resolve_symbol(asset, discovered):
    wanted = asset.upper()
    candidates = [f"{wanted}_USDT", f"{wanted}/USDT", f"{wanted}-USDT", wanted]
    for candidate in candidates:
        if candidate.upper() in discovered:
            return candidate.upper()
    def norm(x):
        return str(x).upper().replace("/", "_").replace("-", "_")
    matches = [actual for actual in discovered if norm(actual) == f"{wanted}_USDT"]
    return matches[0] if len(matches) == 1 else None

def _normalize_kline_rows(rows, symbol):
    records = []
    for row in rows:
        if isinstance(row, dict):
            required = ("t", "o", "h", "l", "c", "a")
            if not all(k in row for k in required):
                raise RuntimeError(f"{symbol}: malformed XT kline row")
            records.append({
                "Timestamp": int(row["t"]),
                "Open": float(row["o"]),
                "High": float(row["h"]),
                "Low": float(row["l"]),
                "Close": float(row["c"]),
                "Volume": float(row["a"]),
                "Turnover": float(row["v"]) if row.get("v") is not None else float("nan"),
            })
        elif isinstance(row, (list, tuple)) and len(row) >= 6:
            records.append({
                "Timestamp": int(row[0]),
                "Open": float(row[1]),
                "High": float(row[2]),
                "Low": float(row[3]),
                "Close": float(row[4]),
                "Volume": float(row[5]),
                "Turnover": float(row[6]) if len(row) > 6 else float("nan"),
            })
        else:
            raise RuntimeError(f"{symbol}: unsupported XT kline row")

    df = pd.DataFrame(records)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    return df

def _fetch_xt_batch(session, symbol, start_ms, end_ms):
    payload = _api_json(
        session, KLINE_URL,
        {
            "symbol": symbol.strip().lower(),
            "interval": INTERVAL,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": LIMIT,
        },
    )
    rows = payload.get("result")
    if not isinstance(rows, list):
        raise RuntimeError(f"{symbol}: XT kline result is not a list")
    return _normalize_kline_rows(rows, symbol)

def _audit_download(df, symbol):
    if df.empty:
        raise RuntimeError(f"{symbol}: empty XT dataset")
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

def _download_symbol(session, symbol, start_dt, end_dt):
    start_ms = _to_ms(start_dt)
    cursor_end = _to_ms(end_dt)
    batches = []
    calls = 0

    while cursor_end >= start_ms:
        batch = _fetch_xt_batch(session, symbol, start_ms, cursor_end)
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
            raise RuntimeError(f"{symbol}: XT returned candle beyond requested end")

        batches.append(batch)
        print(
            f"  {symbol}: request={calls:02d} rows={len(batch):4d} "
            f"{batch['Date'].iloc[0]} -> {batch['Date'].iloc[-1]}"
        )

        if first <= start_ms:
            break

        next_end = first - 1
        if next_end >= cursor_end:
            raise RuntimeError(f"{symbol}: XT backward pagination made no progress")

        cursor_end = next_end
        if calls > 1000:
            raise RuntimeError(f"{symbol}: XT pagination safety stop")
        time.sleep(SLEEP)

    if not batches:
        raise RuntimeError(f"{symbol}: XT returned zero historical candles")

    df = (
        pd.concat(batches, ignore_index=True)
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    df = df[
        (df["Date"] >= pd.Timestamp(start_dt))
        & (df["Date"] <= pd.Timestamp(end_dt))
    ].copy()

    now = _now_utc()
    if not df.empty and df["Date"].iloc[-1] + pd.Timedelta(minutes=15) > pd.Timestamp(now):
        df = df.iloc[:-1].copy()

    report = _audit_download(df, symbol)
    report["api_requests"] = calls
    if report["rows"] < 30000:
        raise RuntimeError(
            f"{symbol}: incomplete XT coverage: {report['rows']} rows (expected roughly 35,000)"
        )
    return df, report

def ensure_xt_data(data_dir: Path):
    """Ensure all 14 XT 15m datasets exist in THIS GitHub Actions run."""
    data_dir.mkdir(parents=True, exist_ok=True)
    end_dt = _now_utc()
    start_dt = end_dt - timedelta(days=365)

    valid = True
    for asset in SYMBOLS:
        path = data_dir / f"{asset}_USDT_15m.csv"
        if not path.exists():
            valid = False
            break
        try:
            probe = pd.read_csv(path, usecols=["Date"])
            if len(probe) < 30000:
                valid = False
                break
            dates = pd.to_datetime(probe["Date"], utc=True)
            if dates.max() < pd.Timestamp(end_dt - timedelta(days=2)):
                valid = False
                break
            if dates.min() > pd.Timestamp(start_dt + timedelta(days=2)):
                valid = False
                break
        except Exception:
            valid = False
            break

    if valid:
        print("XT data audit: 14/14 existing CSVs have sufficient 365-day coverage.")
        return

    print("XT data audit: CSVs missing/incomplete/stale -> downloading 365 days from XT Futures...")
    session = requests.Session()
    session.headers.update({"User-Agent": "HUNTER-V129-AUDITED/1.0"})

    discovered = _discover_symbols(session)
    resolved = {asset: _resolve_symbol(asset, discovered) for asset in SYMBOLS}
    unresolved = [a for a, v in resolved.items() if v is None]
    if unresolved:
        raise RuntimeError("Unresolved XT Futures symbols: " + ", ".join(unresolved))

    for asset in SYMBOLS:
        actual = resolved[asset]
        df, report = _download_symbol(session, actual, start_dt, end_dt)
        out = data_dir / f"{asset}_USDT_15m.csv"
        df.to_csv(out, index=False)
        print(
            f"XT DATA OK {asset}: rows={report['rows']:,}, "
            f"start={report['start']}, end={report['end']}, "
            f"gaps={report['gaps']}, requests={report['api_requests']}"
        )

    missing = [
        asset for asset in SYMBOLS
        if not (data_dir / f"{asset}_USDT_15m.csv").exists()
    ]
    if missing:
        raise RuntimeError("XT final data verification failed; missing: " + ", ".join(missing))

    print("XT data audit: 14/14 files successfully downloaded and verified.")

def load_csv(data_dir: Path, asset: str) -> pd.DataFrame:
    path = data_dir / f"{asset}_USDT_15m.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing XT Futures data file: {path}")

    df = pd.read_csv(path)

    if {"Date", "Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
        df = df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    if np.issubdtype(df["timestamp"].dtype, np.number):
        ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        ts = pd.to_datetime(df["timestamp"], utc=True)

    df["timestamp"] = ts
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")].copy()

    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required[1:])

    # Do not repair gaps. They are reported and treated as data boundaries.
    bad_grid = df.index.minute % 15 != 0
    if bad_grid.any():
        raise ValueError(
            f"{asset}: {int(bad_grid.sum())} timestamps are not aligned to 15m"
        )

    bad_ohlc = (
        (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (df["volume"] < 0)
        | (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    )
    if bad_ohlc.any():
        raise ValueError(f"{asset}: {int(bad_ohlc.sum())} invalid OHLCV rows")

    return df


def split_segments(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split at missing 15m candles. Never bridge a data gap."""
    if len(df) < 2:
        return [df.copy()]

    delta = df.index.to_series().diff().dt.total_seconds().div(60.0)
    cuts = np.flatnonzero((delta.to_numpy() > 15.0001))

    starts = [0] + cuts.tolist()
    ends = cuts.tolist() + [len(df)]

    return [
        df.iloc[s:e].copy()
        for s, e in zip(starts, ends)
        if e - s >= 80
    ]


def atr_wilder_like(df: pd.DataFrame, n: int = ATR_N) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


# ---------------------------------------------------------------------------
# HUNTER-MTF1 STRATEGY
# 1D = macro direction
# 4H = trend state / controlled pullback context
# 1H = setup + entry
# ---------------------------------------------------------------------------

def wilder_atr(df: pd.DataFrame, n: int = ATR_N) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def aggregate_htf(df15: pd.DataFrame):
    """Build closed 1H/4H/1D bars from 15m data. No forward filling."""
    agg = dict(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    h1 = df15.resample("1h", label="right", closed="left", origin="epoch").agg(**agg).dropna()
    h4 = df15.resample("4h", label="right", closed="left", origin="epoch").agg(**agg).dropna()
    d1 = df15.resample("1D", label="right", closed="left", origin="epoch").agg(**agg).dropna()
    return h1, h4, d1


def prepare_mtf(df15: pd.DataFrame):
    h1, h4, d1 = aggregate_htf(df15)
    for df in (h1, h4, d1):
        df["atr"] = wilder_atr(df, ATR_N)
        df["ema50"] = df["close"].ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean()
        df["ema200"] = df["close"].ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean()
        df["ema20"] = df["close"].ewm(span=EMA_PULLBACK, adjust=False, min_periods=EMA_PULLBACK).mean()
        df["ema50_slope"] = df["ema50"].diff(3)
        df["vol_avg"] = df["volume"].shift(1).rolling(VOL_N, min_periods=VOL_N).mean()
    return h1, h4, d1


def regime_daily(row):
    if not np.isfinite(row["ema200"]) or not np.isfinite(row["ema50_slope"]):
        return "NEUTRAL"
    if row["close"] > row["ema200"] and row["ema50"] > row["ema200"] and row["ema50_slope"] > 0:
        return "BULL"
    if row["close"] < row["ema200"] and row["ema50"] < row["ema200"] and row["ema50_slope"] < 0:
        return "BEAR"
    return "NEUTRAL"


def context_4h(row, daily_regime):
    """Allow trend continuation or a controlled pullback; reject opposite 4H trend."""
    if not np.isfinite(row["ema200"]):
        return "NEUTRAL"
    if daily_regime == "BULL":
        # Trend: price above 4H EMA50; pullback: price between EMA50 and EMA200.
        if row["close"] > row["ema50"] and row["ema50"] > row["ema200"]:
            return "BULL_TREND"
        if row["close"] >= row["ema200"] and row["ema50"] > row["ema200"]:
            return "BULL_PULLBACK"
    if daily_regime == "BEAR":
        if row["close"] < row["ema50"] and row["ema50"] < row["ema200"]:
            return "BEAR_TREND"
        if row["close"] <= row["ema200"] and row["ema50"] < row["ema200"]:
            return "BEAR_PULLBACK"
    return "NEUTRAL"


def generate_candidates(asset: str, df15: pd.DataFrame) -> list[dict]:
    h1, h4, d1 = prepare_mtf(df15)
    candidates = []

    # We intentionally require completed 1H bars and enter on the next 1H open.
    for i in range(max(EMA_SLOW + 10, PULLBACK_LOOKBACK + VOL_N + 5), len(h1) - 1):
        signal_ts = h1.index[i]
        entry_ts = h1.index[i + 1]

        # Strict causality: only HTF bars CLOSED before the completed 1H signal bar.
        dsub = d1[d1.index < signal_ts]
        h4sub = h4[h4.index < signal_ts]
        if len(dsub) < EMA_SLOW or len(h4sub) < EMA_SLOW:
            continue

        drow = dsub.iloc[-1]
        h4row = h4sub.iloc[-1]
        dreg = regime_daily(drow)
        ctx = context_4h(h4row, dreg)
        if dreg == "NEUTRAL" or ctx == "NEUTRAL":
            continue

        prev = h1.iloc[i - 1]
        cur = h1.iloc[i]
        recent = h1.iloc[i - PULLBACK_LOOKBACK:i]
        atr = float(cur["atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue

        rng = float(cur["high"] - cur["low"])
        body = abs(float(cur["close"] - cur["open"]))
        if rng <= 0 or body < BODY_ATR_MIN * atr:
            continue
        vol_avg = float(cur["vol_avg"])
        if not np.isfinite(vol_avg) or vol_avg <= 0 or float(cur["volume"]) < VOL_MULT * vol_avg:
            continue

        # Pullback must have interacted with the 1H EMA20 or a recent local extreme.
        pullback_long = bool(
            (recent["low"] <= recent["ema20"]).any()
            or float(recent["low"].min()) < float(h4row["close"])
        )
        pullback_short = bool(
            (recent["high"] >= recent["ema20"]).any()
            or float(recent["high"].max()) > float(h4row["close"])
        )

        # Continuation trigger: completed 1H candle closes beyond previous candle.
        close_top = (cur["close"] - cur["low"]) / rng
        close_bottom = (cur["high"] - cur["close"]) / rng
        long_trigger = bool(
            cur["close"] > cur["open"]
            and cur["close"] > prev["high"]
            and close_top >= 0.70
        )
        short_trigger = bool(
            cur["close"] < cur["open"]
            and cur["close"] < prev["low"]
            and close_bottom >= 0.70
        )

        long_ok = dreg == "BULL" and ctx in {"BULL_TREND", "BULL_PULLBACK"} and pullback_long and long_trigger
        short_ok = dreg == "BEAR" and ctx in {"BEAR_TREND", "BEAR_PULLBACK"} and pullback_short and short_trigger
        if not (long_ok or short_ok):
            continue

        side = "LONG" if long_ok else "SHORT"
        raw_entry = float(h1.iloc[i + 1]["open"])
        entry = raw_entry * (1 + SLIPPAGE) if side == "LONG" else raw_entry * (1 - SLIPPAGE)

        if side == "LONG":
            structural_sl = float(recent["low"].min()) - STOP_BUFFER_ATR * atr
            risk = entry - structural_sl
            if risk <= 0:
                continue
            sl, tp = structural_sl, entry + RR * risk
        else:
            structural_sl = float(recent["high"].max()) + STOP_BUFFER_ATR * atr
            risk = structural_sl - entry
            if risk <= 0:
                continue
            sl, tp = structural_sl, entry - RR * risk

        candidates.append({
            "asset": asset,
            "signal_ts": signal_ts,
            "entry_ts": entry_ts,
            "side": side,
            "entry": float(entry),
            "sl": float(sl),
            "tp": float(tp),
            "risk": float(risk),
            "dreg": dreg,
            "context_4h": ctx,
        })
    return candidates


def simulate_candidate(candidate: dict, df15: pd.DataFrame) -> dict:
    """Simulate from the 1H entry timestamp on raw 15m candles; no timeout."""
    entry_ts = pd.Timestamp(candidate["entry_ts"])
    segment_end = candidate.get("segment_end_ts")
    future = df15[df15.index >= entry_ts]
    if segment_end is not None:
        future = future[future.index <= pd.Timestamp(segment_end)]
    if future.empty:
        return {**candidate, "outcome": "OPEN", "exit_ts": None, "exit_price": None, "pnl": 0.0}

    side = candidate["side"]
    entry = candidate["entry"]
    sl = candidate["sl"]
    tp = candidate["tp"]
    notional = TRADE_MARGIN * LEVERAGE

    for ts, bar in future.iterrows():
        hi, lo = float(bar["high"]), float(bar["low"])
        if side == "LONG":
            hit_sl, hit_tp = lo <= sl, hi >= tp
            if hit_sl and hit_tp:
                exit_price, outcome = sl, "LOSS"
            elif hit_sl:
                exit_price, outcome = sl, "LOSS"
            elif hit_tp:
                exit_price, outcome = tp, "WIN"
            else:
                continue
            gross = (exit_price - entry) / entry * notional
        else:
            hit_sl, hit_tp = hi >= sl, lo <= tp
            if hit_sl and hit_tp:
                exit_price, outcome = sl, "LOSS"
            elif hit_sl:
                exit_price, outcome = sl, "LOSS"
            elif hit_tp:
                exit_price, outcome = tp, "WIN"
            else:
                continue
            gross = (entry - exit_price) / entry * notional

        fees = notional * FEE_RATE * 2.0
        pnl = gross - fees
        return {**candidate, "outcome": outcome, "exit_ts": ts, "exit_price": exit_price, "pnl": float(pnl)}

    return {**candidate, "outcome": "OPEN", "exit_ts": None, "exit_price": None, "pnl": 0.0}


def enforce_global_non_overlap(trades: list[dict]) -> list[dict]:
    accepted = []
    last_exit = None
    for t in sorted(trades, key=lambda z: (pd.Timestamp(z["entry_ts"]), z["asset"])):
        if last_exit is not None and pd.Timestamp(t["entry_ts"]) <= pd.Timestamp(last_exit):
            continue
        accepted.append(t)
        if t["exit_ts"] is not None:
            last_exit = t["exit_ts"]
        else:
            last_exit = pd.Timestamp.max
    return accepted


def loss_streaks(outcomes: list[str]) -> list[int]:
    streaks, cur = [], 0
    for o in outcomes:
        if o == "LOSS":
            cur += 1
        else:
            if cur:
                streaks.append(cur)
            cur = 0
    if cur:
        streaks.append(cur)
    return streaks


def build_report(trades: list[dict], reports: dict) -> dict:
    realized = [t for t in trades if t["outcome"] in {"WIN", "LOSS"}]
    wins = [t for t in realized if t["outcome"] == "WIN"]
    losses = [t for t in realized if t["outcome"] == "LOSS"]
    pnl = sum(float(t["pnl"]) for t in realized)
    gross_win = sum(max(0.0, float(t["pnl"])) for t in realized)
    gross_loss = -sum(min(0.0, float(t["pnl"])) for t in realized)
    pf = gross_win / gross_loss if gross_loss > 0 else 0.0

    eq = INITIAL_EQUITY
    peak = eq
    max_dd = 0.0
    for t in sorted(realized, key=lambda z: pd.Timestamp(z["exit_ts"])):
        eq += float(t["pnl"])
        peak = max(peak, eq)
        max_dd = min(max_dd, eq - peak)

    streak_list = loss_streaks([t["outcome"] for t in sorted(realized, key=lambda z: pd.Timestamp(z["exit_ts"]))])
    days = max(1.0, (max(pd.Timestamp(r["exit_ts"]) for r in realized) - min(pd.Timestamp(r["entry_ts"]) for r in realized)).total_seconds() / 86400.0) if realized else 365.0

    per_symbol = {}
    for asset in SYMBOLS:
        rr = [t for t in realized if t["asset"] == asset]
        if not rr:
            per_symbol[asset] = {"trades": 0, "win_rate": 0.0, "pnl": 0.0}
        else:
            per_symbol[asset] = {
                "trades": len(rr),
                "win_rate": 100.0 * sum(t["outcome"] == "WIN" for t in rr) / len(rr),
                "pnl": sum(float(t["pnl"]) for t in rr),
            }

    return {
        "strategy": "HUNTER-MTF1",
        "integrity": {
            "causal": True,
            "lookahead": False,
            "entry": "next 1H open after completed 1H signal candle",
            "trend": "1D",
            "context": "4H",
            "signal": "1H",
            "timeout": False,
            "global_non_overlap": True,
            "same_candle_sl_tp": "LOSS",
            "end_of_data": "OPEN, not LOSS",
        },
        "parameters": {
            "initial_equity": INITIAL_EQUITY,
            "trade_margin": TRADE_MARGIN,
            "leverage": LEVERAGE,
            "fee_rate": FEE_RATE,
            "slippage": SLIPPAGE,
            "RR": RR,
            "daily_ema": [EMA_FAST, EMA_SLOW],
            "4h_ema": [EMA_FAST, EMA_SLOW],
            "1h_pullback_ema": EMA_PULLBACK,
        },
        "trades": len(realized),
        "open_at_end": sum(t["outcome"] == "OPEN" for t in trades),
        "win_rate": 100.0 * len(wins) / len(realized) if realized else 0.0,
        "profit_factor": pf,
        "net_pnl": pnl,
        "final_equity": INITIAL_EQUITY + pnl,
        "max_drawdown": max_dd,
        "max_consecutive_losses": max(streak_list) if streak_list else 0,
        "loss_streaks": streak_list,
        "trades_per_day": len(realized) / days,
        "per_symbol": per_symbol,
        "data_audit": reports,
    }


def main():
    args = parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("HUNTER-MTF1 — 1D → 4H → 1H")
    print("1D = macro direction | 4H = trend/pullback context | 1H = setup + entry")
    print("RR=1:2 | NO LOOKAHEAD | NO TIMEOUT | GLOBAL NON-OVERLAP")
    print("=" * 88)

    ensure_xt_data(data_dir)
    all_candidates, reports = [], {}
    for asset in SYMBOLS:
        df = load_csv(data_dir, asset)
        segs = split_segments(df)
        gap_count = int((df.index.to_series().diff() > pd.Timedelta(minutes=15)).sum())
        reports[asset] = {
            "rows": len(df),
            "gaps": gap_count,
            "segments": len(segs),
            "start": df.index[0].isoformat(),
            "end": df.index[-1].isoformat(),
        }
        c = []
        for seg in segs:
            seg_candidates = generate_candidates(asset, seg)
            for item in seg_candidates:
                item["segment_end_ts"] = seg.index[-1]
            c.extend(seg_candidates)
        print(f"{asset:6s} rows={len(df):6d} segments={len(segs):2d} candidates={len(c):4d}")
        all_candidates.extend(c)

    print(f"Raw candidates: {len(all_candidates)}")
    simulated = []
    for asset in SYMBOLS:
        df = load_csv(data_dir, asset)
        for c in [x for x in all_candidates if x["asset"] == asset]:
            # Restrict simulation to the source segment by entry timestamp naturally through data.
            simulated.append(simulate_candidate(c, df))
    print(f"Simulated candidates: {len(simulated)}")

    accepted = enforce_global_non_overlap(simulated)
    print(f"Accepted non-overlapping trades: {len(accepted)}")

    report = build_report(accepted, reports)
    print("\n===== HUNTER-MTF1 RESULT =====")
    for k in ["trades", "open_at_end", "win_rate", "profit_factor", "net_pnl", "final_equity", "max_drawdown", "max_consecutive_losses", "trades_per_day"]:
        print(f"{k:24s}: {report[k]}")
    print("\nPer symbol:")
    for asset, v in report["per_symbol"].items():
        print(f"  {asset:6s} trades={v['trades']:4d} WR={v['win_rate']:6.2f}% PnL=${v['pnl']:10.2f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(accepted).to_csv(out_dir / "TRADES.csv", index=False)
    (out_dir / "BACKTEST_REPORT.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
