#!/usr/bin/env python3
"""
HUNTER-V129-AUDITED — XT USDT-M Futures / 15m / 365d

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
OUT_DIR = DATA_DIR / "backtest_v129_audited"

INITIAL_EQUITY = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 50.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003

ATR_N = 14
BODY_N = 20

SL_ATR = 1.5
TP_ATR = 3.0
BE_ATR = 1.5

RR = TP_ATR / SL_ATR


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


def prepare_htf(df15: pd.DataFrame):
    """
    Resampling is explicitly causal:
    HTF bars are right-labelled and a 15m signal at time T may only use
    HTF bars whose closing timestamp is STRICTLY before T.
    """

    x = df15.copy()

    x["atr15"] = atr_wilder_like(x, ATR_N)
    x["body"] = (x["close"] - x["open"]).abs()

    # Prior completed 20-bar average only.
    x["avg_body"] = x["body"].shift(1).rolling(
        BODY_N, min_periods=BODY_N
    ).mean()

    h1 = (
        x.resample(
            "1h",
            label="right",
            closed="right",
            origin="epoch",
        )
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna()
    )

    h4 = (
        x.resample(
            "4h",
            label="right",
            closed="right",
            origin="epoch",
        )
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna()
    )

    h4["ema50"] = h4["close"].ewm(
        span=50, adjust=False, min_periods=50
    ).mean()
    h4["ema200"] = h4["close"].ewm(
        span=200, adjust=False, min_periods=200
    ).mean()

    h4["bull"] = (
        (h4["close"] > h4["ema200"])
        & (h4["ema50"] > h4["ema200"])
    )
    h4["bear"] = (
        (h4["close"] < h4["ema200"])
        & (h4["ema50"] < h4["ema200"])
    )

    return x, h1, h4


def generate_candidates(asset: str, df15: pd.DataFrame) -> list[dict]:
    """
    Candidate timestamp = the COMPLETED displacement candle.
    Actual entry occurs on the next 15m candle open.
    """

    x, h1, h4 = prepare_htf(df15)
    candidates = []

    for i in range(max(250, BODY_N + 5), len(x) - 1):
        signal_ts = x.index[i]
        entry_i = i + 1

        # Critical causal rule:
        # HTF candles must have closed before the signal candle opened.
        h1_sub = h1[h1.index < signal_ts]
        h4_sub = h4[h4.index < signal_ts]

        if len(h1_sub) < 10 or len(h4_sub) < 1:
            continue

        regime = h4_sub.iloc[-1]
        if pd.isna(regime["ema200"]):
            continue

        recent_lows = h1_sub["low"].iloc[-10:-2]
        recent_highs = h1_sub["high"].iloc[-10:-2]

        if len(recent_lows) == 0 or len(recent_highs) == 0:
            continue

        support = float(recent_lows.min())
        resistance = float(recent_highs.max())

        # Sweep is the PREVIOUS completed 15m candle.
        prev = x.iloc[i - 1]
        curr = x.iloc[i]

        sweep_low = (
            prev["low"] < support
            and prev["close"] > support
        )
        sweep_high = (
            prev["high"] > resistance
            and prev["close"] < resistance
        )

        # Displacement is the completed signal candle.
        displacement_up = (
            curr["close"] > curr["open"]
            and curr["body"] > 2.0 * curr["avg_body"]
        )
        displacement_down = (
            curr["close"] < curr["open"]
            and curr["body"] > 2.0 * curr["avg_body"]
        )

        long_ok = bool(regime["bull"] and sweep_low and displacement_up)
        short_ok = bool(regime["bear"] and sweep_high and displacement_down)

        if not long_ok and not short_ok:
            continue

        side = "LONG" if long_ok else "SHORT"

        entry_bar = x.iloc[entry_i]
        raw_entry = float(entry_bar["open"])

        if side == "LONG":
            entry = raw_entry * (1.0 + SLIPPAGE)
        else:
            entry = raw_entry * (1.0 - SLIPPAGE)

        atr = float(curr["atr15"])
        if not np.isfinite(atr) or atr <= 0:
            continue

        if side == "LONG":
            sl = entry - SL_ATR * atr
            tp = entry + TP_ATR * atr
            be_trigger = entry + BE_ATR * atr
        else:
            sl = entry + SL_ATR * atr
            tp = entry - TP_ATR * atr
            be_trigger = entry - BE_ATR * atr

        candidates.append(
            {
                "asset": asset,
                "signal_ts": signal_ts,
                "entry_ts": x.index[entry_i],
                "entry_i": entry_i,
                "side": side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "be_trigger": be_trigger,
            }
        )

    return candidates


def simulate_candidate(candidate: dict, df15: pd.DataFrame) -> dict:
    """Simulate until SL/TP or dataset end. No timeout."""

    i = candidate["entry_i"]
    side = candidate["side"]
    entry = candidate["entry"]
    sl = candidate["sl"]
    tp = candidate["tp"]
    be_trigger = candidate["be_trigger"]

    be_active = False
    exit_i = None
    outcome = "OPEN"
    exit_price = np.nan

    for j in range(i, len(df15)):
        bar = df15.iloc[j]
        high = float(bar["high"])
        low = float(bar["low"])

        if side == "LONG":
            if not be_active and high >= be_trigger:
                be_active = True
                sl = entry

            hit_sl = low <= sl
            hit_tp = high >= tp

            # Conservative priority: SL wins if both occur in one candle.
            if hit_sl:
                outcome = "BE" if be_active else "LOSS"
                exit_price = entry if be_active else sl
                exit_i = j
                break

            if hit_tp:
                outcome = "WIN"
                exit_price = tp
                exit_i = j
                break

        else:
            if not be_active and low <= be_trigger:
                be_active = True
                sl = entry

            hit_sl = high >= sl
            hit_tp = low <= tp

            if hit_sl:
                outcome = "BE" if be_active else "LOSS"
                exit_price = entry if be_active else sl
                exit_i = j
                break

            if hit_tp:
                outcome = "WIN"
                exit_price = tp
                exit_i = j
                break

    notional = TRADE_MARGIN * LEVERAGE

    if outcome == "OPEN":
        pnl = 0.0
        exit_ts = pd.NaT
    else:
        if side == "LONG":
            price_ret = (exit_price - entry) / entry
        else:
            price_ret = (entry - exit_price) / entry

        pnl = (
            notional * price_ret
            - notional * FEE_RATE * 2.0
        )
        exit_ts = df15.index[exit_i]

    result = dict(candidate)
    result.update(
        {
            "exit_ts": exit_ts,
            "exit_i": exit_i,
            "outcome": outcome,
            "exit_price": exit_price,
            "pnl": float(pnl),
            "be_used": bool(be_active),
        }
    )
    return result


def enforce_global_non_overlap(trades: list[dict]) -> list[dict]:
    """
    Portfolio-level lock.

    Candidates are ordered by signal/entry time. Once a trade is accepted,
    no later candidate may enter until the accepted trade has actually closed.
    A candidate on the same candle as the previous exit is rejected.
    """

    accepted = []
    next_allowed_ts = None

    for t in sorted(
        trades,
        key=lambda z: (z["entry_ts"], z["asset"]),
    ):
        if next_allowed_ts is not None:
            if pd.isna(t["entry_ts"]):
                continue
            if t["entry_ts"] <= next_allowed_ts:
                continue

        accepted.append(t)

        if not pd.isna(t["exit_ts"]):
            next_allowed_ts = t["exit_ts"]
        else:
            # An OPEN-at-end trade blocks everything after it.
            next_allowed_ts = pd.Timestamp.max.tz_localize("UTC")

    return accepted


def loss_streaks(outcomes: list[str]) -> list[int]:
    result = []
    n = 0
    for x in outcomes:
        if x == "LOSS":
            n += 1
        else:
            if n:
                result.append(n)
            n = 0
    if n:
        result.append(n)
    return result


def build_report(trades: list[dict], reports: dict) -> dict:
    df = pd.DataFrame(trades)

    if df.empty:
        return {
            "strategy": "HUNTER-V129-AUDITED",
            "trades": 0,
            "realized_trades": 0,
            "open_at_end": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_pnl": 0.0,
            "final_equity": float(INITIAL_EQUITY),
            "max_drawdown": 0.0,
            "max_loss_streak": 0,
            "loss_streak_list": [],
            "trades_per_day": 0.0,
            "data_audit": reports,
        }

    df = df.sort_values("entry_ts").reset_index(drop=True)
    realized = df[df["outcome"] != "OPEN"].copy()

    wins = realized[realized["outcome"] == "WIN"]
    losses = realized[realized["outcome"] == "LOSS"]

    gross_profit = float(wins["pnl"].sum())
    gross_loss = abs(float(losses["pnl"].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else math.inf

    df["cum_pnl"] = df["pnl"].cumsum()
    equity = INITIAL_EQUITY + df["cum_pnl"]
    peak = equity.cummax()
    dd = equity - peak

    streaks = loss_streaks(realized["outcome"].tolist())

    start = df["entry_ts"].min()
    end = df["entry_ts"].max()
    days = (
        (end - start).total_seconds() / 86400.0
        if pd.notna(start) and pd.notna(end) and end > start
        else 0.0
    )

    per_symbol = {}
    for asset in SYMBOLS:
        q = realized[realized["asset"] == asset]
        n = len(q)
        w = int((q["outcome"] == "WIN").sum())
        per_symbol[asset] = {
            "trades": n,
            "win_rate": (w / n * 100.0) if n else 0.0,
            "pnl": float(q["pnl"].sum()) if n else 0.0,
        }

    return {
        "strategy": "HUNTER-V129-AUDITED",
        "rr_nominal": RR,
        "initial_equity": INITIAL_EQUITY,
        "margin": TRADE_MARGIN,
        "leverage": LEVERAGE,
        "trades_including_open": len(df),
        "realized_trades": len(realized),
        "open_at_end": int((df["outcome"] == "OPEN").sum()),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "break_even": int((realized["outcome"] == "BE").sum()),
        "win_rate": (
            len(wins) / len(realized) * 100.0
            if len(realized) else 0.0
        ),
        "profit_factor": pf,
        "net_pnl": float(df["pnl"].sum()),
        "final_equity": float(INITIAL_EQUITY + df["pnl"].sum()),
        "max_drawdown": float(dd.min()),
        "max_loss_streak": max(streaks) if streaks else 0,
        "loss_streak_list": streaks,
        "trades_per_day": (
            len(realized) / days if days > 0 else 0.0
        ),
        "data_audit": reports,
        "per_symbol": per_symbol,
    }


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # GitHub Actions runners are fresh; data must be prepared in the same run.
    ensure_xt_data(data_dir)

    print("=" * 88)
    print("HUNTER-V129-AUDITED — XT USDT-M FUTURES")
    print("=" * 88)
    print("Integrity mode: CAUSAL / NO LOOKAHEAD / NO TIMEOUT / NO OVERLAP")
    print("Entry: next 15m open after completed signal candle")
    print("SL/TP: 1.5 ATR / 3.0 ATR  => nominal RR 1:2")
    print("BE: preserved for audit isolation only")
    print()

    all_candidates = []
    data_reports = {}

    for asset in SYMBOLS:
        df = load_csv(data_dir, asset)

        diffs = df.index.to_series().diff().dropna()
        gaps = diffs[diffs > pd.Timedelta(minutes=15)]

        data_reports[asset] = {
            "rows": int(len(df)),
            "start": df.index[0].isoformat(),
            "end": df.index[-1].isoformat(),
            "gaps": int(len(gaps)),
            "max_gap_minutes": (
                int(gaps.max().total_seconds() / 60.0)
                if len(gaps) else 15
            ),
        }

        print(
            f"{asset:8s} rows={len(df):6d} "
            f"gaps={data_reports[asset]['gaps']:3d}"
        )

        for seg in split_segments(df):
            all_candidates.extend(generate_candidates(asset, seg))

    print()
    print(f"Raw candidates: {len(all_candidates)}")

    simulated = []
    # Re-load by asset once for simulation.
    frames = {asset: load_csv(data_dir, asset) for asset in SYMBOLS}

    for c in all_candidates:
        # Candidate simulation must stay inside its original contiguous segment.
        # Build a segment ending at the candidate entry and extending to dataset end,
        # but do not cross any data gap.
        df = frames[c["asset"]]
        pos = df.index.get_indexer([c["entry_ts"]])[0]
        if pos < 0:
            continue

        # Find segment start by walking backwards across exact 15m intervals.
        start = pos
        while start > 0:
            if df.index[start] - df.index[start - 1] != pd.Timedelta(minutes=15):
                break
            start -= 1

        end = pos + 1
        while end < len(df):
            if df.index[end] - df.index[end - 1] != pd.Timedelta(minutes=15):
                break
            end += 1

        seg = df.iloc[start:end]
        local_i = seg.index.get_indexer([c["entry_ts"]])[0]
        if local_i < 0:
            continue

        c2 = dict(c)
        c2["entry_i"] = local_i
        simulated.append(simulate_candidate(c2, seg))

    print(f"Simulated candidates: {len(simulated)}")

    accepted = enforce_global_non_overlap(simulated)

    print(f"Accepted non-overlapping trades: {len(accepted)}")

    report = build_report(accepted, data_reports)

    trade_df = pd.DataFrame(accepted)
    trade_df.to_csv(out_dir / "TRADES.csv", index=False)

    import json
    (out_dir / "BACKTEST_REPORT.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("===== HUNTER-V129-AUDITED RESULT =====")
    print("=" * 88)
    print(f"Realized trades:       {report['realized_trades']}")
    print(f"Open at dataset end:   {report['open_at_end']}")
    print(f"Win rate:              {report['win_rate']:.2f}%")
    print(f"Profit factor:         {report['profit_factor']:.3f}")
    print(f"Net PnL:               ${report['net_pnl']:,.2f}")
    print(f"Final equity:          ${report['final_equity']:,.2f}")
    print(f"Max drawdown:          ${report['max_drawdown']:,.2f}")
    print(f"Max loss streak:       {report['max_loss_streak']}")
    print(f"Trades / day:          {report['trades_per_day']:.2f}")
    print(f"RR nominal:            {RR:.2f}")
    print()
    print("Per-symbol:")
    for asset, r in report["per_symbol"].items():
        print(
            f"  {asset:8s} trades={r['trades']:4d} "
            f"WR={r['win_rate']:6.2f}% "
            f"PnL=${r['pnl']:10.2f}"
        )

    print()
    print(f"Reports written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
