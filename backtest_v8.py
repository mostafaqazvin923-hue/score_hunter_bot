#!/usr/bin/env python3
"""
HUNTER-V1 — TREND PULLBACK + LIQUIDITY RECLAIM

Project goal
------------
Build a simple, causal XT USDT-M Futures strategy around:
    1D regime -> 4H trend -> 1H pullback/sweep/reclaim -> next 1H open entry.

Hard rules
----------
- Raw source: XT Futures 15m only.
- Signals are built on completed 1H candles.
- 4H and 1D context use only completed candles available at the signal close.
- Entry: next 1H candle OPEN, with adverse slippage.
- Fixed RR 1:2.
- Structural stop: beyond the signal candle's liquidity-sweep extreme.
- No timeout, no break-even, no trailing.
- No lookahead / repaint / future pivot confirmation.
- One portfolio position at a time (strict non-overlap).
- A candle that closes a trade cannot create a new trade.
- Same-candle SL + TP => LOSS.
- Positions still open at dataset end remain OPEN.
- Fixed $100 margin and 50x leverage.

This is a research/backtest engine, not an execution bot.
It intentionally avoids parameter optimization in the first test.
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# -----------------------------
# Project configuration
# -----------------------------
SYMBOLS = [
    "BTC", "ETH", "SOL", "SUI", "AVAX", "NEAR", "ADA", "BNB",
    "APT", "CRV", "ONDO", "PENDLE", "ICP", "WIF",
]

BASE_FUT = "https://fapi.xt.com"
KLINE_URL = f"{BASE_FUT}/future/market/v1/public/q/kline"
SYMBOL_URL = f"{BASE_FUT}/future/market/v1/public/symbol/list"

DAYS = 365
WARMUP_DAYS = 35
INTERVAL = "15m"
INTERVAL_MS = 15 * 60 * 1000
LIMIT = 1500
REQUEST_TIMEOUT = 20
RETRIES = 5
MIN_ROWS = 30000

INITIAL_EQUITY = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
FEE_RATE = 0.0007
SLIPPAGE = 0.0003
RR = 2.0

# Strategy: deliberately small and interpretable.
EMA_FAST_4H = 50
EMA_SLOW_4H = 200
EMA_PULLBACK_1H = 20
ATR_PERIOD_1H = 14
SWEEP_LOOKBACK = 20
BODY_LOOKBACK = 20
MIN_BODY_ATR = 0.45
MIN_CLOSE_LOCATION = 0.65
MIN_STOP_PCT = 0.0010
MAX_STOP_PCT = 0.0300
SWEEP_BUFFER_ATR = 0.08

# Regime/trend structure must be established, not merely touched.
MIN_4H_EMA_GAP_ATR = 0.05
MIN_1D_EMA_GAP_PCT = 0.0020

DATA_DIR = Path("data/xt_hunter_v1")
OUT_DIR = DATA_DIR / "results"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "HUNTER-V1-XT/1.0"})


def request_json(url: str, params: dict) -> dict:
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            obj = r.json()
            if not isinstance(obj, dict):
                raise RuntimeError("non-dict JSON response")
            return obj
        except Exception as exc:
            last = exc
            if attempt < RETRIES:
                time.sleep(min(2.0, 0.4 * attempt))
    raise RuntimeError(f"XT request failed: {url} {params} :: {last}")


def extract_list(obj: dict) -> list:
    result = obj.get("result")
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ("items", "data", "list", "rows"):
            value = result.get(key)
            if isinstance(value, list):
                return value
    for key in ("data", "list", "rows"):
        value = obj.get(key)
        if isinstance(value, list):
            return value
    return []


def parse_kline(rows: list) -> list:
    out = []
    for x in rows:
        try:
            if isinstance(x, dict):
                t = x.get("t")
                o = x.get("o", x.get("open"))
                h = x.get("h", x.get("high"))
                l = x.get("l", x.get("low"))
                c = x.get("c", x.get("close"))
                v = x.get("a", x.get("q", x.get("volume", 0)))
            elif isinstance(x, (list, tuple)) and len(x) >= 6:
                t, o, h, l, c, v = x[:6]
            else:
                continue
            out.append((int(t), float(o), float(h), float(l), float(c), float(v)))
        except (TypeError, ValueError):
            continue
    return out


def resolve_symbol(asset: str, discovered: list) -> str:
    wanted = {asset.lower(), f"{asset.lower()}_usdt", f"{asset.lower()}/usdt"}
    for item in discovered:
        if isinstance(item, str):
            s = item
        elif isinstance(item, dict):
            s = item.get("symbol") or item.get("contract") or item.get("name") or ""
        else:
            continue
        s = str(s)
        if s.lower() in wanted:
            return s
    # Known XT USDT-M convention used by the validated collector.
    return f"{asset.lower()}_usdt"


def fetch_symbol(asset: str, xt_symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Known-good forward pagination with strict in-window cursor progress."""
    rows: List[Tuple[int, float, float, float, float, float]] = []
    cursor = start_ms
    guard = 0

    while cursor <= end_ms:
        guard += 1
        if guard > 1000:
            raise RuntimeError(f"{asset}: pagination guard tripped")

        window_end = min(end_ms, cursor + LIMIT * INTERVAL_MS - 1)
        params = {
            "symbol": xt_symbol,
            "interval": INTERVAL,
            "startTime": cursor,
            "endTime": window_end,
            "limit": LIMIT,
        }

        batch: list = []
        for attempt in range(RETRIES):
            try:
                batch = parse_kline(extract_list(request_json(KLINE_URL, params)))
                if batch:
                    break
            except Exception:
                if attempt == RETRIES - 1:
                    raise
                time.sleep(min(2.0, 0.4 * (attempt + 1)))

        in_window = [x for x in batch if cursor <= x[0] <= window_end]
        if not in_window:
            cursor = window_end + 1
            continue

        rows.extend(in_window)
        mx = max(x[0] for x in in_window)
        next_cursor = mx + 1
        if next_cursor <= cursor:
            raise RuntimeError(f"{asset}: pagination failed to advance")
        cursor = next_cursor
        time.sleep(0.03)

    if not rows:
        raise RuntimeError(f"{asset}: zero XT candles")

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df = df[(df.ts >= start_ms) & (df.ts <= end_ms)].copy()
    if len(df) < MIN_ROWS:
        raise RuntimeError(f"{asset}: only {len(df):,} rows; refusing partial data")

    span = int(df.ts.iloc[-1] - df.ts.iloc[0])
    required = int(DAYS * 86400 * 1000 * 0.90)
    if span < required:
        raise RuntimeError(f"{asset}: insufficient time coverage {span/86400000:.1f}d")

    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("dt").drop(columns=["ts"])
    return df


def ensure_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    data_dir.mkdir(parents=True, exist_ok=True)
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int((DAYS + WARMUP_DAYS) * 86400 * 1000)
    end_ms = (now_ms // INTERVAL_MS) * INTERVAL_MS - 1

    discovered = extract_list(request_json(SYMBOL_URL, {}))
    result: Dict[str, pd.DataFrame] = {}

    for i, asset in enumerate(SYMBOLS, 1):
        path = data_dir / f"{asset}_15m.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["dt"])
            df["dt"] = pd.to_datetime(df["dt"], utc=True)
            df = df.set_index("dt")
            if len(df) >= MIN_ROWS:
                result[asset] = df
                print(f"DATA {i:02d}/{len(SYMBOLS)} {asset:<7} cached rows={len(df):,}")
                continue

        actual = resolve_symbol(asset, discovered)
        print(f"DATA {i:02d}/{len(SYMBOLS)} {asset:<7} downloading {actual}")
        df = fetch_symbol(asset, actual, start_ms, end_ms)
        df.to_csv(path, index_label="dt")
        result[asset] = df
        print(f"       rows={len(df):,} {df.index[0]} -> {df.index[-1]}")

    return result


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    x = df[["open", "high", "low", "close", "volume"]].resample(rule, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return x.dropna(subset=["open", "high", "low", "close"])


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev = df.close.shift(1)
    tr = pd.concat([
        df.high - df.low,
        (df.high - prev).abs(),
        (df.low - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create the causal 1H signal frame from completed 15m candles."""
    h1 = resample_ohlcv(raw, "1h")
    h4 = resample_ohlcv(raw, "4h")
    d1 = resample_ohlcv(raw, "1D")
    h1.index.name = "dt"
    h4.index.name = "dt"
    d1.index.name = "dt"

    # HTF values are shifted one completed candle so the 1H signal can only use
    # a fully completed 4H/1D candle that existed before the signal candle.
    h4["ema50"] = h4.close.ewm(span=EMA_FAST_4H, adjust=False, min_periods=EMA_FAST_4H).mean()
    h4["ema200"] = h4.close.ewm(span=EMA_SLOW_4H, adjust=False, min_periods=EMA_SLOW_4H).mean()
    h4["atr"] = atr(h4, ATR_PERIOD_1H)
    h4["gap_atr"] = (h4.ema50 - h4.ema200).abs() / h4.atr.replace(0, np.nan)
    h4["trend_long"] = (h4.ema50 > h4.ema200) & (h4.close > h4.ema50) & (h4.gap_atr >= MIN_4H_EMA_GAP_ATR)
    h4["trend_short"] = (h4.ema50 < h4.ema200) & (h4.close < h4.ema50) & (h4.gap_atr >= MIN_4H_EMA_GAP_ATR)

    d1["ema50"] = d1.close.ewm(span=50, adjust=False, min_periods=50).mean()
    d1["ema200"] = d1.close.ewm(span=200, adjust=False, min_periods=200).mean()
    d1["gap_pct"] = (d1.ema50 - d1.ema200).abs() / d1.close.replace(0, np.nan)
    d1["regime_long"] = (d1.ema50 > d1.ema200) & (d1.close > d1.ema50) & (d1.gap_pct >= MIN_1D_EMA_GAP_PCT)
    d1["regime_short"] = (d1.ema50 < d1.ema200) & (d1.close < d1.ema50) & (d1.gap_pct >= MIN_1D_EMA_GAP_PCT)

    # Shift HTF state by one completed candle before joining to 1H.
    h4_state = h4[["trend_long", "trend_short"]].shift(1)
    d1_state = d1[["regime_long", "regime_short"]].shift(1)

    h1["atr"] = atr(h1, ATR_PERIOD_1H)
    h1["ema20"] = h1.close.ewm(span=EMA_PULLBACK_1H, adjust=False, min_periods=EMA_PULLBACK_1H).mean()
    h1["prior_low"] = h1.low.shift(1).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min()
    h1["prior_high"] = h1.high.shift(1).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max()
    h1["body"] = (h1.close - h1.open).abs()
    h1["avg_body"] = h1.body.shift(1).rolling(BODY_LOOKBACK, min_periods=BODY_LOOKBACK).mean()
    h1["range"] = h1.high - h1.low
    h1["close_loc"] = (h1.close - h1.low) / h1.range.replace(0, np.nan)

    h1 = pd.merge_asof(
        h1.sort_index().reset_index(),
        h4_state.sort_index().reset_index(),
        on="dt", direction="backward", allow_exact_matches=True,
    ).set_index("dt")
    h1 = pd.merge_asof(
        h1.sort_index().reset_index(),
        d1_state.sort_index().reset_index(),
        on="dt", direction="backward", allow_exact_matches=True,
    ).set_index("dt")

    h1["long_signal"] = (
        h1.trend_long.astype("boolean").fillna(False)
        & h1.regime_long.astype("boolean").fillna(False)
        & (h1.low < h1.prior_low)
        & (h1.close > h1.prior_low)
        & (h1.close > h1.ema20)
        & (h1.body >= MIN_BODY_ATR * h1.atr)
        & (h1.close_loc >= MIN_CLOSE_LOCATION)
    )
    h1["short_signal"] = (
        h1.trend_short.astype("boolean").fillna(False)
        & h1.regime_short.astype("boolean").fillna(False)
        & (h1.high > h1.prior_high)
        & (h1.close < h1.prior_high)
        & (h1.close < h1.ema20)
        & (h1.body >= MIN_BODY_ATR * h1.atr)
        & (h1.close_loc <= (1.0 - MIN_CLOSE_LOCATION))
    )
    return h1.dropna(subset=["atr", "ema20", "prior_low", "prior_high"])


def adverse_entry(open_price: float, side: str) -> float:
    return open_price * (1.0 + SLIPPAGE) if side == "LONG" else open_price * (1.0 - SLIPPAGE)


def net_pnl(entry: float, exit_price: float, side: str) -> float:
    notional = MARGIN * LEVERAGE
    gross = notional * ((exit_price / entry) - 1.0) * (1.0 if side == "LONG" else -1.0)
    fees = notional * FEE_RATE * 2.0
    return gross - fees


def backtest_symbol(asset: str, h1: pd.DataFrame, cutoff: pd.Timestamp) -> Tuple[List[dict], Optional[dict]]:
    trades: List[dict] = []
    open_pos: Optional[dict] = None

    x = h1[h1.index >= cutoff].copy()
    if len(x) < 10:
        return trades, None

    for i in range(1, len(x)):
        bar = x.iloc[i]
        ts = x.index[i]
        prev = x.iloc[i - 1]

        # Existing trade is managed before considering any new signal. If it closes
        # on this candle, this same candle is permanently ineligible for a new entry.
        if open_pos is not None:
            side = open_pos["side"]
            hit_sl = bar.low <= open_pos["sl"] if side == "LONG" else bar.high >= open_pos["sl"]
            hit_tp = bar.high >= open_pos["tp"] if side == "LONG" else bar.low <= open_pos["tp"]
            if hit_sl or hit_tp:
                # Conservative rule: if both are touched, SL wins.
                outcome = "LOSS" if hit_sl else "WIN"
                exit_price = open_pos["sl"] if hit_sl else open_pos["tp"]
                pnl = net_pnl(open_pos["entry"], exit_price, side)
                trades.append({**open_pos, "exit_time": ts, "exit": exit_price, "outcome": outcome, "pnl": pnl})
                open_pos = None
                continue

        if open_pos is not None:
            continue

        # Signal was confirmed on prev 1H close; entry is current 1H OPEN.
        side = None
        if bool(prev.long_signal):
            side = "LONG"
            sweep_extreme = float(prev.low)
            entry = adverse_entry(float(bar.open), side)
            sl = sweep_extreme - float(prev.atr) * SWEEP_BUFFER_ATR
            risk = entry - sl
            if risk <= 0:
                continue
            tp = entry + RR * risk
        elif bool(prev.short_signal):
            side = "SHORT"
            sweep_extreme = float(prev.high)
            entry = adverse_entry(float(bar.open), side)
            sl = sweep_extreme + float(prev.atr) * SWEEP_BUFFER_ATR
            risk = sl - entry
            if risk <= 0:
                continue
            tp = entry - RR * risk
        else:
            continue

        stop_pct = risk / entry
        if not (MIN_STOP_PCT <= stop_pct <= MAX_STOP_PCT):
            continue

        # Entry is at current open. An impossible same-candle pre-entry touch is ignored;
        # only movement after the entry is represented by this candle's OHLC. Conservative
        # handling treats any SL touch and TP touch on the entry candle as SL first.
        hit_sl = bar.low <= sl if side == "LONG" else bar.high >= sl
        hit_tp = bar.high >= tp if side == "LONG" else bar.low <= tp
        if hit_sl or hit_tp:
            outcome = "LOSS" if hit_sl else "WIN"
            exit_price = sl if hit_sl else tp
            pnl = net_pnl(entry, exit_price, side)
            trades.append({
                "asset": asset, "side": side, "signal_time": prev.name,
                "entry_time": ts, "entry": entry, "sl": sl, "tp": tp,
                "exit_time": ts, "exit": exit_price, "outcome": outcome, "pnl": pnl,
            })
        else:
            open_pos = {
                "asset": asset, "side": side, "signal_time": prev.name,
                "entry_time": ts, "entry": entry, "sl": sl, "tp": tp,
            }

    return trades, open_pos


def portfolio_backtest(features: Dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> Tuple[pd.DataFrame, List[dict]]:
    """Strict global non-overlap: one position total across all symbols."""
    events = []
    for asset, frame in features.items():
        for ts, row in frame[frame.index >= cutoff].iterrows():
            if bool(row.long_signal) or bool(row.short_signal):
                events.append((ts, asset))
    events.sort()

    # Build per-symbol signal/entry logic on demand while enforcing a global lock.
    states = {asset: None for asset in features}
    trades: List[dict] = []
    global_pos = None
    event_idx = 0

    # Iterate chronologically over all 1H bars. Every bar manages the open trade first.
    all_times = sorted(set().union(*[set(f.index[f.index >= cutoff]) for f in features.values()]))
    for ts in all_times:
        closed_this_bar = False

        if global_pos is not None:
            asset = global_pos["asset"]
            frame = features[asset]
            if ts in frame.index:
                bar = frame.loc[ts]
                side = global_pos["side"]
                hit_sl = bar.low <= global_pos["sl"] if side == "LONG" else bar.high >= global_pos["sl"]
                hit_tp = bar.high >= global_pos["tp"] if side == "LONG" else bar.low <= global_pos["tp"]
                if hit_sl or hit_tp:
                    outcome = "LOSS" if hit_sl else "WIN"
                    exit_price = global_pos["sl"] if hit_sl else global_pos["tp"]
                    pnl = net_pnl(global_pos["entry"], exit_price, side)
                    trades.append({**global_pos, "exit_time": ts, "exit": exit_price, "outcome": outcome, "pnl": pnl})
                    global_pos = None
                    closed_this_bar = True

        if global_pos is not None or closed_this_bar:
            continue

        # A signal is acted upon on the next 1H bar. Find all signals whose next bar is ts.
        candidates = []
        for asset, frame in features.items():
            if ts not in frame.index:
                continue
            loc = frame.index.get_loc(ts)
            if loc == 0:
                continue
            prev = frame.iloc[loc - 1]
            if bool(prev.long_signal):
                candidates.append((asset, "LONG", prev))
            elif bool(prev.short_signal):
                candidates.append((asset, "SHORT", prev))

        if not candidates:
            continue

        # Deterministic selection only; no ranking/optimization. Prefer the first symbol
        # in the fixed universe order to avoid introducing a hidden cross-sectional optimizer.
        candidates.sort(key=lambda z: SYMBOLS.index(z[0]))
        asset, side, prev = candidates[0]
        bar = features[asset].loc[ts]
        entry = adverse_entry(float(bar.open), side)
        if side == "LONG":
            sl = float(prev.low) - float(prev.atr) * SWEEP_BUFFER_ATR
            risk = entry - sl
            tp = entry + RR * risk
        else:
            sl = float(prev.high) + float(prev.atr) * SWEEP_BUFFER_ATR
            risk = sl - entry
            tp = entry - RR * risk
        if risk <= 0:
            continue
        stop_pct = risk / entry
        if not (MIN_STOP_PCT <= stop_pct <= MAX_STOP_PCT):
            continue

        hit_sl = bar.low <= sl if side == "LONG" else bar.high >= sl
        hit_tp = bar.high >= tp if side == "LONG" else bar.low <= tp
        base = {
            "asset": asset, "side": side, "signal_time": prev.name,
            "entry_time": ts, "entry": entry, "sl": sl, "tp": tp,
        }
        if hit_sl or hit_tp:
            outcome = "LOSS" if hit_sl else "WIN"
            exit_price = sl if hit_sl else tp
            trades.append({**base, "exit_time": ts, "exit": exit_price, "outcome": outcome, "pnl": net_pnl(entry, exit_price, side)})
        else:
            global_pos = base

    opens = [global_pos] if global_pos is not None else []
    return pd.DataFrame(trades), opens


def max_loss_streak(df: pd.DataFrame) -> int:
    streak = best = 0
    for outcome in df.outcome.tolist():
        if outcome == "LOSS":
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def summarize(trades: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    if trades.empty:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "profit_factor": 0.0, "net_pnl": 0.0, "max_dd": 0.0,
            "max_loss_streak": 0, "trades_per_day": 0.0,
        }
    wins = trades[trades.outcome == "WIN"].pnl.sum()
    losses = -trades[trades.outcome == "LOSS"].pnl.sum()
    pf = wins / losses if losses > 0 else math.inf
    equity = INITIAL_EQUITY + trades.sort_values("exit_time").pnl.cumsum()
    dd = equity - equity.cummax()
    span_days = max((trades.exit_time.max() - cutoff).total_seconds() / 86400.0, 1.0)
    return {
        "trades": int(len(trades)),
        "wins": int((trades.outcome == "WIN").sum()),
        "losses": int((trades.outcome == "LOSS").sum()),
        "win_rate": float((trades.outcome == "WIN").mean() * 100.0),
        "profit_factor": float(pf),
        "net_pnl": float(trades.pnl.sum()),
        "max_dd": float(dd.min()),
        "max_loss_streak": int(max_loss_streak(trades.sort_values("exit_time"))),
        "trades_per_day": float(len(trades) / span_days),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_download:
        raw = {}
        for asset in SYMBOLS:
            path = data_dir / f"{asset}_15m.csv"
            if not path.exists():
                raise RuntimeError(f"Missing {path}; remove --no-download to fetch XT data")
            df = pd.read_csv(path, parse_dates=["dt"])
            df["dt"] = pd.to_datetime(df["dt"], utc=True)
            raw[asset] = df.set_index("dt")
    else:
        raw = ensure_data(data_dir)

    features: Dict[str, pd.DataFrame] = {}
    for i, asset in enumerate(SYMBOLS, 1):
        features[asset] = build_features(raw[asset])
        print(f"FEATURE {i:02d}/{len(SYMBOLS)} {asset:<7} 1H rows={len(features[asset]):,}")

    cutoff = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=DAYS))
    trades, opens = portfolio_backtest(features, cutoff)

    if not trades.empty:
        trades = trades.sort_values("exit_time").reset_index(drop=True)
        trades.to_csv(out_dir / "trades.csv", index=False)
    else:
        pd.DataFrame(columns=["asset","side","signal_time","entry_time","entry","sl","tp","exit_time","exit","outcome","pnl"]).to_csv(out_dir / "trades.csv", index=False)

    overall = summarize(trades, cutoff)
    per_symbol = []
    if not trades.empty:
        for asset in SYMBOLS:
            g = trades[trades.asset == asset]
            s = summarize(g, cutoff)
            s["asset"] = asset
            per_symbol.append(s)
    per_symbol_df = pd.DataFrame(per_symbol)
    per_symbol_df.to_csv(out_dir / "per_symbol.csv", index=False)

    report = {
        "strategy": "HUNTER-V1 Trend Pullback + Liquidity Reclaim",
        "data_days": DAYS,
        "symbols": len(SYMBOLS),
        "initial_equity": INITIAL_EQUITY,
        "margin": MARGIN,
        "leverage": LEVERAGE,
        "rr": RR,
        "fee_rate": FEE_RATE,
        "slippage": SLIPPAGE,
        "open_positions_at_end": len(opens),
        **overall,
    }
    pd.DataFrame([report]).to_csv(out_dir / "summary.csv", index=False)

    print("\n" + "=" * 88)
    print("HUNTER-V1 — FINAL BACKTEST")
    print("=" * 88)
    for k, v in report.items():
        print(f"{k:24s}: {v}")
    print("\nPer-symbol:")
    if not per_symbol_df.empty:
        print(per_symbol_df[["asset","trades","win_rate","profit_factor","net_pnl","max_dd","max_loss_streak","trades_per_day"]].to_string(index=False, float_format=lambda z: f"{z:.3f}"))
    print(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()
