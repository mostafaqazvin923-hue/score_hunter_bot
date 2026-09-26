#!/usr/bin/env python3
"""
HUNTER-V2 — CROSS-SECTIONAL EXHAUSTION / REVERSION

Research backtest for XT USDT-M Futures.

Idea (deliberately different from HUNTER-V1):
    1D + 4H market regime -> cross-sectional 1H relative-strength extreme
    -> exhaustion/reversal confirmation -> next 1H open entry.

There is no liquidity sweep, pivot logic, breakout, trailing, BE or timeout.
The signal is based on relative performance across the portfolio, not a
single-symbol structural pattern.

Hard rules:
- Raw XT Futures source: 15m.
- Signals only on completed 1H candles.
- 4H/1D context uses completed candles only.
- Entry at next 1H open with adverse slippage.
- Fixed RR 1:2.
- Fixed $100 margin, 50x leverage.
- Maximum 3 simultaneous positions.
- No same-symbol overlap.
- No timeout / forced exit at dataset end.
- Same-candle SL+TP is LOSS.
- Positions open at dataset end remain OPEN.
- No lookahead / repaint / future leak.

This is a research engine, not an execution bot.
"""

from __future__ import annotations

import argparse
import math
import time
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
MAX_OPEN_POSITIONS = 3

# Cross-sectional signal parameters. No optimization in this first run.
RETURN_LOOKBACK_H = 24
CROSS_SECTION_Z_MIN = 1.00
REVERSAL_RETURN_H = 2
MIN_REVERSAL_ATR = 0.15
ATR_PERIOD = 14
ATR_STOP_MULT = 1.20
MIN_STOP_PCT = 0.0020
MAX_STOP_PCT = 0.0350

DATA_DIR = Path("data/xt_hunter_v2")
OUT_DIR = DATA_DIR / "results"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "HUNTER-V2-XT/1.0"})


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
    return f"{asset.lower()}_usdt"


def fetch_symbol(asset: str, xt_symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
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
        raise RuntimeError(f"{asset}: coverage too short ({span / 86400000:.1f} days)")

    gaps = np.diff(df.ts.to_numpy(dtype=np.int64))
    if len(gaps) and int(gaps.max()) > int(INTERVAL_MS * 1.5):
        raise RuntimeError(f"{asset}: 15m data gap detected")

    df["datetime"] = pd.to_datetime(df.ts, unit="ms", utc=True)
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    x = df.set_index("datetime")[
        ["open", "high", "low", "close", "volume"]
    ].resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    x = x.dropna(subset=["open", "high", "low", "close"]).copy()
    return x


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["atr"] = atr(x, ATR_PERIOD)
    x["ret_24h"] = x["close"].pct_change(RETURN_LOOKBACK_H)
    x["ret_2h"] = x["close"].pct_change(REVERSAL_RETURN_H)
    x["range"] = x["high"] - x["low"]
    x["body"] = (x["close"] - x["open"]).abs()
    x["body_atr"] = x["body"] / x["atr"].replace(0, np.nan)
    x["ema20"] = x["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    return x


def prepare_symbol(asset: str, raw: pd.DataFrame) -> dict:
    h1 = resample_ohlcv(raw, "1h")
    h4 = resample_ohlcv(raw, "4h")
    d1 = resample_ohlcv(raw, "1D")

    h1 = add_context(h1)

    h4["ema50"] = h4["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    h4["ema200"] = h4["close"].ewm(span=200, adjust=False, min_periods=200).mean()
    d1["ema50"] = d1["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    d1["ema200"] = d1["close"].ewm(span=200, adjust=False, min_periods=200).mean()

    # Shift context by one completed higher-timeframe candle relative to the
    # 1H signal. This makes the causal boundary explicit.
    h4_ctx = h4[["close", "ema50", "ema200"]].rename(
        columns={"close": "h4_close", "ema50": "h4_ema50", "ema200": "h4_ema200"}
    ).shift(1)
    d1_ctx = d1[["close", "ema50", "ema200"]].rename(
        columns={"close": "d1_close", "ema50": "d1_ema50", "ema200": "d1_ema200"}
    ).shift(1)

    h1 = h1.join(h4_ctx.reindex(h1.index, method="ffill"))
    h1 = h1.join(d1_ctx.reindex(h1.index, method="ffill"))
    h1["asset"] = asset
    return h1


def build_panel(prepared: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for asset, df in prepared.items():
        x = df.copy()
        x["cross_z"] = np.nan
        parts.append(x)

    # Cross-sectional z-score is calculated only among assets available at the
    # same completed 1H timestamp.
    panel = pd.concat(parts, keys=prepared.keys(), names=["asset", "datetime"])
    panel = panel.reset_index()
    panel["cross_mean"] = panel.groupby("datetime")["ret_24h"].transform("mean")
    panel["cross_std"] = panel.groupby("datetime")["ret_24h"].transform("std", ddof=0)
    panel["cross_z"] = (
        (panel["ret_24h"] - panel["cross_mean"])
        / panel["cross_std"].replace(0, np.nan)
    )
    panel = panel.set_index(["asset", "datetime"]).sort_index()
    return panel


def regime(row: pd.Series) -> int:
    d1_ok = pd.notna(row.get("d1_ema50")) and pd.notna(row.get("d1_ema200"))
    h4_ok = pd.notna(row.get("h4_ema50")) and pd.notna(row.get("h4_ema200"))
    if not (d1_ok and h4_ok):
        return 0
    if row["d1_ema50"] > row["d1_ema200"] and row["h4_ema50"] > row["h4_ema200"]:
        return 1
    if row["d1_ema50"] < row["d1_ema200"] and row["h4_ema50"] < row["h4_ema200"]:
        return -1
    return 0


def make_signal(row: pd.Series) -> int:
    """Return +1 long, -1 short, 0 no signal."""
    z = row.get("cross_z")
    atr_v = row.get("atr")
    if not np.isfinite(z) or not np.isfinite(atr_v) or atr_v <= 0:
        return 0

    reg = regime(row)
    if reg == 0:
        return 0

    # Extreme relative underperformance in a bullish market -> long reversal.
    # Extreme relative outperformance in a bearish market -> short reversal.
    if reg == 1 and z <= -CROSS_SECTION_Z_MIN:
        if row["close"] > row["open"] and row["ret_2h"] > 0:
            if row["body_atr"] >= MIN_REVERSAL_ATR:
                return 1

    if reg == -1 and z >= CROSS_SECTION_Z_MIN:
        if row["close"] < row["open"] and row["ret_2h"] < 0:
            if row["body_atr"] >= MIN_REVERSAL_ATR:
                return -1

    return 0


class Position:
    def __init__(self, asset: str, side: int, entry_time, entry: float, stop: float, tp: float):
        self.asset = asset
        self.side = side
        self.entry_time = entry_time
        self.entry = entry
        self.stop = stop
        self.tp = tp


def execute_exit(pos: Position, bar: pd.Series) -> Optional[Tuple[str, float]]:
    if pos.side == 1:
        hit_sl = bar["low"] <= pos.stop
        hit_tp = bar["high"] >= pos.tp
    else:
        hit_sl = bar["high"] >= pos.stop
        hit_tp = bar["low"] <= pos.tp

    if not hit_sl and not hit_tp:
        return None
    if hit_sl and hit_tp:
        return "LOSS", pos.stop
    if hit_sl:
        return "LOSS", pos.stop
    return "WIN", pos.tp


def trade_pnl(pos: Position, exit_price: float) -> float:
    notional = MARGIN * LEVERAGE
    gross = notional * ((exit_price - pos.entry) / pos.entry) * pos.side
    fees = notional * FEE_RATE * 2.0
    return gross - fees


def build_positions(panel: pd.DataFrame) -> Tuple[List[dict], List[Position]]:
    """Causal portfolio simulation using globally chronological 1H candles."""
    times = sorted(panel.index.get_level_values("datetime").unique())
    by_time = {t: panel.xs(t, level="datetime") for t in times}

    open_pos: Dict[str, Position] = {}
    trades: List[dict] = []
    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    last_close_time = None
    loss_streak = 0
    max_loss_streak = 0

    # Signals from candle t enter at candle t+1 open.
    pending: Dict[pd.Timestamp, List[Tuple[str, int, float, float]]] = {}

    for idx, t in enumerate(times[:-1]):
        next_t = times[idx + 1]
        frame = by_time[t]
        next_frame = by_time.get(next_t)
        if next_frame is None:
            continue

        # 1) Resolve exits on completed candle t before creating new signals.
        for asset in list(open_pos):
            pos = open_pos[asset]
            row = frame.loc[asset]
            result = execute_exit(pos, row)
            if result is None:
                continue
            outcome, exit_price = result
            pnl = trade_pnl(pos, exit_price)
            equity += pnl
            trades.append(
                {
                    "asset": asset,
                    "side": "LONG" if pos.side == 1 else "SHORT",
                    "entry_time": pos.entry_time,
                    "exit_time": t,
                    "entry": pos.entry,
                    "exit": exit_price,
                    "stop": pos.stop,
                    "tp": pos.tp,
                    "outcome": outcome,
                    "pnl": pnl,
                }
            )
            if outcome == "LOSS":
                loss_streak += 1
                max_loss_streak = max(max_loss_streak, loss_streak)
            else:
                loss_streak = 0
            del open_pos[asset]
            last_close_time = t
            peak = max(peak, equity)
            max_dd = min(max_dd, equity - peak)

        # 2) Execute entries scheduled from the previous completed candle.
        entries = pending.pop(t, [])
        # Deterministic ranking: strongest absolute cross-sectional extreme first.
        entries.sort(key=lambda x: (-abs(x[2]), x[0]))
        for asset, side, stop_distance, _z in entries:
            if len(open_pos) >= MAX_OPEN_POSITIONS:
                break
            if asset in open_pos:
                continue
            if last_close_time == t:
                # Explicitly forbid a new entry on the candle that closed a trade.
                continue
            if asset not in next_frame.index and asset not in frame.index:
                continue
            if asset not in frame.index:
                continue
            entry_row = frame.loc[asset]
            raw_entry = float(entry_row["open"] if t in by_time and asset in frame.index else np.nan)
            if not np.isfinite(raw_entry) or raw_entry <= 0:
                continue
            entry = raw_entry * (1.0 + SLIPPAGE * side)
            if side == 1:
                stop = entry - stop_distance
                tp = entry + RR * stop_distance
            else:
                stop = entry + stop_distance
                tp = entry - RR * stop_distance
            if stop <= 0 or tp <= 0:
                continue
            open_pos[asset] = Position(asset, side, t, entry, stop, tp)

        # 3) Generate signals from completed candle t for next candle entry.
        # No signal is allowed on a candle that just closed any trade.
        if last_close_time == t:
            continue
        candidates: List[Tuple[str, int, float, float]] = []
        for asset, row in frame.iterrows():
            if asset in open_pos:
                continue
            side = make_signal(row)
            if side == 0:
                continue
            atr_v = float(row["atr"])
            stop_distance = max(ATR_STOP_MULT * atr_v, MIN_STOP_PCT * float(row["close"]))
            stop_pct = stop_distance / float(row["close"])
            if stop_pct > MAX_STOP_PCT:
                continue
            candidates.append((asset, side, stop_distance, float(abs(row["cross_z"]))))
        candidates.sort(key=lambda x: (-x[3], x[0]))
        capacity = max(0, MAX_OPEN_POSITIONS - len(open_pos))
        for asset, side, stop_distance, zabs in candidates[:capacity]:
            pending.setdefault(next_t, []).append((asset, side, stop_distance, zabs))

    return trades, list(open_pos.values())


def summarize(trades: List[dict], open_positions: List[Position], start_dt, end_dt) -> dict:
    wins = [t["pnl"] for t in trades if t["outcome"] == "WIN"]
    losses = [t["pnl"] for t in trades if t["outcome"] == "LOSS"]
    total = len(trades)
    wr = 100.0 * len(wins) / total if total else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0)

    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    for t in trades:
        equity += t["pnl"]
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    streak = 0
    max_streak = 0
    for t in trades:
        if t["outcome"] == "LOSS":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    days = max((end_dt - start_dt).total_seconds() / 86400.0, 1.0)
    return {
        "initial_equity": INITIAL_EQUITY,
        "final_equity": equity,
        "net_pnl": equity - INITIAL_EQUITY,
        "trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": wr,
        "profit_factor": pf,
        "max_drawdown": max_dd,
        "max_loss_streak": max_streak,
        "trades_per_day": total / days,
        "open_positions_at_end": len(open_positions),
    }


def print_report(summary: dict, trades: List[dict]) -> None:
    print("\nHUNTER-V2 — FINAL BACKTEST")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k} {v:.6f}")
        else:
            print(f"{k} {v}")

    if trades:
        df = pd.DataFrame(trades)
        print("\nPER-SYMBOL")
        for asset, g in df.groupby("asset", sort=True):
            wins = int((g.outcome == "WIN").sum())
            n = len(g)
            wr = 100.0 * wins / n if n else 0.0
            gw = g.loc[g.outcome == "WIN", "pnl"].sum()
            gl = -g.loc[g.outcome == "LOSS", "pnl"].sum()
            pf = gw / gl if gl > 0 else (math.inf if gw > 0 else 0.0)
            print(f"{asset:8s} trades={n:4d} WR={wr:6.2f}% PF={pf:7.3f} NetPnL={g.pnl.sum():10.2f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=DAYS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    days = int(args.days)
    if days < 30:
        raise ValueError("--days must be >= 30")

    end_ms = (int(time.time() * 1000) // INTERVAL_MS) * INTERVAL_MS - 1
    start_ms = end_ms - int((days + WARMUP_DAYS) * 86400 * 1000)

    print("HUNTER-V2 — CROSS-SECTIONAL EXHAUSTION / REVERSION")
    print(f"days={days} symbols={len(SYMBOLS)} timeframe=15m->1h")
    print("Fetching XT Futures data...")

    discovered = []
    try:
        discovered = extract_list(request_json(SYMBOL_URL, {}))
    except Exception as exc:
        print(f"symbol-list warning: {exc}; using default XT symbols")

    prepared: Dict[str, pd.DataFrame] = {}
    for i, asset in enumerate(SYMBOLS, 1):
        xt_symbol = resolve_symbol(asset, discovered)
        print(f"[{i}/{len(SYMBOLS)}] {asset} -> {xt_symbol}")
        raw = fetch_symbol(asset, xt_symbol, start_ms, end_ms)
        prepared[asset] = prepare_symbol(asset, raw)
        print(f"    15m rows={len(raw):,} 1h rows={len(prepared[asset]):,}")

    panel = build_panel(prepared)
    trades, open_positions = build_positions(panel)

    start_dt = pd.to_datetime(start_ms, unit="ms", utc=True).to_pydatetime()
    end_dt = pd.to_datetime(end_ms, unit="ms", utc=True).to_pydatetime()
    summary = summarize(trades, open_positions, start_dt, end_dt)
    print_report(summary, trades)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(OUT_DIR / "trades.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'summary.csv'}")
    print(f"Saved: {OUT_DIR / 'trades.csv'}")


if __name__ == "__main__":
    main()
