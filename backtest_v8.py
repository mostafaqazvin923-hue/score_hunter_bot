#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HUNTER-X VPBB CLEAN PORTFOLIO BACKTEST (LBANK INTEGRATED - PURE 1H PURE EXECUTION)
================================================================================
Pure 1H signal generation without 4H trend restrictions to verify execution.
"""

from __future__ import annotations

import argparse
import math
import time as time_mod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# =========================
# CONFIG & 10 SYMBOLS (PURE 1H)
# =========================

DEFAULT_SYMBOLS = [
    "btc_usdt", "eth_usdt", "sol_usdt", "xrp_usdt", "ada_usdt",
    "avax_usdt", "link_usdt", "near_usdt", "sui_usdt", "dot_usdt",
]

@dataclass
class Config:
    exchange_url: str = "https://api.lbank.info"
    interval: str = "hour1"
    days: int = 365
    initial_equity: float = 10000.0

    # Indicators
    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 200
    rsi_len: int = 14
    atr_len: int = 14
    adx_len: int = 14
    bb_len: int = 20
    bb_std: float = 2.0
    volume_sma_len: int = 20

    # VP
    vp_lookback: int = 96
    vp_rows: int = 48
    vp_value_area: float = 0.70
    vp_near_atr: float = 1.0

    # Signal (Pure 1H - Guaranteed to fire)
    adx_min: float = 5.0
    volume_mult: float = 0.80
    score_min: int = 2         # حداقل امتیاز بسیار پایین برای تضمین اجرای ترید
    swing_lookback: int = 10

    # Risk / trade
    risk_pct: float = 0.005
    max_portfolio_risk_pct: float = 0.02
    weekly_risk_multiplier: float = 0.50
    max_consecutive_losses: int = 3
    max_hold_bars: int = 30
    tp_r: float = 1.5
    sl_atr_buffer: float = 0.50
    max_stop_atr: float = 3.0

    # Costs
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0002

    request_limit: int = 2000
    request_pause_sec: float = 0.15
    timeout_sec: int = 20

    max_open_positions: int = 3


# =========================
# LBANK DATA LOADER
# =========================

def lbank_get_klines(symbol: str, start_ts: int, end_ts: int, cfg: Config) -> pd.DataFrame:
    url = cfg.exchange_url.rstrip("/") + "/v2/kline.do"
    rows: List[list] = []
    cursor = end_ts
    step_sec = 3600
    max_requests = max(20, int(cfg.days * 24 / cfg.request_limit) + 20)

    for _ in range(max_requests):
        params = {
            "symbol": symbol,
            "size": cfg.request_limit,
            "type": cfg.interval,
            "time": str(cursor),
        }
        try:
            r = requests.get(url, params=params, timeout=cfg.timeout_sec)
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            break

        if str(payload.get("result", "")).lower() != "true":
            break

        data = payload.get("data", [])
        if not data:
            break

        rows.extend(data)
        ts = [int(x[0]) for x in data]
        oldest = min(ts)

        if oldest <= start_ts:
            break

        cursor = oldest - step_sec
        time_mod.sleep(cfg.request_pause_sec)

    if not rows:
        raise RuntimeError(f"No data returned from LBank for {symbol}")

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().drop_duplicates("timestamp").sort_values("timestamp")
    df = df[(df["timestamp"].astype("int64") // 10**9 >= start_ts) &
            (df["timestamp"].astype("int64") // 10**9 <= end_ts)]
    df = df.set_index("timestamp")
    return df


def load_symbol(symbol: str, cfg: Config, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{symbol}_{cfg.days}d_1h.csv"

    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp").sort_index()

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=cfg.days + 30)
    df = lbank_get_klines(symbol, int(start.timestamp()), int(now.timestamp()), cfg)
    df.reset_index().to_csv(cache, index=False)
    return df


# =========================
# INDICATORS & VP
# =========================

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).replace([np.inf, -np.inf], np.nan)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1/length, adjust=False, min_periods=length).mean()


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low = df["high"], df["low"]
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = true_range(df)
    atr_w = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    plus = 100 * plus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean() / atr_w
    minus = 100 * minus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean() / atr_w
    dx = 100 * (plus - minus).abs() / (plus + minus).replace(0, np.nan)
    return dx.ewm(alpha=1/length, adjust=False, min_periods=length).mean()


def add_1h_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    x = df.copy()
    x["ema20"] = x["close"].ewm(span=cfg.ema_fast, adjust=False, min_periods=cfg.ema_fast).mean()
    x["rsi"] = rsi(x["close"], cfg.rsi_len)
    x["atr"] = atr(x, cfg.atr_len)
    x["adx"] = adx(x, cfg.adx_len)

    bb_mid = x["close"].rolling(cfg.bb_len).mean()
    bb_sd = x["close"].rolling(cfg.bb_len).std(ddof=0)
    x["bb_mid"] = bb_mid
    x["bb_upper"] = bb_mid + cfg.bb_std * bb_sd
    x["bb_lower"] = bb_mid - cfg.bb_std * bb_sd

    x["vol_sma"] = x["volume"].rolling(cfg.volume_sma_len).mean()
    x["prior_swing_low"] = x["low"].shift(1).rolling(cfg.swing_lookback).min()
    x["prior_swing_high"] = x["high"].shift(1).rolling(cfg.swing_lookback).max()
    return x


def fixed_range_vp(hist: pd.DataFrame, rows: int, value_area: float) -> Tuple[float, float, float]:
    if len(hist) < 5:
        return np.nan, np.nan, np.nan
    lo, hi = float(hist["low"].min()), float(hist["high"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan, np.nan, np.nan

    edges = np.linspace(lo, hi, rows + 1)
    vols = np.zeros(rows, dtype=float)

    for _, r in hist.iterrows():
        h, l, v = float(r["high"]), float(r["low"]), float(r["volume"])
        if not np.isfinite(v) or v <= 0:
            continue
        if h <= l:
            idx = max(0, min(rows - 1, np.searchsorted(edges, float(r["close"], side="right")) - 1))
            vols[idx] += v
            continue
        for j in range(rows):
            overlap = max(0.0, min(h, edges[j + 1]) - max(l, edges[j]))
            if overlap > 0:
                vols[j] += v * (overlap / (h - l))

    if vols.sum() <= 0:
        return np.nan, np.nan, np.nan
    poc_i = int(np.argmax(vols))
    return float((edges[poc_i] + edges[poc_i + 1]) / 2.0), float(edges[-1]), float(edges[0])


def add_vp_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    x = df.copy()
    pocs, vahs, vals = np.full(len(x), np.nan), np.full(len(x), np.nan), np.full(len(x), np.nan)
    arr = x[["high", "low", "close", "volume"]].to_numpy()
    for i in range(cfg.vp_lookback, len(x)):
        hist = pd.DataFrame(arr[i - cfg.vp_lookback:i], columns=["high", "low", "close", "volume"])
        pocs[i], vahs[i], vals[i] = fixed_range_vp(hist, cfg.vp_rows, cfg.vp_value_area)
    x["vp_poc"], x["vp_vah"], x["vp_val"] = pocs, vahs, vals
    return x


# =========================
# SIGNAL GENERATION (PURE 1H)
# =========================

def signal_at(df: pd.DataFrame, i: int, cfg: Config) -> Optional[dict]:
    r = df.iloc[i]
    needed = ["ema20", "rsi", "atr", "adx", "bb_lower", "bb_upper", "vol_sma", "prior_swing_low", "prior_swing_high"]
    if any(pd.isna(r.get(k)) for k in needed):
        return None

    # سیگنال‌های ساده و آزاد مبتنی بر قیمت و RSI
    long_cond = r["rsi"] < 45 or r["close"] < r["bb_lower"]
    short_cond = r["rsi"] > 55 or r["close"] > r["bb_upper"]

    if not long_cond and not short_cond:
        return None

    side = "LONG" if long_cond else "SHORT"
    score = 3
    if score < cfg.score_min:
        return None

    stop = r["prior_swing_low"] if side == "LONG" else r["prior_swing_high"]
    if pd.isna(stop):
        stop = r["close"] - (2.0 * r["atr"] if side == "LONG" else -2.0 * r["atr"])

    return {
        "side": side,
        "score": int(score),
        "signal_index": i,
        "signal_time": df.index[i],
        "stop_pre": float(stop),
        "atr": float(r["atr"]),
    }


# =========================
# GLOBAL PORTFOLIO SIMULATOR
# =========================

class Position:
    def __init__(self, symbol: str, side: str, entry_time: pd.Timestamp, entry: float, stop: float, target: float, qty: float, risk_cash: float, score: int):
        self.symbol = symbol
        self.side = side
        self.entry_time = entry_time
        self.entry = entry
        self.stop = stop
        self.target = target
        self.qty = qty
        self.risk_cash = risk_cash
        self.score = score
        self.bars_held = 0


def run_portfolio_backtest(dfs: Dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    all_timestamps = sorted(list(set().union(*(df.index for df in dfs.values()))))
    equity = cfg.initial_equity
    positions: Dict[str, Position] = {}
    trades: List[dict] = []

    for ts in all_timestamps:
        for symbol in list(positions.keys()):
            df = dfs[symbol]
            if ts not in df.index:
                continue
            row = df.loc[ts]
            pos = positions[symbol]
            pos.bars_held += 1

            high, low = float(row["high"]), float(row["low"])
            hit_sl = low <= pos.stop if pos.side == "LONG" else high >= pos.stop
            hit_tp = high >= pos.target if pos.side == "LONG" else low <= pos.target

            reason, raw_exit = None, None
            if hit_sl:
                reason, raw_exit = "SL", pos.stop
            elif hit_tp:
                reason, raw_exit = "TP", pos.target
            elif pos.bars_held >= cfg.max_hold_bars:
                reason, raw_exit = "TIME", float(row["close"])

            if reason:
                exit_price = raw_exit * (1 - cfg.slippage_rate if pos.side == "LONG" else 1 + cfg.slippage_rate)
                gross = (exit_price - pos.entry) * pos.qty if pos.side == "LONG" else (pos.entry - exit_price) * pos.qty
                net = gross - (pos.entry * pos.qty * cfg.fee_rate) - (abs(exit_price * pos.qty) * cfg.fee_rate)
                equity += net

                trades.append({
                    "symbol": symbol, "side": pos.side, "entry_time": pos.entry_time, "exit_time": ts,
                    "entry": pos.entry, "stop": pos.stop, "target": pos.target, "exit": exit_price,
                    "qty": pos.qty, "risk_cash": pos.risk_cash, "score": pos.score, "bars_held": pos.bars_held,
                    "reason": reason, "net_pnl": net, "r_multiple": net / pos.risk_cash if pos.risk_cash > 0 else 0,
                    "equity_after": equity,
                })
                del positions[symbol]

        if len(positions) < cfg.max_open_positions:
            for symbol, df in dfs.items():
                if symbol in positions or ts not in df.index:
                    continue
                i = df.index.get_loc(ts)
                if i - 1 < 0:
                    continue
                sig = signal_at(df, i - 1, cfg)
                if not sig:
                    continue

                row = df.iloc[i]
                entry_raw = float(row["open"])
                side = sig["side"]
                entry = entry_raw * (1 + cfg.slippage_rate if side == "LONG" else 1 - cfg.slippage_rate)
                stop = sig["stop_pre"]
                stop_dist = (entry - stop) if side == "LONG" else (stop - entry)

                if stop_dist <= 0 or stop_dist > cfg.max_stop_atr * sig["atr"]:
                    continue

                risk_cash = equity * cfg.risk_pct
                qty = risk_cash / stop_dist
                target = entry + cfg.tp_r * stop_dist if side == "LONG" else entry - cfg.tp_r * stop_dist

                positions[symbol] = Position(symbol, side, ts, entry, stop, target, qty, risk_cash, sig["score"])
                if len(positions) >= cfg.max_open_positions:
                    break

    return pd.DataFrame(trades)


# =========================
# MAIN
# =========================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    p.add_argument("--initial-equity", type=float, default=10000.0)
    p.add_argument("--risk-pct", type=float, default=0.005)
    p.add_argument("--out-dir", default="backtest_results")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(days=args.days, initial_equity=args.initial_equity, risk_pct=args.risk_pct)

    print("=" * 78)
    print("HUNTER-X VPBB PORTFOLIO BACKTEST (PURE 1H VERIFICATION)")
    print("=" * 78)

    cache_dir = Path("data_cache")
    dfs = {}
    for n, symbol in enumerate(args.symbols, 1):
        print(f"[{n}/{len(args.symbols)}] Loading {symbol} from LBank...")
        try:
            df = load_symbol(symbol, cfg, cache_dir)
            df = add_1h_indicators(df, cfg)
            df = add_vp_columns(df, cfg)
            dfs[symbol] = df
        except Exception as e:
            print(f"    Error loading {symbol}: {e}")

    if not dfs:
        print("No data available.")
        return

    print("\nRunning Global Synchronized Portfolio Backtest...")
    trades_df = run_portfolio_backtest(dfs, cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(out_dir / "portfolio_trades.csv", index=False)

    print("\n" + "=" * 78)
    print("PORTFOLIO BACKTEST RESULTS")
    print("=" * 78)
    if trades_df.empty:
        print("No trades executed.")
    else:
        wins = trades_df[trades_df.net_pnl > 0]
        net_pnl = trades_df.net_pnl.sum()
        win_rate = len(wins) / len(trades_df) * 100
        print(f"Total Trades  : {len(trades_df)}")
        print(f"Win Rate      : {win_rate:.2f}%")
        print(f"Net PnL       : ${net_pnl:,.2f} ({net_pnl/cfg.initial_equity*100:.2f}%)")
    print(f"\nReport saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
