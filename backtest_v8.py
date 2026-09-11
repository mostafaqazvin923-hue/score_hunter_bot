#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HUNTER-X VPBB PORTFOLIO BACKTEST (FULL SCORE HUNTER PRO LOGIC)
==============================================================
"""

from __future__ import annotations

import argparse
import time as time_mod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

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

    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 200
    rsi_len: int = 14
    atr_len: int = 14
    adx_len: int = 14
    bb_len: int = 20
    bb_std: float = 2.0
    volume_sma_len: int = 20

    # Volume Profile Config
    vp_lookback: int = 96
    vp_rows: int = 48
    vp_value_area: float = 0.70

    # Signal Scoring
    adx_min: float = 18.0
    volume_mult: float = 1.1
    score_min: int = 3        # حداقل امتیاز برای ورود معتبر
    swing_lookback: int = 10

    risk_pct: float = 0.005
    max_hold_bars: int = 30
    tp_r: float = 1.8
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0002

    request_limit: int = 500
    request_pause_sec: float = 0.2
    timeout_sec: int = 20
    max_open_positions: int = 3


def lbank_get_klines(symbol: str, start_ts: int, end_ts: int, cfg: Config) -> pd.DataFrame:
    url = cfg.exchange_url.rstrip("/") + "/v2/kline.do"
    rows: List[list] = []
    cursor = end_ts
    step_sec = 3600
    max_requests = 100

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
        except Exception:
            break

        data = payload.get("data", [])
        if not data:
            break

        rows.extend(data)
        ts = [int(x[0]) for x in data]
        oldest = min(ts)

        if oldest <= start_ts or len(data) < cfg.request_limit:
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
    df = df.set_index("timestamp")
    return df


def load_symbol(symbol: str, cfg: Config, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{symbol}_{cfg.days}d_1h.csv"

    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        if len(df) > 500:
            return df

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=cfg.days + 10)
    df = lbank_get_klines(symbol, int(start.timestamp()), int(now.timestamp()), cfg)
    df.reset_index().to_csv(cache, index=False)
    return df


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


def add_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    x = df.copy()
    x["ema20"] = x["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=cfg.ema_mid, adjust=False).mean()
    x["rsi"] = rsi(x["close"], cfg.rsi_len)
    x["atr"] = atr(x, cfg.atr_len)
    x["adx"] = adx(x, cfg.adx_len)

    bb_mid = x["close"].rolling(cfg.bb_len).mean()
    bb_sd = x["close"].rolling(cfg.bb_len).std(ddof=0)
    x["bb_lower"] = bb_mid - cfg.bb_std * bb_sd
    x["bb_upper"] = bb_mid + cfg.bb_std * bb_sd
    x["vol_sma"] = x["volume"].rolling(cfg.volume_sma_len).mean()

    x["prior_low"] = x["low"].shift(1).rolling(cfg.swing_lookback).min()
    x["prior_high"] = x["high"].shift(1).rolling(cfg.swing_lookback).max()
    return x


def fixed_range_vp(hist: pd.DataFrame, rows: int) -> Tuple[float, float, float]:
    if len(hist) < 5:
        return np.nan, np.nan, np.nan
    lo, hi = float(hist["low"].min()), float(hist["high"].max())
    if hi <= lo:
        return np.nan, np.nan, np.nan

    edges = np.linspace(lo, hi, rows + 1)
    vols = np.zeros(rows, dtype=float)
    for _, r in hist.iterrows():
        h, l, v = float(r["high"]), float(r["low"]), float(r["volume"])
        if v <= 0: continue
        for j in range(rows):
            overlap = max(0.0, min(h, edges[j + 1]) - max(l, edges[j]))
            if overlap > 0:
                vols[j] += v * (overlap / (h - l))

    if vols.sum() <= 0: return np.nan, np.nan, np.nan
    poc_i = int(np.argmax(vols))
    return float((edges[poc_i] + edges[poc_i + 1]) / 2.0), float(edges[-1]), float(edges[0])


def add_vp(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    x = df.copy()
    pocs = np.full(len(x), np.nan)
    arr = x[["high", "low", "close", "volume"]].to_numpy()
    for i in range(cfg.vp_lookback, len(x)):
        hist = pd.DataFrame(arr[i - cfg.vp_lookback:i], columns=["high", "low", "close", "volume"])
        poc, _, _ = fixed_range_vp(hist, cfg.vp_rows)
        pocs[i] = poc
    x["vp_poc"] = pocs
    return x


def signal_at(df: pd.DataFrame, i: int, cfg: Config) -> Optional[dict]:
    r = df.iloc[i]
    needed = ["ema20", "ema50", "rsi", "atr", "adx", "bb_lower", "bb_upper", "vol_sma", "vp_poc", "prior_low", "prior_high"]
    if any(pd.isna(r.get(k)) for k in needed):
        return None

    if r["adx"] < cfg.adx_min or r["volume"] < r["vol_sma"] * cfg.volume_mult:
        return None

    score_long, score_short = 0, 0

    # امتیازدهی لانگ
    if r["close"] > r["ema50"]: score_long += 1
    if r["rsi"] < 45: score_long += 1
    if r["close"] <= r["bb_lower"] or r["close"] <= r["vp_poc"]: score_long += 1

    # امتیازدهی شورت
    if r["close"] < r["ema50"]: score_short += 1
    if r["rsi"] > 55: score_short += 1
    if r["close"] >= r["bb_upper"] or r["close"] >= r["vp_poc"]: score_short += 1

    if score_long >= cfg.score_min and score_long > score_short:
        return {"side": "LONG", "score": score_long, "stop": r["prior_low"]}
    elif score_short >= cfg.score_min and score_short > score_long:
        return {"side": "SHORT", "score": score_short, "stop": r["prior_high"]}

    return None


def run_portfolio_backtest(dfs: Dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    all_timestamps = sorted(list(set().union(*(df.index for df in dfs.values()))))
    equity = cfg.initial_equity
    positions = {}
    trades = []

    for ts in all_timestamps:
        for symbol in list(positions.keys()):
            df = dfs[symbol]
            if ts not in df.index: continue
            row = df.loc[ts]
            pos = positions[symbol]
            pos["bars"] += 1

            hit_sl = row["low"] <= pos["stop"] if pos["side"] == "LONG" else row["high"] >= pos["stop"]
            hit_tp = row["high"] >= pos["target"] if pos["side"] == "LONG" else row["low"] <= pos["target"]

            reason = None
            if hit_sl: reason = "SL"
            elif hit_tp: reason = "TP"
            elif pos["bars"] >= cfg.max_hold_bars: reason = "TIME"

            if reason:
                exit_price = pos["stop"] if reason == "SL" else (pos["target"] if reason == "TP" else row["close"])
                gross = (exit_price - pos["entry"]) * pos["qty"] if pos["side"] == "LONG" else (pos["entry"] - exit_price) * pos["qty"]
                net = gross - (pos["entry"] * pos["qty"] * cfg.fee_rate * 2)
                equity += net

                trades.append({
                    "symbol": symbol, "side": pos["side"], "entry_time": pos["entry_time"],
                    "exit_time": ts, "net_pnl": net, "reason": reason, "equity_after": equity
                })
                del positions[symbol]

        if len(positions) < cfg.max_open_positions:
            for symbol, df in dfs.items():
                if symbol in positions or ts not in df.index: continue
                i = df.index.get_loc(ts)
                if i - 1 < 0: continue

                sig = signal_at(df, i - 1, cfg)
                if not sig: continue

                row = df.iloc[i]
                entry = float(row["open"])
                side = sig["side"]
                stop = sig["stop"]
                stop_dist = abs(entry - stop)

                if pd.isna(stop) or stop_dist <= 0:
                    stop_dist = row["atr"] * 2.0
                    stop = entry - stop_dist if side == "LONG" else entry + stop_dist

                risk_cash = equity * cfg.risk_pct
                qty = risk_cash / stop_dist
                target = entry + cfg.tp_r * stop_dist if side == "LONG" else entry - cfg.tp_r * stop_dist

                positions[symbol] = {
                    "side": side, "entry_time": ts, "entry": entry,
                    "stop": stop, "target": target, "qty": qty, "bars": 0
                }
                if len(positions) >= cfg.max_open_positions: break

    return pd.DataFrame(trades)


def main():
    cfg = Config()
    print("=" * 78)
    print("SCORE HUNTER PRO (VPBB) - MULTI-INDICATOR BACKTEST")
    print("=" * 78)

    cache_dir = Path("data_cache_score_hunter")
    dfs = {}
    for n, symbol in enumerate(DEFAULT_SYMBOLS, 1):
        try:
            df = load_symbol(symbol, cfg, cache_dir)
            df = add_indicators(df, cfg)
            df = add_vp(df, cfg)
            dfs[symbol] = df
            print(f"[{n}/10] Loaded {symbol}: {len(df)} rows")
        except Exception as e:
            print(f"    Error loading {symbol}: {e}")

    if not dfs:
        print("No data available.")
        return

    print("\nRunning VPBB Score Hunter Backtest...")
    trades_df = run_portfolio_backtest(dfs, cfg)

    out_dir = Path("backtest_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(out_dir / "portfolio_trades.csv", index=False)

    print("\n" + "=" * 78)
    print("SCORE HUNTER PRO BACKTEST RESULTS")
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


if __name__ == "__main__":
    main()
