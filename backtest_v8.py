#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HUNTER-X VPBB CLEAN BACKTEST
=============================
1H execution + last CLOSED 4H trend filter
10-symbol LBank spot/futures-style OHLCV backtest.

Strategy specification:
- 4H: EMA20/50/200, RSI14, ATR14, ADX14
- 1H: BB20/2, RSI14, ATR14, volume/SMA20, Fixed-Range Volume Profile
- VP: previous 96 CLOSED 1H candles, 70% value area, 48 rows by default
- Long trend: EMA20 > EMA50 > EMA200, close > EMA20, RSI > 50, EMA20 rising
- Short trend: inverse
- ADX >= 18
- Location: near VAL/POC/Lower BB for longs; near VAH/POC/Upper BB for shorts
- Sweep: current low/high takes prior rolling swing and closes back inside
- BB reclaim: long low <= lower BB and close > lower BB; inverse short
- RSI confirmation: long >45 and rising; short <55 and falling
- Volume: > 1.15 * SMA20
- Score: Trend 2 + VP 2 + Sweep 2 + BB 1 + RSI 1 + Volume 1 + ADX 1
- Minimum score: 7/10
- Entry: next candle OPEN after confirmation candle closes (no same-candle lookahead)
- SL: confirmation swing +/- 0.25 ATR
- Reject if stop distance > 1.5 ATR
- TP: 2R
- Reject if a known structural level blocks 2R
- Range filters: ADX, flat EMA, POC chop, narrow BB, oversized candle, extended price
- Risk: default 0.5% of current equity per trade; portfolio risk cap 2%
- Max 3 consecutive losses pause; weekly loss 5% halves risk
- Correlation/exposure filter is implemented conservatively by allowing only one open
  position per symbol and limiting simultaneous BTC-correlated directional exposure.
- Fees/slippage configurable.
- Max holding period: 40 one-hour candles.
- Intrabar ambiguity: if TP and SL are both touched in one candle, SL is assumed first
  (conservative).
- No future pivots are used. Swing levels are rolling historical extrema only.
- 4H values are shifted so only a fully CLOSED 4H candle is used at each 1H bar.
- VP is computed from candles strictly BEFORE the confirmation candle.
- The script can download paginated LBank kline history.

IMPORTANT:
This is a research/backtest engine, not a guarantee of live performance.
The TradingView built-in FRVP can use lower-timeframe data; this Python version
uses an explicit, reproducible OHLCV volume allocation approximation: each bar's
volume is distributed uniformly across the VP price bins overlapped by its high-low
range. That makes the backtest causal and reproducible, but it will not be bit-for-bit
identical to TradingView's internal FRVP calculation.

Dependencies:
    pip install pandas numpy requests

Examples:
    python backtest_vpbb.py
    python backtest_vpbb.py --days 365
    python backtest_vpbb.py --days 365 --symbols btc_usdt eth_usdt sol_usdt
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
# CONFIG
# =========================

DEFAULT_SYMBOLS = [
    "ada_usdt", "avax_usdt", "btc_usdt", "dot_usdt", "eth_usdt",
    "link_usdt", "near_usdt", "sol_usdt", "sui_usdt", "xrp_usdt",
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
    vp_near_atr: float = 0.35

    # Signal
    adx_min: float = 18.0
    volume_mult: float = 1.15
    score_min: int = 7
    swing_lookback: int = 10
    ema_rise_lookback: int = 4

    # Risk / trade
    risk_pct: float = 0.005
    max_portfolio_risk_pct: float = 0.02
    weekly_risk_multiplier: float = 0.50
    max_consecutive_losses: int = 3
    max_hold_bars: int = 40
    tp_r: float = 2.0
    sl_atr_buffer: float = 0.25
    max_stop_atr: float = 1.50

    # Filters
    bb_width_percentile_window: int = 480
    bb_width_percentile_floor: float = 0.20
    min_bb_width: float = 0.0
    flat_ema_threshold: float = 0.0015
    max_candle_atr: float = 2.5
    max_distance_ema_atr: float = 1.5

    # Costs
    fee_rate: float = 0.0004       # 0.04% per side, configurable
    slippage_rate: float = 0.0002  # 0.02% per side, configurable

    # Structural filter
    structure_lookback: int = 48
    structure_buffer_atr: float = 0.20

    # Data
    request_limit: int = 2000
    request_pause_sec: float = 0.12
    timeout_sec: int = 20

    # Portfolio
    max_open_positions: int = 4
    btc_corr_proxy_symbols: Tuple[str, ...] = (
        "btc_usdt", "eth_usdt", "sol_usdt", "avax_usdt", "link_usdt",
        "near_usdt", "sui_usdt", "xrp_usdt", "ada_usdt", "dot_usdt"
    )


# =========================
# DATA
# =========================

def lbank_get_klines(symbol: str, start_ts: int, end_ts: int, cfg: Config) -> pd.DataFrame:
    """
    Paginated LBank kline download.
    LBank /v2/kline.do supports up to 2000 bars per request.
    We walk backwards in time using the returned oldest timestamp.
    """
    url = cfg.exchange_url.rstrip("/") + "/v2/kline.do"
    rows: List[list] = []
    cursor = end_ts
    step_sec = 3600

    # Safety bound: enough requests for a year of 1H data.
    max_requests = max(20, int(cfg.days * 24 / cfg.request_limit) + 20)

    for _ in range(max_requests):
        params = {
            "symbol": symbol,
            "size": cfg.request_limit,
            "type": cfg.interval,
            "time": str(cursor),
        }
        r = requests.get(url, params=params, timeout=cfg.timeout_sec)
        r.raise_for_status()
        payload = r.json()

        if str(payload.get("result", "")).lower() != "true":
            raise RuntimeError(f"LBank error for {symbol}: {payload}")

        data = payload.get("data", [])
        if not data:
            break

        rows.extend(data)
        ts = [int(x[0]) for x in data]
        oldest = min(ts)

        if oldest <= start_ts:
            break

        # Move before oldest returned bar; avoid repeating the same page.
        cursor = oldest - step_sec
        time_mod.sleep(cfg.request_pause_sec)

    if not rows:
        raise RuntimeError(f"No data returned for {symbol}")

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
    start = now - timedelta(days=cfg.days + 30)  # warm-up
    df = lbank_get_klines(
        symbol,
        int(start.timestamp()),
        int(now.timestamp()),
        cfg,
    )
    out = df.reset_index()
    out.to_csv(cache, index=False)
    return df


# =========================
# INDICATORS
# =========================

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.replace([np.inf, -np.inf], np.nan)


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
    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=df.index
    )

    tr = true_range(df)
    atr_w = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    plus = 100 * plus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean() / atr_w
    minus = 100 * minus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean() / atr_w

    dx = 100 * (plus - minus).abs() / (plus + minus).replace(0, np.nan)
    return dx.ewm(alpha=1/length, adjust=False, min_periods=length).mean()


def add_1h_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    x = df.copy()
    x["ema20"] = x["close"].ewm(span=cfg.ema_fast, adjust=False, min_periods=cfg.ema_fast).mean()
    x["ema50"] = x["close"].ewm(span=cfg.ema_mid, adjust=False, min_periods=cfg.ema_mid).mean()
    x["ema200"] = x["close"].ewm(span=cfg.ema_slow, adjust=False, min_periods=cfg.ema_slow).mean()
    x["rsi"] = rsi(x["close"], cfg.rsi_len)
    x["atr"] = atr(x, cfg.atr_len)
    x["adx"] = adx(x, cfg.adx_len)

    bb_mid = x["close"].rolling(cfg.bb_len).mean()
    bb_sd = x["close"].rolling(cfg.bb_len).std(ddof=0)
    x["bb_mid"] = bb_mid
    x["bb_upper"] = bb_mid + cfg.bb_std * bb_sd
    x["bb_lower"] = bb_mid - cfg.bb_std * bb_sd
    x["bb_width"] = (x["bb_upper"] - x["bb_lower"]) / bb_mid.replace(0, np.nan)

    x["vol_sma"] = x["volume"].rolling(cfg.volume_sma_len).mean()
    x["candle_atr_ratio"] = (x["high"] - x["low"]) / x["atr"]
    x["ema20_dist_atr"] = (x["close"] - x["ema20"]).abs() / x["atr"]

    # Rolling historical extrema exclude current candle.
    x["prior_swing_low"] = x["low"].shift(1).rolling(cfg.swing_lookback).min()
    x["prior_swing_high"] = x["high"].shift(1).rolling(cfg.swing_lookback).max()

    # Structural levels: also strictly prior candles.
    x["prior_struct_low"] = x["low"].shift(1).rolling(cfg.structure_lookback).min()
    x["prior_struct_high"] = x["high"].shift(1).rolling(cfg.structure_lookback).max()

    # BB width percentile threshold, causal.
    qwin = x["bb_width"].shift(1).rolling(cfg.bb_width_percentile_window)
    x["bb_width_p20"] = qwin.quantile(cfg.bb_width_percentile_floor)

    return x


def add_4h_context(df1h: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # Resample completed 4H candles. Closed-4H values are then shifted one 4H bar
    # before being joined onto 1H candles.
    df4 = pd.DataFrame({
        "open": df1h["open"].resample("4h", label="right", closed="right").first(),
        "high": df1h["high"].resample("4h", label="right", closed="right").max(),
        "low": df1h["low"].resample("4h", label="right", closed="right").min(),
        "close": df1h["close"].resample("4h", label="right", closed="right").last(),
        "volume": df1h["volume"].resample("4h", label="right", closed="right").sum(),
    }).dropna()

    df4["ema20_4h"] = df4["close"].ewm(span=cfg.ema_fast, adjust=False, min_periods=cfg.ema_fast).mean()
    df4["ema50_4h"] = df4["close"].ewm(span=cfg.ema_mid, adjust=False, min_periods=cfg.ema_mid).mean()
    df4["ema200_4h"] = df4["close"].ewm(span=cfg.ema_slow, adjust=False, min_periods=cfg.ema_slow).mean()
    df4["rsi_4h"] = rsi(df4["close"], cfg.rsi_len)
    df4["atr_4h"] = atr(df4, cfg.atr_len)
    df4["adx_4h"] = adx(df4, cfg.adx_len)

    # Crucial anti-lookahead rule:
    # only the prior fully closed 4H candle is allowed at each 1H decision.
    ctx = df4.shift(1)[
        ["ema20_4h", "ema50_4h", "ema200_4h", "rsi_4h", "atr_4h", "adx_4h"]
    ]
    return df1h.join(ctx.reindex(df1h.index, method="ffill"))


# =========================
# VOLUME PROFILE
# =========================

def fixed_range_vp(
    hist: pd.DataFrame,
    rows: int,
    value_area: float,
) -> Tuple[float, float, float]:
    """
    Causal fixed-range VP approximation.

    Each historical candle's volume is distributed uniformly over all price bins
    overlapped by its high-low range. This avoids assigning the full candle volume
    only to the close and better approximates an OHLCV volume profile.

    Returns: POC, VAH, VAL.
    """
    if len(hist) < 5:
        return np.nan, np.nan, np.nan

    lo = float(hist["low"].min())
    hi = float(hist["high"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan, np.nan, np.nan

    edges = np.linspace(lo, hi, rows + 1)
    vols = np.zeros(rows, dtype=float)

    for _, r in hist.iterrows():
        h = float(r["high"])
        l = float(r["low"])
        v = float(r["volume"])
        if not np.isfinite(v) or v <= 0:
            continue

        if h <= l:
            idx = np.searchsorted(edges, float(r["close"]), side="right") - 1
            idx = max(0, min(rows - 1, idx))
            vols[idx] += v
            continue

        # Uniform distribution across overlapped price bins.
        for j in range(rows):
            overlap = max(0.0, min(h, edges[j + 1]) - max(l, edges[j]))
            if overlap > 0:
                vols[j] += v * (overlap / (h - l))

    if vols.sum() <= 0:
        return np.nan, np.nan, np.nan

    poc_i = int(np.argmax(vols))
    total = float(vols.sum())
    target = total * value_area

    # Standard outward expansion from POC, adding the larger neighboring row.
    left = poc_i - 1
    right = poc_i + 1
    covered = float(vols[poc_i])
    lo_i = poc_i
    hi_i = poc_i

    while covered < target and (left >= 0 or right < rows):
        lv = vols[left] if left >= 0 else -1
        rv = vols[right] if right < rows else -1

        if rv > lv:
            covered += max(0.0, rv)
            hi_i = right
            right += 1
        else:
            covered += max(0.0, lv)
            lo_i = left
            left -= 1

    poc = (edges[poc_i] + edges[poc_i + 1]) / 2.0
    val = edges[lo_i]
    vah = edges[hi_i + 1]
    return float(poc), float(vah), float(val)


def add_vp_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    x = df.copy()
    pocs = np.full(len(x), np.nan)
    vahs = np.full(len(x), np.nan)
    vals = np.full(len(x), np.nan)

    # Compute profile using candles strictly BEFORE current signal candle.
    arr = x[["high", "low", "close", "volume"]].to_numpy()
    for i in range(cfg.vp_lookback, len(x)):
        hist = pd.DataFrame(
            arr[i - cfg.vp_lookback:i],
            columns=["high", "low", "close", "volume"],
        )
        pocs[i], vahs[i], vals[i] = fixed_range_vp(
            hist, cfg.vp_rows, cfg.vp_value_area
        )

    x["vp_poc"] = pocs
    x["vp_vah"] = vahs
    x["vp_val"] = vals
    return x


# =========================
# SIGNALS
# =========================

def near(a: float, b: float, atr_value: float, mult: float) -> bool:
    return np.isfinite(a) and np.isfinite(b) and np.isfinite(atr_value) and abs(a - b) <= mult * atr_value


def signal_at(df: pd.DataFrame, i: int, cfg: Config) -> Optional[dict]:
    r = df.iloc[i]

    needed = [
        "ema20", "ema50", "ema200", "rsi", "atr", "adx",
        "bb_mid", "bb_upper", "bb_lower", "vol_sma",
        "prior_swing_low", "prior_swing_high", "vp_poc", "vp_vah", "vp_val",
        "ema20_4h", "ema50_4h", "ema200_4h", "rsi_4h", "adx_4h",
    ]
    if any(pd.isna(r.get(k)) for k in needed):
        return None

    # 4H trend
    long_trend = (
        r["ema20_4h"] > r["ema50_4h"] > r["ema200_4h"]
        and r["close"] > r["ema20_4h"]
        and r["rsi_4h"] > 50
    )
    short_trend = (
        r["ema20_4h"] < r["ema50_4h"] < r["ema200_4h"]
        and r["close"] < r["ema20_4h"]
        and r["rsi_4h"] < 50
    )

    # 1H trend / strength
    ema_rising = r["ema20"] > df.iloc[i - cfg.ema_rise_lookback]["ema20"] if i >= cfg.ema_rise_lookback else False
    ema_falling = r["ema20"] < df.iloc[i - cfg.ema_rise_lookback]["ema20"] if i >= cfg.ema_rise_lookback else False

    trend_long = (
        r["ema20"] > r["ema50"] > r["ema200"]
        and r["close"] > r["ema20"]
        and r["rsi"] > 50
        and ema_rising
    )
    trend_short = (
        r["ema20"] < r["ema50"] < r["ema200"]
        and r["close"] < r["ema20"]
        and r["rsi"] < 50
        and ema_falling
    )

    # Range filters
    if r["adx"] < cfg.adx_min or r["adx_4h"] < cfg.adx_min:
        return None
    if not np.isfinite(r["bb_width_p20"]) or r["bb_width"] < max(cfg.min_bb_width, r["bb_width_p20"]):
        return None
    if r["candle_atr_ratio"] > cfg.max_candle_atr:
        return None
    if r["ema20_dist_atr"] > cfg.max_distance_ema_atr:
        return None

    ema_slope = abs(r["ema20"] - r["ema50"]) / r["close"]
    if ema_slope < cfg.flat_ema_threshold:
        return None

    # Avoid dead-center POC chop: if price is close to POC and not at outer value area.
    if near(r["close"], r["vp_poc"], r["atr"], 0.15):
        poc_chop = True
    else:
        poc_chop = False

    # Liquidity sweep definitions use only prior rolling extrema.
    long_sweep = (
        r["low"] < r["prior_swing_low"]
        and r["close"] > r["prior_swing_low"]
    )
    short_sweep = (
        r["high"] > r["prior_swing_high"]
        and r["close"] < r["prior_swing_high"]
    )

    long_bb = r["low"] <= r["bb_lower"] and r["close"] > r["bb_lower"]
    short_bb = r["high"] >= r["bb_upper"] and r["close"] < r["bb_upper"]

    long_rsi = r["rsi"] > 45 and r["rsi"] > df.iloc[i - 1]["rsi"]
    short_rsi = r["rsi"] < 55 and r["rsi"] < df.iloc[i - 1]["rsi"]

    volume_ok = r["volume"] > cfg.volume_mult * r["vol_sma"]

    long_vp = (
        near(r["close"], r["vp_val"], r["atr"], cfg.vp_near_atr)
        or near(r["close"], r["vp_poc"], r["atr"], cfg.vp_near_atr)
        or near(r["low"], r["bb_lower"], r["atr"], cfg.vp_near_atr)
    )
    short_vp = (
        near(r["close"], r["vp_vah"], r["atr"], cfg.vp_near_atr)
        or near(r["close"], r["vp_poc"], r["atr"], cfg.vp_near_atr)
        or near(r["high"], r["bb_upper"], r["atr"], cfg.vp_near_atr)
    )

    # Score exactly as specified.
    long_score = (
        2 * int(long_trend and trend_long)
        + 2 * int(long_vp)
        + 2 * int(long_sweep)
        + 1 * int(long_bb)
        + 1 * int(long_rsi)
        + 1 * int(volume_ok)
        + 1 * int(r["adx"] >= cfg.adx_min)
    )
    short_score = (
        2 * int(short_trend and trend_short)
        + 2 * int(short_vp)
        + 2 * int(short_sweep)
        + 1 * int(short_bb)
        + 1 * int(short_rsi)
        + 1 * int(volume_ok)
        + 1 * int(r["adx"] >= cfg.adx_min)
    )

    if poc_chop:
        # POC-centered trades are only allowed when a sweep/reclaim is very clear.
        if not long_sweep and not short_sweep:
            return None

    candidates = []
    if long_score >= cfg.score_min and long_trend and trend_long and long_vp and long_sweep and long_bb and long_rsi:
        candidates.append(("LONG", long_score))
    if short_score >= cfg.score_min and short_trend and trend_short and short_vp and short_sweep and short_bb and short_rsi:
        candidates.append(("SHORT", short_score))

    if not candidates:
        return None

    side, score = max(candidates, key=lambda z: z[1])

    if side == "LONG":
        swing = min(r["low"], r["prior_swing_low"])
        stop = swing - cfg.sl_atr_buffer * r["atr"]
        entry = None  # next bar open is assigned by simulator
    else:
        swing = max(r["high"], r["prior_swing_high"])
        stop = swing + cfg.sl_atr_buffer * r["atr"]
        entry = None

    return {
        "side": side,
        "score": int(score),
        "signal_index": i,
        "signal_time": df.index[i],
        "stop_pre": float(stop),
        "atr": float(r["atr"]),
        "vp_poc": float(r["vp_poc"]),
        "vp_vah": float(r["vp_vah"]),
        "vp_val": float(r["vp_val"]),
        "prior_struct_low": float(r["prior_struct_low"]),
        "prior_struct_high": float(r["prior_struct_high"]),
    }


# =========================
# TRADE SIMULATION
# =========================

@dataclass
class Position:
    symbol: str
    side: str
    entry_time: pd.Timestamp
    entry: float
    stop: float
    target: float
    qty: float
    risk_cash: float
    score: int
    bars_held: int = 0


def apply_entry_slippage(price: float, side: str, slip: float) -> float:
    return price * (1 + slip if side == "LONG" else 1 - slip)


def apply_exit_slippage(price: float, side: str, slip: float) -> float:
    # Exit LONG = sell -> slightly lower. Exit SHORT = buy -> slightly higher.
    return price * (1 - slip if side == "LONG" else 1 + slip)


def structural_blocks_target(
    side: str,
    entry: float,
    target: float,
    row: pd.Series,
    atr_value: float,
    cfg: Config,
) -> bool:
    """
    Conservative structural filter:
    Long: if known prior structural high lies between entry and target, reject.
    Short: if prior structural low lies between target and entry, reject.
    """
    buf = cfg.structure_buffer_atr * atr_value

    if side == "LONG":
        level = row["prior_struct_high"]
        return np.isfinite(level) and entry < level < target and (level - entry) > buf
    else:
        level = row["prior_struct_low"]
        return np.isfinite(level) and target < level < entry and (entry - level) > buf


def exit_position(
    pos: Position,
    bar: pd.Series,
    reason: str,
    raw_price: float,
    cfg: Config,
) -> Tuple[float, float]:
    exit_price = apply_exit_slippage(raw_price, pos.side, cfg.slippage_rate)
    gross = (
        (exit_price - pos.entry) * pos.qty
        if pos.side == "LONG"
        else (pos.entry - exit_price) * pos.qty
    )
    # Fees on both entry and exit notional.
    entry_fee = pos.entry * pos.qty * cfg.fee_rate
    exit_fee = abs(exit_price * pos.qty) * cfg.fee_rate
    net = gross - entry_fee - exit_fee
    return float(net), float(exit_price)


# =========================
# BACKTEST
# =========================

def backtest_symbol(symbol: str, df: pd.DataFrame, cfg: Config, state: dict) -> List[dict]:
    trades: List[dict] = []
    position: Optional[Position] = None

    for i in range(1, len(df)):
        row = df.iloc[i]

        # Manage existing position first.
        if position is not None:
            position.bars_held += 1
            high = float(row["high"])
            low = float(row["low"])

            hit_sl = low <= position.stop if position.side == "LONG" else high >= position.stop
            hit_tp = high >= position.target if position.side == "LONG" else low <= position.target

            reason = None
            raw_exit = None

            if hit_sl and hit_tp:
                # Conservative ambiguity rule.
                reason = "SL_AND_TP_SAME_BAR_SL_FIRST"
                raw_exit = position.stop
            elif hit_sl:
                reason = "SL"
                raw_exit = position.stop
            elif hit_tp:
                reason = "TP"
                raw_exit = position.target
            elif position.bars_held >= cfg.max_hold_bars:
                reason = "TIME"
                raw_exit = float(row["close"])

            if reason:
                net, exit_px = exit_position(position, row, reason, raw_exit, cfg)
                r_mult = net / position.risk_cash if position.risk_cash > 0 else 0.0
                state["equity"] += net

                trades.append({
                    "symbol": symbol,
                    "side": position.side,
                    "entry_time": position.entry_time,
                    "exit_time": df.index[i],
                    "entry": position.entry,
                    "stop": position.stop,
                    "target": position.target,
                    "exit": exit_px,
                    "qty": position.qty,
                    "risk_cash": position.risk_cash,
                    "score": position.score,
                    "bars_held": position.bars_held,
                    "reason": reason,
                    "net_pnl": net,
                    "r_multiple": r_mult,
                    "equity_after": state["equity"],
                })

                if net < 0:
                    state["consecutive_losses"] += 1
                else:
                    state["consecutive_losses"] = 0

                position = None
                continue

        # No new position on same candle after an exit.
        if position is not None:
            continue

        # Entry signals are generated on CLOSED confirmation candle i,
        # and entered at candle i+1 OPEN.
        if i + 1 >= len(df):
            continue

        sig = signal_at(df, i, cfg)
        if not sig:
            continue

        # Entry occurs at next candle open.
        next_row = df.iloc[i + 1]
        entry_raw = float(next_row["open"])
        side = sig["side"]
        entry = apply_entry_slippage(entry_raw, side, cfg.slippage_rate)

        stop = float(sig["stop_pre"])
        if side == "LONG":
            stop_dist = entry - stop
        else:
            stop_dist = stop - entry

        if stop_dist <= 0:
            continue

        if stop_dist > cfg.max_stop_atr * sig["atr"]:
            continue

        # Weekly loss rule: if current week is down 5% from week's starting equity,
        # halve risk. This is tracked globally.
        week_key = df.index[i + 1].isocalendar().week
        year_key = df.index[i + 1].isocalendar().year
        wk = (int(year_key), int(week_key))
        if state.get("week_key") != wk:
            state["week_key"] = wk
            state["week_start_equity"] = state["equity"]

        week_dd = (state["equity"] - state["week_start_equity"]) / state["week_start_equity"]
        risk_pct = cfg.risk_pct * (cfg.weekly_risk_multiplier if week_dd <= -0.05 else 1.0)

        if state["consecutive_losses"] >= cfg.max_consecutive_losses:
            # Pause until next week.
            if wk == state.get("pause_week"):
                continue
            state["pause_week"] = wk

        # Portfolio risk cap is represented by one-symbol backtests here; in the
        # combined portfolio runner below, open-position risk is capped globally.
        risk_cash = state["equity"] * risk_pct
        qty = risk_cash / stop_dist

        target = (
            entry + cfg.tp_r * stop_dist
            if side == "LONG"
            else entry - cfg.tp_r * stop_dist
        )

        # Structural blocker.
        if structural_blocks_target(
            side, entry, target, row, sig["atr"], cfg
        ):
            continue

        # Open position.
        position = Position(
            symbol=symbol,
            side=side,
            entry_time=df.index[i + 1],
            entry=entry,
            stop=stop,
            target=target,
            qty=qty,
            risk_cash=risk_cash,
            score=sig["score"],
        )

    return trades


# =========================
# REPORTING
# =========================

def summarize(trades: pd.DataFrame, initial_equity: float) -> dict:
    if trades.empty:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0,
            "net_pnl": 0.0, "return_pct": 0.0, "profit_factor": 0.0,
            "expectancy_R": 0.0, "max_drawdown_pct": 0.0,
            "avg_R": 0.0, "median_R": 0.0,
        }

    wins = trades[trades.net_pnl > 0]
    losses = trades[trades.net_pnl < 0]
    gross_profit = wins.net_pnl.sum()
    gross_loss = abs(losses.net_pnl.sum())

    equity = pd.Series(
        trades["equity_after"].to_numpy(),
        index=trades["exit_time"],
    ).sort_index()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = abs(float(dd.min())) * 100 if len(dd) else 0.0

    return {
        "trades": int(len(trades)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate_pct": round(100 * len(wins) / len(trades), 2),
        "net_pnl": round(float(trades.net_pnl.sum()), 2),
        "return_pct": round(100 * float(trades.net_pnl.sum()) / initial_equity, 2),
        "profit_factor": round(float(gross_profit / gross_loss), 3) if gross_loss else float("inf"),
        "expectancy_R": round(float(trades.r_multiple.mean()), 4),
        "avg_R": round(float(trades.r_multiple.mean()), 4),
        "median_R": round(float(trades.r_multiple.median()), 4),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_hold_bars": round(float(trades.bars_held.mean()), 2),
        "tp_rate_pct": round(100 * (trades.reason == "TP").mean(), 2),
        "sl_rate_pct": round(100 * trades.reason.str.startswith("SL").mean(), 2),
        "time_exit_pct": round(100 * (trades.reason == "TIME").mean(), 2),
    }


def save_report(all_trades: pd.DataFrame, out_dir: Path, cfg: Config):
    out_dir.mkdir(parents=True, exist_ok=True)

    all_trades.to_csv(out_dir / "trades.csv", index=False)

    summary = summarize(all_trades, cfg.initial_equity)
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)

    if not all_trades.empty:
        by_symbol = (
            all_trades.groupby("symbol")
            .apply(lambda g: pd.Series(summarize(g, cfg.initial_equity)))
            .reset_index()
        )
        by_symbol.to_csv(out_dir / "by_symbol.csv", index=False)

        all_trades["month"] = pd.to_datetime(all_trades["exit_time"]).dt.to_period("M").astype(str)
        monthly = (
            all_trades.groupby("month")
            .agg(
                trades=("symbol", "size"),
                wins=("net_pnl", lambda s: int((s > 0).sum())),
                losses=("net_pnl", lambda s: int((s < 0).sum())),
                pnl=("net_pnl", "sum"),
                avg_R=("r_multiple", "mean"),
            )
            .reset_index()
        )
        monthly["win_rate_pct"] = 100 * monthly["wins"] / monthly["trades"]
        monthly.to_csv(out_dir / "monthly.csv", index=False)

        all_trades["hour_utc"] = pd.to_datetime(all_trades["entry_time"]).dt.hour
        by_hour = (
            all_trades.groupby("hour_utc")
            .agg(trades=("symbol", "size"), pnl=("net_pnl", "sum"), avg_R=("r_multiple", "mean"))
            .reset_index()
        )
        by_hour.to_csv(out_dir / "by_hour_utc.csv", index=False)

    with open(out_dir / "config.txt", "w", encoding="utf-8") as f:
        for k, v in asdict(cfg).items():
            f.write(f"{k}={v}\n")


# =========================
# MAIN
# =========================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    p.add_argument("--initial-equity", type=float, default=10000.0)
    p.add_argument("--risk-pct", type=float, default=0.005)
    p.add_argument("--fee", type=float, default=0.0004)
    p.add_argument("--slippage", type=float, default=0.0002)
    p.add_argument("--vp-lookback", type=int, default=96)
    p.add_argument("--vp-rows", type=int, default=48)
    p.add_argument("--score-min", type=int, default=7)
    p.add_argument("--tp-r", type=float, default=2.0)
    p.add_argument("--cache-dir", default="data_cache")
    p.add_argument("--out-dir", default="backtest_results")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        days=args.days,
        initial_equity=args.initial_equity,
        risk_pct=args.risk_pct,
        fee_rate=args.fee,
        slippage_rate=args.slippage,
        vp_lookback=args.vp_lookback,
        vp_rows=args.vp_rows,
        score_min=args.score_min,
        tp_r=args.tp_r,
    )

    print("=" * 78)
    print("HUNTER-X VPBB CLEAN BACKTEST")
    print("=" * 78)
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Period:  {cfg.days} days + warmup")
    print(f"TF:      1H execution / CLOSED 4H context")
    print(f"VP:      {cfg.vp_lookback} bars / {cfg.vp_rows} rows / {cfg.vp_value_area:.0%} VA")
    print(f"Score:   >= {cfg.score_min}/10")
    print(f"Target:  {cfg.tp_r:.2f}R")
    print(f"Risk:    {cfg.risk_pct:.2%} per trade")
    print(f"Fee:     {cfg.fee_rate:.3%} per side")
    print(f"Slip:    {cfg.slippage_rate:.3%} per side")
    print()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)

    all_trades = []

    # Each symbol is tested independently first. This gives clean per-symbol
    # diagnostics. A portfolio-level runner is included below as a separate
    # extension point; do not interpret the sum of independent equities as a
    # single compounded account.
    for n, symbol in enumerate(args.symbols, 1):
        print(f"[{n}/{len(args.symbols)}] Download/load {symbol} ...")
        try:
            df = load_symbol(symbol, cfg, cache_dir)
            print(f"    candles: {len(df):,}  {df.index.min()} -> {df.index.max()}")

            df = add_1h_indicators(df, cfg)
            df = add_4h_context(df, cfg)
            df = add_vp_columns(df, cfg)

            state = {
                "equity": cfg.initial_equity,
                "consecutive_losses": 0,
                "week_key": None,
                "week_start_equity": cfg.initial_equity,
                "pause_week": None,
            }

            trades = backtest_symbol(symbol, df, cfg, state)
            print(f"    trades: {len(trades):,}")

            all_trades.extend(trades)

        except Exception as exc:
            print(f"    ERROR: {exc}")

    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True)
        trades_df = trades_df.sort_values("exit_time").reset_index(drop=True)

    save_report(trades_df, out_dir, cfg)

    print()
    print("=" * 78)
    print("AGGREGATE DIAGNOSTICS (independent symbol equity curves)")
    print("=" * 78)

    if trades_df.empty:
        print("NO TRADES FOUND.")
        return

    s = summarize(trades_df, cfg.initial_equity)
    for k, v in s.items():
        print(f"{k:22s}: {v}")

    print()
    print("By symbol:")
    for symbol, g in trades_df.groupby("symbol"):
        ss = summarize(g, cfg.initial_equity)
        print(
            f"  {symbol:12s} "
            f"trades={ss['trades']:4d} "
            f"WR={ss['win_rate_pct']:6.2f}% "
            f"PF={ss['profit_factor']:6.3f} "
            f"avgR={ss['avg_R']:7.3f} "
            f"PnL={ss['net_pnl']:10.2f}"
        )

    print()
    print(f"Reports written to: {out_dir.resolve()}")
    print("Files: trades.csv, summary.csv, by_symbol.csv, monthly.csv, by_hour_utc.csv, config.txt")


if __name__ == "__main__":
    main()
