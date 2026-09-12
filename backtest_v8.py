#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
============================================================
HUNTER-X STRATEGY LAB
============================================================

LBank 1H real-market backtest
4H regime + 1H entries

هدف:
مقایسه چند خانواده استراتژی بدون Lookahead و بدون overfitting.

STRATEGIES:
1) TREND_PULLBACK
2) BREAKOUT_RETEST
3) LIQUIDITY_SWEEP
4) MOMENTUM_CONTINUATION
5) VOLATILITY_EXPANSION
6) TREND_PULLBACK_STRONG

EXECUTION:
- Entry = next 1H candle OPEN
- SL = 1R
- TP = 2R
- If TP + SL touched same candle => SL first
- Max hold = 24 candles
- Fees and slippage included
- One position per symbol at a time

DATA:
- LBank via CCXT
- 1H candles
- 4H generated from 1H candles
- Only CLOSED 4H candles are used

DATA SPLIT:
- TRAIN: first 60%
- VALIDATION: next 20%
- OOS: final 20%

IMPORTANT:
OOS must NOT be used to tune strategy parameters.
============================================================
"""

from __future__ import annotations

import os
import time
import math
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DAYS = int(os.getenv("BACKTEST_DAYS", "365"))

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "ADA": "ADA/USDT",
    "AVAX": "AVAX/USDT",
    "LINK": "LINK/USDT",
    "NEAR": "NEAR/USDT",
    "SUI": "SUI/USDT",
    "DOT": "DOT/USDT",
}

TIMEFRAME = "1h"

# Risk model
SL_R = 1.0
TP_R = 2.0

# Execution assumptions
FEE_PER_SIDE = 0.0006
SLIPPAGE_PER_SIDE = 0.0003

# Maximum holding period
MAX_HOLD_BARS = 24

# One position per symbol
ONE_POSITION_PER_SYMBOL = True

# Cache
CACHE_DIR = Path("lbank_cache")
CACHE_DIR.mkdir(exist_ok=True)

OUTPUT_TRADES = "strategy_lab_trades.csv"
OUTPUT_SUMMARY = "strategy_lab_summary.csv"


# ============================================================
# EXCHANGE
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True,
    "timeout": 30000,
})


# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


def adx(df, period=14):

    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0
    )

    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0
    )

    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr_val = tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    plus_di = (
        100 *
        pd.Series(plus_dm, index=df.index)
        .ewm(alpha=1 / period, adjust=False)
        .mean()
        / atr_val
    )

    minus_di = (
        100 *
        pd.Series(minus_dm, index=df.index)
        .ewm(alpha=1 / period, adjust=False)
        .mean()
        / atr_val
    )

    dx = (
        100 *
        (plus_di - minus_di).abs()
        /
        (plus_di + minus_di).replace(0, np.nan)
    )

    return dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


# ============================================================
# DOWNLOAD LBank 1H
# ============================================================

def download_symbol(name, symbol):

    cache_file = CACHE_DIR / f"{name}_1h.csv"

    # --------------------------------------------------------
    # Use cache if it has enough recent data
    # --------------------------------------------------------

    if cache_file.exists():

        try:
            cached = pd.read_csv(cache_file)

            cached["timestamp"] = pd.to_datetime(
                cached["timestamp"],
                utc=True
            )

            if len(cached) >= int(DAYS * 24 * 0.95):

                print(
                    f"  Using cache: {len(cached)} candles"
                )

                return cached

        except Exception:
            pass

    print(f"  Downloading {name} from LBank...")

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=DAYS + 30)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    all_rows = []

    current_since = start_ms

    retries = 0

    while current_since < end_ms:

        try:

            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe=TIMEFRAME,
                since=current_since,
                limit=1000
            )

            if not ohlcv:
                break

            all_rows.extend(ohlcv)

            last_ts = ohlcv[-1][0]

            next_since = last_ts + 1

            if next_since <= current_since:
                break

            current_since = next_since

            retries = 0

            print(
                f"    downloaded {len(all_rows)} candles",
                end="\r"
            )

            if len(ohlcv) < 1000:
                break

            time.sleep(exchange.rateLimit / 1000)

        except Exception as e:

            retries += 1

            print(
                f"\n    retry {retries}/3: {e}"
            )

            if retries >= 3:
                print(
                    "    stopping this symbol after 3 failures"
                )
                break

            time.sleep(3)

    print()

    if not all_rows:
        raise RuntimeError(
            f"No data returned for {name}"
        )

    df = pd.DataFrame(
        all_rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    )

    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Remove incomplete current candle
    now = pd.Timestamp.now(tz="UTC")

    df = df[
        df["timestamp"] + pd.Timedelta(hours=1)
        <= now
    ]

    df.to_csv(
        cache_file,
        index=False
    )

    print(
        f"  {name}: {len(df)} candles saved"
    )

    return df


# ============================================================
# BUILD 4H
# ============================================================

def build_4h(df1):

    x = df1.set_index("timestamp")

    df4 = x.resample("4h", label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    df4 = df4.dropna().reset_index()

    # Indicators
    df4["ema20"] = ema(df4["close"], 20)
    df4["ema50"] = ema(df4["close"], 50)
    df4["ema200"] = ema(df4["close"], 200)

    df4["rsi"] = rsi(df4["close"], 14)
    df4["atr"] = atr(df4, 14)
    df4["adx"] = adx(df4, 14)

    # IMPORTANT:
    # Shift everything by one completed 4H candle.
    # This means the 1H bar can only see CLOSED 4H data.
    cols = [
        "close",
        "ema20",
        "ema50",
        "ema200",
        "rsi",
        "atr",
        "adx"
    ]

    for c in cols:
        df4[f"closed_{c}"] = df4[c].shift(1)

    df4 = df4.rename(
        columns={
            "timestamp": "4h_timestamp"
        }
    )

    return df4


# ============================================================
# MERGE 4H INTO 1H
# ============================================================

def prepare_dataframe(df1):

    df = df1.copy()

    df["ema9"] = ema(df["close"], 9)
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)

    df["rsi"] = rsi(df["close"], 14)
    df["atr"] = atr(df, 14)
    df["adx"] = adx(df, 14)

    df["vol_sma20"] = df["volume"].rolling(20).mean()

    # Previous structures
    df["prev_high_20"] = (
        df["high"]
        .rolling(20)
        .max()
        .shift(1)
    )

    df["prev_low_20"] = (
        df["low"]
        .rolling(20)
        .min()
        .shift(1)
    )

    df["prev_high_10"] = (
        df["high"]
        .rolling(10)
        .max()
        .shift(1)
    )

    df["prev_low_10"] = (
        df["low"]
        .rolling(10)
        .min()
        .shift(1)
    )

    # Candle measurements
    df["body"] = (
        df["close"] - df["open"]
    ).abs()

    df["range"] = (
        df["high"] - df["low"]
    )

    df["body_atr"] = (
        df["body"] /
        df["atr"].replace(0, np.nan)
    )

    # Location of close inside candle
    df["close_location"] = (
        (df["close"] - df["low"])
        /
        df["range"].replace(0, np.nan)
    )

    # Bollinger
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()

    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std

    df["bb_width"] = (
        (df["bb_upper"] - df["bb_lower"])
        /
        df["bb_mid"]
    )

    # Historical BB width
    df["bb_width_low_50"] = (
        df["bb_width"]
        .rolling(50)
        .quantile(0.25)
    )

    # ATR expansion
    df["atr_sma20"] = (
        df["atr"].rolling(20).mean()
    )

    # Slopes
    df["ema20_slope"] = (
        df["ema20"] -
        df["ema20"].shift(5)
    )

    df["ema50_slope"] = (
        df["ema50"] -
        df["ema50"].shift(5)
    )

    # --------------------------------------------------------
    # 4H
    # --------------------------------------------------------

    df4 = build_4h(df1)

    df = pd.merge_asof(
        df.sort_values("timestamp"),
        df4.sort_values("4h_timestamp"),
        left_on="timestamp",
        right_on="4h_timestamp",
        direction="backward"
    )

    return df


# ============================================================
# 4H REGIME
# ============================================================

def regime(df, i):

    e20 = df["closed_ema20"].iloc[i]
    e50 = df["closed_ema50"].iloc[i]
    e200 = df["closed_ema200"].iloc[i]
    r = df["closed_rsi"].iloc[i]
    a = df["closed_adx"].iloc[i]

    if any(pd.isna(x) for x in [e20, e50, e200, r, a]):
        return 0

    if (
        e20 > e50 > e200
        and r >= 50
        and a >= 18
    ):
        return 1

    if (
        e20 < e50 < e200
        and r <= 50
        and a >= 18
    ):
        return -1

    return 0


# ============================================================
# STRATEGY 1
# TREND PULLBACK
# ============================================================

def signal_trend_pullback(df, i):

    if i < 250:
        return None

    reg = regime(df, i)

    close = df["close"].iloc[i]
    low = df["low"].iloc[i]
    high = df["high"].iloc[i]

    e20 = df["ema20"].iloc[i]
    e50 = df["ema50"].iloc[i]

    r = df["rsi"].iloc[i]
    a = df["adx"].iloc[i]
    atr_v = df["atr"].iloc[i]

    if any(pd.isna(x) for x in [
        close, low, high,
        e20, e50, r, a, atr_v
    ]):
        return None

    # LONG
    if reg == 1:

        pullback = (
            low <= e20 + 0.35 * atr_v
            and
            close > e20
        )

        bullish = (
            close > df["open"].iloc[i]
            and
            df["close_location"].iloc[i] >= 0.65
        )

        momentum = (
            r >= 50
            and
            df["ema20_slope"].iloc[i] > 0
        )

        if pullback and bullish and momentum:

            entry = df["open"].iloc[i + 1]

            swing = min(
                df["low"].iloc[i],
                df["low"].iloc[i - 1],
                df["low"].iloc[i - 2]
            )

            risk = entry - swing

            if risk <= 0:
                return None

            return {
                "direction": "LONG",
                "entry": entry,
                "stop": swing,
                "strategy": "TREND_PULLBACK"
            }

    # SHORT
    if reg == -1:

        pullback = (
            high >= e20 - 0.35 * atr_v
            and
            close < e20
        )

        bearish = (
            close < df["open"].iloc[i]
            and
            df["close_location"].iloc[i] <= 0.35
        )

        momentum = (
            r <= 50
            and
            df["ema20_slope"].iloc[i] < 0
        )

        if pullback and bearish and momentum:

            entry = df["open"].iloc[i + 1]

            swing = max(
                df["high"].iloc[i],
                df["high"].iloc[i - 1],
                df["high"].iloc[i - 2]
            )

            risk = swing - entry

            if risk <= 0:
                return None

            return {
                "direction": "SHORT",
                "entry": entry,
                "stop": swing,
                "strategy": "TREND_PULLBACK"
            }

    return None


# ============================================================
# STRATEGY 2
# BREAKOUT + RETEST
# ============================================================

def signal_breakout_retest(df, i):

    if i < 250:
        return None

    reg = regime(df, i)

    close = df["close"].iloc[i]
    low = df["low"].iloc[i]
    high = df["high"].iloc[i]

    atr_v = df["atr"].iloc[i]

    if pd.isna(atr_v):
        return None

    # Previous breakout must have happened BEFORE current candle.
    prev_close = df["close"].iloc[i - 1]

    level_high = df["prev_high_20"].iloc[i - 1]
    level_low = df["prev_low_20"].iloc[i - 1]

    if reg == 1 and not pd.isna(level_high):

        breakout = prev_close > level_high

        retest = (
            low <= level_high + 0.25 * atr_v
            and
            close > level_high
        )

        bullish = (
            close > df["open"].iloc[i]
            and
            df["close_location"].iloc[i] >= 0.60
        )

        if breakout and retest and bullish:

            entry = df["open"].iloc[i + 1]

            swing = min(
                low,
                df["low"].iloc[i - 1]
            )

            risk = entry - swing

            if risk > 0:

                return {
                    "direction": "LONG",
                    "entry": entry,
                    "stop": swing,
                    "strategy": "BREAKOUT_RETEST"
                }

    if reg == -1 and not pd.isna(level_low):

        breakout = prev_close < level_low

        retest = (
            high >= level_low - 0.25 * atr_v
            and
            close < level_low
        )

        bearish = (
            close < df["open"].iloc[i]
            and
            df["close_location"].iloc[i] <= 0.40
        )

        if breakout and retest and bearish:

            entry = df["open"].iloc[i + 1]

            swing = max(
                high,
                df["high"].iloc[i - 1]
            )

            risk = swing - entry

            if risk > 0:

                return {
                    "direction": "SHORT",
                    "entry": entry,
                    "stop": swing,
                    "strategy": "BREAKOUT_RETEST"
                }

    return None


# ============================================================
# STRATEGY 3
# LIQUIDITY SWEEP
# ============================================================

def signal_liquidity_sweep(df, i):

    if i < 250:
        return None

    reg = regime(df, i)

    low = df["low"].iloc[i]
    high = df["high"].iloc[i]
    close = df["close"].iloc[i]
    open_ = df["open"].iloc[i]

    prev_low = df["prev_low_20"].iloc[i]
    prev_high = df["prev_high_20"].iloc[i]

    r = df["rsi"].iloc[i]
    atr_v = df["atr"].iloc[i]

    if any(pd.isna(x) for x in [
        prev_low,
        prev_high,
        r,
        atr_v
    ]):
        return None

    # LONG liquidity sweep
    if reg == 1:

        sweep = low < prev_low

        reclaim = close > prev_low

        bullish = (
            close > open_
            and
            df["close_location"].iloc[i] >= 0.65
        )

        rsi_confirm = r > 45

        if sweep and reclaim and bullish and rsi_confirm:

            entry = df["open"].iloc[i + 1]

            stop = low - 0.15 * atr_v

            risk = entry - stop

            if risk > 0:

                return {
                    "direction": "LONG",
                    "entry": entry,
                    "stop": stop,
                    "strategy": "LIQUIDITY_SWEEP"
                }

    # SHORT liquidity sweep
    if reg == -1:

        sweep = high > prev_high

        reclaim = close < prev_high

        bearish = (
            close < open_
            and
            df["close_location"].iloc[i] <= 0.35
        )

        rsi_confirm = r < 55

        if sweep and reclaim and bearish and rsi_confirm:

            entry = df["open"].iloc[i + 1]

            stop = high + 0.15 * atr_v

            risk = stop - entry

            if risk > 0:

                return {
                    "direction": "SHORT",
                    "entry": entry,
                    "stop": stop,
                    "strategy": "LIQUIDITY_SWEEP"
                }

    return None


# ============================================================
# STRATEGY 4
# MOMENTUM CONTINUATION
# ============================================================

def signal_momentum(df, i):

    if i < 250:
        return None

    reg = regime(df, i)

    close = df["close"].iloc[i]
    open_ = df["open"].iloc[i]

    atr_v = df["atr"].iloc[i]
    r = df["rsi"].iloc[i]
    adx_v = df["adx"].iloc[i]

    volume = df["volume"].iloc[i]
    vol_avg = df["vol_sma20"].iloc[i]

    if any(pd.isna(x) for x in [
        atr_v,
        r,
        adx_v,
        vol_avg
    ]):
        return None

    body_atr = df["body_atr"].iloc[i]
    location = df["close_location"].iloc[i]

    # LONG
    if reg == 1:

        momentum = (
            close > open_
            and
            body_atr >= 0.65
            and
            location >= 0.75
            and
            volume >= 1.20 * vol_avg
            and
            r >= 55
            and
            adx_v >= 20
        )

        breakout = (
            close >
            df["prev_high_10"].iloc[i]
        )

        if momentum and breakout:

            entry = df["open"].iloc[i + 1]

            stop = min(
                df["low"].iloc[i],
                df["low"].iloc[i - 1]
            )

            risk = entry - stop

            if risk > 0:

                return {
                    "direction": "LONG",
                    "entry": entry,
                    "stop": stop,
                    "strategy": "MOMENTUM_CONTINUATION"
                }

    # SHORT
    if reg == -1:

        momentum = (
            close < open_
            and
            body_atr >= 0.65
            and
            location <= 0.25
            and
            volume >= 1.20 * vol_avg
            and
            r <= 45
            and
            adx_v >= 20
        )

        breakdown = (
            close <
            df["prev_low_10"].iloc[i]
        )

        if momentum and breakdown:

            entry = df["open"].iloc[i + 1]

            stop = max(
                df["high"].iloc[i],
                df["high"].iloc[i - 1]
            )

            risk = stop - entry

            if risk > 0:

                return {
                    "direction": "SHORT",
                    "entry": entry,
                    "stop": stop,
                    "strategy": "MOMENTUM_CONTINUATION"
                }

    return None


# ============================================================
# STRATEGY 5
# VOLATILITY EXPANSION
# ============================================================

def signal_volatility(df, i):

    if i < 250:
        return None

    reg = regime(df, i)

    close = df["close"].iloc[i]
    open_ = df["open"].iloc[i]

    atr_v = df["atr"].iloc[i]
    atr_avg = df["atr_sma20"].iloc[i]

    bb_width = df["bb_width"].iloc[i]
    bb_low = df["bb_width_low_50"].iloc[i]

    if any(pd.isna(x) for x in [
        atr_v,
        atr_avg,
        bb_width,
        bb_low
    ]):
        return None

    volume = df["volume"].iloc[i]
    vol_avg = df["vol_sma20"].iloc[i]

    if pd.isna(vol_avg):
        return None

    expansion = (
        atr_v > 1.15 * atr_avg
        and
        bb_width > bb_low
        and
        volume > 1.15 * vol_avg
    )

    # LONG
    if reg == 1:

        bullish = (
            close > open_
            and
            close > df["prev_high_20"].iloc[i]
            and
            df["close_location"].iloc[i] >= 0.70
        )

        if expansion and bullish:

            entry = df["open"].iloc[i + 1]

            stop = min(
                df["low"].iloc[i],
                df["low"].iloc[i - 1]
            )

            risk = entry - stop

            if risk > 0:

                return {
                    "direction": "LONG",
                    "entry": entry,
                    "stop": stop,
                    "strategy": "VOLATILITY_EXPANSION"
                }

    # SHORT
    if reg == -1:

        bearish = (
            close < open_
            and
            close < df["prev_low_20"].iloc[i]
            and
            df["close_location"].iloc[i] <= 0.30
        )

        if expansion and bearish:

            entry = df["open"].iloc[i + 1]

            stop = max(
                df["high"].iloc[i],
                df["high"].iloc[i - 1]
            )

            risk = stop - entry

            if risk > 0:

                return {
                    "direction": "SHORT",
                    "entry": entry,
                    "stop": stop,
                    "strategy": "VOLATILITY_EXPANSION"
                }

    return None


# ============================================================
# STRATEGY 6
# STRONG TREND PULLBACK
# ============================================================

def signal_strong_pullback(df, i):

    if i < 250:
        return None

    reg = regime(df, i)

    close = df["close"].iloc[i]
    low = df["low"].iloc[i]
    high = df["high"].iloc[i]

    e20 = df["ema20"].iloc[i]
    e50 = df["ema50"].iloc[i]

    r = df["rsi"].iloc[i]
    adx_v = df["adx"].iloc[i]

    if any(pd.isna(x) for x in [
        close,
        e20,
        e50,
        r,
        adx_v
    ]):
        return None

    # LONG
    if reg == 1:

        strong_trend = (
            e20 > e50
            and
            df["ema20_slope"].iloc[i] > 0
            and
            df["ema50_slope"].iloc[i] > 0
            and
            adx_v >= 22
        )

        pullback = (
            low <= e20
            and
            close > e20
            and
            close > e50
        )

        recovery = (
            close > df["open"].iloc[i]
            and
            df["close_location"].iloc[i] >= 0.70
            and
            r >= 52
        )

        if strong_trend and pullback and recovery:

            entry = df["open"].iloc[i + 1]

            stop = min(
                low,
                df["low"].iloc[i - 1]
            )

            risk = entry - stop

            if risk > 0:

                return {
                    "direction": "LONG",
                    "entry": entry,
                    "stop": stop,
                    "strategy": "STRONG_TREND_PULLBACK"
                }

    # SHORT
    if reg == -1:

        strong_trend = (
            e20 < e50
            and
            df["ema20_slope"].iloc[i] < 0
            and
            df["ema50_slope"].iloc[i] < 0
            and
            adx_v >= 22
        )

        pullback = (
            high >= e20
            and
            close < e20
            and
            close < e50
        )

        recovery = (
            close < df["open"].iloc[i]
            and
            df["close_location"].iloc[i] <= 0.30
            and
            r <= 48
        )

        if strong_trend and pullback and recovery:

            entry = df["open"].iloc[i + 1]

            stop = max(
                high,
                df["high"].iloc[i - 1]
            )

            risk = stop - entry

            if risk > 0:

                return {
                    "direction": "SHORT",
                    "entry": entry,
                    "stop": stop,
                    "strategy": "STRONG_TREND_PULLBACK"
                }

    return None


# ============================================================
# STRATEGY MAP
# ============================================================

STRATEGIES = {
    "TREND_PULLBACK": signal_trend_pullback,
    "BREAKOUT_RETEST": signal_breakout_retest,
    "LIQUIDITY_SWEEP": signal_liquidity_sweep,
    "MOMENTUM_CONTINUATION": signal_momentum,
    "VOLATILITY_EXPANSION": signal_volatility,
    "STRONG_TREND_PULLBACK": signal_strong_pullback,
}


# ============================================================
# EXECUTION
# ============================================================

def apply_entry_slippage(entry, direction):

    if direction == "LONG":
        return entry * (1 + SLIPPAGE_PER_SIDE)

    return entry * (1 - SLIPPAGE_PER_SIDE)


def apply_exit_slippage(price, direction):

    if direction == "LONG":
        return price * (1 - SLIPPAGE_PER_SIDE)

    return price * (1 + SLIPPAGE_PER_SIDE)


def trade_pnl_r(gross_r):

    # Approximate fees in R.
    # Calculated from actual notional movement.
    return gross_r


# ============================================================
# SIMULATE ONE SIGNAL
# ============================================================

def simulate_trade(df, signal_index, signal):

    entry_index = signal_index + 1

    if entry_index >= len(df):
        return None

    direction = signal["direction"]

    raw_entry = float(
        df["open"].iloc[entry_index]
    )

    entry = apply_entry_slippage(
        raw_entry,
        direction
    )

    stop_raw = float(signal["stop"])

    if direction == "LONG":

        risk = entry - stop_raw

        if risk <= 0:
            return None

        stop = stop_raw
        target = entry + TP_R * risk

    else:

        risk = stop_raw - entry

        if risk <= 0:
            return None

        stop = stop_raw
        target = entry - TP_R * risk

    # Reject absurdly large stop.
    # Prevent one unusual candle from producing distorted R.
    atr_v = df["atr"].iloc[signal_index]

    if pd.isna(atr_v) or atr_v <= 0:
        return None

    stop_atr = risk / atr_v

    if stop_atr > 3.0:
        return None

    max_exit = min(
        len(df) - 1,
        entry_index + MAX_HOLD_BARS
    )

    result = "TIMEOUT"
    exit_price = float(
        df["close"].iloc[max_exit]
    )

    exit_index = max_exit

    for j in range(
        entry_index,
        max_exit + 1
    ):

        high = float(df["high"].iloc[j])
        low = float(df["low"].iloc[j])

        if direction == "LONG":

            hit_sl = low <= stop
            hit_tp = high >= target

            if hit_sl and hit_tp:
                result = "SL"
                exit_price = stop
                exit_index = j
                break

            if hit_sl:
                result = "SL"
                exit_price = stop
                exit_index = j
                break

            if hit_tp:
                result = "TP"
                exit_price = target
                exit_index = j
                break

        else:

            hit_sl = high >= stop
            hit_tp = low <= target

            if hit_sl and hit_tp:
                result = "SL"
                exit_price = stop
                exit_index = j
                break

            if hit_sl:
                result = "SL"
                exit_price = stop
                exit_index = j
                break

            if hit_tp:
                result = "TP"
                exit_price = target
                exit_index = j
                break

    # Apply exit slippage
    exit_exec = apply_exit_slippage(
        exit_price,
        direction
    )

    # --------------------------------------------------------
    # Gross R
    # --------------------------------------------------------

    if direction == "LONG":

        gross_r = (
            exit_exec - entry
        ) / risk

    else:

        gross_r = (
            entry - exit_exec
        ) / risk

    # --------------------------------------------------------
    # Fees
    # --------------------------------------------------------

    # Approximate fee in price terms.
    total_fee_fraction = (
        FEE_PER_SIDE * 2
    )

    fee_r = (
        entry * total_fee_fraction
    ) / risk

    net_r = gross_r - fee_r

    return {
        "signal_time": df["timestamp"].iloc[signal_index],
        "entry_time": df["timestamp"].iloc[entry_index],
        "exit_time": df["timestamp"].iloc[exit_index],

        "direction": direction,

        "entry": entry,
        "stop": stop,
        "target": target,
        "exit": exit_exec,

        "result": result,

        "gross_r": gross_r,
        "net_r": net_r,

        "hold_bars": exit_index - entry_index,

        "stop_atr": stop_atr,

        "strategy": signal["strategy"],
    }


# ============================================================
# BACKTEST ONE STRATEGY / SYMBOL
# ============================================================

def backtest_strategy(df, strategy_name):

    signal_func = STRATEGIES[strategy_name]

    trades = []

    i = 250

    next_free_index = 0

    while i < len(df) - 2:

        if i < next_free_index:
            i += 1
            continue

        signal = signal_func(df, i)

        if signal is None:
            i += 1
            continue

        trade = simulate_trade(
            df,
            i,
            signal
        )

        if trade is None:
            i += 1
            continue

        trades.append(trade)

        # One position per symbol.
        # Do not allow overlapping trades.
        next_free_index = (
            i
            + 1
            + trade["hold_bars"]
            + 1
        )

        i += 1

    return pd.DataFrame(trades)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(trades):

    if trades.empty:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "timeouts": 0,
            "win_rate": 0.0,
            "resolved_win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "total_r": 0.0,
            "max_drawdown_r": 0.0,
            "avg_r": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "trades_per_day": 0.0,
        }

    r = trades["net_r"].astype(float)

    wins = (
        trades["result"] == "TP"
    ).sum()

    losses = (
        trades["result"] == "SL"
    ).sum()

    timeouts = (
        trades["result"] == "TIMEOUT"
    ).sum()

    resolved = wins + losses

    gross_profit = (
        r[r > 0].sum()
    )

    gross_loss = (
        -r[r < 0].sum()
    )

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    for x in r:

        equity += x

        peak = max(
            peak,
            equity
        )

        dd = peak - equity

        max_dd = max(
            max_dd,
            dd
        )

    if len(trades) > 1:

        days = (
            trades["exit_time"].max()
            -
            trades["signal_time"].min()
        ).total_seconds() / 86400

        days = max(days, 1)

    else:
        days = 1

    avg_win = (
        r[r > 0].mean()
        if wins
        else 0
    )

    avg_loss = (
        r[r < 0].mean()
        if losses
        else 0
    )

    return {
        "trades": len(trades),
        "wins": int(wins),
        "losses": int(losses),
        "timeouts": int(timeouts),

        "win_rate":
            100 * wins / len(trades),

        "resolved_win_rate":
            100 * wins / resolved
            if resolved
            else 0,

        "profit_factor": pf,

        "expectancy_r":
            r.mean(),

        "total_r":
            r.sum(),

        "max_drawdown_r":
            -max_dd,

        "avg_r":
            r.mean(),

        "avg_win_r":
            avg_win,

        "avg_loss_r":
            avg_loss,

        "trades_per_day":
            len(trades) / days,
    }


# ============================================================
# DATA SPLIT
# ============================================================

def add_split_labels(df):

    n = len(df)

    train_end = int(
        n * 0.60
    )

    validation_end = int(
        n * 0.80
    )

    df = df.copy()

    df["split"] = "OOS"

    df.loc[
        :train_end - 1,
        "split"
    ] = "TRAIN"

    df.loc[
        train_end:validation_end - 1,
        "split"
    ] = "VALIDATION"

    return df


# ============================================================
# RUN ALL STRATEGIES
# ============================================================

def run_symbol(name, symbol):

    print()
    print("=" * 70)
    print(f"{name} | {symbol}")
    print("=" * 70)

    df1 = download_symbol(
        name,
        symbol
    )

    if len(df1) < 500:

        print(
            f"{name}: insufficient data "
            f"({len(df1)})"
        )

        return []

    df = prepare_dataframe(df1)

    df = add_split_labels(df)

    results = []

    for strategy_name in STRATEGIES:

        print(
            f"  Testing {strategy_name}..."
        )

        trades = backtest_strategy(
            df,
            strategy_name
        )

        if trades.empty:
            print("    no trades")
            continue

        # Add symbol
        trades["symbol"] = name

        # Determine split from signal candle
        split_map = (
            df.set_index("timestamp")["split"]
            .to_dict()
        )

        trades["split"] = (
            trades["signal_time"]
            .map(split_map)
            .fillna("OOS")
        )

        for split_name in [
            "TRAIN",
            "VALIDATION",
            "OOS"
        ]:

            part = trades[
                trades["split"] == split_name
            ].copy()

            m = calculate_metrics(part)

            row = {
                "strategy": strategy_name,
                "symbol": name,
                "split": split_name,
                **m
            }

            results.append(row)

    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(summary):

    print()
    print("=" * 110)
    print("STRATEGY LAB RESULTS")
    print("=" * 110)

    # Validation ranking
    val = summary[
        summary["split"] == "VALIDATION"
    ].copy()

    if not val.empty:

        agg = (
            val.groupby("strategy")
            .agg({
                "trades": "sum",
                "wins": "sum",
                "losses": "sum",
                "timeouts": "sum",
                "total_r": "sum",
                "expectancy_r": "mean",
                "profit_factor": "mean",
            })
            .reset_index()
        )

        print()
        print("VALIDATION RANKING")
        print("-" * 110)

        agg = agg.sort_values(
            [
                "total_r",
                "profit_factor"
            ],
            ascending=False
        )

        print(
            agg.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.3f}"
            )
        )

    # OOS
    oos = summary[
        summary["split"] == "OOS"
    ].copy()

    if not oos.empty:

        print()
        print("OOS RESULTS")
        print("-" * 110)

        agg_oos = (
            oos.groupby("strategy")
            .agg({
                "trades": "sum",
                "wins": "sum",
                "losses": "sum",
                "timeouts": "sum",
                "total_r": "sum",
                "expectancy_r": "mean",
                "profit_factor": "mean",
            })
            .reset_index()
        )

        agg_oos = agg_oos.sort_values(
            "total_r",
            ascending=False
        )

        print(
            agg_oos.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.3f}"
            )
        )


# ============================================================
# FINAL PORTFOLIO REPORT
# ============================================================

def print_portfolio_report(trades):

    print()
    print("=" * 110)
    print("PORTFOLIO REPORT")
    print("=" * 110)

    if trades.empty:
        print("NO TRADES")
        return

    for split in [
        "TRAIN",
        "VALIDATION",
        "OOS"
    ]:

        part = trades[
            trades["split"] == split
        ]

        if part.empty:
            continue

        print()
        print(f"===== {split} =====")

        m = calculate_metrics(part)

        for k, v in m.items():

            if isinstance(v, float):
                print(
                    f"{k:22s}: {v:.3f}"
                )
            else:
                print(
                    f"{k:22s}: {v}"
                )

    # Strategy OOS
    print()
    print("OOS BY STRATEGY")
    print("-" * 90)

    oos = trades[
        trades["split"] == "OOS"
    ]

    for strategy in STRATEGIES:

        part = oos[
            oos["strategy"] == strategy
        ]

        m = calculate_metrics(part)

        print(
            f"{strategy:25s} "
            f"Trades={m['trades']:4d} "
            f"WR={m['win_rate']:6.2f}% "
            f"PF={m['profit_factor']:5.2f} "
            f"R={m['total_r']:8.2f} "
            f"Exp={m['expectancy_r']:7.3f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("HUNTER-X STRATEGY LAB")
    print("=" * 80)

    print(
        f"Exchange      : LBank"
    )

    print(
        f"Period        : {DAYS} days"
    )

    print(
        f"Symbols       : {len(SYMBOLS)}"
    )

    print(
        f"SL            : {SL_R}R"
    )

    print(
        f"TP            : {TP_R}R"
    )

    print(
        f"Fee / side    : {FEE_PER_SIDE}"
    )

    print(
        f"Slippage/side : {SLIPPAGE_PER_SIDE}"
    )

    print(
        f"Max hold      : {MAX_HOLD_BARS}H"
    )

    print(
        "Data split    : 60% TRAIN / 20% VALIDATION / 20% OOS"
    )

    print()
    print(
        "NO LOOKAHEAD MODE ENABLED"
    )

    all_results = []

    for name, symbol in SYMBOLS.items():

        try:

            rows = run_symbol(
                name,
                symbol
            )

            all_results.extend(rows)

        except Exception as e:

            print()
            print(
                f"ERROR {name}: "
                f"{type(e).__name__}: {e}"
            )

    if not all_results:

        print()
        print(
            "NO RESULTS."
        )

        return

    summary = pd.DataFrame(
        all_results
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    print_results(
        summary
    )

    # --------------------------------------------------------
    # Re-run all strategies to create full trade dataset.
    # --------------------------------------------------------

    all_trades = []

    print()
    print("=" * 80)
    print("BUILDING FULL TRADE REPORT")
    print("=" * 80)

    for name, symbol in SYMBOLS.items():

        try:

            df1 = download_symbol(
                name,
                symbol
            )

            df = prepare_dataframe(
                df1
            )

            df = add_split_labels(
                df
            )

            split_map = (
                df.set_index("timestamp")["split"]
                .to_dict()
            )

            for strategy_name in STRATEGIES:

                trades = backtest_strategy(
                    df,
                    strategy_name
                )

                if trades.empty:
                    continue

                trades["symbol"] = name

                trades["split"] = (
                    trades["signal_time"]
                    .map(split_map)
                    .fillna("OOS")
                )

                all_trades.append(
                    trades
                )

        except Exception as e:

            print(
                f"Trade report error {name}: {e}"
            )

    if all_trades:

        trades = pd.concat(
            all_trades,
            ignore_index=True
        )

        trades = trades.sort_values(
            "entry_time"
        ).reset_index(
            drop=True
        )

        trades.to_csv(
            OUTPUT_TRADES,
            index=False
        )

        print_portfolio_report(
            trades
        )

    print()
    print("=" * 80)
    print("FILES")
    print("=" * 80)

    print(
        f"Summary : {OUTPUT_SUMMARY}"
    )

    print(
        f"Trades  : {OUTPUT_TRADES}"
    )

    print()
    print("DONE.")


if __name__ == "__main__":
    main()
