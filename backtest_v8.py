#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLE-1
Confluence Liquidity Engine
Causal Crypto Futures Backtest

IMPORTANT:
- Research system, NOT a profitability guarantee.
- RR = exactly 1:2
- Entry = next 15m candle OPEN
- No timeout
- No lookahead
- No centered rolling
- No future exit timestamp used for active-position management
"""

import time
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DAYS = 365
WARMUP_DAYS = 70

TIMEFRAME = "15m"
LIMIT = 1000

# Trading model
TRADE_MARGIN = 100.0
LEVERAGE = 50.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003

# EXACT RR
RR = 2.0

# Portfolio
MAX_OPEN_POSITIONS = 3

# 4-loss breaker is OFF in the baseline.
# We want to measure natural loss streak first.
BREAKER_ENABLED = False
BREAKER_AFTER_LOSSES = 4

# Indicators
ATR_LEN = 14
RVOL_LEN = 20
EMA_FAST = 50
EMA_SLOW = 200
ADX_LEN = 14

# Initial research thresholds.
# These are NOT optimized to force 50% WR.
MIN_SCORE_REVERSAL = 9
MIN_SCORE_CONTINUATION = 8

# Requested universe.
# Actual LBank symbols are discovered dynamically.
TARGET_BASES = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "ADA",
    "AVAX",
    "LINK",
    "DOGE",
    "DOT",
]

OUT_DIR = Path("cle1_results")
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Candidate:

    symbol: str
    setup: str

    signal_time: pd.Timestamp
    entry_time: pd.Timestamp

    side: str

    entry: float
    sl: float
    tp: float

    score: int
    reason: str


@dataclass
class Trade:

    symbol: str
    setup: str
    side: str

    signal_time: str
    entry_time: str
    exit_time: str

    entry: float
    sl: float
    tp: float
    exit: float

    result: str
    pnl: float

    score: int
    holding_hours: float


# ============================================================
# EXCHANGE
# ============================================================

def make_exchange():

    return ccxt.lbank({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {
            "defaultType": "swap"
        }
    })


# ============================================================
# DYNAMIC LBank SYMBOL DISCOVERY
# ============================================================

def discover_symbols(exchange):

    exchange.load_markets()

    selected = []

    for symbol, market in exchange.markets.items():

        try:

            if not market.get("swap"):
                continue

            if not market.get("linear"):
                continue

            if market.get("quote") != "USDT":
                continue

            if market.get("base") not in TARGET_BASES:
                continue

            if market.get("active") is False:
                continue

            selected.append(symbol)

        except Exception:
            continue

    by_base = {}

    for symbol in selected:

        base = exchange.markets[symbol].get("base")

        if base not in by_base:
            by_base[base] = symbol

    ordered = [
        by_base[b]
        for b in TARGET_BASES
        if b in by_base
    ]

    print()
    print("=" * 70)
    print("DISCOVERED LBank LINEAR USDT SWAPS")
    print("=" * 70)

    for symbol in ordered:
        print(symbol)

    missing = [
        b for b in TARGET_BASES
        if b not in by_base
    ]

    if missing:

        print()
        print("Missing requested bases:")
        print(", ".join(missing))

    print("=" * 70)

    return ordered


# ============================================================
# OHLCV DOWNLOAD
# ============================================================

def fetch_ohlcv_paginated(
    exchange,
    symbol,
    since_ms,
    until_ms
):

    rows = []

    current_since = since_ms

    while current_since < until_ms:

        try:

            batch = exchange.fetch_ohlcv(
                symbol,
                timeframe=TIMEFRAME,
                since=current_since,
                limit=LIMIT
            )

        except Exception as e:

            print(
                f"ERROR {symbol}: "
                f"{type(e).__name__}: {e}"
            )

            break

        if not batch:
            break

        rows.extend(batch)

        last_ts = batch[-1][0]

        if last_ts >= until_ms:
            break

        next_since = last_ts + 1

        if next_since <= current_since:
            break

        current_since = next_since

        time.sleep(0.20)

    if not rows:

        return pd.DataFrame(
            columns=[
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df["ts"] = pd.to_datetime(
        df["ts"],
        unit="ms",
        utc=True
    )

    df = (
        df
        .drop_duplicates("ts")
        .sort_values("ts")
    )

    df = df[
        (df["ts"].astype("int64") // 10**6 >= since_ms)
        &
        (df["ts"].astype("int64") // 10**6 <= until_ms)
    ]

    df = df.set_index("ts")

    df = df[
        [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ].astype(float)

    # --------------------------------------------------------
    # REMOVE CURRENT OPEN 15m CANDLE
    # --------------------------------------------------------

    tf_ms = 15 * 60 * 1000

    now_ms = exchange.milliseconds()

    last_complete_open = (
        (now_ms // tf_ms) * tf_ms
        - tf_ms
    )

    df = df[
        (df.index.astype("int64") // 10**6)
        <= last_complete_open
    ]

    return df


# ============================================================
# INDICATORS
# ============================================================

def ema(series, length):

    return series.ewm(
        span=length,
        adjust=False,
        min_periods=length
    ).mean()


def true_range(df):

    previous_close = df["close"].shift(1)

    a = df["high"] - df["low"]

    b = (
        df["high"] - previous_close
    ).abs()

    c = (
        df["low"] - previous_close
    ).abs()

    return pd.concat(
        [a, b, c],
        axis=1
    ).max(axis=1)


def atr(df, length=14):

    tr = true_range(df)

    return tr.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()


def rvol(df, length=20):

    average_volume = (
        df["volume"]
        .rolling(
            length,
            min_periods=length
        )
        .mean()
    )

    return (
        df["volume"]
        / average_volume
    )


def adx(df, length=14):

    up_move = df["high"].diff()

    down_move = -df["low"].diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move)
            & (up_move > 0),
            up_move,
            0.0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move)
            & (down_move > 0),
            down_move,
            0.0
        ),
        index=df.index
    )

    tr = true_range(df)

    atr_value = tr.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / length,
            adjust=False,
            min_periods=length
        ).mean()
        / atr_value
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / length,
            adjust=False,
            min_periods=length
        ).mean()
        / atr_value
    )

    denominator = (
        plus_di + minus_di
    ).replace(0, np.nan)

    dx = (
        100
        * (plus_di - minus_di).abs()
        / denominator
    )

    return dx.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()


def daily_vwap(df):

    day = df.index.floor("D")

    pv = (
        df["close"]
        * df["volume"]
    )

    cumulative_pv = (
        pv.groupby(day)
        .cumsum()
    )

    cumulative_volume = (
        df["volume"]
        .groupby(day)
        .cumsum()
    )

    return (
        cumulative_pv
        / cumulative_volume.replace(
            0,
            np.nan
        )
    )


# ============================================================
# CONFIRMED PIVOTS
# ============================================================

def confirmed_pivots(
    df,
    left=2,
    right=2
):

    highs = df["high"]
    lows = df["low"]

    pivot_high = pd.Series(
        False,
        index=df.index
    )

    pivot_low = pd.Series(
        False,
        index=df.index
    )

    # IMPORTANT:
    # At candle i, we are confirming a pivot that occurred
    # right candles earlier.
    #
    # No centered rolling.
    # No future leakage.

    for i in range(
        left + right,
        len(df)
    ):

        p = i - right

        left_highs = highs.iloc[
            p - left:p
        ]

        right_highs = highs.iloc[
            p + 1:p + right + 1
        ]

        left_lows = lows.iloc[
            p - left:p
        ]

        right_lows = lows.iloc[
            p + 1:p + right + 1
        ]

        current_high = highs.iloc[p]
        current_low = lows.iloc[p]

        if (
            len(left_highs) == left
            and len(right_highs) == right
        ):

            if (
                current_high > left_highs.max()
                and
                current_high >= right_highs.max()
            ):

                pivot_high.iloc[i] = True

            if (
                current_low < left_lows.min()
                and
                current_low <= right_lows.min()
            ):

                pivot_low.iloc[i] = True

    return pivot_high, pivot_low


def add_structure(df):

    df = df.copy()

    ph, pl = confirmed_pivots(
        df,
        left=2,
        right=2
    )

    df["pivot_high_confirmed"] = ph
    df["pivot_low_confirmed"] = pl

    # The actual pivot occurred 2 candles earlier.
    df["pivot_high_price"] = np.where(
        ph,
        df["high"].shift(2),
        np.nan
    )

    df["pivot_low_price"] = np.where(
        pl,
        df["low"].shift(2),
        np.nan
    )

    df["last_swing_high"] = (
        pd.Series(
            df["pivot_high_price"],
            index=df.index
        )
        .ffill()
    )

    df["last_swing_low"] = (
        pd.Series(
            df["pivot_low_price"],
            index=df.index
        )
        .ffill()
    )

    return df


# ============================================================
# FVG
# ============================================================

def fvg_flags(df):

    bullish = (
        df["low"]
        > df["high"].shift(2)
    )

    bearish = (
        df["high"]
        < df["low"].shift(2)
    )

    return (
        bullish.fillna(False),
        bearish.fillna(False)
    )


# ============================================================
# PREPARE 15m
# ============================================================

def prepare_15m(df):

    df = df.copy()

    df["atr"] = atr(
        df,
        ATR_LEN
    )

    df["rvol"] = rvol(
        df,
        RVOL_LEN
    )

    df["ema50"] = ema(
        df["close"],
        EMA_FAST
    )

    df["ema200"] = ema(
        df["close"],
        EMA_SLOW
    )

    df["adx"] = adx(
        df,
        ADX_LEN
    )

    df["vwap"] = daily_vwap(df)

    df["range"] = (
        df["high"]
        - df["low"]
    )

    df["body"] = (
        df["close"]
        - df["open"]
    ).abs()

    df["body_ratio"] = (
        df["body"]
        / df["range"].replace(
            0,
            np.nan
        )
    )

    df["close_location"] = (
        df["close"]
        - df["low"]
    ) / df["range"].replace(
        0,
        np.nan
    )

    bull_fvg, bear_fvg = fvg_flags(df)

    df["bull_fvg"] = bull_fvg
    df["bear_fvg"] = bear_fvg

    return df


# ============================================================
# HTF
# ============================================================

def resample_ohlcv(
    df,
    rule
):

    out = (
        df.resample(
            rule,
            label="right",
            closed="right"
        )
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
        .dropna()
    )

    return out


def prepare_htf(df):

    df = df.copy()

    df["ema50"] = ema(
        df["close"],
        50
    )

    df["ema200"] = ema(
        df["close"],
        200
    )

    df["adx"] = adx(
        df,
        14
    )

    df["atr"] = atr(
        df,
        14
    )

    return df


def map_htf_context(
    base,
    h1,
    h4
):

    h1 = prepare_htf(h1)
    h4 = prepare_htf(h4)

    columns = [
        "close",
        "ema50",
        "ema200",
        "adx",
        "atr"
    ]

    result = base.copy()

    h1_map = (
        h1[columns]
        .reindex(
            result.index,
            method="ffill"
        )
    )

    h4_map = (
        h4[columns]
        .reindex(
            result.index,
            method="ffill"
        )
    )

    for column in columns:

        result[
            f"h1_{column}"
        ] = h1_map[column]

        result[
            f"h4_{column}"
        ] = h4_map[column]

    return result


# ============================================================
# DISPLACEMENT
# ============================================================

def displacement_up(row):

    if not np.isfinite(row["atr"]):
        return False

    return (
        row["body_ratio"] >= 0.60
        and
        row["close_location"] >= 0.75
        and
        row["range"] >= 1.10 * row["atr"]
        and
        row["rvol"] >= 1.10
    )


def displacement_down(row):

    if not np.isfinite(row["atr"]):
        return False

    return (
        row["body_ratio"] >= 0.60
        and
        row["close_location"] <= 0.25
        and
        row["range"] >= 1.10 * row["atr"]
        and
        row["rvol"] >= 1.10
    )


# ============================================================
# CONTEXT
# ============================================================

def get_context(
    df,
    i
):

    row = df.iloc[i]
    previous = df.iloc[i - 1]

    # --------------------------------------------------------
    # 4H REGIME
    # --------------------------------------------------------

    lookback = max(
        0,
        i - 8
    )

    h4_ema50_previous = (
        df["h4_ema50"]
        .iloc[lookback]
    )

    bull_regime = (
        row["h4_close"]
        > row["h4_ema200"]
        and
        row["h4_ema50"]
        > row["h4_ema200"]
        and
        row["h4_ema50"]
        >= h4_ema50_previous
    )

    bear_regime = (
        row["h4_close"]
        < row["h4_ema200"]
        and
        row["h4_ema50"]
        < row["h4_ema200"]
        and
        row["h4_ema50"]
        <= h4_ema50_previous
    )

    # --------------------------------------------------------
    # 1H REGIME / STRUCTURE
    # --------------------------------------------------------

    h1_bull = (
        row["h1_close"]
        > row["h1_ema200"]
        and
        row["h1_ema50"]
        > row["h1_ema200"]
    )

    h1_bear = (
        row["h1_close"]
        < row["h1_ema200"]
        and
        row["h1_ema50"]
        < row["h1_ema200"]
    )

    swing_high = row[
        "last_swing_high"
    ]

    swing_low = row[
        "last_swing_low"
    ]

    # --------------------------------------------------------
    # BOS
    # --------------------------------------------------------

    bull_bos = (
        pd.notna(swing_high)
        and
        row["close"] > swing_high
        and
        previous["close"] <= swing_high
    )

    bear_bos = (
        pd.notna(swing_low)
        and
        row["close"] < swing_low
        and
        previous["close"] >= swing_low
    )

    # --------------------------------------------------------
    # LIQUIDITY SWEEP
    # --------------------------------------------------------

    sweep_low = (
        pd.notna(swing_low)
        and
        row["low"] < swing_low
        and
        row["close"] > swing_low
    )

    sweep_high = (
        pd.notna(swing_high)
        and
        row["high"] > swing_high
        and
        row["close"] < swing_high
    )

    # --------------------------------------------------------
    # VOLUME / VOLATILITY
    # --------------------------------------------------------

    volume_ok = (
        pd.notna(row["rvol"])
        and
        row["rvol"] >= 1.05
    )

    volume_strong = (
        pd.notna(row["rvol"])
        and
        row["rvol"] >= 1.25
    )

    volatility_expansion = (
        pd.notna(row["atr"])
        and
        row["range"]
        >= 1.15 * row["atr"]
    )

    return {

        "bull_regime":
            bull_regime,

        "bear_regime":
            bear_regime,

        "h1_bull":
            h1_bull,

        "h1_bear":
            h1_bear,

        "bull_bos":
            bull_bos,

        "bear_bos":
            bear_bos,

        "sweep_low":
            sweep_low,

        "sweep_high":
            sweep_high,

        "bull_fvg":
            bool(row["bull_fvg"]),

        "bear_fvg":
            bool(row["bear_fvg"]),

        "volume_ok":
            volume_ok,

        "volume_strong":
            volume_strong,

        "volatility_expansion":
            volatility_expansion,
    }


# ============================================================
# CONFLUENCE SCORE
# ============================================================

def score_reversal(
    row,
    context,
    side
):

    score = 0
    reasons = []

    if side == "LONG":

        if context["bull_regime"]:
            score += 2
            reasons.append("4H_BULL")

        if context["h1_bull"]:
            score += 2
            reasons.append("1H_BULL")

        if context["sweep_low"]:
            score += 2
            reasons.append("SELLSIDE_SWEEP")

        if row["close"] > row["open"]:
            score += 1
            reasons.append("BULL_RECLAIM")

        if displacement_up(row):
            score += 2
            reasons.append("DISPLACEMENT")

        if context["bull_fvg"]:
            score += 1
            reasons.append("BULL_FVG")

        if row["close"] > row["vwap"]:
            score += 1
            reasons.append("VWAP")

        if context["volume_strong"]:
            score += 1
            reasons.append("RVOL")

        if context["volatility_expansion"]:
            score += 1
            reasons.append("VOL_EXPANSION")

    else:

        if context["bear_regime"]:
            score += 2
            reasons.append("4H_BEAR")

        if context["h1_bear"]:
            score += 2
            reasons.append("1H_BEAR")

        if context["sweep_high"]:
            score += 2
            reasons.append("BUYSIDE_SWEEP")

        if row["close"] < row["open"]:
            score += 1
            reasons.append("BEAR_RECLAIM")

        if displacement_down(row):
            score += 2
            reasons.append("DISPLACEMENT")

        if context["bear_fvg"]:
            score += 1
            reasons.append("BEAR_FVG")

        if row["close"] < row["vwap"]:
            score += 1
            reasons.append("VWAP")

        if context["volume_strong"]:
            score += 1
            reasons.append("RVOL")

        if context["volatility_expansion"]:
            score += 1
            reasons.append("VOL_EXPANSION")

    return score, ",".join(reasons)


def score_continuation(
    row,
    context,
    side
):

    score = 0
    reasons = []

    if side == "LONG":

        if context["bull_regime"]:
            score += 2
            reasons.append("4H_BULL")

        if context["h1_bull"]:
            score += 2
            reasons.append("1H_BULL")

        if context["bull_bos"]:
            score += 2
            reasons.append("BOS")

        if displacement_up(row):
            score += 2
            reasons.append("DISPLACEMENT")

        if context["bull_fvg"]:
            score += 1
            reasons.append("BULL_FVG")

        if row["close"] > row["vwap"]:
            score += 1
            reasons.append("VWAP")

        if context["volume_strong"]:
            score += 1
            reasons.append("RVOL")

        if context["volatility_expansion"]:
            score += 1
            reasons.append("VOL_EXPANSION")

    else:

        if context["bear_regime"]:
            score += 2
            reasons.append("4H_BEAR")

        if context["h1_bear"]:
            score += 2
            reasons.append("1H_BEAR")

        if context["bear_bos"]:
            score += 2
            reasons.append("BOS")

        if displacement_down(row):
            score += 2
            reasons.append("DISPLACEMENT")

        if context["bear_fvg"]:
            score += 1
            reasons.append("BEAR_FVG")

        if row["close"] < row["vwap"]:
            score += 1
            reasons.append("VWAP")

        if context["volume_strong"]:
            score += 1
            reasons.append("RVOL")

        if context["volatility_expansion"]:
            score += 1
            reasons.append("VOL_EXPANSION")

    return score, ",".join(reasons)


# ============================================================
# CANDIDATE CREATION
# ============================================================

def make_candidate(
    df,
    i,
    symbol,
    setup,
    side,
    score,
    reason
):

    if i + 1 >= len(df):
        return None

    row = df.iloc[i]

    signal_time = df.index[i]

    # ENTRY = NEXT CANDLE OPEN
    entry_time = df.index[i + 1]

    entry = float(
        df["open"].iloc[i + 1]
    )

    atr_value = float(
        row["atr"]
    )

    if (
        not np.isfinite(entry)
        or
        not np.isfinite(atr_value)
        or
        atr_value <= 0
    ):
        return None

    swing_low = row[
        "last_swing_low"
    ]

    swing_high = row[
        "last_swing_high"
    ]

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    if side == "LONG":

        if pd.notna(swing_low):

            structural_sl = min(
                float(swing_low),
                float(row["low"])
            )

        else:

            structural_sl = float(
                row["low"]
            )

        sl = (
            structural_sl
            - 0.15 * atr_value
        )

        if sl >= entry:
            return None

        risk = entry - sl

        tp = (
            entry
            + RR * risk
        )

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    else:

        if pd.notna(swing_high):

            structural_sl = max(
                float(swing_high),
                float(row["high"])
            )

        else:

            structural_sl = float(
                row["high"]
            )

        sl = (
            structural_sl
            + 0.15 * atr_value
        )

        if sl <= entry:
            return None

        risk = sl - entry

        tp = (
            entry
            - RR * risk
        )

    return Candidate(
        symbol=symbol,
        setup=setup,
        signal_time=signal_time,
        entry_time=entry_time,
        side=side,
        entry=entry,
        sl=sl,
        tp=tp,
        score=score,
        reason=reason
    )


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(
    df,
    symbol,
    test_start
):

    candidates = []

    # Enough warmup for HTF indicators.
    start_i = 300

    for i in range(
        start_i,
        len(df) - 1
    ):

        signal_time = df.index[i]

        if signal_time < test_start:
            continue

        row = df.iloc[i]

        context = get_context(
            df,
            i
        )

        # ====================================================
        # SETUP A
        # LIQUIDITY SWEEP REVERSAL
        # ====================================================

        if context["sweep_low"]:

            score, reason = score_reversal(
                row,
                context,
                "LONG"
            )

            if score >= MIN_SCORE_REVERSAL:

                candidate = make_candidate(
                    df=df,
                    i=i,
                    symbol=symbol,
                    setup="REVERSAL",
                    side="LONG",
                    score=score,
                    reason=reason
                )

                if candidate:
                    candidates.append(
                        candidate
                    )

        if context["sweep_high"]:

            score, reason = score_reversal(
                row,
                context,
                "SHORT"
            )

            if score >= MIN_SCORE_REVERSAL:

                candidate = make_candidate(
                    df=df,
                    i=i,
                    symbol=symbol,
                    setup="REVERSAL",
                    side="SHORT",
                    score=score,
                    reason=reason
                )

                if candidate:
                    candidates.append(
                        candidate
                    )

        # ====================================================
        # SETUP B
        # TREND CONTINUATION
        # ====================================================

        if context["bull_bos"]:

            score, reason = score_continuation(
                row,
                context,
                "LONG"
            )

            if score >= MIN_SCORE_CONTINUATION:

                candidate = make_candidate(
                    df=df,
                    i=i,
                    symbol=symbol,
                    setup="CONTINUATION",
                    side="LONG",
                    score=score,
                    reason=reason
                )

                if candidate:
                    candidates.append(
                        candidate
                    )

        if context["bear_bos"]:

            score, reason = score_continuation(
                row,
                context,
                "SHORT"
            )

            if score >= MIN_SCORE_CONTINUATION:

                candidate = make_candidate(
                    df=df,
                    i=i,
                    symbol=symbol,
                    setup="CONTINUATION",
                    side="SHORT",
                    score=score,
                    reason=reason
                )

                if candidate:
                    candidates.append(
                        candidate
                    )

    # ========================================================
    # SAME SYMBOL + SAME ENTRY TIME
    # KEEP HIGHEST SCORE
    # ========================================================

    candidates.sort(
        key=lambda x: (
            x.entry_time,
            -x.score
        )
    )

    unique = {}

    for candidate in candidates:

        key = (
            candidate.symbol,
            candidate.entry_time
        )

        if key not in unique:
            unique[key] = candidate

    candidates = list(
        unique.values()
    )

    candidates.sort(
        key=lambda x: x.entry_time
    )

    return candidates


# ============================================================
# EXECUTION
# ============================================================

def entry_slippage(
    price,
    side
):

    if side == "LONG":

        return price * (
            1 + SLIPPAGE
        )

    return price * (
        1 - SLIPPAGE
    )


def exit_slippage(
    price,
    side
):

    if side == "LONG":

        return price * (
            1 - SLIPPAGE
        )

    return price * (
        1 + SLIPPAGE
    )


def calculate_pnl(
    side,
    entry,
    exit_price
):

    notional = (
        TRADE_MARGIN
        * LEVERAGE
    )

    if side == "LONG":

        gross = (
            exit_price - entry
        ) / entry * notional

    else:

        gross = (
            entry - exit_price
        ) / entry * notional

    fees = (
        notional
        * FEE_RATE
        * 2
    )

    return gross - fees


# ============================================================
# PORTFOLIO BACKTEST
# ============================================================

def portfolio_sim(
    all_symbol_data,
    all_candidates
):

    events = []

    for symbol, candidates in all_candidates.items():

        for candidate in candidates:

            events.append(
                candidate
            )

    events.sort(
        key=lambda x: x.entry_time
    )

    active = {}

    trades = []

    last_exit_by_symbol = {}

    loss_streak = 0

    cursor = {
        symbol: 0
        for symbol in all_symbol_data
    }

    # --------------------------------------------------------
    # MANAGE ACTIVE POSITION
    # --------------------------------------------------------

    def manage_position_until(
        symbol,
        until_time
    ):

        nonlocal loss_streak

        if symbol not in active:
            return

        df = all_symbol_data[
            symbol
        ]

        position = active[
            symbol
        ]

        while cursor[symbol] < len(df):

            i = cursor[symbol]

            timestamp = df.index[i]

            if timestamp < position["entry_time"]:

                cursor[symbol] += 1

                continue

            # Do not consume the candle belonging to
            # the new candidate event.
            if timestamp >= until_time:
                break

            row = df.iloc[i]

            side = position["side"]

            # ------------------------------------------------
            # STOP / TARGET
            # ------------------------------------------------

            if side == "LONG":

                hit_sl = (
                    row["low"]
                    <= position["sl"]
                )

                hit_tp = (
                    row["high"]
                    >= position["tp"]
                )

            else:

                hit_sl = (
                    row["high"]
                    >= position["sl"]
                )

                hit_tp = (
                    row["low"]
                    <= position["tp"]
                )

            # ------------------------------------------------
            # EXIT
            # ------------------------------------------------

            if hit_sl or hit_tp:

                # Conservative:
                # if both touched, SL first.
                if hit_sl:

                    result = "LOSS"

                    raw_exit = (
                        position["sl"]
                    )

                else:

                    result = "WIN"

                    raw_exit = (
                        position["tp"]
                    )

                exit_price = exit_slippage(
                    raw_exit,
                    side
                )

                pnl = calculate_pnl(
                    side=side,
                    entry=position["entry"],
                    exit_price=exit_price
                )

                holding_hours = max(
                    0.0,
                    (
                        timestamp
                        - position["entry_time"]
                    ).total_seconds()
                    / 3600
                )

                trades.append(
                    Trade(
                        symbol=symbol,
                        setup=position["setup"],
                        side=side,

                        signal_time=str(
                            position["signal_time"]
                        ),

                        entry_time=str(
                            position["entry_time"]
                        ),

                        exit_time=str(
                            timestamp
                        ),

                        entry=position["entry"],
                        sl=position["sl"],
                        tp=position["tp"],
                        exit=exit_price,

                        result=result,
                        pnl=pnl,

                        score=position["score"],

                        holding_hours=holding_hours
                    )
                )

                if result == "LOSS":

                    loss_streak += 1

                else:

                    loss_streak = 0

                last_exit_by_symbol[
                    symbol
                ] = timestamp

                del active[
                    symbol
                ]

                cursor[symbol] += 1

                return

            cursor[symbol] += 1

    # ========================================================
    # PROCESS EVENTS CHRONOLOGICALLY
    # ========================================================

    for candidate in events:

        # First process any existing positions that
        # should have already closed BEFORE this entry.
        for symbol in list(active.keys()):

            manage_position_until(
                symbol,
                candidate.entry_time
            )

        # ----------------------------------------------------
        # SAME SYMBOL OVERLAP LOCK
        # ----------------------------------------------------

        if candidate.symbol in active:
            continue

        # ----------------------------------------------------
        # NO SAME-CANDLE REENTRY
        # ----------------------------------------------------

        if (
            candidate.symbol
            in last_exit_by_symbol
        ):

            if (
                last_exit_by_symbol[
                    candidate.symbol
                ]
                >= candidate.entry_time
            ):

                continue

        # ----------------------------------------------------
        # MAX 3 POSITIONS
        # ----------------------------------------------------

        if (
            len(active)
            >= MAX_OPEN_POSITIONS
        ):

            continue

        # ----------------------------------------------------
        # OPTIONAL LOSS BREAKER
        # ----------------------------------------------------

        if BREAKER_ENABLED:

            if (
                loss_streak
                >= BREAKER_AFTER_LOSSES
            ):

                continue

        # ----------------------------------------------------
        # ENTRY
        # ----------------------------------------------------

        entry_price = entry_slippage(
            candidate.entry,
            candidate.side
        )

        active[
            candidate.symbol
        ] = {

            "symbol":
                candidate.symbol,

            "setup":
                candidate.setup,

            "side":
                candidate.side,

            "signal_time":
                candidate.signal_time,

            "entry_time":
                candidate.entry_time,

            "entry":
                entry_price,

            "sl":
                candidate.sl,

            "tp":
                candidate.tp,

            "score":
                candidate.score,
        }

    # ========================================================
    # MANAGE REMAINING POSITIONS
    # ========================================================

    for symbol in list(active.keys()):

        manage_position_until(
            symbol,
            pd.Timestamp.max.tz_localize(
                "UTC"
            )
        )

    return trades


# ============================================================
# REPORTING
# ============================================================

def calculate_loss_streaks(
    results
):

    streaks = []

    current = 0

    for result in results:

        if result == "LOSS":

            current += 1

        else:

            if current > 0:

                streaks.append(
                    current
                )

            current = 0

    if current > 0:

        streaks.append(
            current
        )

    return streaks


def summarize(
    trades,
    start,
    end
):

    if not trades:

        return {

            "trades": 0,

            "win_rate_pct": 0,

            "loss_rate_pct": 0,

            "be_rate_pct": 0,

            "net_pnl": 0,

            "profit_factor": 0,

            "avg_win": 0,

            "avg_loss": 0,

            "max_drawdown": 0,

            "max_consecutive_losses": 0,

            "loss_streak_list": "",

            "trades_per_day": 0,
        }

    df = pd.DataFrame(
        [
            asdict(t)
            for t in trades
        ]
    )

    wins = df[
        df["result"] == "WIN"
    ]["pnl"]

    losses = df[
        df["result"] == "LOSS"
    ]["pnl"]

    bes = df[
        df["result"] == "BE"
    ]["pnl"]

    equity = (
        df
        .sort_values("exit_time")
        ["pnl"]
        .cumsum()
    )

    peak = equity.cummax()

    drawdown = (
        equity - peak
    )

    streaks = calculate_loss_streaks(
        df
        .sort_values("exit_time")
        ["result"]
        .tolist()
    )

    if losses.sum() < 0:

        profit_factor = (
            wins.sum()
            / abs(losses.sum())
        )

    else:

        profit_factor = float("inf")

    days = max(
        1.0,
        (
            end - start
        ).total_seconds()
        / 86400
    )

    return {

        "trades":
            int(len(df)),

        "win_rate_pct":
            float(
                100
                * len(wins)
                / len(df)
            ),

        "loss_rate_pct":
            float(
                100
                * len(losses)
                / len(df)
            ),

        "be_rate_pct":
            float(
                100
                * len(bes)
                / len(df)
            ),

        "net_pnl":
            float(
                df["pnl"].sum()
            ),

        "profit_factor":
            float(
                profit_factor
            ),

        "avg_win":
            float(
                wins.mean()
            )
            if len(wins)
            else 0.0,

        "avg_loss":
            float(
                losses.mean()
            )
            if len(losses)
            else 0.0,

        "max_drawdown":
            float(
                drawdown.min()
            ),

        "max_consecutive_losses":
            int(
                max(streaks)
                if streaks
                else 0
            ),

        "loss_streak_list":
            ",".join(
                map(
                    str,
                    streaks
                )
            ),

        "trades_per_day":
            float(
                len(df) / days
            ),
    }


# ============================================================
# EXTRA REPORTS
# ============================================================

def save_reports(
    trades,
    candidates,
    stage_rows,
    test_start,
    test_end
):

    # --------------------------------------------------------
    # CANDIDATES
    # --------------------------------------------------------

    candidate_rows = [
        asdict(c)
        for c in candidates
    ]

    pd.DataFrame(
        candidate_rows
    ).to_csv(
        OUT_DIR
        / "cle1_candidates.csv",
        index=False
    )

    # --------------------------------------------------------
    # TRADES
    # --------------------------------------------------------

    trade_rows = [
        asdict(t)
        for t in trades
    ]

    pd.DataFrame(
        trade_rows
    ).to_csv(
        OUT_DIR
        / "cle1_trades.csv",
        index=False
    )

    # --------------------------------------------------------
    # STAGE COUNTS
    # --------------------------------------------------------

    pd.DataFrame(
        stage_rows
    ).to_csv(
        OUT_DIR
        / "cle1_stage_counts.csv",
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = summarize(
        trades,
        test_start,
        test_end
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        OUT_DIR
        / "cle1_summary.csv",
        index=False
    )

    if not trades:
        return

    td = pd.DataFrame(
        trade_rows
    )

    td["exit_time"] = pd.to_datetime(
        td["exit_time"],
        utc=True
    )

    # --------------------------------------------------------
    # MONTHLY
    # --------------------------------------------------------

    td["month"] = (
        td["exit_time"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        td.groupby("month")
        .agg(
            trades=("pnl", "size"),

            wins=(
                "result",
                lambda x:
                (x == "WIN").sum()
            ),

            losses=(
                "result",
                lambda x:
                (x == "LOSS").sum()
            ),

            pnl=("pnl", "sum")
        )
        .reset_index()
    )

    monthly["win_rate_pct"] = (
        100
        * monthly["wins"]
        / monthly["trades"]
    )

    monthly.to_csv(
        OUT_DIR
        / "cle1_monthly.csv",
        index=False
    )

    # --------------------------------------------------------
    # SYMBOL
    # --------------------------------------------------------

    by_symbol = (
        td.groupby("symbol")
        .agg(
            trades=("pnl", "size"),

            wins=(
                "result",
                lambda x:
                (x == "WIN").sum()
            ),

            losses=(
                "result",
                lambda x:
                (x == "LOSS").sum()
            ),

            pnl=("pnl", "sum")
        )
        .reset_index()
    )

    by_symbol["win_rate_pct"] = (
        100
        * by_symbol["wins"]
        / by_symbol["trades"]
    )

    by_symbol.to_csv(
        OUT_DIR
        / "cle1_by_symbol.csv",
        index=False
    )

    # --------------------------------------------------------
    # SETUP
    # --------------------------------------------------------

    by_setup = (
        td.groupby("setup")
        .agg(
            trades=("pnl", "size"),

            wins=(
                "result",
                lambda x:
                (x == "WIN").sum()
            ),

            losses=(
                "result",
                lambda x:
                (x == "LOSS").sum()
            ),

            pnl=("pnl", "sum")
        )
        .reset_index()
    )

    by_setup["win_rate_pct"] = (
        100
        * by_setup["wins"]
        / by_setup["trades"]
    )

    by_setup.to_csv(
        OUT_DIR
        / "cle1_by_setup.csv",
        index=False
    )

    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    by_direction = (
        td.groupby("side")
        .agg(
            trades=("pnl", "size"),

            wins=(
                "result",
                lambda x:
                (x == "WIN").sum()
            ),

            losses=(
                "result",
                lambda x:
                (x == "LOSS").sum()
            ),

            pnl=("pnl", "sum")
        )
        .reset_index()
    )

    by_direction["win_rate_pct"] = (
        100
        * by_direction["wins"]
        / by_direction["trades"]
    )

    by_direction.to_csv(
        OUT_DIR
        / "cle1_by_direction.csv",
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    exchange = make_exchange()

    now = pd.Timestamp(
        datetime.now(timezone.utc)
    )

    # Last completely closed 15m candle.
    test_end = (
        now.floor("15min")
        - pd.Timedelta(
            minutes=15
        )
    )

    test_start = (
        test_end
        - pd.Timedelta(
            days=DAYS
        )
    )

    fetch_start = (
        test_start
        - pd.Timedelta(
            days=WARMUP_DAYS
        )
    )

    since_ms = int(
        fetch_start.timestamp()
        * 1000
    )

    until_ms = int(
        test_end.timestamp()
        * 1000
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 78)
    print("CLE-1 — CONFLUENCE LIQUIDITY ENGINE")
    print("=" * 78)

    print(
        "Test:",
        test_start,
        "->",
        test_end
    )

    print(
        "Warmup:",
        WARMUP_DAYS,
        "days"
    )

    print(
        "RR:",
        "1:2 FIXED"
    )

    print(
        "Timeout:",
        "DISABLED"
    )

    print(
        "Lookahead:",
        "NONE BY DESIGN"
    )

    print(
        "Entry:",
        "NEXT 15m OPEN"
    )

    print(
        "Max Open Positions:",
        MAX_OPEN_POSITIONS
    )

    print(
        "Breaker:",
        "ON"
        if BREAKER_ENABLED
        else "OFF"
    )

    print("=" * 78)

    # ========================================================
    # DISCOVER SYMBOLS
    # ========================================================

    symbols = discover_symbols(
        exchange
    )

    if not symbols:

        raise RuntimeError(
            "NO LBank linear USDT swap symbols discovered"
        )

    all_data = {}

    all_candidates = {}

    stage_rows = []

    # ========================================================
    # EACH SYMBOL
    # ========================================================

    for symbol in symbols:

        print()
        print(
            f"Fetching {symbol}"
        )

        df = fetch_ohlcv_paginated(
            exchange,
            symbol,
            since_ms,
            until_ms
        )

        print(
            "  candles:",
            len(df)
        )

        if len(df) < 1000:

            print(
                "  insufficient data"
            )

            continue

        # ----------------------------------------------------
        # 15m
        # ----------------------------------------------------

        df = prepare_15m(df)

        # ----------------------------------------------------
        # 1H / 4H
        # ----------------------------------------------------

        h1 = resample_ohlcv(
            df,
            "1h"
        )

        h4 = resample_ohlcv(
            df,
            "4h"
        )

        # ----------------------------------------------------
        # MAP COMPLETED HTF CONTEXT
        # ----------------------------------------------------

        df = map_htf_context(
            df,
            h1,
            h4
        )

        # ----------------------------------------------------
        # MARKET STRUCTURE
        # ----------------------------------------------------

        df = add_structure(
            df
        )

        df = df.replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )

        # ----------------------------------------------------
        # CANDIDATES
        # ----------------------------------------------------

        candidates = generate_candidates(
            df,
            symbol,
            test_start
        )

        # ====================================================
        # RESEARCH DIAGNOSTICS
        # ====================================================

        test_df = df[
            df.index >= test_start
        ]

        raw_directional = int(
            (
                test_df["close"]
                != test_df["open"]
            ).sum()
        )

        sweep_events = int(
            (
                (
                    test_df["low"]
                    <
                    test_df[
                        "last_swing_low"
                    ]
                )
                |
                (
                    test_df["high"]
                    >
                    test_df[
                        "last_swing_high"
                    ]
                )
            )
            .fillna(False)
            .sum()
        )

        bos_events = int(
            (
                (
                    test_df["close"]
                    >
                    test_df[
                        "last_swing_high"
                    ]
                )
                |
                (
                    test_df["close"]
                    <
                    test_df[
                        "last_swing_low"
                    ]
                )
            )
            .fillna(False)
            .sum()
        )

        reversal_count = sum(
            c.setup == "REVERSAL"
            for c in candidates
        )

        continuation_count = sum(
            c.setup == "CONTINUATION"
            for c in candidates
        )

        stage_rows.append({

            "symbol":
                symbol,

            "test_candles":
                len(test_df),

            "raw_directional_candles":
                raw_directional,

            "sweep_events":
                sweep_events,

            "bos_events":
                bos_events,

            "accepted_reversal_candidates":
                reversal_count,

            "accepted_continuation_candidates":
                continuation_count,

            "accepted_total_candidates":
                len(candidates)
        })

        all_data[
            symbol
        ] = df

        all_candidates[
            symbol
        ] = candidates

        print(
            "  candidates:",
            len(candidates)
        )

        print(
            "  reversal:",
            reversal_count
        )

        print(
            "  continuation:",
            continuation_count
        )

    # ========================================================
    # CHECK
    # ========================================================

    if not all_data:

        raise RuntimeError(
            "NO USABLE LBank DATA"
        )

    # ========================================================
    # FLATTEN CANDIDATES
    # ========================================================

    all_candidates_flat = []

    for candidates in all_candidates.values():

        all_candidates_flat.extend(
            candidates
        )

    all_candidates_flat.sort(
        key=lambda x:
        x.entry_time
    )

    print()
    print("=" * 78)
    print(
        "TOTAL CANDIDATES:",
        len(all_candidates_flat)
    )
    print("=" * 78)

    # ========================================================
    # PORTFOLIO SIMULATION
    # ========================================================

    trades = portfolio_sim(
        all_data,
        all_candidates
    )

    # ========================================================
    # SAVE REPORTS
    # ========================================================

    save_reports(
        trades,
        all_candidates_flat,
        stage_rows,
        test_start,
        test_end
    )

    # ========================================================
    # FINAL
    # ========================================================

    summary = summarize(
        trades,
        test_start,
        test_end
    )

    print()
    print("=" * 78)
    print("FINAL CLE-1 REPORT")
    print("=" * 78)

    print(
        "Total Trades:",
        summary["trades"]
    )

    print(
        "Trades / Day:",
        round(
            summary["trades_per_day"],
            3
        )
    )

    print(
        "Win Rate:",
        round(
            summary["win_rate_pct"],
            2
        ),
        "%"
    )

    print(
        "Loss Rate:",
        round(
            summary["loss_rate_pct"],
            2
        ),
        "%"
    )

    print(
        "BE Rate:",
        round(
            summary["be_rate_pct"],
            2
        ),
        "%"
    )

    print(
        "Net PnL:",
        round(
            summary["net_pnl"],
            2
        )
    )

    print(
        "Profit Factor:",
        round(
            summary["profit_factor"],
            3
        )
        if np.isfinite(
            summary["profit_factor"]
        )
        else "INF"
    )

    print(
        "Average Win:",
        round(
            summary["avg_win"],
            2
        )
    )

    print(
        "Average Loss:",
        round(
            summary["avg_loss"],
            2
        )
    )

    print(
        "Max Drawdown:",
        round(
            summary["max_drawdown"],
            2
        )
    )

    print(
        "Max Consecutive Losses:",
        summary[
            "max_consecutive_losses"
        ]
    )

    print(
        "Loss Streak List:",
        summary[
            "loss_streak_list"
        ]
    )

    print("=" * 78)

    print(
        "RR = 1:2 FIXED"
    )

    print(
        "Timeout = DISABLED"
    )

    print(
        "Lookahead = NONE"
    )

    print(
        "Entry = NEXT 15m OPEN"
    )

    print("=" * 78)

    print(
        "CSV reports:",
        OUT_DIR.resolve()
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This run is a research measurement."
    )

    print(
        "Do NOT change parameters simply to force"
        " the Win Rate to 50%."
    )

    print(
        "Use cle1_stage_counts.csv to identify"
        " where opportunities are being removed."
    )

    print("=" * 78)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
