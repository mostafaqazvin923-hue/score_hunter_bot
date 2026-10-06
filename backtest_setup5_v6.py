#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SETUP 5 V6 — SOURCE-SEQUENCE / TRUE OB LIMIT BACKTEST

V6 is a structural rebuild after V1-V5 failed to produce a positive OOS edge.

IMPORTANT:
This is a mechanical research translation of the supplied Setup 5 PDF.
Where the PDF is qualitative, numeric thresholds below are explicitly research
translations and are NOT claimed to be literal PDF values.

CORE SEQUENCE
-------------
LONG:
downtrend
-> confirmed 4H demand zone
-> first reaction to HTF zone
-> bullish CHOCH
-> lowest confirmed valley / CHOCH-causing valley as OB
-> two aligned liquidity lows
-> liquidity sweep
-> return to OB
-> TRUE OB LIMIT FILL

SHORT is the exact inverse.

V6 FUNDAMENTAL CHANGE
---------------------
Previous versions approximated an OB entry using the next candle open.

V6 instead models an actual resting OB limit:
    LONG  = proximal/top edge of bullish OB
    SHORT = proximal/bottom edge of bearish OB

The signal candle must actually trade through that OB price.

INTEGRITY RULES
---------------
- No lookahead / future leak.
- Confirmed pivots require PIVOT candles on both sides.
- HTF pivot is usable only after the HTF candle containing the pivot has closed.
- First HTF-zone touch after zone confirmation is used.
- CHOCH is confirmed by a CLOSED 15m candle.
- OB is formed from confirmed structure before CHOCH.
- Liquidity pivots are confirmed before the sweep.
- Sweep is known only after the sweep candle closes.
- Structural target must be confirmed before entry.
- Entry candle is not checked for SL/TP.
- If SL and TP are both hit on the same later candle, SL wins.
- Max one simultaneous trade PER SYMBOL.
- Different symbols may have simultaneous trades.
- No same-symbol re-entry on the exit candle.
- No timeout.
- No breakeven.
- No trailing.
- No partial exits.
- No artificial stop-after-loss rule.
- Unresolved trades are reported separately.
- No OOS parameter optimization.

ACCOUNT
-------
Initial capital : $1,000
Margin          : $100
Leverage        : 50x
Notional        : $5,000
RR              : 1:2
Fee             : 0.07% each side
Slippage        : 0.03%

DATA
----
Binance USD-M Futures
Execution TF: 15m
HTF: 4h
Test: 365 days
Warmup: 90 days
"""

from __future__ import annotations

import io
import os
import time
import zipfile

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "SUIUSDT",
    "AVAXUSDT",
    "NEARUSDT",
    "ADAUSDT",
    "BNBUSDT",
    "APTUSDT",
    "CRVUSDT",
    "ONDOUSDT",
    "PENDLEUSDT",
    "ICPUSDT",
    "WIFUSDT",
]

EXEC_TF = "15m"
HTF_TF = "4h"

TEST_DAYS = 365
WARMUP_DAYS = 90

# Account
INITIAL_CAPITAL = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE

# Fixed RR
RR = 2.0

# Costs
FEE_RATE = 0.0007
SLIPPAGE = 0.0003

# Structure
PIVOT = 2

# Liquidity alignment research translation
ALIGN_TOL = 0.004

# OB stop buffer research translation
SL_BUFFER = 0.0005

# Risk sanity bounds
MIN_RISK = 0.0005
MAX_RISK = 0.08

# Search windows
MAX_REACTION_BARS = 96
MAX_CHOCH_BARS = 160
MAX_LIQ_BARS = 32
MAX_SWEEP_BARS = 96
MAX_RETURN_BARS = 96

# CHOCH displacement research translation
CHOCH_BODY_ATR = 0.50
CHOCH_EXT_ATR = 0.10
MIN_DISPLACEMENT_RANGE_ATR = 0.90

# Binance archive
ARCHIVE_BASE = "https://data.binance.vision/data/futures/um"

# Optional fixed end:
# BACKTEST_END=2026-10-05
BACKTEST_END = os.getenv("BACKTEST_END", "").strip()

# Chronological split
TRAIN_FRAC = 0.50
VALID_FRAC = 0.20
# OOS = remaining 30%

MAX_WORKERS = min(6, len(SYMBOLS))
REQUEST_TIMEOUT = 30

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "setup5-v6-research/1.0"
})


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Candidate:
    symbol: str
    direction: str

    signal_idx: int
    entry_idx: int

    zone_idx: int
    reaction_idx: int
    choch_idx: int
    ob_idx: int

    liq1_idx: int
    liq2_idx: int
    sweep_idx: int

    target_idx: int

    entry: float
    sl: float
    tp: float
    target: float
    risk: float

    choch_body_atr: float
    choch_ext_atr: float
    displacement_range_atr: float

    sweep_pen_atr: float
    structural_room_R: float

    ifc: int


@dataclass
class Trade:
    symbol: str
    direction: str

    signal_idx: int
    entry_idx: int
    exit_idx: int

    entry_time: str
    exit_time: str

    entry: float
    sl: float
    tp: float
    exit: float

    pnl: float
    result: str
    R: float

    zone_idx: int
    reaction_idx: int
    choch_idx: int
    ob_idx: int

    liq1_idx: int
    liq2_idx: int
    sweep_idx: int

    target_idx: int
    target: float

    choch_body_atr: float
    choch_ext_atr: float
    displacement_range_atr: float

    sweep_pen_atr: float
    structural_room_R: float

    ifc: int


# ============================================================
# TIME / DATA
# ============================================================

def utc_now():
    return pd.Timestamp.now(tz="UTC")


def get_end_time():
    if BACKTEST_END:
        d = pd.Timestamp(BACKTEST_END, tz="UTC")
        return d + pd.Timedelta(hours=23, minutes=45)

    # Previous completed 15m candle.
    return utc_now().floor("D") - pd.Timedelta(minutes=15)


def month_range(start, end):
    x = pd.Timestamp(
        start.year,
        start.month,
        1,
        tz="UTC",
    )

    last = pd.Timestamp(
        end.year,
        end.month,
        1,
        tz="UTC",
    )

    while x <= last:
        yield x
        x += pd.offsets.MonthBegin(1)


def parse_zip(content):
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        csv_name = next(
            name
            for name in z.namelist()
            if name.lower().endswith(".csv")
        )

        raw = pd.read_csv(
            z.open(csv_name),
            header=None,
        ).iloc[:, :12]

    raw.columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_volume",
        "taker_buy_quote",
        "ignore",
    ]

    for c in ["open_time", "close_time"]:
        raw[c] = pd.to_numeric(
            raw[c],
            errors="coerce",
        )

    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        raw[c] = pd.to_numeric(
            raw[c],
            errors="coerce",
        )

    raw = raw.dropna(
        subset=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    raw["open_time"] = pd.to_datetime(
        raw["open_time"],
        unit="ms",
        utc=True,
    )

    raw["close_time"] = pd.to_datetime(
        raw["close_time"],
        unit="ms",
        utc=True,
    )

    return (
        raw
        .sort_values("open_time")
        .drop_duplicates("open_time")
        .reset_index(drop=True)
    )


def fetch_data(symbol, timeframe, start, end):
    parts = []

    # Monthly archives
    for month in month_range(start, end):

        url = (
            f"{ARCHIVE_BASE}/monthly/klines/"
            f"{symbol}/{timeframe}/"
            f"{symbol}-{timeframe}-{month:%Y-%m}.zip"
        )

        try:
            response = SESSION.get(
                url,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                parts.append(
                    parse_zip(response.content)
                )

        except Exception:
            pass

    # Daily fallback.
    # Especially useful around the current/latest month.
    if not parts or (end - start).days <= 45:

        day = pd.Timestamp(
            start.date(),
            tz="UTC",
        )

        while day <= end:

            url = (
                f"{ARCHIVE_BASE}/daily/klines/"
                f"{symbol}/{timeframe}/"
                f"{symbol}-{timeframe}-{day:%Y-%m-%d}.zip"
            )

            try:
                response = SESSION.get(
                    url,
                    timeout=REQUEST_TIMEOUT,
                )

                if response.status_code == 200:
                    parts.append(
                        parse_zip(response.content)
                    )

            except Exception:
                pass

            day += pd.Timedelta(days=1)

    if not parts:
        raise RuntimeError(
            f"No Binance data: {symbol} {timeframe}"
        )

    df = (
        pd.concat(parts, ignore_index=True)
        .sort_values("open_time")
        .drop_duplicates("open_time")
        .reset_index(drop=True)
    )

    df = df[
        (df["open_time"] >= start)
        & (df["open_time"] <= end)
        & (df["close_time"] <= utc_now())
    ].reset_index(drop=True)

    return df


# ============================================================
# DATA INTEGRITY
# ============================================================

def validate_data(df, interval_minutes, symbol, timeframe):

    # 15m:
    # 365 days + 90 warmup is around 43k candles.
    #
    # 4h:
    # 365 days + 90 warmup is around 2700 candles.
    #
    # Do NOT incorrectly demand 30k rows from 4h.
    minimum_rows = (
        30000
        if timeframe == "15m"
        else 1800
    )

    if len(df) < minimum_rows:
        raise RuntimeError(
            f"{symbol} {timeframe}: "
            f"too few rows {len(df)}"
        )

    if not df["open_time"].is_monotonic_increasing:
        raise RuntimeError(
            f"{symbol} {timeframe}: "
            "timestamps not monotonic"
        )

    if df["open_time"].duplicated().any():
        raise RuntimeError(
            f"{symbol} {timeframe}: "
            "duplicate timestamps"
        )

    gaps = df["open_time"].diff().dropna()

    bad = gaps[
        gaps != pd.Timedelta(
            minutes=interval_minutes
        )
    ]

    if not bad.empty:
        raise RuntimeError(
            f"{symbol} {timeframe}: "
            f"gap detected: {bad.max()}"
        )

    invalid = (
        (df["high"] < df["low"])
        | (df["open"] > df["high"])
        | (df["open"] < df["low"])
        | (df["close"] > df["high"])
        | (df["close"] < df["low"])
    )

    if invalid.any():
        raise RuntimeError(
            f"{symbol} {timeframe}: "
            "invalid OHLC"
        )


# ============================================================
# INDICATORS / PIVOTS
# ============================================================

def calculate_atr(df, period=14):

    previous_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(
        period,
        min_periods=period,
    ).mean()


def calculate_pivots(df, p=PIVOT):

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    pivot_high = np.zeros(
        len(df),
        dtype=bool,
    )

    pivot_low = np.zeros(
        len(df),
        dtype=bool,
    )

    for i in range(
        p,
        len(df) - p,
    ):

        pivot_high[i] = (
            highs[i] > max(highs[i-p:i])
            and highs[i] >= max(highs[i+1:i+p+1])
        )

        pivot_low[i] = (
            lows[i] < min(lows[i-p:i])
            and lows[i] <= min(lows[i+1:i+p+1])
        )

    return pivot_high, pivot_low


def confirmed_last(flag, p=PIVOT):

    result = np.full(
        len(flag),
        -1,
        dtype=np.int32,
    )

    last = -1

    for i in range(len(flag)):

        pivot_index = i - p

        if pivot_index >= 0 and flag[pivot_index]:
            last = pivot_index

        result[i] = last

    return result


def confirmed_second(flag, p=PIVOT):

    result = np.full(
        len(flag),
        -1,
        dtype=np.int32,
    )

    last = -1
    second = -1

    for i in range(len(flag)):

        pivot_index = i - p

        if pivot_index >= 0 and flag[pivot_index]:

            second = last
            last = pivot_index

        result[i] = second

    return result


def map_htf_to_execution(htf, execution_times):

    # A 4H candle becomes usable only after its close.
    return np.searchsorted(
        htf["close_time"].astype("int64").to_numpy(),
        execution_times.astype("int64").to_numpy(),
        side="right",
    ) - 1


# ============================================================
# TREND
# ============================================================

def is_downtrend(
    df,
    last_high,
    second_high,
    last_low,
    second_low,
    i,
):

    h1 = last_high[i]
    h2 = second_high[i]
    l1 = last_low[i]
    l2 = second_low[i]

    if min(h1, h2, l1, l2) < 0:
        return False

    return (
        df["high"].iloc[h1]
        < df["high"].iloc[h2]
        and
        df["low"].iloc[l1]
        < df["low"].iloc[l2]
    )


def is_uptrend(
    df,
    last_high,
    second_high,
    last_low,
    second_low,
    i,
):

    h1 = last_high[i]
    h2 = second_high[i]
    l1 = last_low[i]
    l2 = second_low[i]

    if min(h1, h2, l1, l2) < 0:
        return False

    return (
        df["high"].iloc[h1]
        > df["high"].iloc[h2]
        and
        df["low"].iloc[l1]
        > df["low"].iloc[l2]
    )


# ============================================================
# HTF ZONE
# ============================================================

def get_htf_zone(htf, pivot_index, direction):

    if direction == "LONG":

        # Demand:
        # pivot candle Low -> candle body top.
        zone_low = float(
            htf["low"].iloc[pivot_index]
        )

        zone_high = float(
            max(
                htf["open"].iloc[pivot_index],
                htf["close"].iloc[pivot_index],
            )
        )

    else:

        # Supply:
        # candle body bottom -> pivot candle High.
        zone_low = float(
            min(
                htf["open"].iloc[pivot_index],
                htf["close"].iloc[pivot_index],
            )
        )

        zone_high = float(
            htf["high"].iloc[pivot_index]
        )

    return zone_low, zone_high


def find_first_zone_reaction(
    execution,
    start_index,
    zone_low,
    zone_high,
):

    stop = min(
        len(execution) - 1,
        start_index + MAX_REACTION_BARS,
    )

    for i in range(
        start_index,
        stop,
    ):

        touched = (
            execution["low"].iloc[i] <= zone_high
            and
            execution["high"].iloc[i] >= zone_low
        )

        if touched:
            return i

    return None


def zone_invalidated_before_choch(
    execution,
    start_index,
    end_index,
    zone_low,
    zone_high,
    direction,
):

    for i in range(
        start_index,
        end_index,
    ):

        if direction == "LONG":

            if execution["close"].iloc[i] < zone_low:
                return True

        else:

            if execution["close"].iloc[i] > zone_high:
                return True

    return False


# ============================================================
# CHOCH
# ============================================================

def find_choch(
    execution,
    pivot_high,
    pivot_low,
    last_high,
    last_low,
    reaction_index,
    direction,
):

    stop = min(
        len(execution) - 2,
        reaction_index + MAX_CHOCH_BARS,
    )

    for i in range(
        reaction_index + 1,
        stop,
    ):

        atr_value = execution["atr"].iloc[i]

        if not np.isfinite(atr_value):
            continue

        atr_value = max(
            float(atr_value),
            1e-12,
        )

        candle_body = abs(
            float(
                execution["close"].iloc[i]
                - execution["open"].iloc[i]
            )
        ) / atr_value

        candle_range = (
            float(
                execution["high"].iloc[i]
                - execution["low"].iloc[i]
            )
            / atr_value
        )

        # ----------------------------------------------------
        # LONG CHOCH
        # ----------------------------------------------------

        if direction == "LONG":

            previous_lower_high = last_high[i]

            if previous_lower_high < 0:
                continue

            level = float(
                execution["high"].iloc[
                    previous_lower_high
                ]
            )

            extension = (
                float(
                    execution["close"].iloc[i]
                )
                - level
            ) / atr_value

            if (
                execution["close"].iloc[i] > level
                and
                candle_body >= CHOCH_BODY_ATR
                and
                extension >= CHOCH_EXT_ATR
                and
                candle_range >= MIN_DISPLACEMENT_RANGE_ATR
            ):

                # The PDF emphasizes meaningful structure
                # between HTF reaction and CHOCH.
                middle_lows = [
                    reaction_index + 1 + x
                    for x in np.flatnonzero(
                        pivot_low[
                            reaction_index + 1:i
                        ]
                    )
                    if (
                        reaction_index
                        + 1
                        + x
                        + PIVOT
                        <= i
                    )
                ]

                if not middle_lows:
                    continue

                return (
                    i,
                    candle_body,
                    extension,
                    candle_range,
                )

        # ----------------------------------------------------
        # SHORT CHOCH
        # ----------------------------------------------------

        else:

            previous_higher_low = last_low[i]

            if previous_higher_low < 0:
                continue

            level = float(
                execution["low"].iloc[
                    previous_higher_low
                ]
            )

            extension = (
                level
                - float(
                    execution["close"].iloc[i]
                )
            ) / atr_value

            if (
                execution["close"].iloc[i] < level
                and
                candle_body >= CHOCH_BODY_ATR
                and
                extension >= CHOCH_EXT_ATR
                and
                candle_range >= MIN_DISPLACEMENT_RANGE_ATR
            ):

                middle_highs = [
                    reaction_index + 1 + x
                    for x in np.flatnonzero(
                        pivot_high[
                            reaction_index + 1:i
                        ]
                    )
                    if (
                        reaction_index
                        + 1
                        + x
                        + PIVOT
                        <= i
                    )
                ]

                if not middle_highs:
                    continue

                return (
                    i,
                    candle_body,
                    extension,
                    candle_range,
                )

    return None


# ============================================================
# ORDER BLOCK
# ============================================================

def find_order_block(
    execution,
    pivot_high,
    pivot_low,
    reaction_index,
    choch_index,
    direction,
):

    if direction == "LONG":

        candidates = [
            reaction_index + x
            for x in np.flatnonzero(
                pivot_low[
                    reaction_index:choch_index
                ]
            )
            if (
                reaction_index
                + x
                + PIVOT
                <= choch_index
            )
        ]

        if not candidates:
            return None

        # Lowest confirmed valley before CHOCH.
        ob_index = min(
            candidates,
            key=lambda x:
                execution["low"].iloc[x]
        )

        ob_low = float(
            execution["low"].iloc[ob_index]
        )

        ob_high = float(
            max(
                execution["open"].iloc[ob_index],
                execution["close"].iloc[ob_index],
            )
        )

        sl = ob_low * (
            1.0 - SL_BUFFER
        )

    else:

        candidates = [
            reaction_index + x
            for x in np.flatnonzero(
                pivot_high[
                    reaction_index:choch_index
                ]
            )
            if (
                reaction_index
                + x
                + PIVOT
                <= choch_index
            )
        ]

        if not candidates:
            return None

        # Highest confirmed peak before CHOCH.
        ob_index = max(
            candidates,
            key=lambda x:
                execution["high"].iloc[x]
        )

        ob_low = float(
            min(
                execution["open"].iloc[ob_index],
                execution["close"].iloc[ob_index],
            )
        )

        ob_high = float(
            execution["high"].iloc[ob_index]
        )

        sl = ob_high * (
            1.0 + SL_BUFFER
        )

    return (
        ob_index,
        ob_low,
        ob_high,
        sl,
    )


# ============================================================
# LIQUIDITY
# ============================================================

def find_liquidity(
    execution,
    pivot_high,
    pivot_low,
    choch_index,
    ob_index,
    direction,
):

    if direction == "LONG":
        flags = pivot_low
    else:
        flags = pivot_high

    candidates = [
        choch_index + 1 + x
        for x in np.flatnonzero(
            flags[choch_index + 1:]
        )
        if (
            choch_index
            + 1
            + x
            + PIVOT
            < len(execution)
        )
    ]

    # Liquidity must form close enough to the OB.
    candidates = [
        x
        for x in candidates
        if x - ob_index <= MAX_LIQ_BARS
    ]

    if len(candidates) < 2:
        return None

    for j in range(
        1,
        len(candidates),
    ):

        first = candidates[j - 1]
        second = candidates[j]

        if direction == "LONG":

            p1 = float(
                execution["low"].iloc[first]
            )

            p2 = float(
                execution["low"].iloc[second]
            )

        else:

            p1 = float(
                execution["high"].iloc[first]
            )

            p2 = float(
                execution["high"].iloc[second]
            )

        level = (
            p1 + p2
        ) / 2.0

        difference = (
            abs(p1 - p2)
            / max(
                abs(level),
                1e-12,
            )
        )

        if difference <= ALIGN_TOL:

            return (
                first,
                second,
                level,
            )

    return None


# ============================================================
# LIQUIDITY SWEEP
# ============================================================

def find_sweep(
    execution,
    liquidity_second_index,
    liquidity_level,
    direction,
):

    stop = min(
        len(execution) - 1,
        liquidity_second_index
        + MAX_SWEEP_BARS,
    )

    for i in range(
        liquidity_second_index + PIVOT,
        stop,
    ):

        atr_value = max(
            float(
                execution["atr"].iloc[i]
            ),
            1e-12,
        )

        if direction == "LONG":

            valid = (
                execution["low"].iloc[i]
                < liquidity_level
                and
                execution["close"].iloc[i]
                >= liquidity_level
            )

            penetration = (
                liquidity_level
                - float(
                    execution["low"].iloc[i]
                )
            ) / atr_value

        else:

            valid = (
                execution["high"].iloc[i]
                > liquidity_level
                and
                execution["close"].iloc[i]
                <= liquidity_level
            )

            penetration = (
                float(
                    execution["high"].iloc[i]
                )
                - liquidity_level
            ) / atr_value

        if valid:

            return (
                i,
                penetration,
            )

    return None


# ============================================================
# STRUCTURAL TARGET
# ============================================================

def find_structural_target(
    execution,
    pivot_high,
    pivot_low,
    choch_index,
    entry_signal_index,
    direction,
):

    if direction == "LONG":
        flags = pivot_high
    else:
        flags = pivot_low

    candidates = [
        choch_index + 1 + x
        for x in np.flatnonzero(
            flags[
                choch_index + 1:
                entry_signal_index
            ]
        )
        if (
            choch_index
            + 1
            + x
            + PIVOT
            < entry_signal_index
        )
    ]

    if not candidates:
        return None

    if direction == "LONG":

        target_index = max(
            candidates,
            key=lambda x:
                execution["high"].iloc[x]
        )

        target_price = float(
            execution["high"].iloc[
                target_index
            ]
        )

    else:

        target_index = min(
            candidates,
            key=lambda x:
                execution["low"].iloc[x]
        )

        target_price = float(
            execution["low"].iloc[
                target_index
            ]
        )

    return (
        target_index,
        target_price,
    )


# ============================================================
# IFC
# ============================================================

def detect_ifc(
    execution,
    ob_index,
    entry_signal_index,
    direction,
):

    # IFC/FVG is optional.
    # It is logged only and NEVER required for entry.

    stop = max(
        ob_index + 1,
        entry_signal_index - 1,
    )

    for i in range(
        ob_index + 1,
        stop,
    ):

        if direction == "LONG":

            if (
                execution["low"].iloc[i + 1]
                >
                execution["high"].iloc[i - 1]
            ):
                return 1

        else:

            if (
                execution["high"].iloc[i + 1]
                <
                execution["low"].iloc[i - 1]
            ):
                return 1

    return 0


# ============================================================
# BUILD V6 EVENTS
# ============================================================

def build_candidates(
    symbol,
    execution,
    htf,
):

    execution = execution.copy()

    execution["atr"] = calculate_atr(
        execution
    )

    pivot_high, pivot_low = calculate_pivots(
        execution
    )

    last_high = confirmed_last(
        pivot_high
    )

    second_high = confirmed_second(
        pivot_high
    )

    last_low = confirmed_last(
        pivot_low
    )

    second_low = confirmed_second(
        pivot_low
    )

    htf_pivot_high, htf_pivot_low = (
        calculate_pivots(htf)
    )

    htf_last_high = confirmed_last(
        htf_pivot_high
    )

    htf_last_low = confirmed_last(
        htf_pivot_low
    )

    htf_map = map_htf_to_execution(
        htf,
        execution["open_time"],
    )

    events = []
    seen = set()

    # Enough warmup for pivots and ATR.
    for i in range(
        100,
        len(execution) - 2,
    ):

        if not np.isfinite(
            execution["atr"].iloc[i]
        ):
            continue

        htf_index = int(
            htf_map[i]
        )

        if htf_index < 0:
            continue

        for direction in (
            "LONG",
            "SHORT",
        ):

            if direction == "LONG":

                trend = is_downtrend(
                    execution,
                    last_high,
                    second_high,
                    last_low,
                    second_low,
                    i,
                )

                zone_index = int(
                    htf_last_low[htf_index]
                )

            else:

                trend = is_uptrend(
                    execution,
                    last_high,
                    second_high,
                    last_low,
                    second_low,
                    i,
                )

                zone_index = int(
                    htf_last_high[htf_index]
                )

            if not trend:
                continue

            if zone_index < 0:
                continue

            # The HTF pivot itself must already be confirmed.
            if (
                htf["close_time"].iloc[
                    zone_index
                ]
                >
                execution["open_time"].iloc[i]
            ):
                continue

            zone_low, zone_high = (
                get_htf_zone(
                    htf,
                    zone_index,
                    direction,
                )
            )

            # ------------------------------------------------
            # FIRST TOUCH AFTER HTF ZONE BECOMES USABLE
            # ------------------------------------------------

            zone_available_index = int(
                np.searchsorted(
                    execution["open_time"]
                    .astype("int64")
                    .to_numpy(),

                    htf["close_time"]
                    .iloc[zone_index]
                    .value,

                    side="right",
                )
            )

            if (
                zone_available_index
                >= len(execution) - 1
            ):
                continue

            reaction_index = (
                find_first_zone_reaction(
                    execution,
                    zone_available_index,
                    zone_low,
                    zone_high,
                )
            )

            if reaction_index is None:
                continue

            # Do not use a reaction that occurred
            # before the current structural context.
            if reaction_index < i:
                continue

            # Trend must still be present immediately
            # before the reaction.
            if direction == "LONG":

                trend_at_reaction = (
                    is_downtrend(
                        execution,
                        last_high,
                        second_high,
                        last_low,
                        second_low,
                        reaction_index,
                    )
                )

            else:

                trend_at_reaction = (
                    is_uptrend(
                        execution,
                        last_high,
                        second_high,
                        last_low,
                        second_low,
                        reaction_index,
                    )
                )

            if not trend_at_reaction:
                continue

            # ------------------------------------------------
            # CHOCH
            # ------------------------------------------------

            choch_result = find_choch(
                execution,
                pivot_high,
                pivot_low,
                last_high,
                last_low,
                reaction_index,
                direction,
            )

            if choch_result is None:
                continue

            (
                choch_index,
                choch_body_atr,
                choch_ext_atr,
                displacement_range_atr,
            ) = choch_result

            # ------------------------------------------------
            # HTF ZONE INVALIDATION
            # ------------------------------------------------

            if zone_invalidated_before_choch(
                execution,
                reaction_index + 1,
                choch_index,
                zone_low,
                zone_high,
                direction,
            ):
                continue

            # ------------------------------------------------
            # ORDER BLOCK
            # ------------------------------------------------

            ob_result = find_order_block(
                execution,
                pivot_high,
                pivot_low,
                reaction_index,
                choch_index,
                direction,
            )

            if ob_result is None:
                continue

            (
                ob_index,
                ob_low,
                ob_high,
                sl,
            ) = ob_result

            # OB must be before CHOCH.
            if ob_index >= choch_index:
                continue

            # ------------------------------------------------
            # LIQUIDITY
            # ------------------------------------------------

            liquidity_result = (
                find_liquidity(
                    execution,
                    pivot_high,
                    pivot_low,
                    choch_index,
                    ob_index,
                    direction,
                )
            )

            if liquidity_result is None:
                continue

            (
                liquidity_first,
                liquidity_second,
                liquidity_level,
            ) = liquidity_result

            # ------------------------------------------------
            # SWEEP
            # ------------------------------------------------

            sweep_result = find_sweep(
                execution,
                liquidity_second,
                liquidity_level,
                direction,
            )

            if sweep_result is None:
                continue

            (
                sweep_index,
                sweep_penetration_atr,
            ) = sweep_result

            # ------------------------------------------------
            # EVENT DEDUPLICATION
            # ------------------------------------------------

            key = (
                direction,
                zone_index,
                reaction_index,
                choch_index,
                ob_index,
                liquidity_first,
                liquidity_second,
                sweep_index,
            )

            if key in seen:
                continue

            seen.add(key)

            events.append(
                {
                    "direction": direction,
                    "zone": zone_index,
                    "reaction": reaction_index,
                    "choch": choch_index,
                    "ob": (
                        ob_index,
                        ob_low,
                        ob_high,
                        sl,
                    ),
                    "liquidity": (
                        liquidity_first,
                        liquidity_second,
                        liquidity_level,
                    ),
                    "sweep": (
                        sweep_index,
                        sweep_penetration_atr,
                    ),
                    "choch_body_atr":
                        choch_body_atr,
                    "choch_ext_atr":
                        choch_ext_atr,
                    "displacement_range_atr":
                        displacement_range_atr,
                }
            )

    # ========================================================
    # CONVERT STRUCTURAL EVENTS TO TRUE OB LIMIT CANDIDATES
    # ========================================================

    candidates = []

    for event in events:

        direction = event["direction"]

        (
            ob_index,
            ob_low,
            ob_high,
            sl,
        ) = event["ob"]

        (
            liquidity_first,
            liquidity_second,
            liquidity_level,
        ) = event["liquidity"]

        (
            sweep_index,
            sweep_penetration_atr,
        ) = event["sweep"]

        # ----------------------------------------------------
        # TRUE OB LIMIT PRICE
        #
        # LONG:
        # proximal/top edge of bullish OB.
        #
        # SHORT:
        # proximal/bottom edge of bearish OB.
        # ----------------------------------------------------

        if direction == "LONG":
            entry_price = ob_high
        else:
            entry_price = ob_low

        # ----------------------------------------------------
        # FIRST RETURN TO OB AFTER SWEEP
        # ----------------------------------------------------

        signal_index = None

        stop = min(
            len(execution) - 1,
            sweep_index + 1 + MAX_RETURN_BARS,
        )

        for r in range(
            sweep_index + 1,
            stop,
        ):

            touches_ob = (
                execution["low"].iloc[r]
                <= ob_high
                and
                execution["high"].iloc[r]
                >= ob_low
            )

            if touches_ob:

                signal_index = r
                break

        if signal_index is None:
            continue

        # The limit order is filled at the actual OB price.
        # It is NOT replaced with next candle open.
        entry_index = signal_index

        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        if direction == "LONG":

            risk = (
                entry_price
                - sl
            )

        else:

            risk = (
                sl
                - entry_price
            )

        if risk <= 0:
            continue

        risk_pct = (
            risk
            / entry_price
        )

        if (
            risk_pct < MIN_RISK
            or
            risk_pct > MAX_RISK
        ):
            continue

        # ----------------------------------------------------
        # STRUCTURAL TARGET
        #
        # Must already be confirmed before entry.
        # ----------------------------------------------------

        target_result = (
            find_structural_target(
                execution,
                pivot_high,
                pivot_low,
                event["choch"],
                signal_index,
                direction,
            )
        )

        if target_result is None:
            continue

        (
            target_index,
            target_price,
        ) = target_result

        # ----------------------------------------------------
        # STRUCTURAL ROOM
        # ----------------------------------------------------

        if direction == "LONG":

            structural_room_R = (
                target_price
                - entry_price
            ) / risk

        else:

            structural_room_R = (
                entry_price
                - target_price
            ) / risk

        # TP must fit inside the already-confirmed
        # structural target.
        if structural_room_R < RR:
            continue

        # ----------------------------------------------------
        # FIXED 1:2 TARGET
        # ----------------------------------------------------

        if direction == "LONG":

            tp = (
                entry_price
                + RR * risk
            )

        else:

            tp = (
                entry_price
                - RR * risk
            )

        # ----------------------------------------------------
        # IFC
        # Optional diagnostic only.
        # ----------------------------------------------------

        ifc_present = detect_ifc(
            execution,
            ob_index,
            signal_index,
            direction,
        )

        candidate = Candidate(
            symbol=symbol,
            direction=direction,

            signal_idx=signal_index,
            entry_idx=entry_index,

            zone_idx=event["zone"],
            reaction_idx=event["reaction"],
            choch_idx=event["choch"],
            ob_idx=ob_index,

            liq1_idx=liquidity_first,
            liq2_idx=liquidity_second,
            sweep_idx=sweep_index,

            target_idx=target_index,

            entry=float(entry_price),
            sl=float(sl),
            tp=float(tp),
            target=float(target_price),
            risk=float(risk),

            choch_body_atr=float(
                event["choch_body_atr"]
            ),

            choch_ext_atr=float(
                event["choch_ext_atr"]
            ),

            displacement_range_atr=float(
                event["displacement_range_atr"]
            ),

            sweep_pen_atr=float(
                sweep_penetration_atr
            ),

            structural_room_R=float(
                structural_room_R
            ),

            ifc=int(ifc_present),
        )

        candidates.append(candidate)

    candidates.sort(
        key=lambda x: x.entry_idx
    )

    return execution, candidates


# ============================================================
# CANDIDATE AUDIT
# ============================================================

def audit_candidate(candidate):

    assert (
        candidate.zone_idx
        < candidate.reaction_idx
        < candidate.choch_idx
    )

    assert (
        candidate.ob_idx
        < candidate.choch_idx
    )

    assert (
        candidate.liq1_idx
        < candidate.liq2_idx
        < candidate.sweep_idx
        < candidate.entry_idx
    )

    assert (
        candidate.target_idx
        < candidate.entry_idx
    )

    # True limit fill:
    signal and entry are the same candle.
    assert (
        candidate.entry_idx
        == candidate.signal_idx
    )

    if candidate.direction == "LONG":

        assert (
            candidate.tp
            > candidate.entry
            > candidate.sl
        )

    else:

        assert (
            candidate.sl
            > candidate.entry
            > candidate.tp
        )

    # Fixed RR before costs.
    long_r = (
        candidate.tp
        - candidate.entry
    )

    short_r = (
        candidate.entry
        - candidate.tp
    )

    if candidate.direction == "LONG":

        assert np.isclose(
            long_r,
            RR * candidate.risk,
            rtol=1e-9,
            atol=1e-9,
        )

    else:

        assert np.isclose(
            short_r,
            RR * candidate.risk,
            rtol=1e-9,
            atol=1e-9,
        )


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate(
    execution,
    candidates,
):

    trades = []
    unresolved = 0

    # One open trade per symbol.
    # Since this function processes one symbol,
    # one occupied index is enough.
    occupied_until = -1

    for candidate in candidates:

        # No same-symbol overlap.
        if (
            candidate.entry_idx
            <= occupied_until
        ):
            continue

        exit_index = None
        exit_price = None
        result = None

        # IMPORTANT:
        # Do NOT check the entry candle for SL/TP.
        # This avoids same-candle entry/exit ambiguity.
        for j in range(
            candidate.entry_idx + 1,
            len(execution),
        ):

            high = float(
                execution["high"].iloc[j]
            )

            low = float(
                execution["low"].iloc[j]
            )

            if candidate.direction == "LONG":

                # Conservative:
                # SL wins if both are touched.
                if low <= candidate.sl:

                    exit_index = j
                    exit_price = candidate.sl
                    result = "LOSS"

                    break

                if high >= candidate.tp:

                    exit_index = j
                    exit_price = candidate.tp
                    result = "WIN"

                    break

            else:

                if high >= candidate.sl:

                    exit_index = j
                    exit_price = candidate.sl
                    result = "LOSS"

                    break

                if low <= candidate.tp:

                    exit_index = j
                    exit_price = candidate.tp
                    result = "WIN"

                    break

        # ----------------------------------------------------
        # UNRESOLVED
        # ----------------------------------------------------

        if exit_index is None:

            unresolved += 1

            # Position remains open until dataset end.
            occupied_until = len(execution) - 1

            continue

        # ----------------------------------------------------
        # PNL
        # ----------------------------------------------------

        if candidate.direction == "LONG":

            gross_move = (
                exit_price
                - candidate.entry
            )

        else:

            gross_move = (
                candidate.entry
                - exit_price
            )

        gross_pnl = (
            gross_move
            / candidate.entry
            * NOTIONAL
        )

        # Entry + exit fee.
        fees = (
            2.0
            * FEE_RATE
            * NOTIONAL
        )

        pnl = (
            gross_pnl
            - fees
        )

        # R before fees.
        nominal_risk_money = (
            candidate.risk
            / candidate.entry
            * NOTIONAL
        )

        realized_R = (
            gross_pnl
            / nominal_risk_money
        )

        trade = Trade(

            symbol=candidate.symbol,
            direction=candidate.direction,

            signal_idx=candidate.signal_idx,
            entry_idx=candidate.entry_idx,
            exit_idx=exit_index,

            entry_time=str(
                execution["open_time"].iloc[
                    candidate.entry_idx
                ]
            ),

            exit_time=str(
                execution["open_time"].iloc[
                    exit_index
                ]
            ),

            entry=candidate.entry,
            sl=candidate.sl,
            tp=candidate.tp,
            exit=float(exit_price),

            pnl=float(pnl),
            result=result,
            R=float(realized_R),

            zone_idx=candidate.zone_idx,
            reaction_idx=candidate.reaction_idx,
            choch_idx=candidate.choch_idx,
            ob_idx=candidate.ob_idx,

            liq1_idx=candidate.liq1_idx,
            liq2_idx=candidate.liq2_idx,
            sweep_idx=candidate.sweep_idx,

            target_idx=candidate.target_idx,
            target=candidate.target,

            choch_body_atr=candidate.choch_body_atr,
            choch_ext_atr=candidate.choch_ext_atr,
            displacement_range_atr=candidate.displacement_range_atr,

            sweep_pen_atr=candidate.sweep_pen_atr,
            structural_room_R=candidate.structural_room_R,

            ifc=candidate.ifc,
        )

        trades.append(trade)

        # Same-symbol re-entry only after exit candle.
        occupied_until = exit_index

    return trades, unresolved


# ============================================================
# TRADE AUDIT
# ============================================================

def audit_trades(trades):

    by_symbol = {}

    for trade in trades:

        by_symbol.setdefault(
            trade.symbol,
            [],
        ).append(trade)

    for symbol, symbol_trades in by_symbol.items():

        symbol_trades.sort(
            key=lambda x: x.entry_idx
        )

        for previous, current in zip(
            symbol_trades,
            symbol_trades[1:],
        ):

            # Earliest re-entry:
            # candle AFTER the previous exit candle.
            assert (
                current.entry_idx
                > previous.exit_idx
            ), (
                f"{symbol}: "
                "same-symbol overlap/re-entry "
                f"prev_exit={previous.exit_idx} "
                f"current_entry={current.entry_idx}"
            )


# ============================================================
# METRICS
# ============================================================

def metrics(trades):

    if not trades:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "wr": 0.0,
            "pf": 0.0,
            "net": 0.0,
            "max_streak": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
        }

    ordered = sorted(
        trades,
        key=lambda x: (
            x.exit_time,
            x.symbol,
        ),
    )

    wins = [
        t.pnl
        for t in ordered
        if t.pnl > 0
    ]

    losses = [
        t.pnl
        for t in ordered
        if t.pnl <= 0
    ]

    gross_profit = sum(wins)
    gross_loss = abs(
        sum(losses)
    )

    current_streak = 0
    max_streak = 0

    for trade in ordered:

        if trade.result == "LOSS":
            current_streak += 1

        else:
            current_streak = 0

        max_streak = max(
            max_streak,
            current_streak,
        )

    return {
        "trades": len(ordered),

        "wins": len(wins),
        "losses": len(losses),

        "wr": (
            100.0
            * len(wins)
            / len(ordered)
        ),

        "pf": (
            gross_profit
            / gross_loss
            if gross_loss > 0
            else float("inf")
        ),

        "net": float(
            sum(t.pnl for t in ordered)
        ),

        "max_streak": max_streak,

        "avg_win": (
            float(np.mean(wins))
            if wins
            else 0.0
        ),

        "avg_loss": (
            float(np.mean(losses))
            if losses
            else 0.0
        ),

        "expectancy": float(
            np.mean(
                [t.pnl for t in ordered]
            )
        ),
    }


# ============================================================
# TRAIN / VALIDATION / OOS
# ============================================================

def split_periods(
    trades,
    start,
    end,
):

    train_end = (
        start
        + (end - start)
        * TRAIN_FRAC
    )

    valid_end = (
        train_end
        + (end - start)
        * VALID_FRAC
    )

    def entry_timestamp(trade):

        return pd.Timestamp(
            trade.entry_time,
            tz="UTC",
        )

    train = [
        t
        for t in trades
        if entry_timestamp(t) < train_end
    ]

    validation = [
        t
        for t in trades
        if (
            train_end
            <= entry_timestamp(t)
            < valid_end
        )
    ]

    oos = [
        t
        for t in trades
        if entry_timestamp(t) >= valid_end
    ]

    return (
        train,
        validation,
        oos,
    )


# ============================================================
# SYMBOL PROCESSOR
# ============================================================

def process_symbol(
    symbol,
    start,
    end,
):

    execution = fetch_data(
        symbol,
        EXEC_TF,
        start - pd.Timedelta(
            days=WARMUP_DAYS
        ),
        end,
    )

    htf = fetch_data(
        symbol,
        HTF_TF,
        start - pd.Timedelta(
            days=WARMUP_DAYS + 15
        ),
        end,
    )

    validate_data(
        execution,
        15,
        symbol,
        "15m",
    )

    validate_data(
        htf,
        240,
        symbol,
        "4h",
    )

    execution, candidates = (
        build_candidates(
            symbol,
            execution,
            htf,
        )
    )

    for candidate in candidates:
        audit_candidate(candidate)

    trades, unresolved = simulate(
        execution,
        candidates,
    )

    audit_trades(trades)

    return (
        symbol,
        candidates,
        trades,
        unresolved,
    )


# ============================================================
# PRINT
# ============================================================

def print_metrics(
    label,
    trades,
):

    m = metrics(trades)

    print(
        f"{label}: "
        f"trades={m['trades']} "
        f"W/L={m['wins']}/{m['losses']} "
        f"WR={m['wr']:.2f}% "
        f"PF={m['pf']:.3f} "
        f"Net=${m['net']:.2f} "
        f"MaxStreak={m['max_streak']}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    end = get_end_time()

    start = (
        end
        - pd.Timedelta(
            days=TEST_DAYS
        )
        + pd.Timedelta(
            minutes=15
        )
    )

    print(
        "SETUP 5 V6 — "
        "SOURCE-SEQUENCE / TRUE OB LIMIT BACKTEST"
    )

    print(
        f"UTC: {start} -> {end}"
    )

    print(
        f"Symbols: {len(SYMBOLS)} "
        f"TF: {EXEC_TF}/{HTF_TF} "
        f"RR=1:{RR:.0f}"
    )

    print(
        "Execution: TRUE proximal OB limit fill"
    )

    print(
        "No next-open approximation"
    )

    print(
        "Per-symbol overlap only"
    )

    print(
        "Different symbols may overlap"
    )

    all_candidates = []
    all_trades = []

    unresolved_total = 0

    failures = []

    start_runtime = time.time()

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                process_symbol,
                symbol,
                start,
                end,
            ): symbol

            for symbol in SYMBOLS
        }

        for future in as_completed(futures):

            symbol = futures[future]

            try:

                (
                    _,
                    candidates,
                    trades,
                    unresolved,
                ) = future.result()

                all_candidates.extend(
                    candidates
                )

                all_trades.extend(
                    trades
                )

                unresolved_total += unresolved

                print(
                    f"{symbol}: "
                    f"candidates={len(candidates)} "
                    f"trades={len(trades)} "
                    f"unresolved={unresolved}"
                )

            except Exception as exc:

                failures.append(
                    (
                        symbol,
                        repr(exc),
                    )
                )

                print(
                    symbol,
                    "FAILED",
                    exc,
                )

    # Never silently ignore symbol failures.
    if failures:

        message = "; ".join(
            f"{symbol}: {error}"
            for symbol, error
            in failures
        )

        raise RuntimeError(message)

    # Global chronological order.
    all_trades.sort(
        key=lambda x: (
            x.exit_time,
            x.symbol,
        )
    )

    audit_trades(
        all_trades
    )

    # --------------------------------------------------------
    # SPLITS
    # --------------------------------------------------------

    train, validation, oos = (
        split_periods(
            all_trades,
            start,
            end,
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print("FULL")
    print_metrics(
        "FULL",
        all_trades,
    )

    print("TRAIN")
    print_metrics(
        "TRAIN",
        train,
    )

    print("VALIDATION")
    print_metrics(
        "VALIDATION",
        validation,
    )

    print("OOS LOCKED")
    print_metrics(
        "OOS LOCKED",
        oos,
    )

    # --------------------------------------------------------
    # OOS DIRECTION
    # --------------------------------------------------------

    print()

    for direction in (
        "LONG",
        "SHORT",
    ):

        direction_trades = [
            t
            for t in oos
            if t.direction == direction
        ]

        print_metrics(
            f"OOS {direction}",
            direction_trades,
        )

    # --------------------------------------------------------
    # OOS BY SYMBOL
    # --------------------------------------------------------

    print()

    for symbol in SYMBOLS:

        symbol_trades = [
            t
            for t in oos
            if t.symbol == symbol
        ]

        print_metrics(
            f"OOS {symbol}",
            symbol_trades,
        )

    # --------------------------------------------------------
    # IFC DIAGNOSTIC
    # --------------------------------------------------------

    print()

    for value in (0, 1):

        subset = [
            t
            for t in oos
            if t.ifc == value
        ]

        print_metrics(
            f"OOS IFC={value}",
            subset,
        )

    # --------------------------------------------------------
    # UNRESOLVED
    # --------------------------------------------------------

    print(
        f"\nUnresolved={unresolved_total}"
    )

    # --------------------------------------------------------
    # TARGET CHECK
    # --------------------------------------------------------

    oos_metrics = metrics(
        oos
    )

    print(
        "\nTARGET CHECK "
        "— DESCRIPTIVE ONLY"
    )

    print(
        "OOS trades >=100 : "
        f"{'PASS' if oos_metrics['trades'] >= 100 else 'FAIL'}"
    )

    print(
        "OOS WR >=50%     : "
        f"{'PASS' if oos_metrics['wr'] >= 50 else 'FAIL'}"
    )

    print(
        "OOS PF >=1.20    : "
        f"{'PASS' if oos_metrics['pf'] >= 1.20 else 'FAIL'}"
    )

    print(
        "OOS streak <=4   : "
        f"{'PASS' if oos_metrics['max_streak'] <= 4 else 'FAIL'}"
    )

    print(
        "OOS Net >0       : "
        f"{'PASS' if oos_metrics['net'] > 0 else 'FAIL'}"
    )

    # --------------------------------------------------------
    # CSV OUTPUTS
    # --------------------------------------------------------

    pd.DataFrame(
        [
            asdict(c)
            for c in all_candidates
        ]
    ).to_csv(
        "setup5_v6_candidates.csv",
        index=False,
    )

    pd.DataFrame(
        [
            asdict(t)
            for t in all_trades
        ]
    ).to_csv(
        "setup5_v6_trades.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "split": "TRAIN",
                **metrics(train),
            },
            {
                "split": "VALIDATION",
                **metrics(validation),
            },
            {
                "split": "OOS",
                **metrics(oos),
            },
        ]
    ).to_csv(
        "setup5_v6_summary.csv",
        index=False,
    )

    print(
        f"\nRuntime: "
        f"{time.time() - start_runtime:.1f}s"
    )


if __name__ == "__main__":
    main()
