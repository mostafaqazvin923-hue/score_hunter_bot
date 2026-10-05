#!/usr/bin/env python3
"""
SETUP 5 V1
Causal Binance USD-M Futures backtest

Execution timeframe : 15m
Higher timeframe     : 4h
Warmup               : 90 days
Test                 : 365 days

Integrity rules:
- No lookahead / future leak
- No same-candle re-entry
- Max 1 simultaneous trade per symbol
- Different symbols may overlap
- Entry only on next 15m candle open
- SL/TP only
- No timeout
- No breakeven
- No trailing
- No partial exits
- Fixed RR 1:2
- Conservative same-candle SL/TP handling: SL wins
- Unresolved trades are explicitly reported
"""

import io
import os
import sys
import time
import zipfile
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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

EXEC_INTERVAL = "15m"
HTF_INTERVAL = "4h"

WARMUP_DAYS = 90
TEST_DAYS = 365

PIVOT = 2

# Setup 5 V1 mechanical translations
ALIGN_TOL = 0.004
MAX_LIQ_TO_OB_BARS = 24

# 0.05% structural stop buffer
SL_BUFFER = 0.0005

# Risk limits
MIN_RISK = 0.0005
MAX_RISK = 0.08

# Trade model
RR = 2.0
INITIAL_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN_PER_TRADE * LEVERAGE

# Conservative cost assumptions
FEE_RATE = 0.0007
SLIPPAGE = 0.0003

# Binance public archive
BASE_URL = "https://data.binance.vision/data/futures/um"

# Cache
CACHE_DIR = Path(
    os.getenv("BINANCE_CACHE_DIR", ".cache_binance_um")
)

# Parallel downloading
DOWNLOAD_WORKERS = int(
    os.getenv("DOWNLOAD_WORKERS", "6")
)

REQUEST_TIMEOUT = 60
MAX_RETRIES = 4

# Optional reproducibility:
#
# BACKTEST_END=2026-10-03
#
# means last execution candle = 2026-10-03 23:45 UTC.
#
# If empty, script uses the last completed UTC day.
BACKTEST_END = os.getenv(
    "BACKTEST_END",
    ""
).strip()


# ============================================================
# CONSTANTS
# ============================================================

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "setup5-v1-causal-backtester/1.0"
    }
)


# ============================================================
# TIME
# ============================================================

def utc_now():
    return pd.Timestamp.now(tz="UTC")


def parse_backtest_end():
    """
    Default:
        previous completed UTC day, 23:45

    Optional:
        BACKTEST_END=YYYY-MM-DD
    """
    if BACKTEST_END:
        ts = pd.Timestamp(BACKTEST_END)

        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        return (
            ts.normalize()
            + pd.Timedelta(
                hours=23,
                minutes=45
            )
        )

    # Last complete UTC day, last 15m candle.
    return (
        utc_now().normalize()
        - pd.Timedelta(minutes=15)
    )


OOS_END = parse_backtest_end()

OOS_START = (
    OOS_END
    - pd.Timedelta(days=TEST_DAYS)
    + pd.Timedelta(minutes=15)
)

DATA_START = (
    OOS_START
    - pd.Timedelta(days=WARMUP_DAYS)
)


# ============================================================
# DATE HELPERS
# ============================================================

def month_range(start, end):
    current = start.normalize().replace(day=1)
    last = end.normalize().replace(day=1)

    while current <= last:
        yield current
        current = current + pd.offsets.MonthBegin(1)


def day_range(start, end):
    current = start.normalize()
    last = end.normalize()

    while current <= last:
        yield current
        current += pd.Timedelta(days=1)


# ============================================================
# BINANCE ARCHIVE
# ============================================================

def archive_url(
    symbol,
    interval,
    timestamp,
    daily
):
    if daily:
        date_str = timestamp.strftime(
            "%Y-%m-%d"
        )

        return (
            f"{BASE_URL}/daily/klines/"
            f"{symbol}/{interval}/"
            f"{symbol}-{interval}-{date_str}.zip"
        )

    month_str = timestamp.strftime(
        "%Y-%m"
    )

    return (
        f"{BASE_URL}/monthly/klines/"
        f"{symbol}/{interval}/"
        f"{symbol}-{interval}-{month_str}.zip"
    )


def cache_path(
    symbol,
    interval,
    key,
    daily
):
    folder = (
        CACHE_DIR
        / interval
        / symbol
        / ("daily" if daily else "monthly")
    )

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    return folder / f"{key}.csv"


def download_archive(
    url,
    cache_file
):
    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------
    if (
        cache_file.exists()
        and cache_file.stat().st_size > 100
    ):
        try:
            return pd.read_csv(
                cache_file
            )
        except Exception:
            cache_file.unlink(
                missing_ok=True
            )

    last_error = None

    # --------------------------------------------------------
    # Download with retry
    # --------------------------------------------------------
    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):
        try:
            response = SESSION.get(
                url,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()

            with zipfile.ZipFile(
                io.BytesIO(
                    response.content
                )
            ) as zf:

                csv_files = [
                    name
                    for name in zf.namelist()
                    if name.lower().endswith(".csv")
                ]

                if not csv_files:
                    raise RuntimeError(
                        f"No CSV in archive: {url}"
                    )

                with zf.open(
                    csv_files[0]
                ) as fh:

                    raw = pd.read_csv(
                        fh,
                        header=None
                    )

            raw.to_csv(
                cache_file,
                index=False
            )

            return raw

        except Exception as exc:
            last_error = exc

            if attempt < MAX_RETRIES:
                time.sleep(
                    1.5 * attempt
                )

    raise RuntimeError(
        f"Download failed after "
        f"{MAX_RETRIES} attempts:\n"
        f"{url}\n"
        f"{last_error}"
    )


# ============================================================
# DATA NORMALIZATION
# ============================================================

def normalize_raw(raw):
    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
            ]
        )

    raw = raw.iloc[:, :12].copy()

    raw.columns = KLINE_COLUMNS

    # Binance USD-M Futures archive timestamps
    # are milliseconds.
    raw["open_time"] = pd.to_numeric(
        raw["open_time"],
        errors="coerce"
    )

    raw["close_time"] = pd.to_numeric(
        raw["close_time"],
        errors="coerce"
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        raw[column] = pd.to_numeric(
            raw[column],
            errors="coerce"
        )

    raw["open_time"] = pd.to_datetime(
        raw["open_time"],
        unit="ms",
        utc=True,
        errors="coerce"
    )

    raw["close_time"] = pd.to_datetime(
        raw["close_time"],
        unit="ms",
        utc=True,
        errors="coerce"
    )

    raw = raw.dropna(
        subset=[
            "open_time",
            "close_time",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    return raw[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
        ]
    ].copy()


# ============================================================
# LOAD BINANCE KLINES
# ============================================================

def load_klines(
    symbol,
    interval,
    start,
    end
):
    parts = []

    months = list(
        month_range(
            start,
            end
        )
    )

    # --------------------------------------------------------
    # Monthly archives
    #
    # We avoid using the final month archive because it can
    # still be incomplete. The final month is obtained through
    # daily archives.
    # --------------------------------------------------------
    for month in months:

        is_final_month = (
            month.year == end.year
            and month.month == end.month
        )

        if is_final_month:
            continue

        key = month.strftime(
            "%Y-%m"
        )

        path = cache_path(
            symbol,
            interval,
            key,
            daily=False
        )

        raw = download_archive(
            archive_url(
                symbol,
                interval,
                month,
                daily=False
            ),
            path
        )

        if raw is not None:
            normalized = normalize_raw(
                raw
            )

            if not normalized.empty:
                parts.append(
                    normalized
                )

    # --------------------------------------------------------
    # First and final months through daily archives.
    # This also guarantees precise boundary control.
    # --------------------------------------------------------
    month_keys = {
        (
            start.year,
            start.month
        ),
        (
            end.year,
            end.month
        ),
    }

    for year, month in sorted(
        month_keys
    ):

        month_start = pd.Timestamp(
            year=year,
            month=month,
            day=1,
            tz="UTC"
        )

        next_month = (
            month_start
            + pd.offsets.MonthBegin(1)
        )

        month_end = (
            next_month
            - pd.Timedelta(days=1)
        )

        daily_start = max(
            start.normalize(),
            month_start
        )

        daily_end = min(
            end.normalize(),
            month_end
        )

        for day in day_range(
            daily_start,
            daily_end
        ):

            key = day.strftime(
                "%Y-%m-%d"
            )

            path = cache_path(
                symbol,
                interval,
                key,
                daily=True
            )

            raw = download_archive(
                archive_url(
                    symbol,
                    interval,
                    day,
                    daily=True
                ),
                path
            )

            if raw is not None:
                normalized = normalize_raw(
                    raw
                )

                if not normalized.empty:
                    parts.append(
                        normalized
                    )

    if not parts:
        raise RuntimeError(
            f"No Binance data found: "
            f"{symbol} {interval}"
        )

    df = pd.concat(
        parts,
        ignore_index=True
    )

    # --------------------------------------------------------
    # De-duplicate
    # --------------------------------------------------------
    df = (
        df.drop_duplicates(
            subset=["open_time"]
        )
        .sort_values(
            "open_time"
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Never use current incomplete candle.
    # --------------------------------------------------------
    now = utc_now()

    df = df[
        df["close_time"] <= now
    ].copy()

    # --------------------------------------------------------
    # Exact requested range.
    # --------------------------------------------------------
    df = df[
        (df["open_time"] >= start)
        &
        (df["open_time"] <= end)
    ].copy()

    if df.empty:
        raise RuntimeError(
            f"Empty filtered dataset: "
            f"{symbol} {interval}"
        )

    return (
        df.sort_values("open_time")
        .reset_index(drop=True)
    )


# ============================================================
# DATA INTEGRITY
# ============================================================

def validate_data(
    df,
    expected_minutes,
    symbol,
    interval
):
    if df.empty:
        raise RuntimeError(
            f"{symbol} {interval}: empty"
        )

    if df["open_time"].duplicated().any():
        raise RuntimeError(
            f"{symbol} {interval}: duplicate candles"
        )

    # OHLC integrity
    bad_ohlc = (
        (df["high"] < df["low"])
        |
        (
            df["high"]
            < df[
                [
                    "open",
                    "close"
                ]
            ].max(axis=1)
        )
        |
        (
            df["low"]
            > df[
                [
                    "open",
                    "close"
                ]
            ].min(axis=1)
        )
    )

    if bad_ohlc.any():
        raise RuntimeError(
            f"{symbol} {interval}: invalid OHLC"
        )

    # Gap check
    delta = (
        df["open_time"]
        .diff()
        .dropna()
    )

    expected = pd.Timedelta(
        minutes=expected_minutes
    )

    gaps = delta[
        delta > expected * 1.5
    ]

    if not gaps.empty:
        raise RuntimeError(
            f"{symbol} {interval}: "
            f"data gap detected. "
            f"largest={gaps.max()}"
        )


# ============================================================
# CONFIRMED PIVOTS
# ============================================================

def confirmed_pivots(
    df,
    pivot=PIVOT
):
    highs = df[
        "high"
    ].to_numpy(
        dtype=float
    )

    lows = df[
        "low"
    ].to_numpy(
        dtype=float
    )

    n = len(df)

    pivot_high = np.zeros(
        n,
        dtype=bool
    )

    pivot_low = np.zeros(
        n,
        dtype=bool
    )

    for i in range(
        pivot,
        n - pivot
    ):

        h = highs[i]
        l = lows[i]

        left_high = highs[
            i - pivot:i
        ]

        right_high = highs[
            i + 1:i + pivot + 1
        ]

        left_low = lows[
            i - pivot:i
        ]

        right_low = lows[
            i + 1:i + pivot + 1
        ]

        # Confirmed pivot high
        if (
            h > left_high.max()
            and
            h >= right_high.max()
        ):
            pivot_high[i] = True

        # Confirmed pivot low
        if (
            l < left_low.min()
            and
            l <= right_low.max()
        ):
            # This line is intentionally replaced below.
            pass

    # Recalculate lows correctly.
    pivot_low[:] = False

    for i in range(
        pivot,
        n - pivot
    ):

        l = lows[i]

        left_low = lows[
            i - pivot:i
        ]

        right_low = lows[
            i + 1:i + pivot + 1
        ]

        if (
            l < left_low.min()
            and
            l <= right_low.min()
        ):
            pivot_low[i] = True

    return (
        pivot_high,
        pivot_low
    )


def pivots_known_before(
    pivot_high,
    pivot_low,
    index,
    pivot=PIVOT
):
    """
    A pivot centered at i becomes known only after
    PIVOT candles to its right have closed.

    Therefore only pivot centers <= index-PIVOT
    are allowed.
    """

    last_known_center = (
        index - pivot
    )

    if last_known_center < 0:
        return [], []

    highs = np.flatnonzero(
        pivot_high[
            :last_known_center + 1
        ]
    ).tolist()

    lows = np.flatnonzero(
        pivot_low[
            :last_known_center + 1
        ]
    ).tolist()

    return highs, lows


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Candidate:
    symbol: str
    direction: str

    reaction_idx: int

    htf_zone_idx: int
    htf_known_idx: int

    choch_idx: int

    ob_idx: int

    liq_idx1: int
    liq_idx2: int
    liq_level: float

    sweep_idx: int

    entry_signal_idx: int
    entry_idx: int

    structural_target_idx: int

    ob_low: float
    ob_high: float

    stop: float
    target: float

    entry_price: float


@dataclass
class Trade:
    symbol: str
    direction: str

    entry_idx: int
    entry_time: str

    entry_price: float

    stop: float
    target: float

    exit_idx: int | None
    exit_time: str | None
    exit_price: float | None

    outcome: str

    pnl: float
    fees: float
    r_multiple: float


# ============================================================
# BASIC HELPERS
# ============================================================

def candle_touches_zone(
    row,
    zone_low,
    zone_high
):
    return (
        float(row["low"])
        <= zone_high
        and
        float(row["high"])
        >= zone_low
    )


def last_two(values):
    if len(values) < 2:
        return None

    return (
        values[-2],
        values[-1]
    )


# ============================================================
# HTF ZONES
# ============================================================

def get_htf_zones(
    htf,
    htf_pivot_high,
    htf_pivot_low,
    direction
):
    zones = []

    pivot_array = (
        htf_pivot_low
        if direction == "LONG"
        else htf_pivot_high
    )

    centers = np.flatnonzero(
        pivot_array
    )

    for center in centers:

        known_idx = (
            center + PIVOT
        )

        if known_idx >= len(htf):
            continue

        row = htf.iloc[
            center
        ]

        if direction == "LONG":

            zone_low = float(
                row["low"]
            )

            zone_high = max(
                float(row["open"]),
                float(row["close"])
            )

        else:

            zone_low = min(
                float(row["open"]),
                float(row["close"])
            )

            zone_high = float(
                row["high"]
            )

        zones.append(
            {
                "center_idx": int(center),
                "known_idx": int(known_idx),
                "known_time": htf.iloc[
                    known_idx
                ]["close_time"],
                "zone_low": zone_low,
                "zone_high": zone_high,
            }
        )

    return zones


# ============================================================
# HTF REACTION
# ============================================================

def find_reaction(
    df,
    zone,
    start_idx,
    direction
):
    zone_low = zone[
        "zone_low"
    ]

    zone_high = zone[
        "zone_high"
    ]

    for i in range(
        start_idx,
        len(df)
    ):

        row = df.iloc[i]

        # First valid touch
        if candle_touches_zone(
            row,
            zone_low,
            zone_high
        ):
            return i

        # Zone invalidation before reaction
        if direction == "LONG":
            if float(row["close"]) < zone_low:
                return None

        else:
            if float(row["close"]) > zone_high:
                return None

    return None


# ============================================================
# TREND
# ============================================================

def trend_ok(
    df,
    pivot_high,
    pivot_low,
    index,
    direction
):
    highs, lows = pivots_known_before(
        pivot_high,
        pivot_low,
        index
    )

    last_highs = last_two(
        highs
    )

    last_lows = last_two(
        lows
    )

    if (
        last_highs is None
        or
        last_lows is None
    ):
        return False

    sh2, sh1 = last_highs
    sl2, sl1 = last_lows

    if direction == "LONG":

        # Existing downtrend:
        # lower swing highs + lower swing lows
        return (
            float(
                df.iloc[sh1]["high"]
            )
            <
            float(
                df.iloc[sh2]["high"]
            )
            and
            float(
                df.iloc[sl1]["low"]
            )
            <
            float(
                df.iloc[sl2]["low"]
            )
        )

    # Existing uptrend
    return (
        float(
            df.iloc[sh1]["high"]
        )
        >
        float(
            df.iloc[sh2]["high"]
        )
        and
        float(
            df.iloc[sl1]["low"]
        )
        >
        float(
            df.iloc[sl2]["low"]
        )
    )


# ============================================================
# CHOCH
# ============================================================

def find_choch(
    df,
    pivot_high,
    pivot_low,
    reaction_idx,
    direction
):
    # CHOCH cannot occur on reaction candle.
    for t in range(
        reaction_idx + 1,
        len(df)
    ):

        highs, lows = pivots_known_before(
            pivot_high,
            pivot_low,
            t
        )

        if direction == "LONG":

            if not highs:
                continue

            last_swing_high = highs[-1]

            # Bullish CHOCH
            if (
                float(
                    df.iloc[t]["close"]
                )
                >
                float(
                    df.iloc[
                        last_swing_high
                    ]["high"]
                )
            ):
                return t

        else:

            if not lows:
                continue

            last_swing_low = lows[-1]

            # Bearish CHOCH
            if (
                float(
                    df.iloc[t]["close"]
                )
                <
                float(
                    df.iloc[
                        last_swing_low
                    ]["low"]
                )
            ):
                return t

    return None


# ============================================================
# ORDER BLOCK
# ============================================================

def find_order_block(
    df,
    pivot_high,
    pivot_low,
    reaction_idx,
    choch_idx,
    direction
):
    highs, lows = pivots_known_before(
        pivot_high,
        pivot_low,
        choch_idx
    )

    if direction == "LONG":

        valleys = [
            i
            for i in lows
            if reaction_idx < i < choch_idx
        ]

        if not valleys:
            return None

        # Lowest confirmed valley
        ob_idx = min(
            valleys,
            key=lambda i:
            float(
                df.iloc[i]["low"]
            )
        )

        row = df.iloc[
            ob_idx
        ]

        ob_low = float(
            row["low"]
        )

        ob_high = max(
            float(row["open"]),
            float(row["close"])
        )

        return (
            ob_idx,
            ob_low,
            ob_high
        )

    peaks = [
        i
        for i in highs
        if reaction_idx < i < choch_idx
    ]

    if not peaks:
        return None

    # Highest confirmed peak
    ob_idx = max(
        peaks,
        key=lambda i:
        float(
            df.iloc[i]["high"]
        )
    )

    row = df.iloc[
        ob_idx
    ]

    ob_low = min(
        float(row["open"]),
        float(row["close"])
    )

    ob_high = float(
        row["high"]
    )

    return (
        ob_idx,
        ob_low,
        ob_high
    )


# ============================================================
# LIQUIDITY + SWEEP
# ============================================================

def find_liquidity_and_sweep(
    df,
    pivot_high,
    pivot_low,
    choch_idx,
    ob_low,
    ob_high,
    direction
):
    """
    Requirements:
      CHOCH
        ->
      at least two aligned valleys/peaks
        ->
      liquidity sweep
        ->
      return to OB
    """

    for sweep_idx in range(
        choch_idx + 1,
        len(df)
    ):

        highs, lows = pivots_known_before(
            pivot_high,
            pivot_low,
            sweep_idx
        )

        pivots = (
            lows
            if direction == "LONG"
            else highs
        )

        post_choch = [
            i
            for i in pivots
            if i > choch_idx
            and i < sweep_idx
        ]

        if len(post_choch) < 2:
            continue

        # Most recent pair first
        for pos in range(
            len(post_choch) - 1,
            0,
            -1
        ):

            idx1 = post_choch[
                pos - 1
            ]

            idx2 = post_choch[
                pos
            ]

            if direction == "LONG":

                p1 = float(
                    df.iloc[idx1]["low"]
                )

                p2 = float(
                    df.iloc[idx2]["low"]
                )

            else:

                p1 = float(
                    df.iloc[idx1]["high"]
                )

                p2 = float(
                    df.iloc[idx2]["high"]
                )

            level = (
                p1 + p2
            ) / 2.0

            if abs(p1 - p2) / max(
                abs(level),
                1e-12
            ) > ALIGN_TOL:
                continue

            # Liquidity-to-OB temporal limit.
            if (
                sweep_idx - idx2
                >
                MAX_LIQ_TO_OB_BARS
            ):
                continue

            row = df.iloc[
                sweep_idx
            ]

            if direction == "LONG":

                swept = (
                    float(row["low"])
                    < level
                    and
                    float(row["close"])
                    >= level
                )

            else:

                swept = (
                    float(row["high"])
                    > level
                    and
                    float(row["close"])
                    <= level
                )

            if swept:
                return (
                    idx1,
                    idx2,
                    level,
                    sweep_idx
                )

    return None


# ============================================================
# RETURN TO OB
# ============================================================

def find_ob_return(
    df,
    sweep_idx,
    ob_low,
    ob_high
):
    """
    The signal candle is the first return to the OB
    after the liquidity sweep.

    Actual entry occurs at the NEXT candle open.
    """

    for t in range(
        sweep_idx + 1,
        len(df) - 1
    ):

        row = df.iloc[t]

        if candle_touches_zone(
            row,
            ob_low,
            ob_high
        ):
            return t

    return None


# ============================================================
# STRUCTURAL TARGET
# ============================================================

def find_structural_target(
    df,
    pivot_high,
    pivot_low,
    choch_idx,
    entry_signal_idx,
    direction
):
    """
    Conservative causal translation:

    Only confirmed swing peaks/valleys that were known
    before the entry signal candle are allowed.

    This prevents the target filter from seeing future price.
    """

    highs, lows = pivots_known_before(
        pivot_high,
        pivot_low,
        entry_signal_idx
    )

    if direction == "LONG":

        peaks = [
            i
            for i in highs
            if choch_idx < i < entry_signal_idx
        ]

        if not peaks:
            return None

        idx = max(
            peaks,
            key=lambda i:
            float(
                df.iloc[i]["high"]
            )
        )

        return (
            idx,
            float(
                df.iloc[idx]["high"]
            )
        )

    valleys = [
        i
        for i in lows
        if choch_idx < i < entry_signal_idx
    ]

    if not valleys:
        return None

    idx = min(
        valleys,
        key=lambda i:
        float(
            df.iloc[i]["low"]
        )
    )

    return (
        idx,
        float(
            df.iloc[idx]["low"]
        )
    )


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def build_candidates(
    symbol,
    df,
    htf,
    htf_pivot_high,
    htf_pivot_low,
    pivot_high,
    pivot_low
):
    candidates = []

    for direction in (
        "LONG",
        "SHORT"
    ):

        zones = get_htf_zones(
            htf,
            htf_pivot_high,
            htf_pivot_low,
            direction
        )

        for zone in zones:

            known_time = zone[
                "known_time"
            ]

            # HTF information must be known before reaction.
            start_idx = int(
                df[
                    "open_time"
                ].searchsorted(
                    known_time,
                    side="right"
                )
            )

            if start_idx >= len(df):
                continue

            reaction_idx = find_reaction(
                df,
                zone,
                start_idx,
                direction
            )

            if reaction_idx is None:
                continue

            # Downtrend / uptrend requirement.
            if not trend_ok(
                df,
                pivot_high,
                pivot_low,
                reaction_idx,
                direction
            ):
                continue

            # CHOCH
            choch_idx = find_choch(
                df,
                pivot_high,
                pivot_low,
                reaction_idx,
                direction
            )

            if choch_idx is None:
                continue

            # OB
            ob = find_order_block(
                df,
                pivot_high,
                pivot_low,
                reaction_idx,
                choch_idx,
                direction
            )

            if ob is None:
                continue

            (
                ob_idx,
                ob_low,
                ob_high
            ) = ob

            # Liquidity + sweep
            liquidity = (
                find_liquidity_and_sweep(
                    df,
                    pivot_high,
                    pivot_low,
                    choch_idx,
                    ob_low,
                    ob_high,
                    direction
                )
            )

            if liquidity is None:
                continue

            (
                liq_idx1,
                liq_idx2,
                liq_level,
                sweep_idx
            ) = liquidity

            # Strict causal sequence.
            if not (
                choch_idx
                <
                liq_idx1
                <
                liq_idx2
                <
                sweep_idx
            ):
                continue

            # First return to OB.
            entry_signal_idx = find_ob_return(
                df,
                sweep_idx,
                ob_low,
                ob_high
            )

            if entry_signal_idx is None:
                continue

            if entry_signal_idx <= sweep_idx:
                continue

            # Actual entry = next candle OPEN.
            entry_idx = (
                entry_signal_idx + 1
            )

            if entry_idx >= len(df):
                continue

            raw_open = float(
                df.iloc[
                    entry_idx
                ]["open"]
            )

            if direction == "LONG":

                entry_price = (
                    raw_open
                    * (1.0 + SLIPPAGE)
                )

                stop = (
                    ob_low
                    * (1.0 - SL_BUFFER)
                )

                risk = (
                    entry_price - stop
                )

                if risk <= 0:
                    continue

                target = (
                    entry_price
                    + RR * risk
                )

            else:

                entry_price = (
                    raw_open
                    * (1.0 - SLIPPAGE)
                )

                stop = (
                    ob_high
                    * (1.0 + SL_BUFFER)
                )

                risk = (
                    stop - entry_price
                )

                if risk <= 0:
                    continue

                target = (
                    entry_price
                    - RR * risk
                )

            risk_pct = (
                risk / entry_price
            )

            if not (
                MIN_RISK
                <= risk_pct
                <= MAX_RISK
            ):
                continue

            # Structural target must be known BEFORE entry.
            structural = (
                find_structural_target(
                    df,
                    pivot_high,
                    pivot_low,
                    choch_idx,
                    entry_signal_idx,
                    direction
                )
            )

            if structural is None:
                continue

            (
                structural_idx,
                structural_price
            ) = structural

            if structural_idx >= entry_signal_idx:
                continue

            # 1:2 TP must remain inside structural target.
            if direction == "LONG":

                if target > structural_price:
                    continue

            else:

                if target < structural_price:
                    continue

            candidate = Candidate(
                symbol=symbol,
                direction=direction,
                reaction_idx=reaction_idx,
                htf_zone_idx=zone[
                    "center_idx"
                ],
                htf_known_idx=zone[
                    "known_idx"
                ],
                choch_idx=choch_idx,
                ob_idx=ob_idx,
                liq_idx1=liq_idx1,
                liq_idx2=liq_idx2,
                liq_level=liq_level,
                sweep_idx=sweep_idx,
                entry_signal_idx=entry_signal_idx,
                entry_idx=entry_idx,
                structural_target_idx=structural_idx,
                ob_low=ob_low,
                ob_high=ob_high,
                stop=stop,
                target=target,
                entry_price=entry_price,
            )

            candidates.append(
                candidate
            )

    # --------------------------------------------------------
    # Deduplicate identical entry points.
    # --------------------------------------------------------
    unique = {}

    for candidate in candidates:

        key = (
            candidate.symbol,
            candidate.direction,
            candidate.entry_idx
        )

        if key not in unique:
            unique[key] = candidate

    return sorted(
        unique.values(),
        key=lambda x: (
            x.entry_idx,
            x.direction
        )
    )


# ============================================================
# CANDIDATE AUDIT
# ============================================================

def audit_candidates(
    candidates,
    df,
    htf
):
    for c in candidates:

        # HTF information known before execution.
        assert (
            c.htf_known_idx
            <
            len(htf)
        )

        # OB before CHOCH
        assert (
            c.ob_idx
            <
            c.choch_idx
        )

        # Liquidity before sweep
        assert (
            c.liq_idx1
            <
            c.liq_idx2
            <
            c.sweep_idx
        )

        # Sweep before entry signal
        assert (
            c.sweep_idx
            <
            c.entry_signal_idx
            <
            c.entry_idx
        )

        # Entry is exactly next candle.
        assert (
            c.entry_idx
            ==
            c.entry_signal_idx + 1
        )

        # Structural target must be known before entry.
        assert (
            c.structural_target_idx
            <
            c.entry_signal_idx
        )

        # RR 1:2.
        if c.direction == "LONG":

            risk = (
                c.entry_price
                - c.stop
            )

            expected_target = (
                c.entry_price
                + RR * risk
            )

        else:

            risk = (
                c.stop
                - c.entry_price
            )

            expected_target = (
                c.entry_price
                - RR * risk
            )

        assert risk > 0

        assert np.isclose(
            c.target,
            expected_target,
            rtol=1e-10,
            atol=1e-10
        )

        assert (
            c.entry_idx
            <
            len(df)
        )


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_candidates(
    symbol,
    df,
    candidates
):
    trades = []

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This lock is PER SYMBOL.
    #
    # Different symbols are allowed to overlap.
    # --------------------------------------------------------
    occupied_until = -1

    for candidate in sorted(
        candidates,
        key=lambda x: x.entry_idx
    ):

        # No same-symbol overlap.
        #
        # If previous trade exits at J,
        # new entry can only happen at J+1.
        if (
            candidate.entry_idx
            <= occupied_until
        ):
            continue

        entry_idx = (
            candidate.entry_idx
        )

        entry_row = df.iloc[
            entry_idx
        ]

        entry_price = (
            candidate.entry_price
        )

        exit_idx = None
        exit_price = None

        outcome = "UNRESOLVED"

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Entry candle is NOT checked for SL/TP.
        #
        # This avoids using intrabar information on the
        # same candle whose OPEN created the entry.
        # ----------------------------------------------------
        for j in range(
            entry_idx + 1,
            len(df)
        ):

            row = df.iloc[j]

            high = float(
                row["high"]
            )

            low = float(
                row["low"]
            )

            if candidate.direction == "LONG":

                hit_sl = (
                    low
                    <= candidate.stop
                )

                hit_tp = (
                    high
                    >= candidate.target
                )

                # Conservative:
                # if both occur in the same candle,
                # SL wins.
                if hit_sl:

                    exit_idx = j
                    exit_price = (
                        candidate.stop
                    )
                    outcome = "LOSS"
                    break

                if hit_tp:

                    exit_idx = j
                    exit_price = (
                        candidate.target
                    )
                    outcome = "WIN"
                    break

            else:

                hit_sl = (
                    high
                    >= candidate.stop
                )

                hit_tp = (
                    low
                    <= candidate.target
                )

                if hit_sl:

                    exit_idx = j
                    exit_price = (
                        candidate.stop
                    )
                    outcome = "LOSS"
                    break

                if hit_tp:

                    exit_idx = j
                    exit_price = (
                        candidate.target
                    )
                    outcome = "WIN"
                    break

        # ----------------------------------------------------
        # Unresolved trade
        # ----------------------------------------------------
        if exit_idx is None:

            trades.append(
                Trade(
                    symbol=symbol,
                    direction=candidate.direction,
                    entry_idx=entry_idx,
                    entry_time=str(
                        entry_row["open_time"]
                    ),
                    entry_price=entry_price,
                    stop=candidate.stop,
                    target=candidate.target,
                    exit_idx=None,
                    exit_time=None,
                    exit_price=None,
                    outcome="UNRESOLVED",
                    pnl=0.0,
                    fees=0.0,
                    r_multiple=0.0,
                )
            )

            occupied_until = (
                len(df) - 1
            )

            continue

        exit_row = df.iloc[
            exit_idx
        ]

        # ----------------------------------------------------
        # Gross PnL
        # ----------------------------------------------------
        if candidate.direction == "LONG":

            gross_pnl = (
                (
                    exit_price
                    - entry_price
                )
                / entry_price
            ) * NOTIONAL

        else:

            gross_pnl = (
                (
                    entry_price
                    - exit_price
                )
                / entry_price
            ) * NOTIONAL

        # Two-sided fees
        fees = (
            NOTIONAL
            * FEE_RATE
            * 2.0
        )

        pnl = (
            gross_pnl
            - fees
        )

        # ----------------------------------------------------
        # R multiple
        # ----------------------------------------------------
        risk_dollars = (
            abs(
                entry_price
                - candidate.stop
            )
            / entry_price
        ) * NOTIONAL

        if risk_dollars > 0:
            r_multiple = (
                pnl
                / risk_dollars
            )
        else:
            r_multiple = 0.0

        trades.append(
            Trade(
                symbol=symbol,
                direction=candidate.direction,
                entry_idx=entry_idx,
                entry_time=str(
                    entry_row["open_time"]
                ),
                entry_price=entry_price,
                stop=candidate.stop,
                target=candidate.target,
                exit_idx=exit_idx,
                exit_time=str(
                    exit_row["open_time"]
                ),
                exit_price=exit_price,
                outcome=outcome,
                pnl=pnl,
                fees=fees,
                r_multiple=r_multiple,
            )
        )

        # ----------------------------------------------------
        # Re-entry restriction:
        #
        # exit on J
        # next possible entry = J+1
        #
        # Therefore candidate.entry_idx <= J is blocked.
        # ----------------------------------------------------
        occupied_until = exit_idx

    return trades


# ============================================================
# TRADE AUDIT
# ============================================================

def audit_trades(
    trades
):
    ordered = sorted(
        trades,
        key=lambda x: (
            x.entry_idx,
            x.symbol
        )
    )

    previous = None

    for trade in ordered:

        assert trade.outcome in {
            "WIN",
            "LOSS",
            "UNRESOLVED"
        }

        if trade.exit_idx is not None:

            assert (
                trade.exit_idx
                >
                trade.entry_idx
            )

        if previous is not None:

            if (
                previous.exit_idx is not None
                and
                trade.entry_idx
                <= previous.exit_idx
            ):
                raise AssertionError(
                    "Same-symbol overlap detected."
                )

        previous = trade


# ============================================================
# REPORTING
# ============================================================

def max_loss_streak(
    trades
):
    streak = 0
    maximum = 0

    ordered = sorted(
        trades,
        key=lambda x: (
            x.exit_time or "",
            x.symbol
        )
    )

    for trade in ordered:

        if trade.outcome == "LOSS":

            streak += 1

            maximum = max(
                maximum,
                streak
            )

        elif trade.outcome == "WIN":

            streak = 0

    return maximum


def summarize(
    trades
):
    resolved = [
        t
        for t in trades
        if t.outcome
        in {
            "WIN",
            "LOSS"
        }
    ]

    wins = [
        t
        for t in resolved
        if t.outcome == "WIN"
    ]

    losses = [
        t
        for t in resolved
        if t.outcome == "LOSS"
    ]

    gross_profit = sum(
        max(t.pnl, 0.0)
        for t in resolved
    )

    gross_loss = sum(
        min(t.pnl, 0.0)
        for t in resolved
    )

    if gross_loss < 0:
        profit_factor = (
            gross_profit
            /
            abs(gross_loss)
        )
    else:
        profit_factor = (
            float("inf")
            if gross_profit > 0
            else 0.0
        )

    net_pnl = sum(
        t.pnl
        for t in resolved
    )

    final_equity = (
        INITIAL_CAPITAL
        + net_pnl
    )

    # --------------------------------------------------------
    # Sequential realized-equity DD.
    #
    # Cross-symbol simultaneous exposure is allowed.
    # This DD is based on realized trade exits and is not
    # presented as an intrabar mark-to-market portfolio DD.
    # --------------------------------------------------------
    equity = INITIAL_CAPITAL
    peak = equity
    max_dd = 0.0

    for trade in sorted(
        resolved,
        key=lambda x: (
            x.exit_time,
            x.symbol
        )
    ):

        equity += trade.pnl

        peak = max(
            peak,
            equity
        )

        drawdown = (
            peak
            - equity
        )

        max_dd = max(
            max_dd,
            drawdown
        )

    win_rate = (
        100.0
        * len(wins)
        / len(resolved)
        if resolved
        else 0.0
    )

    return {
        "trades": len(resolved),
        "wins": len(wins),
        "losses": len(losses),
        "unresolved": (
            len(trades)
            - len(resolved)
        ),
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "net_pnl": net_pnl,
        "final_equity": final_equity,
        "max_drawdown": max_dd,
        "max_loss_streak": max_loss_streak(
            resolved
        ),
        "avg_win": (
            np.mean(
                [
                    t.pnl
                    for t in wins
                ]
            )
            if wins
            else 0.0
        ),
        "avg_loss": (
            np.mean(
                [
                    t.pnl
                    for t in losses
                ]
            )
            if losses
            else 0.0
        ),
        "expectancy": (
            np.mean(
                [
                    t.pnl
                    for t in resolved
                ]
            )
            if resolved
            else 0.0
        ),
    }


def print_summary(
    all_trades,
    all_candidates
):
    summary = summarize(
        all_trades
    )

    print()
    print("=" * 90)
    print(
        "SETUP 5 V1 — FINAL SUMMARY"
    )
    print("=" * 90)

    print(
        f"Period UTC : "
        f"{OOS_START} -> {OOS_END}"
    )

    print(
        f"Symbols    : "
        f"{len(SYMBOLS)}"
    )

    print(
        f"Candidates : "
        f"{len(all_candidates)}"
    )

    print(
        f"Trades     : "
        f"{summary['trades']}"
    )

    print(
        f"W/L        : "
        f"{summary['wins']}/"
        f"{summary['losses']}"
    )

    print(
        f"Unresolved : "
        f"{summary['unresolved']}"
    )

    print(
        f"Win rate   : "
        f"{summary['win_rate_pct']:.2f}%"
    )

    print(
        f"PF         : "
        f"{summary['profit_factor']:.3f}"
    )

    print(
        f"Net PnL    : "
        f"${summary['net_pnl']:.2f}"
    )

    print(
        f"Final Eq.  : "
        f"${summary['final_equity']:.2f}"
    )

    print(
        f"Max DD     : "
        f"${summary['max_drawdown']:.2f}"
    )

    print(
        f"Max streak : "
        f"{summary['max_loss_streak']}"
    )

    print(
        f"Avg win    : "
        f"${summary['avg_win']:.2f}"
    )

    print(
        f"Avg loss   : "
        f"${summary['avg_loss']:.2f}"
    )

    print(
        f"Expectancy : "
        f"${summary['expectancy']:.2f}"
    )

    print()
    print("TARGET CHECK")
    print(
        "WR >= 50%        : "
        f"{'PASS' if summary['win_rate_pct'] >= 50 else 'FAIL'}"
    )

    print(
        "Max streak <= 4  : "
        f"{'PASS' if summary['max_loss_streak'] <= 4 else 'FAIL'}"
    )

    print(
        "RR                : "
        "fixed 1:2"
    )

    print("=" * 90)


# ============================================================
# SYMBOL PROCESSING
# ============================================================

def process_symbol(
    symbol
):
    print(
        f"[{symbol}] "
        f"downloading 15m + 4h ...",
        flush=True
    )

    # --------------------------------------------------------
    # Execution data
    # --------------------------------------------------------
    df = load_klines(
        symbol,
        EXEC_INTERVAL,
        DATA_START,
        OOS_END
    )

    # --------------------------------------------------------
    # HTF data
    #
    # IMPORTANT:
    # Direct Binance 4h archive.
    # No resampling.
    # --------------------------------------------------------
    htf = load_klines(
        symbol,
        HTF_INTERVAL,
        DATA_START,
        OOS_END
    )

    validate_data(
        df,
        15,
        symbol,
        EXEC_INTERVAL
    )

    validate_data(
        htf,
        240,
        symbol,
        HTF_INTERVAL
    )

    # --------------------------------------------------------
    # Confirmed pivots
    # --------------------------------------------------------
    pivot_high, pivot_low = (
        confirmed_pivots(
            df,
            PIVOT
        )
    )

    htf_pivot_high, htf_pivot_low = (
        confirmed_pivots(
            htf,
            PIVOT
        )
    )

    # --------------------------------------------------------
    # Candidate generation
    # --------------------------------------------------------
    candidates = build_candidates(
        symbol,
        df,
        htf,
        htf_pivot_high,
        htf_pivot_low,
        pivot_high,
        pivot_low
    )

    # --------------------------------------------------------
    # Only OOS entries.
    # --------------------------------------------------------
    filtered_candidates = []

    for candidate in candidates:

        entry_time = df.iloc[
            candidate.entry_idx
        ]["open_time"]

        if (
            OOS_START
            <= entry_time
            <= OOS_END
        ):
            filtered_candidates.append(
                candidate
            )

    candidates = (
        filtered_candidates
    )

    # --------------------------------------------------------
    # Integrity audit
    # --------------------------------------------------------
    audit_candidates(
        candidates,
        df,
        htf
    )

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------
    trades = simulate_candidates(
        symbol,
        df,
        candidates
    )

    audit_trades(
        trades
    )

    summary = summarize(
        trades
    )

    print(
        f"[{symbol}] "
        f"candles={len(df):,} "
        f"HTF={len(htf):,} "
        f"candidates={len(candidates)} "
        f"trades={summary['trades']} "
        f"unresolved={summary['unresolved']} "
        f"WR={summary['win_rate_pct']:.2f}% "
        f"PF={summary['profit_factor']:.3f} "
        f"PnL=${summary['net_pnl']:.2f}",
        flush=True
    )

    return (
        symbol,
        candidates,
        trades
    )


# ============================================================
# MAIN
# ============================================================

def main():

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 90)
    print("SETUP 5 V1")
    print("=" * 90)

    print(
        f"Data start : {DATA_START}"
    )

    print(
        f"OOS start  : {OOS_START}"
    )

    print(
        f"OOS end    : {OOS_END}"
    )

    print(
        f"Execution  : {EXEC_INTERVAL}"
    )

    print(
        f"HTF        : {HTF_INTERVAL}"
    )

    print(
        f"Warmup     : {WARMUP_DAYS} days"
    )

    print(
        f"Test       : {TEST_DAYS} days"
    )

    print(
        f"Capital    : ${INITIAL_CAPITAL:.2f}"
    )

    print(
        f"Margin     : ${MARGIN_PER_TRADE:.2f}"
    )

    print(
        f"Leverage   : {LEVERAGE:.1f}x"
    )

    print(
        f"Notional   : ${NOTIONAL:.2f}"
    )

    print(
        f"RR         : 1:{RR:.1f}"
    )

    print(
        f"Fee        : {FEE_RATE:.5f}"
    )

    print(
        f"Slippage   : {SLIPPAGE:.5f}"
    )

    print("=" * 90)
    print()

    all_candidates = []
    all_trades = []

    failures = []

    workers = max(
        1,
        min(
            DOWNLOAD_WORKERS,
            len(SYMBOLS)
        )
    )

    # --------------------------------------------------------
    # Parallel symbol processing
    # --------------------------------------------------------
    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {
            executor.submit(
                process_symbol,
                symbol
            ): symbol
            for symbol in SYMBOLS
        }

        for future in as_completed(
            futures
        ):

            symbol = futures[
                future
            ]

            try:

                (
                    returned_symbol,
                    candidates,
                    trades
                ) = future.result()

                all_candidates.extend(
                    candidates
                )

                all_trades.extend(
                    trades
                )

            except Exception as exc:

                failures.append(
                    (
                        symbol,
                        repr(exc)
                    )
                )

                print(
                    f"[{symbol}] ERROR: "
                    f"{exc}",
                    file=sys.stderr,
                    flush=True
                )

                traceback.print_exc()

    # --------------------------------------------------------
    # Never publish an incomplete result.
    # --------------------------------------------------------
    if failures:

        print()
        print("=" * 90)
        print(
            "FATAL: ONE OR MORE SYMBOLS FAILED"
        )
        print("=" * 90)

        for symbol, error in failures:

            print(
                f"{symbol}: {error}"
            )

        raise SystemExit(2)

    # --------------------------------------------------------
    # Deterministic ordering
    # --------------------------------------------------------
    all_candidates.sort(
        key=lambda x: (
            x.entry_idx,
            x.symbol,
            x.direction
        )
    )

    all_trades.sort(
        key=lambda x: (
            x.entry_time,
            x.symbol,
            x.direction
        )
    )

    # --------------------------------------------------------
    # Per-symbol audit
    #
    # Cross-symbol overlap is explicitly allowed.
    # --------------------------------------------------------
    trades_by_symbol = {}

    for trade in all_trades:

        trades_by_symbol.setdefault(
            trade.symbol,
            []
        ).append(
            trade
        )

    for symbol, symbol_trades in (
        trades_by_symbol.items()
    ):

        audit_trades(
            symbol_trades
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------
    print_summary(
        all_trades,
        all_candidates
    )

    # ========================================================
    # CSV OUTPUT
    # ========================================================

    output_dir = Path(
        "backtest_results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Trades
    pd.DataFrame(
        [
            asdict(trade)
            for trade in all_trades
        ]
    ).to_csv(
        output_dir
        / "setup5_v1_trades.csv",
        index=False
    )

    # Candidates
    pd.DataFrame(
        [
            asdict(candidate)
            for candidate in all_candidates
        ]
    ).to_csv(
        output_dir
        / "setup5_v1_candidates.csv",
        index=False
    )

    # Summary
    summary = summarize(
        all_trades
    )

    summary.update(
        {
            "oos_start": str(
                OOS_START
            ),
            "oos_end": str(
                OOS_END
            ),
            "test_days": TEST_DAYS,
            "warmup_days": WARMUP_DAYS,
            "rr": RR,
            "initial_capital":
                INITIAL_CAPITAL,
            "margin_per_trade":
                MARGIN_PER_TRADE,
            "leverage":
                LEVERAGE,
            "notional":
                NOTIONAL,
            "fee_rate":
                FEE_RATE,
            "slippage":
                SLIPPAGE,
            "symbols":
                len(SYMBOLS),
        }
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        output_dir
        / "setup5_v1_summary.csv",
        index=False
    )

    print()
    print(
        "Results written to:"
    )
    print(
        "backtest_results/"
    )
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
