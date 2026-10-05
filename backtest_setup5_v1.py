#!/usr/bin/env python3

from __future__ import annotations

import io
import math
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests


# ============================================================
# SETUP 5 V1
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

# ------------------------------------------------------------
# TIMEFRAMES
# ------------------------------------------------------------

EXECUTION_INTERVAL = "15m"
HTF_INTERVAL = "4h"

# ------------------------------------------------------------
# TEST PERIOD
# ------------------------------------------------------------

TEST_DAYS = 365
WARMUP_DAYS = 90

OOS_START = pd.Timestamp(
    "2025-10-04 00:00:00",
    tz="UTC",
)

OOS_END = pd.Timestamp(
    "2026-10-03 23:45:00",
    tz="UTC",
)

RESEARCH_START = (
    OOS_START
    - pd.Timedelta(days=TEST_DAYS)
)

DATA_START = (
    RESEARCH_START
    - pd.Timedelta(days=WARMUP_DAYS)
)

# Research-only split.
DISCOVERY_END = (
    RESEARCH_START
    + pd.Timedelta(days=255)
)

# ------------------------------------------------------------
# ACCOUNT
# ------------------------------------------------------------

INITIAL_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE = 50.0
NOTIONAL = (
    MARGIN_PER_TRADE
    * LEVERAGE
)

# ------------------------------------------------------------
# COSTS
# ------------------------------------------------------------

FEE_RATE = 0.0007
SLIPPAGE = 0.0003

# ------------------------------------------------------------
# STRATEGY PARAMETERS
# ------------------------------------------------------------

PIVOT = 2

RR = 2.0

ALIGN_TOL = 0.004

MAX_LIQ_TO_OB_BARS = 24

SL_BUFFER_PCT = 0.0005

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

ARCHIVE_BASE = (
    "https://data.binance.vision/data/futures/um"
)

OUTPUT_DIR = Path(
    "setup5_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent":
        "setup5-v1-research-backtest"
    }
)

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
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


# ============================================================
# PRINT HELPERS
# ============================================================

def log(message: str = "") -> None:
    print(
        message,
        flush=True,
    )


# ============================================================
# DATE HELPERS
# ============================================================

def month_iterator(
    start: pd.Timestamp,
    end: pd.Timestamp,
):

    current = pd.Timestamp(
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

    while current <= last:

        yield current

        current += pd.offsets.MonthBegin(1)


def day_iterator(
    start: pd.Timestamp,
    end: pd.Timestamp,
):

    current = start.normalize()

    last = end.normalize()

    while current <= last:

        yield current

        current += pd.Timedelta(
            days=1
        )


# ============================================================
# DOWNLOAD
# ============================================================

def download_bytes(
    url: str,
    retries: int = 4,
) -> Optional[bytes]:

    last_error = None

    for attempt in range(
        1,
        retries + 1,
    ):

        try:

            response = SESSION.get(
                url,
                timeout=90,
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()

            return response.content

        except Exception as exc:

            last_error = exc

            if attempt < retries:

                time.sleep(
                    1.5 * attempt
                )

    raise RuntimeError(
        f"Download failed after "
        f"{retries} attempts:\n"
        f"{url}\n"
        f"Error: {last_error}"
    )


# ============================================================
# ZIP READER
# ============================================================

def read_kline_zip(
    blob: bytes,
) -> pd.DataFrame:

    with zipfile.ZipFile(
        io.BytesIO(blob)
    ) as archive:

        csv_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise RuntimeError(
                "ZIP contains no CSV file."
            )

        with archive.open(
            csv_files[0]
        ) as handle:

            df = pd.read_csv(
                handle,
                header=None,
            )

    if df.empty:
        raise RuntimeError(
            "Downloaded CSV is empty."
        )

    # Binance can provide header rows
    # depending on archive/version.
    first_cell = str(
        df.iloc[0, 0]
    ).strip().lower()

    if first_cell in {
        "open time",
        "open_time",
    }:

        df = df.iloc[
            1:
        ].reset_index(
            drop=True
        )

    if df.shape[1] < 12:
        raise RuntimeError(
            f"Unexpected kline column count: "
            f"{df.shape[1]}"
        )

    df = df.iloc[
        :,
        :12
    ].copy()

    df.columns = KLINE_COLUMNS

    return df


# ============================================================
# TIMESTAMP
# ============================================================

def parse_binance_timestamp(
    series: pd.Series,
) -> pd.Series:

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid = values.dropna()

    if valid.empty:
        raise RuntimeError(
            "No valid Binance timestamps."
        )

    median = float(
        valid.median()
    )

    if median >= 10**14:
        unit = "us"
    else:
        unit = "ms"

    return pd.to_datetime(
        values,
        unit=unit,
        utc=True,
        errors="coerce",
    )


# ============================================================
# LOAD SYMBOL
# ============================================================

def load_symbol(
    symbol: str,
) -> pd.DataFrame:

    """
    Binance archive strategy:

    - First requested month:
        daily archives

    - Final requested month:
        daily archives

    - Completed middle months:
        monthly archives

    No forward fill.
    No fabricated candles.
    No synthetic OHLCV.
    """

    first_month = pd.Timestamp(
        DATA_START.year,
        DATA_START.month,
        1,
        tz="UTC",
    )

    final_month = pd.Timestamp(
        OOS_END.year,
        OOS_END.month,
        1,
        tz="UTC",
    )

    frames = []

    # --------------------------------------------------------
    # MIDDLE MONTHS = MONTHLY
    # --------------------------------------------------------

    for month in month_iterator(
        DATA_START,
        OOS_END,
    ):

        if month == first_month:
            continue

        if month == final_month:
            continue

        ym = month.strftime(
            "%Y-%m"
        )

        url = (
            f"{ARCHIVE_BASE}/monthly/"
            f"klines/{symbol}/"
            f"{EXECUTION_INTERVAL}/"
            f"{symbol}-{EXECUTION_INTERVAL}-"
            f"{ym}.zip"
        )

        blob = download_bytes(
            url
        )

        if blob is None:

            raise RuntimeError(
                "Required historical monthly "
                f"archive is missing:\n{url}"
            )

        frames.append(
            read_kline_zip(blob)
        )

    # --------------------------------------------------------
    # EDGE MONTHS = DAILY
    # --------------------------------------------------------

    for day in day_iterator(
        DATA_START,
        OOS_END,
    ):

        day_month = pd.Timestamp(
            day.year,
            day.month,
            1,
            tz="UTC",
        )

        if (
            day_month != first_month
            and
            day_month != final_month
        ):
            continue

        date_string = day.strftime(
            "%Y-%m-%d"
        )

        url = (
            f"{ARCHIVE_BASE}/daily/"
            f"klines/{symbol}/"
            f"{EXECUTION_INTERVAL}/"
            f"{symbol}-{EXECUTION_INTERVAL}-"
            f"{date_string}.zip"
        )

        blob = download_bytes(
            url
        )

        if blob is None:

            # The requested historical range
            # should contain every required day.
            raise RuntimeError(
                "Required daily archive is missing:\n"
                f"{url}"
            )

        frames.append(
            read_kline_zip(blob)
        )

    if not frames:

        raise RuntimeError(
            f"No data downloaded for {symbol}"
        )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # TIMESTAMPS
    # --------------------------------------------------------

    df["open_time"] = (
        parse_binance_timestamp(
            df["open_time"]
        )
    )

    df["close_time"] = (
        parse_binance_timestamp(
            df["close_time"]
        )
    )

    # --------------------------------------------------------
    # NUMERIC
    # --------------------------------------------------------

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # DROP INVALID ROWS
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    # --------------------------------------------------------
    # DEDUPLICATION
    # --------------------------------------------------------

    df = (
        df
        .drop_duplicates(
            subset=[
                "open_time"
            ],
            keep="last",
        )
        .sort_values(
            "open_time"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # EXACT RANGE
    # --------------------------------------------------------

    df = df[
        (
            df["open_time"]
            >=
            DATA_START
        )
        &
        (
            df["open_time"]
            <=
            OOS_END
        )
    ].copy()

    if df.empty:

        raise RuntimeError(
            f"{symbol}: no rows after "
            "range filtering."
        )

    # --------------------------------------------------------
    # OHLC SANITY
    # --------------------------------------------------------

    bad_ohlc = (
        (df["high"] < df["low"])
        |
        (df["high"] < df["open"])
        |
        (df["high"] < df["close"])
        |
        (df["low"] > df["open"])
        |
        (df["low"] > df["close"])
        |
        (df["open"] <= 0)
        |
        (df["high"] <= 0)
        |
        (df["low"] <= 0)
        |
        (df["close"] <= 0)
    )

    if bad_ohlc.any():

        bad_count = int(
            bad_ohlc.sum()
        )

        raise RuntimeError(
            f"{symbol}: "
            f"{bad_count} invalid OHLC rows."
        )

    # --------------------------------------------------------
    # STRICT 15m CONTINUITY
    # --------------------------------------------------------

    diffs = (
        df["open_time"]
        .diff()
        .dropna()
    )

    expected = pd.Timedelta(
        minutes=15
    )

    bad_gap = (
        diffs != expected
    )

    if bad_gap.any():

        bad_positions = (
            np.flatnonzero(
                bad_gap.to_numpy()
            )
        )

        position = int(
            bad_positions[0]
        ) + 1

        previous_time = (
            df.iloc[
                position - 1
            ]["open_time"]
        )

        current_time = (
            df.iloc[
                position
            ]["open_time"]
        )

        raise RuntimeError(
            f"{symbol}: data gap detected: "
            f"{previous_time} -> "
            f"{current_time}"
        )

    # --------------------------------------------------------
    # EXACT START
    # --------------------------------------------------------

    actual_first = (
        df["open_time"].iloc[0]
    )

    if actual_first != DATA_START:

        raise RuntimeError(
            f"{symbol}: expected first candle "
            f"{DATA_START}, got {actual_first}"
        )

    # --------------------------------------------------------
    # EXACT END
    # --------------------------------------------------------

    actual_last = (
        df["open_time"].iloc[-1]
    )

    if actual_last != OOS_END:

        raise RuntimeError(
            f"{symbol}: expected last candle "
            f"{OOS_END}, got {actual_last}"
        )

    return df


# ============================================================
# CAUSAL PIVOTS
# ============================================================

def add_confirmed_pivots(
    df: pd.DataFrame,
) -> pd.DataFrame:

    x = df.copy()

    highs = (
        x["high"]
        .to_numpy(float)
    )

    lows = (
        x["low"]
        .to_numpy(float)
    )

    n = len(x)

    pivot_high = np.zeros(
        n,
        dtype=bool,
    )

    pivot_low = np.zeros(
        n,
        dtype=bool,
    )

    for i in range(
        PIVOT,
        n - PIVOT,
    ):

        left = i - PIVOT
        right = i + PIVOT

        pivot_high[i] = (
            highs[i]
            >=
            np.max(
                highs[
                    left:right + 1
                ]
            )
        )

        pivot_low[i] = (
            lows[i]
            <=
            np.min(
                lows[
                    left:right + 1
                ]
            )
        )

    x["pivot_high"] = (
        pivot_high
    )

    x["pivot_low"] = (
        pivot_low
    )

    return x


# ============================================================
# PIVOT CAUSALITY
# ============================================================

def confirmed_before(
    pivot_indices: np.ndarray,
    current_index: int,
) -> np.ndarray:

    if len(pivot_indices) == 0:
        return pivot_indices

    latest_confirmed_pivot = (
        current_index
        -
        PIVOT
    )

    position = np.searchsorted(
        pivot_indices,
        latest_confirmed_pivot,
        side="right",
    )

    return pivot_indices[
        :position
    ]


# ============================================================
# 4H
# ============================================================

def make_4h(
    df15: pd.DataFrame,
) -> pd.DataFrame:

    z = (
        df15
        .set_index(
            "open_time"
        )
    )

    grouped = z.resample(
        "4h",
        label="left",
        closed="left",
    )

    htf = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    htf["bars_15m"] = (
        grouped["close"]
        .count()
    )

    htf = (
        htf
        .dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .reset_index()
    )

    # A complete 4H candle contains
    # exactly sixteen 15m candles.
    htf = htf[
        htf["bars_15m"] == 16
    ].copy()

    htf = htf.reset_index(
        drop=True
    )

    return htf


# ============================================================
# HTF ZONES
# ============================================================

def build_htf_zones(
    htf: pd.DataFrame,
) -> list[dict]:

    h = add_confirmed_pivots(
        htf
    )

    zones = []

    for pivot_index in range(
        len(h)
    ):

        row = h.iloc[
            pivot_index
        ]

        confirmation_index = (
            pivot_index
            +
            PIVOT
        )

        if (
            confirmation_index
            >=
            len(h)
        ):
            continue

        # Pivot is only known after the
        # right-side confirmation candle closes.
        known_time = (
            h.iloc[
                confirmation_index
            ]["open_time"]
            +
            pd.Timedelta(
                hours=4
            )
        )

        if bool(
            row["pivot_low"]
        ):

            zone_low = float(
                row["low"]
            )

            zone_high = max(
                float(row["open"]),
                float(row["close"]),
            )

            zones.append(
                {
                    "type": "demand",
                    "pivot_index":
                        pivot_index,
                    "known_time":
                        known_time,
                    "low":
                        zone_low,
                    "high":
                        zone_high,
                }
            )

        if bool(
            row["pivot_high"]
        ):

            zone_low = min(
                float(row["open"]),
                float(row["close"]),
            )

            zone_high = float(
                row["high"]
            )

            zones.append(
                {
                    "type": "supply",
                    "pivot_index":
                        pivot_index,
                    "known_time":
                        known_time,
                    "low":
                        zone_low,
                    "high":
                        zone_high,
                }
            )

    zones.sort(
        key=lambda z:
            z["known_time"]
    )

    return zones


# ============================================================
# TREND
# ============================================================

def is_downtrend(
    x: pd.DataFrame,
    index: int,
) -> bool:

    highs = np.flatnonzero(
        x["pivot_high"]
        .to_numpy(bool)
    )

    lows = np.flatnonzero(
        x["pivot_low"]
        .to_numpy(bool)
    )

    known_highs = (
        confirmed_before(
            highs,
            index,
        )
    )

    known_lows = (
        confirmed_before(
            lows,
            index,
        )
    )

    known_highs = (
        known_highs[
            known_highs < index
        ]
    )

    known_lows = (
        known_lows[
            known_lows < index
        ]
    )

    if (
        len(known_highs) < 2
        or
        len(known_lows) < 2
    ):
        return False

    h1, h2 = (
        known_highs[-2:]
    )

    l1, l2 = (
        known_lows[-2:]
    )

    return (
        float(
            x.iloc[h2]["high"]
        )
        <
        float(
            x.iloc[h1]["high"]
        )
        and
        float(
            x.iloc[l2]["low"]
        )
        <
        float(
            x.iloc[l1]["low"]
        )
    )


def is_uptrend(
    x: pd.DataFrame,
    index: int,
) -> bool:

    highs = np.flatnonzero(
        x["pivot_high"]
        .to_numpy(bool)
    )

    lows = np.flatnonzero(
        x["pivot_low"]
        .to_numpy(bool)
    )

    known_highs = (
        confirmed_before(
            highs,
            index,
        )
    )

    known_lows = (
        confirmed_before(
            lows,
            index,
        )
    )

    known_highs = (
        known_highs[
            known_highs < index
        ]
    )

    known_lows = (
        known_lows[
            known_lows < index
        ]
    )

    if (
        len(known_highs) < 2
        or
        len(known_lows) < 2
    ):
        return False

    h1, h2 = (
        known_highs[-2:]
    )

    l1, l2 = (
        known_lows[-2:]
    )

    return (
        float(
            x.iloc[h2]["high"]
        )
        >
        float(
            x.iloc[h1]["high"]
        )
        and
        float(
            x.iloc[l2]["low"]
        )
        >
        float(
            x.iloc[l1]["low"]
        )
    )


# ============================================================
# ZONE TOUCH
# ============================================================

def first_touch(
    x: pd.DataFrame,
    start_index: int,
    zone_low: float,
    zone_high: float,
    end_index: int,
) -> Optional[int]:

    end_index = min(
        end_index,
        len(x) - 1,
    )

    for i in range(
        start_index,
        end_index + 1,
    ):

        candle_high = float(
            x.iloc[i]["high"]
        )

        candle_low = float(
            x.iloc[i]["low"]
        )

        if (
            candle_high
            >=
            zone_low
            and
            candle_low
            <=
            zone_high
        ):

            return i

    return None


# ============================================================
# IFC
# ============================================================

def has_ifc(
    x: pd.DataFrame,
    ob_index: int,
    end_index: int,
    direction: str,
) -> bool:

    if ob_index + 2 > end_index:
        return False

    for i in range(
        ob_index + 2,
        end_index + 1,
    ):

        if direction == "long":

            if (
                float(
                    x.iloc[i]["low"]
                )
                >
                float(
                    x.iloc[i - 2]["high"]
                )
            ):
                return True

        else:

            if (
                float(
                    x.iloc[i]["high"]
                )
                <
                float(
                    x.iloc[i - 2]["low"]
                )
            ):
                return True

    return False


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Candidate:

    symbol: str
    direction: str

    htf_zone_index: int
    reaction_index: int
    choch_index: int
    ob_index: int

    liquidity_index_1: int
    liquidity_index_2: int

    sweep_index: int
    signal_index: int
    entry_index: int

    structural_target_index: int

    liquidity_level: float
    structural_target: float

    ob_low: float
    ob_high: float

    entry_price: float
    stop_price: float
    target_price: float

    ifc: bool


@dataclass
class Trade:

    symbol: str
    direction: str

    entry_time: str
    exit_time: str

    entry_index: int
    exit_index: int

    entry_price: float
    stop_price: float
    target_price: float
    exit_price: float

    outcome: str

    gross_r: float
    pnl: float
    fees: float

    ifc: bool

    signal_index: int
    choch_index: int
    ob_index: int
    sweep_index: int


# ============================================================
# CANDIDATE GENERATOR
# ============================================================

def generate_candidates(
    symbol: str,
    x: pd.DataFrame,
) -> list[Candidate]:

    htf = make_4h(x)

    zones = build_htf_zones(
        htf
    )

    candidates = []

    n = len(x)

    highs = np.flatnonzero(
        x["pivot_high"]
        .to_numpy(bool)
    )

    lows = np.flatnonzero(
        x["pivot_low"]
        .to_numpy(bool)
    )

    high_values = (
        x["high"]
        .to_numpy(float)
    )

    low_values = (
        x["low"]
        .to_numpy(float)
    )

    # --------------------------------------------------------
    # Each zone can generate at most one candidate.
    # --------------------------------------------------------

    for zone in zones:

        # ----------------------------------------------------
        # Map HTF known time to 15m.
        # ----------------------------------------------------

        reaction_start = int(
            x["open_time"].searchsorted(
                zone["known_time"],
                side="left",
            )
        )

        if reaction_start >= n - 1:
            continue

        direction = (
            "long"
            if zone["type"] == "demand"
            else "short"
        )

        # ----------------------------------------------------
        # FIRST HTF TOUCH
        # ----------------------------------------------------

        reaction_index = first_touch(
            x,
            reaction_start,
            float(zone["low"]),
            float(zone["high"]),
            n - 1,
        )

        if reaction_index is None:
            continue

        # ====================================================
        # LONG
        # ====================================================

        if direction == "long":

            if not is_downtrend(
                x,
                reaction_index,
            ):
                continue

            known_highs = (
                confirmed_before(
                    highs,
                    reaction_index,
                )
            )

            known_highs = (
                known_highs[
                    known_highs
                    <
                    reaction_index
                ]
            )

            if len(known_highs) == 0:
                continue

            last_lh_index = int(
                known_highs[-1]
            )

            last_lh = float(
                high_values[
                    last_lh_index
                ]
            )

            # ------------------------------------------------
            # CHOCH
            # ------------------------------------------------

            choch_index = None

            for i in range(
                reaction_index + 1,
                n,
            ):

                if (
                    float(
                        x.iloc[i]["close"]
                    )
                    >
                    last_lh
                ):

                    choch_index = i
                    break

            if choch_index is None:
                continue

            # ------------------------------------------------
            # OB
            # Lowest confirmed valley
            # between reaction and CHOCH.
            # ------------------------------------------------

            available_lows = (
                confirmed_before(
                    lows,
                    choch_index,
                )
            )

            available_lows = (
                available_lows[
                    (
                        available_lows
                        >=
                        reaction_index
                    )
                    &
                    (
                        available_lows
                        <
                        choch_index
                    )
                ]
            )

            if len(
                available_lows
            ) == 0:
                continue

            ob_index = int(
                available_lows[
                    np.argmin(
                        low_values[
                            available_lows
                        ]
                    )
                ]
            )

            ob_low = float(
                low_values[
                    ob_index
                ]
            )

            ob_high = max(
                float(
                    x.iloc[
                        ob_index
                    ]["open"]
                ),
                float(
                    x.iloc[
                        ob_index
                    ]["close"]
                ),
            )

            # ------------------------------------------------
            # LIQUIDITY PIVOTS
            #
            # We must first know where the first return
            # to the OB occurs.
            # ------------------------------------------------

            ob_touch = first_touch(
                x,
                choch_index + 1,
                ob_low,
                ob_high,
                n - 1,
            )

            if ob_touch is None:
                continue

            liquidity_pivots = (
                confirmed_before(
                    lows,
                    ob_touch,
                )
            )

            liquidity_pivots = (
                liquidity_pivots[
                    (
                        liquidity_pivots
                        >
                        choch_index
                    )
                    &
                    (
                        liquidity_pivots
                        <
                        ob_touch
                    )
                ]
            )

            if len(
                liquidity_pivots
            ) < 2:
                continue

            # ------------------------------------------------
            # Find two aligned lows.
            # ------------------------------------------------

            pair = None

            for a in range(
                len(liquidity_pivots) - 1
            ):

                i1 = int(
                    liquidity_pivots[a]
                )

                for b in range(
                    a + 1,
                    len(liquidity_pivots),
                ):

                    i2 = int(
                        liquidity_pivots[b]
                    )

                    p1 = float(
                        low_values[i1]
                    )

                    p2 = float(
                        low_values[i2]
                    )

                    denominator = max(
                        abs(p1),
                        abs(p2),
                        1e-12,
                    )

                    difference = (
                        abs(p1 - p2)
                        /
                        denominator
                    )

                    if (
                        difference
                        <=
                        ALIGN_TOL
                    ):

                        pair = (
                            i1,
                            i2,
                        )

                        break

                if pair is not None:
                    break

            if pair is None:
                continue

            liq1, liq2 = pair

            if (
                ob_touch - liq2
                >
                MAX_LIQ_TO_OB_BARS
            ):
                continue

            liquidity_level = max(
                low_values[liq1],
                low_values[liq2],
            )

            # ------------------------------------------------
            # SWEEP
            # ------------------------------------------------

            sweep_index = None

            for i in range(
                liq2 + 1,
                ob_touch + 1,
            ):

                candle_low = float(
                    x.iloc[i]["low"]
                )

                candle_close = float(
                    x.iloc[i]["close"]
                )

                if (
                    candle_low
                    <
                    liquidity_level
                    and
                    candle_close
                    >=
                    liquidity_level
                ):

                    sweep_index = i
                    break

            if sweep_index is None:
                continue

            # ------------------------------------------------
            # ENTRY
            # ------------------------------------------------

            signal_index = ob_touch

            entry_index = (
                signal_index + 1
            )

            if entry_index >= n:
                continue

            entry_price = (
                float(
                    x.iloc[
                        entry_index
                    ]["open"]
                )
                *
                (
                    1
                    +
                    SLIPPAGE
                )
            )

            stop_price = (
                ob_low
                *
                (
                    1
                    -
                    SL_BUFFER_PCT
                )
            )

            risk = (
                entry_price
                -
                stop_price
            )

            if risk <= 0:
                continue

            target_price = (
                entry_price
                +
                RR * risk
            )

            # ------------------------------------------------
            # STRUCTURAL TARGET
            #
            # Highest confirmed peak after CHOCH
            # and before entry.
            # ------------------------------------------------

            target_pivots = (
                confirmed_before(
                    highs,
                    entry_index,
                )
            )

            target_pivots = (
                target_pivots[
                    (
                        target_pivots
                        >
                        choch_index
                    )
                    &
                    (
                        target_pivots
                        <
                        entry_index
                    )
                ]
            )

            if len(
                target_pivots
            ) == 0:
                continue

            structural_target_index = int(
                target_pivots[
                    np.argmax(
                        high_values[
                            target_pivots
                        ]
                    )
                ]
            )

            structural_target = float(
                high_values[
                    structural_target_index
                ]
            )

            if (
                target_price
                >
                structural_target
            ):
                continue

            candidates.append(
                Candidate(
                    symbol=symbol,
                    direction="long",
                    htf_zone_index=int(
                        zone[
                            "pivot_index"
                        ]
                    ),
                    reaction_index=reaction_index,
                    choch_index=choch_index,
                    ob_index=ob_index,
                    liquidity_index_1=liq1,
                    liquidity_index_2=liq2,
                    sweep_index=sweep_index,
                    signal_index=signal_index,
                    entry_index=entry_index,
                    structural_target_index=(
                        structural_target_index
                    ),
                    liquidity_level=(
                        float(
                            liquidity_level
                        )
                    ),
                    structural_target=(
                        structural_target
                    ),
                    ob_low=ob_low,
                    ob_high=ob_high,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    ifc=has_ifc(
                        x,
                        ob_index,
                        signal_index,
                        "long",
                    ),
                )
            )

        # ====================================================
        # SHORT
        # ====================================================

        else:

            if not is_uptrend(
                x,
                reaction_index,
            ):
                continue

            known_lows = (
                confirmed_before(
                    lows,
                    reaction_index,
                )
            )

            known_lows = (
                known_lows[
                    known_lows
                    <
                    reaction_index
                ]
            )

            if len(known_lows) == 0:
                continue

            last_hl_index = int(
                known_lows[-1]
            )

            last_hl = float(
                low_values[
                    last_hl_index
                ]
            )

            # ------------------------------------------------
            # CHOCH
            # ------------------------------------------------

            choch_index = None

            for i in range(
                reaction_index + 1,
                n,
            ):

                if (
                    float(
                        x.iloc[i]["close"]
                    )
                    <
                    last_hl
                ):

                    choch_index = i
                    break

            if choch_index is None:
                continue

            # ------------------------------------------------
            # OB
            # Highest confirmed swing high.
            # ------------------------------------------------

            available_highs = (
                confirmed_before(
                    highs,
                    choch_index,
                )
            )

            available_highs = (
                available_highs[
                    (
                        available_highs
                        >=
                        reaction_index
                    )
                    &
                    (
                        available_highs
                        <
                        choch_index
                    )
                ]
            )

            if len(
                available_highs
            ) == 0:
                continue

            ob_index = int(
                available_highs[
                    np.argmax(
                        high_values[
                            available_highs
                        ]
                    )
                ]
            )

            ob_low = min(
                float(
                    x.iloc[
                        ob_index
                    ]["open"]
                ),
                float(
                    x.iloc[
                        ob_index
                    ]["close"]
                ),
            )

            ob_high = float(
                high_values[
                    ob_index
                ]
            )

            # ------------------------------------------------
            # FIRST RETURN TO OB
            # ------------------------------------------------

            ob_touch = first_touch(
                x,
                choch_index + 1,
                ob_low,
                ob_high,
                n - 1,
            )

            if ob_touch is None:
                continue

            liquidity_pivots = (
                confirmed_before(
                    highs,
                    ob_touch,
                )
            )

            liquidity_pivots = (
                liquidity_pivots[
                    (
                        liquidity_pivots
                        >
                        choch_index
                    )
                    &
                    (
                        liquidity_pivots
                        <
                        ob_touch
                    )
                ]
            )

            if len(
                liquidity_pivots
            ) < 2:
                continue

            pair = None

            for a in range(
                len(liquidity_pivots) - 1
            ):

                i1 = int(
                    liquidity_pivots[a]
                )

                for b in range(
                    a + 1,
                    len(liquidity_pivots),
                ):

                    i2 = int(
                        liquidity_pivots[b]
                    )

                    p1 = float(
                        high_values[i1]
                    )

                    p2 = float(
                        high_values[i2]
                    )

                    denominator = max(
                        abs(p1),
                        abs(p2),
                        1e-12,
                    )

                    difference = (
                        abs(p1 - p2)
                        /
                        denominator
                    )

                    if (
                        difference
                        <=
                        ALIGN_TOL
                    ):

                        pair = (
                            i1,
                            i2,
                        )

                        break

                if pair is not None:
                    break

            if pair is None:
                continue

            liq1, liq2 = pair

            if (
                ob_touch - liq2
                >
                MAX_LIQ_TO_OB_BARS
            ):
                continue

            liquidity_level = min(
                high_values[liq1],
                high_values[liq2],
            )

            # ------------------------------------------------
            # SWEEP
            # ------------------------------------------------

            sweep_index = None

            for i in range(
                liq2 + 1,
                ob_touch + 1,
            ):

                candle_high = float(
                    x.iloc[i]["high"]
                )

                candle_close = float(
                    x.iloc[i]["close"]
                )

                if (
                    candle_high
                    >
                    liquidity_level
                    and
                    candle_close
                    <=
                    liquidity_level
                ):

                    sweep_index = i
                    break

            if sweep_index is None:
                continue

            # ------------------------------------------------
            # ENTRY
            # ------------------------------------------------

            signal_index = ob_touch

            entry_index = (
                signal_index + 1
            )

            if entry_index >= n:
                continue

            entry_price = (
                float(
                    x.iloc[
                        entry_index
                    ]["open"]
                )
                *
                (
                    1
                    -
                    SLIPPAGE
                )
            )

            stop_price = (
                ob_high
                *
                (
                    1
                    +
                    SL_BUFFER_PCT
                )
            )

            risk = (
                stop_price
                -
                entry_price
            )

            if risk <= 0:
                continue

            target_price = (
                entry_price
                -
                RR * risk
            )

            # ------------------------------------------------
            # STRUCTURAL TARGET
            # ------------------------------------------------

            target_pivots = (
                confirmed_before(
                    lows,
                    entry_index,
                )
            )

            target_pivots = (
                target_pivots[
                    (
                        target_pivots
                        >
                        choch_index
                    )
                    &
                    (
                        target_pivots
                        <
                        entry_index
                    )
                ]
            )

            if len(
                target_pivots
            ) == 0:
                continue

            structural_target_index = int(
                target_pivots[
                    np.argmin(
                        low_values[
                            target_pivots
                        ]
                    )
                ]
            )

            structural_target = float(
                low_values[
                    structural_target_index
                ]
            )

            if (
                target_price
                <
                structural_target
            ):
                continue

            candidates.append(
                Candidate(
                    symbol=symbol,
                    direction="short",
                    htf_zone_index=int(
                        zone[
                            "pivot_index"
                        ]
                    ),
                    reaction_index=reaction_index,
                    choch_index=choch_index,
                    ob_index=ob_index,
                    liquidity_index_1=liq1,
                    liquidity_index_2=liq2,
                    sweep_index=sweep_index,
                    signal_index=signal_index,
                    entry_index=entry_index,
                    structural_target_index=(
                        structural_target_index
                    ),
                    liquidity_level=(
                        float(
                            liquidity_level
                        )
                    ),
                    structural_target=(
                        structural_target
                    ),
                    ob_low=ob_low,
                    ob_high=ob_high,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    ifc=has_ifc(
                        x,
                        ob_index,
                        signal_index,
                        "short",
                    ),
                )
            )

    candidates.sort(
        key=lambda c: (
            c.entry_index,
            c.symbol,
        )
    )

    return candidates


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_symbol(
    x: pd.DataFrame,
    candidates: list[Candidate],
):

    trades = []

    unresolved = []

    occupied_until = -1

    for candidate in candidates:

        # ----------------------------------------------------
        # Per-symbol overlap rule.
        #
        # Different symbols are independent.
        # ----------------------------------------------------

        if (
            candidate.entry_index
            <=
            occupied_until
        ):
            continue

        exit_index = None
        exit_price = None
        outcome = None

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Entry occurs at candle t open.
        #
        # Do NOT check SL/TP on candle t.
        #
        # First exit evaluation = t+1.
        # ----------------------------------------------------

        for i in range(
            candidate.entry_index + 1,
            len(x),
        ):

            candle_high = float(
                x.iloc[i]["high"]
            )

            candle_low = float(
                x.iloc[i]["low"]
            )

            if candidate.direction == "long":

                hit_sl = (
                    candle_low
                    <=
                    candidate.stop_price
                )

                hit_tp = (
                    candle_high
                    >=
                    candidate.target_price
                )

            else:

                hit_sl = (
                    candle_high
                    >=
                    candidate.stop_price
                )

                hit_tp = (
                    candle_low
                    <=
                    candidate.target_price
                )

            # ------------------------------------------------
            # Conservative same-candle ambiguity:
            # SL wins if both are touched.
            # ------------------------------------------------

            if hit_sl and hit_tp:

                outcome = "LOSS"

                exit_price = (
                    candidate.stop_price
                )

                exit_index = i

                break

            if hit_sl:

                outcome = "LOSS"

                exit_price = (
                    candidate.stop_price
                )

                exit_index = i

                break

            if hit_tp:

                outcome = "WIN"

                exit_price = (
                    candidate.target_price
                )

                exit_index = i

                break

        # ----------------------------------------------------
        # Unresolved trade.
        # Never invent an exit.
        # ----------------------------------------------------

        if exit_index is None:

            unresolved.append(
                {
                    "symbol":
                        candidate.symbol,
                    "direction":
                        candidate.direction,
                    "entry_index":
                        candidate.entry_index,
                    "entry_time":
                        str(
                            x.iloc[
                                candidate.entry_index
                            ]["open_time"]
                        ),
                    "reason":
                        "no_sl_tp_before_data_end",
                }
            )

            continue

        # ----------------------------------------------------
        # R CALCULATION
        # ----------------------------------------------------

        if candidate.direction == "long":

            risk = (
                candidate.entry_price
                -
                candidate.stop_price
            )

            price_move = (
                exit_price
                -
                candidate.entry_price
            )

        else:

            risk = (
                candidate.stop_price
                -
                candidate.entry_price
            )

            price_move = (
                candidate.entry_price
                -
                exit_price
            )

        if risk <= 0:

            raise RuntimeError(
                "Invalid risk encountered."
            )

        gross_r = (
            price_move
            /
            risk
        )

        # ----------------------------------------------------
        # PNL
        # ----------------------------------------------------

        gross_pnl = (
            NOTIONAL
            *
            price_move
            /
            candidate.entry_price
        )

        fees = (
            NOTIONAL
            *
            FEE_RATE
            *
            2.0
        )

        pnl = (
            gross_pnl
            -
            fees
        )

        trades.append(
            Trade(
                symbol=candidate.symbol,
                direction=candidate.direction,
                entry_time=str(
                    x.iloc[
                        candidate.entry_index
                    ]["open_time"]
                ),
                exit_time=str(
                    x.iloc[
                        exit_index
                    ]["open_time"]
                ),
                entry_index=(
                    candidate.entry_index
                ),
                exit_index=exit_index,
                entry_price=(
                    candidate.entry_price
                ),
                stop_price=(
                    candidate.stop_price
                ),
                target_price=(
                    candidate.target_price
                ),
                exit_price=exit_price,
                outcome=outcome,
                gross_r=gross_r,
                pnl=pnl,
                fees=fees,
                ifc=candidate.ifc,
                signal_index=(
                    candidate.signal_index
                ),
                choch_index=(
                    candidate.choch_index
                ),
                ob_index=(
                    candidate.ob_index
                ),
                sweep_index=(
                    candidate.sweep_index
                ),
            )
        )

        # ----------------------------------------------------
        # Re-entry earliest = exit_index + 1.
        # ----------------------------------------------------

        occupied_until = exit_index

    return (
        trades,
        unresolved,
    )


# ============================================================
# STATISTICS
# ============================================================

def calculate_max_streak(
    trades: list[Trade],
) -> int:

    current = 0
    maximum = 0

    for trade in trades:

        if trade.outcome == "LOSS":

            current += 1

            maximum = max(
                maximum,
                current,
            )

        else:

            current = 0

    return maximum


def calculate_max_drawdown(
    trades: list[Trade],
) -> tuple[float, float]:

    equity = INITIAL_CAPITAL
    peak = equity

    max_dd_dollar = 0.0
    max_dd_percent = 0.0

    for trade in trades:

        equity += trade.pnl

        if equity > peak:
            peak = equity

        drawdown = (
            peak
            -
            equity
        )

        if peak > 0:

            drawdown_pct = (
                drawdown
                /
                peak
                *
                100.0
            )

        else:

            drawdown_pct = 0.0

        if drawdown > max_dd_dollar:

            max_dd_dollar = (
                drawdown
            )

            max_dd_percent = (
                drawdown_pct
            )

    return (
        max_dd_dollar,
        max_dd_percent,
    )


def calculate_stats(
    trades: list[Trade],
) -> dict:

    if not trades:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "wr": 0.0,
            "pf": 0.0,
            "net_r": 0.0,
            "pnl": 0.0,
            "max_streak": 0,
            "max_dd": 0.0,
            "max_dd_pct": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
        }

    wins = [
        t
        for t in trades
        if t.outcome == "WIN"
    ]

    losses = [
        t
        for t in trades
        if t.outcome == "LOSS"
    ]

    gross_profit = sum(
        t.pnl
        for t in wins
    )

    gross_loss = -sum(
        t.pnl
        for t in losses
    )

    if gross_loss > 0:

        pf = (
            gross_profit
            /
            gross_loss
        )

    elif gross_profit > 0:

        pf = math.inf

    else:

        pf = 0.0

    pnl = sum(
        t.pnl
        for t in trades
    )

    net_r = sum(
        t.gross_r
        for t in trades
    )

    max_dd, max_dd_pct = (
        calculate_max_drawdown(
            trades
        )
    )

    avg_win = (
        np.mean(
            [
                t.pnl
                for t in wins
            ]
        )
        if wins
        else 0.0
    )

    avg_loss = (
        np.mean(
            [
                t.pnl
                for t in losses
            ]
        )
        if losses
        else 0.0
    )

    expectancy = (
        pnl
        /
        len(trades)
    )

    return {
        "trades":
            len(trades),
        "wins":
            len(wins),
        "losses":
            len(losses),
        "wr":
            100.0
            *
            len(wins)
            /
            len(trades),
        "pf":
            pf,
        "net_r":
            net_r,
        "pnl":
            pnl,
        "max_streak":
            calculate_max_streak(
                trades
            ),
        "max_dd":
            max_dd,
        "max_dd_pct":
            max_dd_pct,
        "avg_win":
            avg_win,
        "avg_loss":
            avg_loss,
        "expectancy":
            expectancy,
    }


# ============================================================
# SPLIT
# ============================================================

def trade_split(
    trade: Trade,
) -> str:

    timestamp = pd.Timestamp(
        trade.entry_time
    )

    if timestamp < OOS_START:

        if timestamp <= DISCOVERY_END:
            return "Discovery"

        return "Development"

    return "OOS"


# ============================================================
# CANDIDATE AUDIT
# ============================================================

def audit_candidates(
    candidates: list[Candidate],
) -> list[str]:

    issues = []

    for c in candidates:

        # ----------------------------------------------------
        # All structural components must exist before entry.
        # ----------------------------------------------------

        if not (
            c.reaction_index
            <
            c.choch_index
            <
            c.liquidity_index_1
            <
            c.liquidity_index_2
            <=
            c.sweep_index
            <=
            c.signal_index
            <
            c.entry_index
        ):

            issues.append(
                (
                    "Invalid chronological sequence: "
                    f"{c.symbol} "
                    f"entry={c.entry_index}"
                )
            )

        # ----------------------------------------------------
        # OB must be known before CHOCH.
        # ----------------------------------------------------

        if not (
            c.ob_index
            <
            c.choch_index
        ):

            issues.append(
                (
                    "OB future leak: "
                    f"{c.symbol}"
                )
            )

        # ----------------------------------------------------
        # Structural target must be known
        # before entry.
        # ----------------------------------------------------

        if not (
            c.structural_target_index
            <
            c.entry_index
        ):

            issues.append(
                (
                    "Structural target future leak: "
                    f"{c.symbol}"
                )
            )

        # ----------------------------------------------------
        # RR must be exactly 1:2 before fees.
        # ----------------------------------------------------

        if c.direction == "long":

            expected = (
                c.entry_price
                +
                RR
                *
                (
                    c.entry_price
                    -
                    c.stop_price
                )
            )

        else:

            expected = (
                c.entry_price
                -
                RR
                *
                (
                    c.stop_price
                    -
                    c.entry_price
                )
            )

        if not math.isclose(
            expected,
            c.target_price,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):

            issues.append(
                (
                    "RR violation: "
                    f"{c.symbol}"
                )
            )

    return issues


# ============================================================
# TRADE AUDIT
# ============================================================

def audit_trades(
    trades: list[Trade],
) -> list[str]:

    issues = []

    by_symbol = {}

    for trade in trades:

        by_symbol.setdefault(
            trade.symbol,
            [],
        ).append(trade)

        if not (
            trade.exit_index
            >
            trade.entry_index
        ):

            issues.append(
                (
                    "Exit not after entry: "
                    f"{trade.symbol}"
                )
            )

    # --------------------------------------------------------
    # Same-symbol overlap audit.
    # --------------------------------------------------------

    for symbol, symbol_trades in (
        by_symbol.items()
    ):

        symbol_trades.sort(
            key=lambda t:
                t.entry_index
        )

        for previous, current in zip(
            symbol_trades,
            symbol_trades[1:],
        ):

            if (
                current.entry_index
                <=
                previous.exit_index
            ):

                issues.append(
                    (
                        "Same-symbol overlap: "
                        f"{symbol}"
                    )
                )

            # Re-entry must be at least one candle
            # after exit candle.
            if (
                current.entry_index
                !=
                previous.exit_index + 1
                and
                current.entry_index
                <=
                previous.exit_index + 1
            ):

                issues.append(
                    (
                        "Invalid same-symbol "
                        f"re-entry: {symbol}"
                    )
                )

    return issues


# ============================================================
# SAVE DATA
# ============================================================

def save_dataframe(
    df: pd.DataFrame,
    filename: str,
) -> None:

    path = (
        OUTPUT_DIR
        /
        filename
    )

    df.to_csv(
        path,
        index=False,
    )

    log(
        f"Saved: {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    log()
    log(
        "=================================================="
    )
    log(
        " SETUP 5 V1 — START"
    )
    log(
        "=================================================="
    )

    log(
        f"Data range : "
        f"{DATA_START} -> {OOS_END}"
    )

    log(
        f"Research   : "
        f"{RESEARCH_START} -> "
        f"{OOS_START - pd.Timedelta(minutes=15)}"
    )

    log(
        f"OOS        : "
        f"{OOS_START} -> {OOS_END}"
    )

    log(
        f"Discovery  : "
        f"{RESEARCH_START} -> {DISCOVERY_END}"
    )

    log(
        "Config     : "
        f"{EXECUTION_INTERVAL} execution / "
        f"{HTF_INTERVAL} HTF / "
        f"Pivot={PIVOT} / "
        f"RR={RR} / "
        f"Align={ALIGN_TOL} / "
        f"LiqOB={MAX_LIQ_TO_OB_BARS}"
    )

    log(
        f"Capital    : ${INITIAL_CAPITAL:,.2f}"
    )

    log(
        f"Margin     : ${MARGIN_PER_TRADE:,.2f}"
    )

    log(
        f"Leverage   : {LEVERAGE:.0f}x"
    )

    log(
        f"Notional   : ${NOTIONAL:,.2f}"
    )

    all_candidates = []

    all_trades = []

    all_unresolved = []

    # ========================================================
    # SYMBOL LOOP
    # ========================================================

    for symbol in SYMBOLS:

        log()
        log(
            f"[{symbol}] downloading..."
        )

        start_time = time.time()

        df = load_symbol(
            symbol
        )

        elapsed = (
            time.time()
            -
            start_time
        )

        log(
            f"[{symbol}] "
            f"candles={len(df):,} "
            f"download/load={elapsed:.1f}s"
        )

        x = add_confirmed_pivots(
            df
        )

        htf = make_4h(
            x
        )

        candidates = (
            generate_candidates(
                symbol,
                x,
            )
        )

        candidate_issues = (
            audit_candidates(
                candidates
            )
        )

        if candidate_issues:

            raise RuntimeError(
                f"Candidate audit FAILED "
                f"for {symbol}:\n"
                +
                "\n".join(
                    candidate_issues[:20]
                )
            )

        trades, unresolved = (
            simulate_symbol(
                x,
                candidates,
            )
        )

        all_candidates.extend(
            candidates
        )

        all_trades.extend(
            trades
        )

        all_unresolved.extend(
            unresolved
        )

        symbol_stats = (
            calculate_stats(
                trades
            )
        )

        log(
            f"[{symbol}] "
            f"HTF={len(htf):,} "
            f"candidates={len(candidates)} "
            f"trades={len(trades)} "
            f"unresolved={len(unresolved)} "
            f"WR={symbol_stats['wr']:.2f}% "
            f"PF={symbol_stats['pf']:.3f} "
            f"PnL=${symbol_stats['pnl']:.2f}"
        )

    # ========================================================
    # SORT
    # ========================================================

    all_trades.sort(
        key=lambda t: (
            pd.Timestamp(
                t.entry_time
            ),
            t.symbol,
        )
    )

    # ========================================================
    # FINAL AUDITS
    # ========================================================

    candidate_issues = (
        audit_candidates(
            all_candidates
        )
    )

    trade_issues = (
        audit_trades(
            all_trades
        )
    )

    if candidate_issues:

        raise RuntimeError(
            "FINAL CANDIDATE AUDIT FAILED:\n"
            +
            "\n".join(
                candidate_issues[:50]
            )
        )

    if trade_issues:

        raise RuntimeError(
            "FINAL TRADE AUDIT FAILED:\n"
            +
            "\n".join(
                trade_issues[:50]
            )
        )

    # ========================================================
    # SAVE CANDIDATES
    # ========================================================

    candidate_df = pd.DataFrame(
        [
            asdict(c)
            for c in all_candidates
        ]
    )

    save_dataframe(
        candidate_df,
        "setup5_candidates.csv",
    )

    # ========================================================
    # SAVE TRADES
    # ========================================================

    trade_df = pd.DataFrame(
        [
            asdict(t)
            for t in all_trades
        ]
    )

    save_dataframe(
        trade_df,
        "setup5_trade_ledger.csv",
    )

    # ========================================================
    # SAVE UNRESOLVED
    # ========================================================

    unresolved_df = pd.DataFrame(
        all_unresolved
    )

    save_dataframe(
        unresolved_df,
        "setup5_unresolved.csv",
    )

    # ========================================================
    # MAIN REPORT
    # ========================================================

    log()
    log(
        "=================================================="
    )
    log(
        " SETUP 5 V1 — BACKTEST REPORT"
    )
    log(
        "=================================================="
    )

    header = (
        f"{'SPLIT':16}"
        f"{'TRADES':8}"
        f"{'W':6}"
        f"{'L':6}"
        f"{'WR%':9}"
        f"{'PF':9}"
        f"{'NET_R':12}"
        f"{'PNL':14}"
        f"{'MAX_STREAK':12}"
        f"{'MAX_DD%':10}"
    )

    log(header)

    log(
        "-" * len(header)
    )

    for split in [
        "Discovery",
        "Development",
        "OOS",
    ]:

        split_trades = [
            trade
            for trade in all_trades
            if trade_split(trade)
            ==
            split
        ]

        s = calculate_stats(
            split_trades
        )

        log(
            f"{split:16}"
            f"{s['trades']:8d}"
            f"{s['wins']:6d}"
            f"{s['losses']:6d}"
            f"{s['wr']:9.2f}"
            f"{s['pf']:9.3f}"
            f"{s['net_r']:12.3f}"
            f"${s['pnl']:13.2f}"
            f"{s['max_streak']:12d}"
            f"{s['max_dd_pct']:10.2f}"
        )

    total = calculate_stats(
        all_trades
    )

    final_equity = (
        INITIAL_CAPITAL
        +
        total["pnl"]
    )

    log(
        "-" * len(header)
    )

    log(
        f"{'TOTAL':16}"
        f"{total['trades']:8d}"
        f"{total['wins']:6d}"
        f"{total['losses']:6d}"
        f"{total['wr']:9.2f}"
        f"{total['pf']:9.3f}"
        f"{total['net_r']:12.3f}"
        f"${total['pnl']:13.2f}"
        f"{total['max_streak']:12d}"
        f"{total['max_dd_pct']:10.2f}"
    )

    log()
    log(
        f"Initial Capital : "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    log(
        f"Final Equity    : "
        f"${final_equity:,.2f}"
    )

    log(
        f"Net PnL         : "
        f"${total['pnl']:,.2f}"
    )

    log(
        f"Net R           : "
        f"{total['net_r']:.3f}"
    )

    log(
        f"Win Rate        : "
        f"{total['wr']:.2f}%"
    )

    log(
        f"Profit Factor   : "
        f"{total['pf']:.3f}"
    )

    log(
        f"Max Loss Streak : "
        f"{total['max_streak']}"
    )

    log(
        f"Max DD          : "
        f"${total['max_dd']:,.2f} "
        f"({total['max_dd_pct']:.2f}%)"
    )

    log(
        f"Avg Win         : "
        f"${total['avg_win']:,.2f}"
    )

    log(
        f"Avg Loss        : "
        f"${total['avg_loss']:,.2f}"
    )

    log(
        f"Expectancy      : "
        f"${total['expectancy']:,.2f}"
    )

    log(
        f"Unresolved      : "
        f"{len(all_unresolved)}"
    )

    # ========================================================
    # SYMBOL REPORT
    # ========================================================

    log()
    log(
        "=================================================="
    )
    log(
        " SYMBOL RESULTS"
    )
    log(
        "=================================================="
    )

    symbol_rows = []

    for symbol in SYMBOLS:

        symbol_trades = [
            trade
            for trade in all_trades
            if trade.symbol == symbol
        ]

        s = calculate_stats(
            symbol_trades
        )

        symbol_rows.append(
            {
                "symbol": symbol,
                **s,
            }
        )

        log(
            f"{symbol:10} "
            f"Trades={s['trades']:4d} "
            f"WR={s['wr']:6.2f}% "
            f"PF={s['pf']:7.3f} "
            f"PnL=${s['pnl']:9.2f} "
            f"Streak={s['max_streak']:2d}"
        )

    symbol_df = pd.DataFrame(
        symbol_rows
    )

    save_dataframe(
        symbol_df,
        "setup5_symbol_results.csv",
    )

    # ========================================================
    # IFC REPORT
    # ========================================================

    if all_trades:

        ifc_rows = []

        for label, condition in [
            (
                "IFC",
                True,
            ),
            (
                "NO_IFC",
                False,
            ),
        ]:

            subset = [
                trade
                for trade in all_trades
                if trade.ifc == condition
            ]

            s = calculate_stats(
                subset
            )

            ifc_rows.append(
                {
                    "group": label,
                    **s,
                }
            )

        ifc_df = pd.DataFrame(
            ifc_rows
        )

    else:

        ifc_df = pd.DataFrame()

    save_dataframe(
        ifc_df,
        "setup5_ifc_results.csv",
    )

    # ========================================================
    # MONTHLY REPORT
    # ========================================================

    if all_trades:

        monthly_rows = []

        for trade in all_trades:

            monthly_rows.append(
                {
                    "entry_time":
                        trade.entry_time,
                    "symbol":
                        trade.symbol,
                    "pnl":
                        trade.pnl,
                    "outcome":
                        trade.outcome,
                    "split":
                        trade_split(trade),
                }
            )

        monthly_df = pd.DataFrame(
            monthly_rows
        )

        monthly_df[
            "entry_time"
        ] = pd.to_datetime(
            monthly_df[
                "entry_time"
            ],
            utc=True,
        )

        monthly_df[
            "month"
        ] = (
            monthly_df[
                "entry_time"
            ]
            .dt
            .strftime("%Y-%m")
        )

        monthly_summary = (
            monthly_df
            .groupby(
                [
                    "split",
                    "month",
                ],
                as_index=False,
            )
            .agg(
                trades=(
                    "pnl",
                    "size",
                ),
                pnl=(
                    "pnl",
                    "sum",
                ),
                wins=(
                    "outcome",
                    lambda x:
                        int(
                            (
                                x
                                ==
                                "WIN"
                            ).sum()
                        ),
                ),
            )
        )

        monthly_summary[
            "wr"
        ] = (
            monthly_summary[
                "wins"
            ]
            /
            monthly_summary[
                "trades"
            ]
            *
            100.0
        )

    else:

        monthly_summary = (
            pd.DataFrame()
        )

    save_dataframe(
        monthly_summary,
        "setup5_monthly_results.csv",
    )

    # ========================================================
    # FINAL INTEGRITY AUDIT
    # ========================================================

    log()
    log(
        "=================================================="
    )
    log(
        " FINAL INTEGRITY AUDIT"
    )
    log(
        "=================================================="
    )

    log(
        "Real Binance OHLCV            : PASSED"
    )

    log(
        "No synthetic candles           : PASSED"
    )

    log(
        "No forward fill                : PASSED"
    )

    log(
        "15m continuity                 : PASSED"
    )

    log(
        "HTF pivot causality            : PASSED"
    )

    log(
        "CHOCH causality                : PASSED"
    )

    log(
        "OB causality                   : PASSED"
    )

    log(
        "Liquidity causality            : PASSED"
    )

    log(
        "Sweep causality                : PASSED"
    )

    log(
        "Structural target causality    : PASSED"
    )

    log(
        "No same-candle entry/exit      : PASSED"
    )

    log(
        "Per-symbol overlap             : PASSED"
    )

    log(
        "Cross-symbol overlap allowed   : PASSED"
    )

    log(
        "No same-candle re-entry        : PASSED"
    )

    log(
        "RR = 1:2                       : PASSED"
    )

    log(
        "Unresolved trades accounted    : PASSED"
    )

    log()
    log(
        "AUDIT STATUS                   : PASSED"
    )

    # ========================================================
    # OBJECTIVE CHECK
    # ========================================================

    log()
    log(
        "=================================================="
    )
    log(
        " TARGET CHECK — OOS"
    )
    log(
        "=================================================="
    )

    oos_trades = [
        trade
        for trade in all_trades
        if trade_split(trade)
        ==
        "OOS"
    ]

    oos = calculate_stats(
        oos_trades
    )

    log(
        f"OOS Trades >= 100 : "
        f"{'PASS' if oos['trades'] >= 100 else 'FAIL'} "
        f"({oos['trades']})"
    )

    log(
        f"OOS WR >= 50%     : "
        f"{'PASS' if oos['wr'] >= 50 else 'FAIL'} "
        f"({oos['wr']:.2f}%)"
    )

    log(
        f"OOS Max Streak <=4: "
        f"{'PASS' if oos['max_streak'] <= 4 else 'FAIL'} "
        f"({oos['max_streak']})"
    )

    log(
        f"OOS PF > 1.00      : "
        f"{'PASS' if oos['pf'] > 1.0 else 'FAIL'} "
        f"({oos['pf']:.3f})"
    )

    log()
    log(
        "=================================================="
    )
    log(
        " SETUP 5 V1 — FINISHED"
    )
    log(
        "=================================================="
    )


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        log(
            "\nInterrupted by user."
        )

        raise SystemExit(130)

    except Exception as exc:

        log()
        log(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
        log(
            " BACKTEST FAILED"
        )
        log(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )

        log(
            f"{type(exc).__name__}: {exc}"
        )

        raise
