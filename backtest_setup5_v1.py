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
# Mechanical research translation of the uploaded PDF
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

INTERVAL = "15m"
HTF_INTERVAL = "4h"

PIVOT = 2

INITIAL_CAPITAL = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE

RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003

# Mechanical research translations.
# These numeric values are NOT claimed as literal PDF values.
ALIGN_TOL = 0.004
MAX_LIQ_TO_OB_BARS = 24
SL_BUFFER_PCT = 0.0005

WARMUP_DAYS = 90
RESEARCH_DAYS = 365

# Reserved OOS
OOS_START = pd.Timestamp(
    "2025-10-04 00:00:00",
    tz="UTC",
)

OOS_END = pd.Timestamp(
    "2026-10-03 23:45:00",
    tz="UTC",
)

RESEARCH_START = (
    OOS_START -
    pd.Timedelta(days=RESEARCH_DAYS)
)

DATA_START = (
    RESEARCH_START -
    pd.Timedelta(days=WARMUP_DAYS)
)

# Research split only.
# OOS remains untouched.
DISCOVERY_END = (
    RESEARCH_START +
    pd.Timedelta(days=255)
)

ARCHIVE_BASE = (
    "https://data.binance.vision/data/futures/um"
)

OUTPUT_DIR = Path("setup5_outputs")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent":
        "setup5-v1-backtest"
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
# DATA DOWNLOAD
# ============================================================

def month_iter(
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


def daily_iter(
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    current = start.normalize()
    last = end.normalize()

    while current <= last:
        yield current
        current += pd.Timedelta(days=1)


def download(
    url: str,
    retries: int = 3,
) -> Optional[bytes]:

    last_error = None

    for attempt in range(retries):

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

            if attempt + 1 < retries:
                time.sleep(
                    1.5 * (attempt + 1)
                )

    raise RuntimeError(
        f"Download failed: {url} :: {last_error}"
    )


def read_zip(
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
                "Archive contains no CSV"
            )

        with archive.open(
            csv_files[0]
        ) as handle:

            df = pd.read_csv(
                handle,
                header=None,
            )

    if len(df) > 0:

        first_value = (
            str(df.iloc[0, 0])
            .strip()
            .lower()
        )

        if first_value in {
            "open time",
            "open_time",
        }:
            df = df.iloc[1:].reset_index(
                drop=True
            )

    df = df.iloc[:, :12].copy()
    df.columns = KLINE_COLUMNS

    return df


def convert_timestamp(
    series: pd.Series,
) -> pd.Series:

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid = values.dropna()

    if valid.empty:
        raise RuntimeError(
            "No valid timestamp values"
        )

    unit = (
        "us"
        if valid.median() > 10**14
        else "ms"
    )

    return pd.to_datetime(
        values,
        unit=unit,
        utc=True,
    )


# ============================================================
# CORRECTED BINANCE LOADER
# ============================================================

def load_symbol(
    symbol: str,
) -> pd.DataFrame:

    """
    IMPORTANT FIX

    The previous version tried to download:

        BTCUSDT-15m-2026-10.zip

    Binance does not necessarily publish the current/final
    month as a monthly archive.

    Therefore:

    1. First requested month  -> daily archives
    2. Final requested month  -> daily archives
    3. Completed middle months -> monthly archives

    No synthetic candles.
    No forward fill.
    No silent gap removal.
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

    first_available_month = None

    # --------------------------------------------------------
    # MONTHLY ARCHIVES
    # --------------------------------------------------------

    for month in month_iter(
        DATA_START,
        OOS_END,
    ):

        # First and final month are intentionally loaded
        # through daily archives.
        if month in {
            first_month,
            final_month,
        }:
            continue

        year_month = month.strftime(
            "%Y-%m"
        )

        url = (
            f"{ARCHIVE_BASE}/monthly/"
            f"klines/{symbol}/{INTERVAL}/"
            f"{symbol}-{INTERVAL}-"
            f"{year_month}.zip"
        )

        blob = download(url)

        if blob is None:

            # Symbol may not have existed yet.
            if first_available_month is None:
                continue

            raise RuntimeError(
                "Archive missing after "
                f"first availability: {url}"
            )

        if first_available_month is None:
            first_available_month = month

        frames.append(
            read_zip(blob)
        )

    # --------------------------------------------------------
    # DAILY ARCHIVES FOR EDGE MONTHS
    # --------------------------------------------------------

    edge_months = {
        first_month.strftime("%Y-%m"),
        final_month.strftime("%Y-%m"),
    }

    for day in daily_iter(
        DATA_START,
        OOS_END,
    ):

        if day.strftime("%Y-%m") not in edge_months:
            continue

        date_string = day.strftime(
            "%Y-%m-%d"
        )

        url = (
            f"{ARCHIVE_BASE}/daily/"
            f"klines/{symbol}/{INTERVAL}/"
            f"{symbol}-{INTERVAL}-"
            f"{date_string}.zip"
        )

        blob = download(url)

        if blob is None:

            day_month = pd.Timestamp(
                day.year,
                day.month,
                1,
                tz="UTC",
            )

            # Before the symbol existed:
            # missing archive is acceptable.
            if (
                first_available_month is None
                or day_month < first_available_month
            ):
                continue

            raise RuntimeError(
                "Daily archive missing after "
                f"first availability: {url}"
            )

        if first_available_month is None:
            first_available_month = pd.Timestamp(
                day.year,
                day.month,
                1,
                tz="UTC",
            )

        frames.append(
            read_zip(blob)
        )

    if not frames:
        raise RuntimeError(
            f"No Binance data found for {symbol}"
        )

    # --------------------------------------------------------
    # CLEAN DATA
    # --------------------------------------------------------

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df["open_time"] = convert_timestamp(
        df["open_time"]
    )

    df["close_time"] = convert_timestamp(
        df["close_time"]
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    df = (
        df
        .drop_duplicates(
            "open_time"
        )
        .sort_values(
            "open_time"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # EXACT DATA RANGE
    # --------------------------------------------------------

    df = df[
        (df["open_time"] >= DATA_START)
        &
        (df["open_time"] <= OOS_END)
    ].copy()

    if df.empty:
        raise RuntimeError(
            f"No rows in requested range "
            f"for {symbol}"
        )

    # --------------------------------------------------------
    # STRICT 15-MINUTE CONTINUITY
    # --------------------------------------------------------

    differences = (
        df["open_time"]
        .diff()
        .dropna()
    )

    expected = pd.Timedelta(
        minutes=15
    )

    invalid = differences[
        differences != expected
    ]

    if not invalid.empty:

        bad_index = invalid.index[0]

        previous_time = df.loc[
            bad_index - 1,
            "open_time",
        ]

        current_time = df.loc[
            bad_index,
            "open_time",
        ]

        raise RuntimeError(
            f"15m data gap for {symbol}: "
            f"{previous_time} -> "
            f"{current_time} "
            f"({current_time - previous_time})"
        )

    # --------------------------------------------------------
    # EXACT OOS COVERAGE
    # --------------------------------------------------------

    actual_last = df[
        "open_time"
    ].iloc[-1]

    if actual_last != OOS_END:

        raise RuntimeError(
            f"Incomplete OOS coverage "
            f"for {symbol}: "
            f"expected {OOS_END}, "
            f"got {actual_last}"
        )

    return df


# ============================================================
# CAUSAL PIVOTS
# ============================================================

def confirmed_pivots(
    df: pd.DataFrame,
    pivot: int = PIVOT,
) -> pd.DataFrame:

    x = df.copy()

    highs = x["high"].to_numpy(
        dtype=float
    )

    lows = x["low"].to_numpy(
        dtype=float
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
        pivot,
        n - pivot,
    ):

        pivot_high[i] = (
            highs[i]
            >=
            np.max(
                highs[
                    i - pivot :
                    i + pivot + 1
                ]
            )
        )

        pivot_low[i] = (
            lows[i]
            <=
            np.min(
                lows[
                    i - pivot :
                    i + pivot + 1
                ]
            )
        )

    x["pivot_high"] = pivot_high
    x["pivot_low"] = pivot_low

    return x


# ============================================================
# 4H AGGREGATION
# ============================================================

def aggregate_4h(
    df15: pd.DataFrame,
) -> pd.DataFrame:

    z = df15.set_index(
        "open_time"
    )

    grouped = z.resample(
        HTF_INTERVAL,
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

    htf["count_15m"] = (
        grouped["close"]
        .count()
    )

    htf = (
        htf
        .dropna()
        .reset_index()
    )

    # 4H requires exactly sixteen 15m candles.
    htf = htf[
        htf["count_15m"] == 16
    ].copy()

    if htf.empty:
        raise RuntimeError(
            "No complete 4H candles"
        )

    return htf


# ============================================================
# HELPERS
# ============================================================

def known_pivots(
    pivot_indices: np.ndarray,
    current_index: int,
) -> np.ndarray:

    if len(pivot_indices) == 0:
        return pivot_indices

    # Pivot i is confirmed at i + PIVOT.
    max_pivot = (
        current_index - PIVOT
    )

    position = np.searchsorted(
        pivot_indices,
        max_pivot,
        side="right",
    )

    return pivot_indices[
        :position
    ]


def aligned(
    first: float,
    second: float,
) -> bool:

    denominator = max(
        abs(first),
        abs(second),
        1e-12,
    )

    return (
        abs(first - second)
        / denominator
        <= ALIGN_TOL
    )


def trend_down(
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

    known_highs = known_pivots(
        highs,
        index,
    )

    known_lows = known_pivots(
        lows,
        index,
    )

    known_highs = known_highs[
        known_highs < index
    ]

    known_lows = known_lows[
        known_lows < index
    ]

    if (
        len(known_highs) < 2
        or len(known_lows) < 2
    ):
        return False

    h1, h2 = known_highs[-2:]
    l1, l2 = known_lows[-2:]

    return (
        float(x.iloc[h2]["high"])
        <
        float(x.iloc[h1]["high"])
        and
        float(x.iloc[l2]["low"])
        <
        float(x.iloc[l1]["low"])
    )


def trend_up(
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

    known_highs = known_pivots(
        highs,
        index,
    )

    known_lows = known_pivots(
        lows,
        index,
    )

    known_highs = known_highs[
        known_highs < index
    ]

    known_lows = known_lows[
        known_lows < index
    ]

    if (
        len(known_highs) < 2
        or len(known_lows) < 2
    ):
        return False

    h1, h2 = known_highs[-2:]
    l1, l2 = known_lows[-2:]

    return (
        float(x.iloc[h2]["high"])
        >
        float(x.iloc[h1]["high"])
        and
        float(x.iloc[l2]["low"])
        >
        float(x.iloc[l1]["low"])
    )


def first_zone_touch(
    x: pd.DataFrame,
    start_index: int,
    zone_low: float,
    zone_high: float,
    end_index: int,
) -> Optional[int]:

    highs = x["high"].to_numpy(
        dtype=float
    )

    lows = x["low"].to_numpy(
        dtype=float
    )

    end_index = min(
        end_index,
        len(x) - 1,
    )

    for i in range(
        start_index,
        end_index + 1,
    ):

        if (
            highs[i] >= zone_low
            and
            lows[i] <= zone_high
        ):
            return i

    return None


# ============================================================
# HTF ZONES
# ============================================================

def build_htf_zones(
    htf: pd.DataFrame,
) -> list[dict]:

    hp = confirmed_pivots(
        htf,
        PIVOT,
    )

    zones = []

    for i in range(
        len(hp)
    ):

        row = hp.iloc[i]

        if (
            not row["pivot_low"]
            and
            not row["pivot_high"]
        ):
            continue

        confirmation_index = (
            i + PIVOT
        )

        if (
            confirmation_index
            >= len(hp)
        ):
            continue

        confirmation_time = (
            pd.Timestamp(
                hp.iloc[
                    confirmation_index
                ]["open_time"]
            )
            +
            pd.Timedelta(
                hours=4
            )
        )

        if row["pivot_low"]:

            body_high = max(
                float(row["open"]),
                float(row["close"]),
            )

            zones.append(
                {
                    "type": "demand",
                    "pivot_index": i,
                    "known_time":
                        confirmation_time,
                    "low":
                        float(row["low"]),
                    "high":
                        body_high,
                }
            )

        if row["pivot_high"]:

            body_low = min(
                float(row["open"]),
                float(row["close"]),
            )

            zones.append(
                {
                    "type": "supply",
                    "pivot_index": i,
                    "known_time":
                        confirmation_time,
                    "low":
                        body_low,
                    "high":
                        float(row["high"]),
                }
            )

    return zones


# ============================================================
# IFC
# ============================================================

def detect_ifc(
    x: pd.DataFrame,
    ob_index: int,
    end_index: int,
    direction: str,
) -> bool:

    for i in range(
        ob_index + 2,
        min(
            end_index,
            len(x) - 1,
        ) + 1,
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
# CANDIDATE
# ============================================================

@dataclass
class Candidate:

    symbol: str
    direction: str

    signal_index: int
    entry_index: int

    htf_zone_index: int
    reaction_index: int
    choch_index: int

    ob_index: int

    liquidity_index_1: int
    liquidity_index_2: int

    sweep_index: int

    structural_target_index: int

    liquidity_level: float
    structural_target: float

    ob_low: float
    ob_high: float

    entry_price: float
    stop_price: float
    target_price: float

    ifc: bool


# ============================================================
# TRADE
# ============================================================

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
# CANDIDATE GENERATION
# ============================================================

def make_candidates(
    symbol: str,
    x: pd.DataFrame,
    htf: pd.DataFrame,
) -> list[Candidate]:

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

    used_signal_indices = set()

    # --------------------------------------------------------
    # PROCESS HTF ZONES
    # --------------------------------------------------------

    for zone in sorted(
        zones,
        key=lambda z:
            z["known_time"],
    ):

        known_time = zone[
            "known_time"
        ]

        # The zone cannot be used until the
        # confirming 4H candle has closed.
        reaction_start = int(
            x["open_time"].searchsorted(
                known_time,
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
        # FIRST HTF ZONE TOUCH
        # ----------------------------------------------------

        reaction_index = (
            first_zone_touch(
                x,
                reaction_start,
                float(zone["low"]),
                float(zone["high"]),
                n - 1,
            )
        )

        if reaction_index is None:
            continue

        # ====================================================
        # LONG
        # ====================================================

        if direction == "long":

            # Setup starts from an existing downtrend.
            if not trend_down(
                x,
                reaction_index,
            ):
                continue

            previous_highs = (
                known_pivots(
                    highs,
                    reaction_index,
                )
            )

            previous_highs = (
                previous_highs[
                    previous_highs
                    < reaction_index
                ]
            )

            if len(previous_highs) == 0:
                continue

            last_lh_index = int(
                previous_highs[-1]
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
            # Lowest confirmed valley before CHOCH.
            # ------------------------------------------------

            available_lows = (
                known_pivots(
                    lows,
                    choch_index,
                )
            )

            available_lows = (
                available_lows[
                    (
                        available_lows
                        >= reaction_index
                    )
                    &
                    (
                        available_lows
                        < choch_index
                    )
                ]
            )

            if len(available_lows) == 0:
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
            # FIRST RETURN TO OB
            # ------------------------------------------------

            ob_touch = (
                first_zone_touch(
                    x,
                    choch_index + 1,
                    ob_low,
                    ob_high,
                    n - 1,
                )
            )

            if ob_touch is None:
                continue

            # ------------------------------------------------
            # ALIGNED LIQUIDITY LOWS
            # ------------------------------------------------

            liquidity_pivots = (
                known_pivots(
                    lows,
                    ob_touch,
                )
            )

            liquidity_pivots = (
                liquidity_pivots[
                    (
                        liquidity_pivots
                        > choch_index
                    )
                    &
                    (
                        liquidity_pivots
                        < ob_touch
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

                first_index = int(
                    liquidity_pivots[a]
                )

                for b in range(
                    a + 1,
                    len(liquidity_pivots),
                ):

                    second_index = int(
                        liquidity_pivots[b]
                    )

                    if aligned(
                        low_values[
                            first_index
                        ],
                        low_values[
                            second_index
                        ],
                    ):

                        pair = (
                            first_index,
                            second_index,
                        )

                        break

                if pair is not None:
                    break

            if pair is None:
                continue

            liq1, liq2 = pair

            liquidity_level = max(
                low_values[liq1],
                low_values[liq2],
            )

            # ------------------------------------------------
            # LIQUIDITY SWEEP
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

            if (
                ob_touch - liq2
                >
                MAX_LIQ_TO_OB_BARS
            ):
                continue

            # ------------------------------------------------
            # ENTRY
            # ------------------------------------------------

            signal_index = ob_touch

            if (
                signal_index
                in used_signal_indices
            ):
                continue

            if signal_index >= n - 1:
                continue

            entry_index = (
                signal_index + 1
            )

            entry_price = (
                float(
                    x.iloc[
                        entry_index
                    ]["open"]
                )
                *
                (1 + SLIPPAGE)
            )

            stop_price = (
                ob_low
                *
                (1 - SL_BUFFER_PCT)
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
            # ------------------------------------------------

            target_pivots = (
                known_pivots(
                    highs,
                    entry_index,
                )
            )

            target_pivots = (
                target_pivots[
                    (
                        target_pivots
                        > choch_index
                    )
                    &
                    (
                        target_pivots
                        < entry_index
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

            structural_target = float
