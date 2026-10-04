#!/usr/bin/env python3

"""
SETUP 4 V1
HTF Zone -> Reaction -> OB -> Aligned Liquidity
-> Liquidity Break -> OB Retest -> Entry

Research translation of Setup 4 from the supplied PDF.

IMPORTANT:
- The PDF describes the structure qualitatively.
- Numeric values below are frozen mechanical translations.
- No OOS tuning.
- Real Binance Futures OHLCV only.
- No synthetic candles.
- No forward filling.
- No silent gap removal.
- No artificial timeout.
- Fixed RR = 1:2.
- Maximum one simultaneous open trade per symbol.
- Different symbols may overlap.
- Re-entry on the same symbol is forbidden until the previous
  trade has closed.
- If a trade closes on candle J, earliest re-entry is J+1.
- Entry candle is not checked for SL/TP to avoid intrabar ambiguity.
- If SL and TP are both touched on the same later candle,
  SL is conservatively assumed first.
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

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

INTERVAL = "15m"

MONTHLY_BASE_URL = (
    "https://data.binance.vision/data/"
    "futures/um/monthly/klines"
)

DAILY_BASE_URL = (
    "https://data.binance.vision/data/"
    "futures/um/daily/klines"
)

# ------------------------------------------------------------
# Fresh Setup 4 research/OOS window.
# ------------------------------------------------------------

OOS_START = pd.Timestamp(
    "2025-10-04 00:00:00",
    tz="UTC",
)

OOS_END = pd.Timestamp(
    "2026-10-03 23:00:00",
    tz="UTC",
)

RESEARCH_START = (
    OOS_START - pd.Timedelta(days=365)
)

WARMUP_START = (
    RESEARCH_START - pd.Timedelta(days=90)
)

# ------------------------------------------------------------
# Portfolio / execution.
# ------------------------------------------------------------

INITIAL_CAPITAL = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE

RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003

# ------------------------------------------------------------
# Mechanical research translation.
# ------------------------------------------------------------

PIVOT = 2

HTF_ATR_PERIOD = 14
HTF_ZONE_ATR_MULT = 0.50

MAX_REACTION_BARS = 24
ALIGN_TOL = 0.004
MAX_LIQUIDITY_DISTANCE_BARS = 96

SL_BUFFER_PCT = 0.0005

MIN_RISK_PCT = 0.0005
MAX_RISK_PCT = 0.08

OUTPUT_DIR = Path(
    "setup4_v1_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent":
            "Mozilla/5.0 setup4-v1-research"
    }
)


# ============================================================
# TIME HELPERS
# ============================================================

def as_utc_timestamp(value):
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")


def month_range(start, end):
    """
    Calendar month iterator.

    Does NOT call to_period() on timezone-aware timestamps.
    """

    start_ts = as_utc_timestamp(start)
    end_ts = as_utc_timestamp(end)

    current = pd.Timestamp(
        year=start_ts.year,
        month=start_ts.month,
        day=1,
        tz="UTC",
    )

    last = pd.Timestamp(
        year=end_ts.year,
        month=end_ts.month,
        day=1,
        tz="UTC",
    )

    months = []

    while current <= last:

        months.append(
            (
                current.year,
                current.month,
            )
        )

        current = (
            current
            + pd.offsets.MonthBegin(1)
        )

    return months


def day_range(start, end):
    start_ts = as_utc_timestamp(start)
    end_ts = as_utc_timestamp(end)

    return pd.date_range(
        start=start_ts.normalize(),
        end=end_ts.normalize(),
        freq="D",
        tz="UTC",
    )


# ============================================================
# BINANCE HTTP
# ============================================================

def http_get(url):
    last_error = None

    for attempt in range(4):

        try:

            response = SESSION.get(
                url,
                timeout=60,
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()

            return response.content

        except requests.RequestException as exc:

            last_error = exc

            if attempt == 3:
                raise

            time.sleep(
                1.5 * (attempt + 1)
            )

    raise last_error


def fetch_month(
    symbol,
    year,
    month,
):
    filename = (
        f"{symbol}-{INTERVAL}-"
        f"{year:04d}-{month:02d}.zip"
    )

    url = (
        f"{MONTHLY_BASE_URL}/"
        f"{symbol}/{INTERVAL}/"
        f"{filename}"
    )

    return http_get(url)


def fetch_day(
    symbol,
    day,
):
    day = as_utc_timestamp(day)

    date_string = day.strftime(
        "%Y-%m-%d"
    )

    filename = (
        f"{symbol}-{INTERVAL}-"
        f"{date_string}.zip"
    )

    url = (
        f"{DAILY_BASE_URL}/"
        f"{symbol}/{INTERVAL}/"
        f"{filename}"
    )

    return http_get(url)


# ============================================================
# ARCHIVE PARSER
# ============================================================

def parse_archive(blob):
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
                "Binance archive contains no CSV file."
            )

        with archive.open(
            csv_files[0]
        ) as handle:

            raw = pd.read_csv(
                handle,
                header=None,
            )

    if raw.empty:
        return pd.DataFrame()

    columns = [
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

    raw = raw.iloc[
        :,
        :len(columns),
    ]

    raw.columns = columns[
        :raw.shape[1]
    ]

    raw["open_time"] = pd.to_numeric(
        raw["open_time"],
        errors="coerce",
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        raw[col] = pd.to_numeric(
            raw[col],
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

    raw["time"] = pd.to_datetime(
        raw["open_time"],
        unit="ms",
        utc=True,
    )

    return raw[
        [
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()


# ============================================================
# DATA FETCH
# ============================================================

def fetch_symbol(symbol):
    """
    Fetch WARMUP_START -> OOS_END.

    Completed calendar months:
        monthly Binance archive.

    Current/incomplete month:
        daily Binance archives.

    Missing required archive = fatal.

    No synthetic candles.
    No forward fill.
    No silent dropping.
    """

    start = as_utc_timestamp(
        WARMUP_START
    )

    end = as_utc_timestamp(
        OOS_END
    )

    today = pd.Timestamp.now(
        tz="UTC"
    ).normalize()

    frames = []

    for year, month in month_range(
        start,
        end,
    ):

        month_start = pd.Timestamp(
            year=year,
            month=month,
            day=1,
            tz="UTC",
        )

        next_month = (
            month_start
            + pd.offsets.MonthBegin(1)
        )

        requested_start = max(
            start,
            month_start,
        )

        requested_end = min(
            end,
            next_month
            - pd.Timedelta(minutes=15),
        )

        # A month is complete only when the next calendar
        # month has started.
        month_is_complete = (
            next_month <= today
        )

        # ----------------------------------------------------
        # Completed month -> monthly archive.
        # ----------------------------------------------------

        if month_is_complete:

            blob = fetch_month(
                symbol,
                year,
                month,
            )

            if blob is None:

                raise RuntimeError(
                    "Missing Binance monthly archive "
                    f"for completed month: "
                    f"{symbol} "
                    f"{year:04d}-{month:02d}"
                )

            frame = parse_archive(
                blob
            )

            if not frame.empty:
                frames.append(frame)

        # ----------------------------------------------------
        # Current/incomplete month -> daily archives.
        # ----------------------------------------------------

        else:

            for day in day_range(
                requested_start,
                requested_end,
            ):

                day = as_utc_timestamp(
                    day
                )

                # Never request future dates.
                if day.normalize() > today:
                    continue

                blob = fetch_day(
                    symbol
