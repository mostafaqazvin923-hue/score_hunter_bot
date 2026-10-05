#!/usr/bin/env python3
"""
SETUP 4 V2 — HTF Zone -> Reaction -> OB -> Aligned Liquidity
           -> Liquidity Break -> OB Retest

Binance USD-M Futures
15m execution / 4h context

SOURCE BASIS
------------
Mechanical research translation of Setup 4 from:
"10 ستاپ برتر.pdf"

The PDF is qualitative. Numeric parameters below are frozen
mechanical translations and are NOT claimed to be literal PDF values.

IMPORTANT RESEARCH PROTOCOL
---------------------------
Fresh OOS for Setup 4 V2:
    2024-10-04 through 2025-10-03 UTC

The previous Setup 4 V1 OOS:
    2025-10-04 through 2026-10-03 UTC

is NOT used for V2 development.

RESEARCH WINDOWS
----------------
Research period:
    365 days immediately before the fresh OOS.

Discovery:
    first half of research period.

Development:
    second half of research period.

OOS:
    completely separate reserved period.

CAUSALITY
---------
- 4H pivots are only usable after PIVOT completed 4H candles.
- A 4H zone is only available after its pivot is confirmed.
- Every structural index must be strictly before entry.
- Entry is at the NEXT 15m candle open after the completed OB retest.
- Entry candle itself is never checked for SL/TP.
- If SL and TP are both touched on the same later candle, SL wins.
- No timeout.
- No breakeven.
- No trailing.
- No synthetic candles.
- No forward fill.
- No same-candle re-entry.

OVERLAP RULE
------------
Maximum one simultaneous open trade per symbol.

Different symbols may have simultaneous open trades.

A symbol may not re-enter until the previous trade has
actually closed. Re-entry on the same exit candle is forbidden.

RISK
----
Initial capital : $1,000
Margin          : $100
Leverage        : 50x
Notional        : $5,000
RR              : 1:2
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

BASE_URL = (
    "https://data.binance.vision/data/futures/um/monthly/klines"
)

INTERVAL = "15m"

# ============================================================
# FRESH OOS — LOCKED FOR V2
# ============================================================

OOS_START = pd.Timestamp(
    "2024-10-04 00:00:00",
    tz="UTC",
)

OOS_END = pd.Timestamp(
    "2025-10-03 23:00:00",
    tz="UTC",
)

# Research period is exactly 365 days immediately before OOS.
RESEARCH_END = OOS_START - pd.Timedelta(minutes=15)
RESEARCH_START = (
    OOS_START - pd.Timedelta(days=365)
)

# Additional warmup before research.
WARMUP_START = (
    RESEARCH_START - pd.Timedelta(days=90)
)

# Split research into two equal calendar halves.
RESEARCH_MID = (
    RESEARCH_START
    + (OOS_START - RESEARCH_START) / 2
)

# ============================================================
# CAPITAL / COSTS
# ============================================================

INITIAL_CAPITAL = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE

RR = 2.0
FEE_RATE = 0.0007
SLIPPAGE = 0.0003

# ============================================================
# MECHANICAL TRANSLATION
# ============================================================

PIVOT = 2

HTF_ATR_PERIOD = 14
HTF_ZONE_ATR_MULT = 0.50

MAX_REACTION_BARS = 24

ALIGN_TOL = 0.004

MAX_LIQUIDITY_DISTANCE_BARS = 96

SL_BUFFER_PCT = 0.0005

MIN_RISK_PCT = 0.0005
MAX_RISK_PCT = 0.08

# ============================================================
# OUTPUT
# ============================================================

OUT = Path("setup4_v2_outputs")
OUT.mkdir(exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "setup4-v2-research/1.0"
    }
)


# ============================================================
# DATA HELPERS
# ============================================================

def month_range(start, end):
    """
    Return all calendar months intersecting [start, end].
    """

    cur = pd.Timestamp(start).to_period("M")
    last = pd.Timestamp(end).to_period("M")

    result = []

    while cur <= last:
        result.append(
            (cur.year, cur.month)
        )
        cur += 1

    return result


def fetch_month(symbol, year, month):
    filename = (
        f"{symbol}-{INTERVAL}-"
        f"{year:04d}-{month:02d}.zip"
    )

    url = (
        f"{BASE_URL}/{symbol}/"
        f"{INTERVAL}/{filename}"
    )

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

        except Exception:

            if attempt == 3:
                raise

            time.sleep(
                1.5 * (attempt + 1)
            )

    return None


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
                "Archive has no CSV file"
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

    raw["close_time"] = pd.to_numeric(
        raw["close_time"],
        errors="coerce",
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
    ]


def fetch_symbol(symbol):
    """
    Fetch real Binance USD-M Futures 15m data.

    404 before first available archive:
        allowed.

    404 after first available archive:
        fatal.

    No forward-fill.
    No synthetic candles.
    No silent gap removal.
    """

    start = WARMUP_START
    end = OOS_END

    frames = []
    first_available = False

    for year, month in month_range(
        start,
        end,
    ):

        blob = fetch_month(
            symbol,
            year,
            month,
        )

        if blob is None:

            if not first_available:
                continue

            raise RuntimeError(
                "Missing Binance archive after "
                f"first availability: "
                f"{symbol} "
                f"{year:04d}-{month:02d}"
            )

        first_available = True

        frame = parse_archive(blob)

        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise RuntimeError(
            f"No historical archive found for {symbol}"
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df = (
        df
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    df = df[
        (df.time >= start)
        & (df.time <= end)
    ].copy()

    if df.empty:
        raise RuntimeError(
            f"No usable rows for {symbol}"
        )

    # The requested historical endpoint is fully completed.
    # Still remove an actually incomplete final candle if encountered.
    now = pd.Timestamp.now(tz="UTC")

    if (
        not df.empty
        and
        df.time.iloc[-1]
        + pd.Timedelta(minutes=15)
        > now
    ):
        df = df.iloc[:-1].copy()

    if df.empty:
        raise RuntimeError(
            f"No usable rows after incomplete "
            f"candle removal for {symbol}"
        )

    # Strict 15m continuity.
    expected = pd.date_range(
        start=df.time.iloc[0],
        end=df.time.iloc[-1],
        freq="15min",
        tz="UTC",
    )

    actual = pd.DatetimeIndex(
        df.time
    )

    missing = expected.difference(
        actual
    )

    if len(missing):

        raise RuntimeError(
            f"15m data gap for {symbol}: "
            f"{len(missing)} missing candles; "
            f"first={missing[0]}"
        )

    return df.reset_index(
        drop=True
    )


# ============================================================
# INDICATORS
# ============================================================

def atr(df, period):
    previous_close = df.close.shift(1)

    true_range = pd.concat(
        [
            df.high - df.low,
            (
                df.low
                - previous_close
            ).abs(),
            (
                df.high
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(
        period,
        min_periods=period,
    ).mean()


# ============================================================
# CAUSAL PIVOTS
# ============================================================

def confirmed_pivots(df):
    """
    pivot_high[i] / pivot_low[i] identify the structural pivot
    located at i.

    A pivot at i becomes KNOWABLE only at i + PIVOT.

    We keep the pivot at its structural index and explicitly
    carry the confirmation index separately through the logic.
    """

    x = df.copy()

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

        left_highs = x.high.iloc[
            i - PIVOT:i
        ]

        right_highs = x.high.iloc[
            i + 1:i + 1 + PIVOT
        ]

        left_lows = x.low.iloc[
            i - PIVOT:i
        ]

        right_lows = x.low.iloc[
            i + 1:i + 1 + PIVOT
        ]

        if (
            x.high.iloc[i]
            > left_highs.max()
            and
            x.high.iloc[i]
            >= right_highs.max()
        ):
            pivot_high[i] = True

        if (
            x.low.iloc[i]
            < left_lows.min()
            and
            x.low.iloc[i]
            <= right_lows.min()
        ):
            pivot_low[i] = True

    x["pivot_high"] = pivot_high
    x["pivot_low"] = pivot_low

    return x


def make_htf(df):
    """
    Build completed 4H candles.

    A 4H candle is used only after its own 4H close.
    """

    htf = (
        df
        .set_index("time")[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]
        .resample(
            "4h",
            label="right",
            closed="right",
        )
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )

    htf["atr"] = atr(
        htf,
        HTF_ATR_PERIOD,
    )

    htf = confirmed_pivots(
        htf
    )

    return htf


# ============================================================
# HTF ZONES — CAUSAL
# ============================================================

def build_zones(htf):
    """
    A confirmed 4H pivot creates a zone.

    IMPORTANT V1 FIX:
    The old code inspected pivot_high at the confirmation row
    while reading the price from i-PIVOT. That could use the wrong
    pivot state.

    V2 explicitly checks the structural pivot at pivot_i and only
    makes the zone available at confirm_i = pivot_i + PIVOT.
    """

    zones = []

    for pivot_i in range(
        PIVOT,
        len(htf) - PIVOT,
    ):

        confirm_i = (
            pivot_i + PIVOT
        )

        row_at_confirmation = htf.iloc[
            confirm_i
        ]

        atr_value = float(
            row_at_confirmation.atr
        )

        if not np.isfinite(
            atr_value
        ):
            continue

        if atr_value <= 0:
            continue

        if bool(
            htf.pivot_high.iloc[
                pivot_i
            ]
        ):

            price = float(
                htf.high.iloc[pivot_i]
            )

            zones.append(
                {
                    "side": "SHORT",
                    "pivot_i": int(
                        pivot_i
                    ),
                    "confirm_i": int(
                        confirm_i
                    ),
                    "pivot_time":
                        htf.time.iloc[
                            pivot_i
                        ],
                    "confirm_time":
                        htf.time.iloc[
                            confirm_i
                        ],
                    "center": price,
                    "low": (
                        price
                        - HTF_ZONE_ATR_MULT
                        * atr_value
                    ),
                    "high": (
                        price
                        + HTF_ZONE_ATR_MULT
                        * atr_value
                    ),
                }
            )

        if bool(
            htf.pivot_low.iloc[
                pivot_i
            ]
        ):

            price = float(
                htf.low.iloc[pivot_i]
            )

            zones.append(
                {
                    "side": "LONG",
                    "pivot_i": int(
                        pivot_i
                    ),
                    "confirm_i": int(
                        confirm_i
                    ),
                    "pivot_time":
                        htf.time.iloc[
                            pivot_i
                        ],
                    "confirm_time":
                        htf.time.iloc[
                            confirm_i
                        ],
                    "center": price,
                    "low": (
                        price
                        - HTF_ZONE_ATR_MULT
                        * atr_value
                    ),
                    "high": (
                        price
                        + HTF_ZONE_ATR_MULT
                        * atr_value
                    ),
                }
            )

    zones.sort(
        key=lambda z: z[
            "confirm_i"
        ]
    )

    return zones


# ============================================================
# STRUCTURAL HELPERS
# ============================================================

def aligned(a, b):
    reference = max(
        abs((a + b) / 2.0),
        1e-12,
    )

    return (
        abs(a - b)
        / reference
        <= ALIGN_TOL
    )


def known_pivots(
    x,
    current_i,
):
    """
    Return every pivot whose confirmation is known at current_i.

    Pivot j becomes usable only when:
        j + PIVOT <= current_i
    """

    max_pivot = (
        current_i - PIVOT
    )

    if max_pivot < PIVOT:
        return [], []

    highs = [
        j
        for j in range(
            PIVOT,
            max_pivot + 1,
        )
        if bool(
            x.pivot_high.iloc[j]
        )
    ]

    lows = [
        j
        for j in range(
            PIVOT,
            max_pivot + 1,
        )
        if bool(
            x.pivot_low.iloc[j]
        )
    ]

    return highs, lows


def first_zone_touch(
    x,
    start_i,
    zone,
):
    end_i = min(
        len(x) - 1,
        start_i + MAX_REACTION_BARS,
    )

    for j in range(
        start_i,
        end_i + 1,
    ):

        if (
            x.low.iloc[j]
            <= zone["high"]
            and
            x.high.iloc[j]
            >= zone["low"]
        ):
            return j

    return None


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def make_candidates(
    symbol,
    x,
    htf,
    zones,
):
    """
    Generate all structurally valid candidates over the FULL
    research + OOS data range.

    No performance filtering is performed here.

    This is important because Discovery/Development must actually
    contain historical trades rather than being accidentally
    replaced by the OOS window.
    """

    candidates = []

    for zone in zones:

        zone_time = zone[
            "confirm_time"
        ]

        future_rows = x.index[
            x.time > zone_time
        ]

        if len(future_rows) == 0:
            continue

        start_i = int(
            future_rows[0]
        )

        touch_i = first_zone_touch(
            x,
            start_i,
            zone,
        )

        if touch_i is None:
            continue

        reaction_end = min(
            len(x) - 1,
            touch_i + MAX_REACTION_BARS,
        )

        # Search the first valid structural sequence after reaction.
        for current_i in range(
            touch_i + 1,
            reaction_end + 1,
        ):

            known_highs, known_lows = (
                known_pivots(
                    x,
                    current_i,
                )
            )

            # ==================================================
            # SHORT
            # ==================================================

            if zone["side"] == "SHORT":

                lower_highs = [
                    p
                    for p in known_highs
                    if (
                        touch_i
                        < p
                        < current_i
                    )
                ]

                if not lower_highs:
                    continue

                ob_i = lower_highs[-1]

                # OB pivot itself must already be confirmed.
                if (
                    ob_i + PIVOT
                    > current_i
                ):
                    continue

                older_highs = [
                    p
                    for p in known_highs
                    if p < ob_i
                ]

                if older_highs:

                    previous_high = (
                        older_highs[-1]
