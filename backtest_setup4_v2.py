#!/usr/bin/env python3
"""
SETUP 4 V2
HTF Zone -> Reaction -> OB -> Aligned Liquidity
-> Liquidity Break -> OB Retest

Binance USD-M Futures
15m execution / 4h context

SOURCE
------
Mechanical research translation of Setup 4 from:
"10 ستاپ برتر.pdf"

The PDF is qualitative.
Numeric parameters in this code are frozen mechanical research
translations and are NOT claimed to be literal PDF specifications.

RESEARCH PROTOCOL
-----------------
Previous Setup 4 V1 OOS:
    2025-10-04 -> 2026-10-03

Fresh Setup 4 V2 OOS:
    2024-10-04 -> 2025-10-03

The fresh OOS is not used for development/tuning.

RESEARCH:
    365 days immediately before fresh OOS.

DISCOVERY:
    first half of research period.

DEVELOPMENT:
    second half of research period.

RISK
----
Initial capital : $1,000
Margin          : $100
Leverage        : 50x
Notional        : $5,000
RR              : 1:2

INTEGRITY
---------
- No lookahead.
- Confirmed 4H pivots only.
- Confirmed 15m pivots only.
- No same-candle re-entry.
- No same-symbol overlap.
- Different symbols may overlap.
- No synthetic candles.
- No forward fill.
- No artificial timeout.
- No trailing.
- No breakeven.
- No future target information.
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
    "https://data.binance.vision/"
    "data/futures/um/monthly/klines"
)

INTERVAL = "15m"

# ============================================================
# FRESH OOS
# ============================================================

OOS_START = pd.Timestamp(
    "2024-10-04 00:00:00",
    tz="UTC",
)

OOS_END = pd.Timestamp(
    "2025-10-03 23:00:00",
    tz="UTC",
)

# ============================================================
# RESEARCH
# ============================================================

RESEARCH_START = (
    OOS_START
    - pd.Timedelta(days=365)
)

RESEARCH_END = (
    OOS_START
    - pd.Timedelta(minutes=15)
)

WARMUP_START = (
    RESEARCH_START
    - pd.Timedelta(days=90)
)

RESEARCH_MID = (
    RESEARCH_START
    + (
        OOS_START
        - RESEARCH_START
    ) / 2
)

# ============================================================
# CAPITAL
# ============================================================

INITIAL_CAPITAL = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE

RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003

# ============================================================
# STRUCTURAL PARAMETERS
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

OUT = Path(
    "setup4_v2_outputs"
)

OUT.mkdir(
    exist_ok=True
)

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent":
            "setup4-v2-research/1.0"
    }
)


# ============================================================
# DATE / MONTH HELPERS
# ============================================================

def month_range(start, end):
    """
    Return all calendar months intersecting [start, end].

    Convert UTC timestamps to naive UTC before Period conversion
    so pandas does not emit the timezone warning.
    """

    start_ts = pd.Timestamp(start)

    end_ts = pd.Timestamp(end)

    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")

    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")

    start_naive = (
        start_ts
        .tz_localize(None)
    )

    end_naive = (
        end_ts
        .tz_localize(None)
    )

    cur = start_naive.to_period("M")
    last = end_naive.to_period("M")

    result = []

    while cur <= last:

        result.append(
            (
                cur.year,
                cur.month,
            )
        )

        cur += 1

    return result


# ============================================================
# DATA DOWNLOAD
# ============================================================

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
        f"{BASE_URL}/"
        f"{symbol}/"
        f"{INTERVAL}/"
        f"{filename}"
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
                1.5
                * (attempt + 1)
            )

    return None


def parse_archive(blob):
    with zipfile.ZipFile(
        io.BytesIO(blob)
    ) as archive:

        csv_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(
                ".csv"
            )
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

    404 after first availability:
        fatal.

    No synthetic candles.
    No forward-fill.
    No silent gap removal.
    """

    frames = []

    first_available = False

    for year, month in month_range(
        WARMUP_START,
        OOS_END,
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
                "first availability: "
                f"{symbol} "
                f"{year:04d}-{month:02d}"
            )

        first_available = True

        frame = parse_archive(
            blob
        )

        if not frame.empty:
            frames.append(frame)

    if not frames:

        raise RuntimeError(
            f"No historical archive found "
            f"for {symbol}"
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df = (
        df
        .drop_duplicates(
            "time"
        )
        .sort_values(
            "time"
        )
        .reset_index(
            drop=True
        )
    )

    df = df[
        (df.time >= WARMUP_START)
        &
        (df.time <= OOS_END)
    ].copy()

    if df.empty:

        raise RuntimeError(
            f"No usable rows for {symbol}"
        )

    # Remove only an actually incomplete final candle.
    now = pd.Timestamp.now(
        tz="UTC"
    )

    if (
        not df.empty
        and
        (
            df.time.iloc[-1]
            + pd.Timedelta(
                minutes=15
            )
            > now
        )
    ):

        df = df.iloc[
            :-1
        ].copy()

    if df.empty:

        raise RuntimeError(
            f"No usable rows after "
            f"incomplete candle removal "
            f"for {symbol}"
        )

    # Strict continuity.
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
            f"15m data gap for "
            f"{symbol}: "
            f"{len(missing)} missing "
            f"candles; "
            f"first={missing[0]}"
        )

    return df.reset_index(
        drop=True
    )


# ============================================================
# ATR
# ============================================================

def atr(
    df,
    period,
):
    previous_close = (
        df.close.shift(1)
    )

    true_range = pd.concat(
        [
            df.high - df.low,

            (
                df.high
                - previous_close
            ).abs(),

            (
                df.low
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(
        axis=1
    )

    return true_range.rolling(
        period,
        min_periods=period,
    ).mean()


# ============================================================
# CONFIRMED PIVOTS
# ============================================================

def confirmed_pivots(df):
    """
    Structural pivot at i becomes usable only at i + PIVOT.

    pivot_high / pivot_low remain attached to the structural
    candle. The calling logic explicitly enforces the confirmation
    delay.
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
            i + 1:
            i + 1 + PIVOT
        ]

        left_lows = x.low.iloc[
            i - PIVOT:i
        ]

        right_lows = x.low.iloc[
            i + 1:
            i + 1 + PIVOT
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
            <= right_lows.max()
        ):

            pivot_low[i] = True

    x["pivot_high"] = (
        pivot_high
    )

    x["pivot_low"] = (
        pivot_low
    )

    return x


# ============================================================
# 4H CONTEXT
# ============================================================

def make_htf(df):
    """
    Build completed 4H candles from real 15m data.
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
# HTF ZONES
# ============================================================

def build_zones(htf):
    """
    A structural 4H pivot becomes an available HTF zone only
    after PIVOT completed 4H candles.

    The ATR used for the zone is taken at confirmation time.
    """

    zones = []

    for pivot_i in range(
        PIVOT,
        len(htf) - PIVOT,
    ):

        confirm_i = (
            pivot_i + PIVOT
        )

        atr_value = float(
            htf.atr.iloc[
                confirm_i
            ]
        )

        if not np.isfinite(
            atr_value
        ):
            continue

        if atr_value <= 0:
            continue

        pivot_time = (
            htf.time.iloc[
                pivot_i
            ]
        )

        confirm_time = (
            htf.time.iloc[
                confirm_i
            ]
        )

        if bool(
            htf.pivot_high.iloc[
                pivot_i
            ]
        ):

            price = float(
                htf.high.iloc[
                    pivot_i
                ]
            )

            zones.append(
                {
                    "side":
                        "SHORT",

                    "pivot_i":
                        int(pivot_i),

                    "confirm_i":
                        int(confirm_i),

                    "pivot_time":
                        pivot_time,

                    "confirm_time":
                        confirm_time,

                    "center":
                        price,

                    "low":
                        (
                            price
                            - (
                                HTF_ZONE_ATR_MULT
                                * atr_value
                            )
                        ),

                    "high":
                        (
                            price
                            + (
                                HTF_ZONE_ATR_MULT
                                * atr_value
                            )
                        ),
                }
            )

        if bool(
            htf.pivot_low.iloc[
                pivot_i
            ]
        ):

            price = float(
                htf.low.iloc[
                    pivot_i
                ]
            )

            zones.append(
                {
                    "side":
                        "LONG",

                    "pivot_i":
                        int(pivot_i),

                    "confirm_i":
                        int(confirm_i),

                    "pivot_time":
                        pivot_time,

                    "confirm_time":
                        confirm_time,

                    "center":
                        price,

                    "low":
                        (
                            price
                            - (
                                HTF_ZONE_ATR_MULT
                                * atr_value
                            )
                        ),

                    "high":
                        (
                            price
                            + (
                                HTF_ZONE_ATR_MULT
                                * atr_value
                            )
                        ),
                }
            )

    zones.sort(
        key=lambda z: z[
            "confirm_time"
        ]
    )

    return zones


# ============================================================
# STRUCTURAL HELPERS
# ============================================================

def aligned(
    a,
    b,
):
    reference = max(
        abs(
            (a + b)
            / 2.0
        ),
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
    Return 15m pivots whose confirmation is already known.

    Pivot j is usable only if:
        j + PIVOT <= current_i
    """

    max_pivot = (
        current_i
        - PIVOT
    )

    if max_pivot < PIVOT:
        return [], []

    highs = []

    lows = []

    for j in range(
        PIVOT,
        max_pivot + 1,
    ):

        if bool(
            x["pivot_high"].iloc[j]
        ):
            highs.append(j)

        if bool(
            x["pivot_low"].iloc[j]
        ):
            lows.append(j)

    return highs, lows


def first_zone_touch(
    x,
    start_i,
    zone,
):
    """
    First 15m candle touching the confirmed HTF zone.

    Reaction search is limited to MAX_REACTION_BARS.
    """

    end_i = min(
        len(x) - 1,
        start_i
        + MAX_REACTION_BARS,
    )

    for j in range(
        start_i,
        end_i + 1,
    ):

        candle_low = float(
            x.low.iloc[j]
        )

        candle_high = float(
            x.high.iloc[j]
        )

        if (
            candle_low
            <= zone["high"]
            and
            candle_high
            >= zone["low"]
        ):

            return j

    return None


# ============================================================
# CANDIDATES
# ============================================================

def make_candidates(
    symbol,
    x,
    htf,
    zones,
):
    """
    Generate structural candidates from the full available
    research + OOS data.

    x:
        15m dataframe with confirmed 15m pivots.

    htf:
        4H dataframe with confirmed 4H pivots.

    No performance filtering is performed here.
    """

    candidates = []

    for zone in zones:

        zone_confirm_time = (
            zone["confirm_time"]
        )

        future_rows = x.index[
            x.time
            > zone_confirm_time
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
            touch_i
            + MAX_REACTION_BARS,
        )

        # ----------------------------------------------------
        # Search structural reaction.
        # ----------------------------------------------------

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

            # =================================================
            # SHORT
            # =================================================

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

                ob_i = (
                    lower_highs[-1]
                )

                # Explicit causal confirmation.
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
                    )

                    if (
                        float(
                            x.high.iloc[
                                ob_i
                            ]
                        )
                        >=
                        float(
                            x.high.iloc[
                                previous_high
                            ]
                        )
                    ):
                        continue

                aligned_highs = [
                    p
                    for p in known_highs
                    if (
                        ob_i
                        < p
                        < current_i
                    )
                ]

                if len(
                    aligned_highs
                ) < 2:
                    continue

                h1 = aligned_highs[
                    -2
                ]

                h2 = aligned_highs[
                    -1
                ]

                if not aligned(
                    float(
                        x.high.iloc[h1]
                    ),
                    float(
                        x.high.iloc[h2]
                    ),
                ):
                    continue

                liquidity_level = max(
                    float(
                        x.high.iloc[h1]
                    ),
                    float(
                        x.high.iloc[h2]
                    ),
                )

                break_i = None

                last_break = min(
                    len(x) - 1,
                    h2
                    + MAX_LIQUIDITY_DISTANCE_BARS,
                )

                for b in range(
                    h2 + 1,
                    last_break + 1,
                ):

                    if (
                        float(
                            x.high.iloc[b]
                        )
                        > liquidity_level
                    ):

                        break_i = b
                        break

                if break_i is None:
                    continue

                # OB midpoint.
                midpoint = (
                    float(
                        x.low.iloc[ob_i]
                    )
                    +
                    float(
                        x.high.iloc[ob_i]
                    )
                ) / 2.0

                retest_i = None

                for e in range(
                    break_i + 1,
                    len(x),
                ):

                    candle_low = float(
                        x.low.iloc[e]
                    )

                    candle_high = float(
                        x.high.iloc[e]
                    )

                    if (
                        candle_low
                        <= midpoint
                        <= candle_high
                    ):

                        retest_i = e
                        break

                if retest_i is None:
                    continue

                entry_i = (
                    retest_i + 1
                )

                if (
                    entry_i
                    >= len(x)
                ):
                    continue

                entry = float(
                    x.open.iloc[
                        entry_i
                    ]
                )

                sl = (
                    float(
                        x.high.iloc[
                            ob_i
                        ]
                    )
                    * (
                        1.0
                        + SL_BUFFER_PCT
                    )
                )

                risk = (
                    sl
                    - entry
                )

                if risk <= 0:
                    continue

                risk_pct = (
                    risk
                    / entry
                )

                if not (
                    MIN_RISK_PCT
                    <= risk_pct
                    <= MAX_RISK_PCT
                ):
                    continue

                tp = (
                    entry
                    - RR * risk
                )

                if h2 <= ob_i:
                    continue

                structural_target = float(
                    x.low.iloc[
                        ob_i:
                        h2 + 1
                    ].min()
                )

                # SHORT structural viability.
                if (
                    tp
                    < structural_target
                ):
                    continue

                candidates.append(
                    {
                        "symbol":
                            symbol,

                        "side":
                            "SHORT",

                        "zone_pivot_i":
                            zone["pivot_i"],

                        "zone_confirm_i":
                            zone["confirm_i"],

                        "zone_pivot_time":
                            zone["pivot_time"],

                        "zone_confirm_time":
                            zone["confirm_time"],

                        "touch_i":
                            touch_i,

                        "ob_i":
                            ob_i,

                        "align1_i":
                            h1,

                        "align2_i":
                            h2,

                        "break_i":
                            break_i,

                        "retest_i":
                            retest_i,

                        "entry_i":
                            entry_i,

                        "entry":
                            entry,

                        "sl":
                            sl,

                        "tp":
                            tp,

                        "structural_target":
                            structural_target,
                    }
                )

                break

            # =================================================
            # LONG
            # =================================================

            else:

                higher_lows = [
                    p
                    for p in known_lows
                    if (
                        touch_i
                        < p
                        < current_i
                    )
                ]

                if not higher_lows:
                    continue

                ob_i = (
                    higher_lows[-1]
                )

                if (
                    ob_i + PIVOT
                    > current_i
                ):
                    continue

                older_lows = [
                    p
                    for p in known_lows
                    if p < ob_i
                ]

                if older_lows:

                    previous_low = (
                        older_lows[-1]
                    )

                    if (
                        float(
                            x.low.iloc[
                                ob_i
                            ]
                        )
                        <=
                        float(
                            x.low.iloc[
                                previous_low
                            ]
                        )
                    ):
                        continue

                aligned_lows = [
                    p
                    for p in known_lows
                    if (
                        ob_i
                        < p
                        < current_i
                    )
                ]

                if len(
                    aligned_lows
                ) < 2:
                    continue

                l1 = aligned_lows[
                    -2
                ]

                l2 = aligned_lows[
                    -1
                ]

                if not aligned(
                    float(
                        x.low.iloc[l1]
                    ),
                    float(
                        x.low.iloc[l2]
                    ),
                ):
                    continue

                liquidity_level = min(
                    float(
                        x.low.iloc[l1]
                    ),
                    float(
                        x.low.iloc[l2]
                    ),
                )

                break_i = None

                last_break = min(
                    len(x) - 1,
                    l2
                    + MAX_LIQUIDITY_DISTANCE_BARS,
                )

                for b in range(
                    l2 + 1,
                    last_break + 1,
                ):

                    if (
                        float(
                            x.low.iloc[b]
                        )
                        < liquidity_level
                    ):

                        break_i = b
                        break

                if break_i is None:
                    continue

                midpoint = (
                    float(
                        x.low.iloc[ob_i]
                    )
                    +
                    float(
                        x.high.iloc[ob_i]
                    )
                ) / 2.0

                retest_i = None

                for e in range(
                    break_i + 1,
                    len(x),
                ):

                    candle_low = float(
                        x.low.iloc[e]
                    )

                    candle_high = float(
                        x.high.iloc[e]
                    )

                    if (
                        candle_low
                        <= midpoint
                        <= candle_high
                    ):

                        retest_i = e
                        break

                if retest_i is None:
                    continue

                entry_i = (
                    retest_i + 1
                )

                if (
                    entry_i
                    >= len(x)
                ):
                    continue

                entry = float(
                    x.open.iloc[
                        entry_i
                    ]
                )

                sl = (
                    float(
                        x.low.iloc[
                            ob_i
                        ]
                    )
                    * (
                        1.0
                        - SL_BUFFER_PCT
                    )
                )

                risk = (
                    entry
                    - sl
                )

                if risk <= 0:
                    continue

                risk_pct = (
                    risk
                    / entry
                )

                if not (
                    MIN_RISK_PCT
                    <= risk_pct
                    <= MAX_RISK_PCT
                ):
                    continue

                tp = (
                    entry
                    + RR * risk
                )

                if l2 <= ob_i:
                    continue

                structural_target = float(
                    x.high.iloc[
                        ob_i:
                        l2 + 1
                    ].max()
                )

                # LONG structural viability.
                if (
                    tp
                    > structural_target
                ):
                    continue

                candidates.append(
                    {
                        "symbol":
                            symbol,

                        "side":
                            "LONG",

                        "zone_pivot_i":
                            zone["pivot_i"],

                        "zone_confirm_i":
                            zone["confirm_i"],

                        "zone_pivot_time":
                            zone["pivot_time"],

                        "zone_confirm_time":
                            zone["confirm_time"],

                        "touch_i":
                            touch_i,

                        "ob_i":
                            ob_i,

                        "align1_i":
                            l1,

                        "align2_i":
                            l2,

                        "break_i":
                            break_i,

                        "retest_i":
                            retest_i,

                        "entry_i":
                            entry_i,

                        "entry":
                            entry,

                        "sl":
                            sl,

                        "tp":
                            tp,

                        "structural_target":
                            structural_target,
                    }
                )

                break

    candidates.sort(
        key=lambda item: (
            item["entry_i"],
            item["side"],
        )
    )

    return candidates


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate(
    symbol,
    x,
    candidates,
):
    """
    Per-symbol overlap rule:

    - maximum one open trade per symbol
    - different symbols can overlap
    - no same-candle re-entry
    - entry candle is not checked for SL/TP
    - SL wins when both SL and TP are touched
    - no timeout
    """

    closed = []

    unresolved = []

    last_exit_i = -1

    for candidate in candidates:

        entry_i = candidate[
            "entry_i"
        ]

        if (
            entry_i
            <= last_exit_i
        ):
            continue

        exit_i = None

        result = None

        exit_price = None

        # Never check entry candle.
        for j in range(
            entry_i + 1,
            len(x),
        ):

            high = float(
                x.high.iloc[j]
            )

            low = float(
                x.low.iloc[j]
            )

            if (
                candidate["side"]
                == "LONG"
            ):

                hit_sl = (
                    low
                    <= candidate["sl"]
                )

                hit_tp = (
                    high
                    >= candidate["tp"]
                )

            else:

                hit_sl = (
                    high
                    >= candidate["sl"]
                )

                hit_tp = (
                    low
                    <= candidate["tp"]
                )

            if (
                hit_sl
                or hit_tp
            ):

                exit_i = j

                # Conservative rule.
                if hit_sl:

                    result = "LOSS"

                    exit_price = (
                        candidate["sl"]
                    )

                else:

                    result = "WIN"

                    exit_price = (
                        candidate["tp"]
                    )

                break

        if exit_i is None:

            unresolved.append(
                {
                    **candidate,

                    "exit_i":
                        None,

                    "result":
                        "UNRESOLVED",

                    "entry_time":
                        x.time.iloc[
                            entry_i
                        ],
                }
            )

            # No artificial timeout.
            # Same symbol remains blocked.
            last_exit_i = (
                len(x) - 1
            )

            continue

        if (
            candidate["side"]
            == "LONG"
        ):

            gross_return = (
                exit_price
                - candidate["entry"]
            ) / candidate["entry"]

        else:

            gross_return = (
                candidate["entry"]
                - exit_price
            ) / candidate["entry"]

        gross_pnl = (
            NOTIONAL
            * gross_return
        )

        fees = (
            NOTIONAL
            * FEE_RATE
            * 2.0
        )

        pnl = (
            gross_pnl
            - fees
        )

        risk_dollars = (
            NOTIONAL
            * abs(
                candidate["entry"]
                - candidate["sl"]
            )
            / candidate["entry"]
        )

        r_multiple = (
            pnl / risk_dollars
            if risk_dollars > 0
            else np.nan
        )

        closed.append(
            {
                **candidate,

                "exit_i":
                    exit_i,

                "exit_price":
                    exit_price,

                "result":
                    result,

                "pnl":
                    pnl,

                "r_multiple":
                    r_multiple,

                "entry_time":
                    x.time.iloc[
                        entry_i
                    ],

                "exit_time":
                    x.time.iloc[
                        exit_i
                    ],
            }
        )

        last_exit_i = exit_i

    return (
        closed,
        unresolved,
    )


# ============================================================
# STATISTICS
# ============================================================

def stats(trades):

    if not trades:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "wr": 0.0,
            "pf": 0.0,
            "net_R": 0.0,
            "pnl": 0.0,
            "max_streak": 0,
        }

    wins = sum(
        t["result"] == "WIN"
        for t in trades
    )

    losses = sum(
        t["result"] == "LOSS"
        for t in trades
    )

    gross_profit = sum(
        max(
            0.0,
            float(t["pnl"]),
        )
        for t in trades
    )

    gross_loss = sum(
        min(
            0.0,
            float(t["pnl"]),
        )
        for t in trades
    )

    streak = 0

    max_streak = 0

    for trade in sorted(
        trades,
        key=lambda t: t[
            "exit_time"
        ],
    ):

        if (
            trade["result"]
            == "LOSS"
        ):

            streak += 1

            max_streak = max(
                max_streak,
                streak,
            )

        else:

            streak = 0

    return {
        "trades":
            len(trades),

        "wins":
            wins,

        "losses":
            losses,

        "wr":
            (
                100.0
                * wins
                / len(trades)
            ),

        "pf":
            (
                gross_profit
                / abs(gross_loss)
                if gross_loss < 0
                else (
                    float("inf")
                    if gross_profit > 0
                    else 0.0
                )
            ),

        "net_R":
            sum(
                float(
                    t["r_multiple"]
                )
                for t in trades
            ),

        "pnl":
            sum(
                float(t["pnl"])
                for t in trades
            ),

        "max_streak":
            max_streak,
    }


def max_drawdown(
    trades
):
    equity = INITIAL_CAPITAL

    peak = equity

    max_dd = 0.0

    for trade in sorted(
        trades,
        key=lambda t: t[
            "exit_time"
        ],
    ):

        equity += float(
            trade["pnl"]
        )

        peak = max(
            peak,
            equity,
        )

        max_dd = max(
            max_dd,
            peak - equity,
        )

    return max_dd


# ============================================================
# SPLIT CLASSIFICATION
# ============================================================

def classify_entry_time(
    entry_time
):
    t = pd.Timestamp(
        entry_time
    )

    if t.tzinfo is None:

        t = t.tz_localize(
            "UTC"
        )

    else:

        t = t.tz_convert(
            "UTC"
        )

    if (
        RESEARCH_START
        <= t
        < RESEARCH_MID
    ):
        return "Discovery"

    if (
        RESEARCH_MID
        <= t
        < OOS_START
    ):
        return "Development"

    if (
        OOS_START
        <= t
        <= OOS_END
    ):
        return "Validation_OOS"

    return "Outside"


# ============================================================
# SPLIT REPORT
# ============================================================

def split_report(
    closed_trades,
    unresolved,
):
    groups = {
        "Discovery": [],
        "Development": [],
        "Validation_OOS": [],
    }

    for trade in closed_trades:

        split = classify_entry_time(
            trade["entry_time"]
        )

        if split in groups:

            groups[
                split
            ].append(
                trade
            )

    rows = []

    for split_name in [
        "Discovery",
        "Development",
        "Validation_OOS",
    ]:

        result = stats(
            groups[
                split_name
            ]
        )

        unresolved_count = sum(
            1
            for trade in unresolved
            if (
                classify_entry_time(
                    trade[
                        "entry_time"
                    ]
                )
                == split_name
            )
        )

        rows.append(
            {
                "split":
                    split_name,

                "trades":
                    result["trades"],

                "wins":
                    result["wins"],

                "losses":
                    result["losses"],

                "WR_%":
                    result["wr"],

                "PF":
                    result["pf"],

                "net_R":
                    result["net_R"],

                "PnL_$":
                    result["pnl"],

                "max_streak":
                    result[
                        "max_streak"
                    ],

                "unresolved":
                    unresolved_count,
            }
        )

    total = stats(
        closed_trades
    )

    rows.append(
        {
            "split":
                "TOTAL",

            "trades":
                total["trades"],

            "wins":
                total["wins"],

            "losses":
                total["losses"],

            "WR_%":
                total["wr"],

            "PF":
                total["pf"],

            "net_R":
                total["net_R"],

            "PnL_$":
                total["pnl"],

            "max_streak":
                total["max_streak"],

            "unresolved":
                len(unresolved),
        }
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

def diagnostic_rows(
    symbol,
    x,
    zones,
    candidates,
    closed,
    unresolved,
):
    rows = []

    split_ranges = [
        (
            "Discovery",
            RESEARCH_START,
            RESEARCH_MID,
        ),
        (
            "Development",
            RESEARCH_MID,
            OOS_START,
        ),
        (
            "Validation_OOS",
            OOS_START,
            OOS_END
            + pd.Timedelta(
                minutes=1
            ),
        ),
    ]

    for (
        split_name,
        start,
        end,
    ) in split_ranges:

        split_candidates = [
            c
            for c in candidates
            if (
                start
                <= x.time.iloc[
                    c["entry_i"]
                ]
                < end
            )
        ]

        split_closed = [
            t
            for t in closed
            if (
                start
                <= t["entry_time"]
                < end
            )
        ]

        split_unresolved = [
            t
            for t in unresolved
            if (
                start
                <= t["entry_time"]
                < end
            )
        ]

        split_zones = [
            z
            for z in zones
            if (
                start
                <= z["confirm_time"]
                < end
            )
        ]

        candidate_times = [
            x.time.iloc[
                c["entry_i"]
            ]
            for c
            in split_candidates
        ]

        rows.append(
            {
                "symbol":
                    symbol,

                "split":
                    split_name,

                "data_first":
                    str(
                        x.time.iloc[0]
                    ),

                "data_last":
                    str(
                        x.time.iloc[-1]
                    ),

                "candles":
                    len(x),

                "htf_zones":
                    len(
                        split_zones
                    ),

                "candidates":
                    len(
                        split_candidates
                    ),

                "closed":
                    len(
                        split_closed
                    ),

                "unresolved":
                    len(
                        split_unresolved
                    ),

                "earliest_candidate":
                    (
                        str(
                            min(
                                candidate_times
                            )
                        )
                        if candidate_times
                        else ""
                    ),

                "latest_candidate":
                    (
                        str(
                            max(
                                candidate_times
                            )
                        )
                        if candidate_times
                        else ""
                    ),
            }
        )

    return rows


def monthly_candidate_rows(
    symbol,
    x,
    candidates,
):
    rows = []

    for candidate in candidates:

        t = pd.Timestamp(
            x.time.iloc[
                candidate["entry_i"]
            ]
        )

        split = classify_entry_time(
            t
        )

        if split == "Outside":
            continue

        rows.append(
            {
                "symbol":
                    symbol,

                "split":
                    split,

                "entry_time":
                    t,

                "year_month":
                    t.strftime(
                        "%Y-%m"
                    ),

                "side":
                    candidate["side"],
            }
        )

    return rows


# ============================================================
# FINAL INTEGRITY AUDIT
# ============================================================

def audit(
    closed_trades,
    unresolved,
):
    errors = []

    all_trades = (
        list(closed_trades)
        + list(unresolved)
    )

    for trade in all_trades:

        # ----------------------------------------------------
        # 1. Structural timestamps
        # ----------------------------------------------------

        zone_pivot_time = pd.Timestamp(
            trade[
                "zone_pivot_time"
            ]
        )

        zone_confirm_time = pd.Timestamp(
            trade[
                "zone_confirm_time"
            ]
        )

        touch_time = pd.Timestamp(
            trade[
                "touch_time"
            ]
        )

        ob_time = pd.Timestamp(
            trade[
                "ob_time"
            ]
        )

        align1_time = pd.Timestamp(
            trade[
                "align1_time"
            ]
        )

        align2_time = pd.Timestamp(
            trade[
                "align2_time"
            ]
        )

        break_time = pd.Timestamp(
            trade[
                "break_time"
            ]
        )

        retest_time = pd.Timestamp(
            trade[
                "retest_time"
            ]
        )

        entry_time = pd.Timestamp(
            trade[
                "entry_time"
            ]
        )

        structural_times = [
            zone_pivot_time,
            zone_confirm_time,
            touch_time,
            ob_time,
            align1_time,
            align2_time,
            break_time,
            retest_time,
        ]

        # Zone pivot itself precedes confirmation.
        if not (
            zone_pivot_time
            < zone_confirm_time
        ):

            errors.append(
                "HTF pivot confirmation "
                "ordering violation"
            )

        # Every structure event must precede entry.
        for structural_time in (
            structural_times
        ):

            if not (
                structural_time
                < entry_time
            ):

                errors.append(
                    "future structural "
                    "information"
                )

        # Explicit sequence.
        ordered = [
            zone_confirm_time,
            touch_time,
            ob_time,
            align1_time,
            align2_time,
            break_time,
            retest_time,
            entry_time,
        ]

        if not all(
            a < b
            for a, b in zip(
                ordered,
                ordered[1:],
            )
        ):

            errors.append(
                "structural sequence "
                "ordering violation"
            )

        # ----------------------------------------------------
        # 2. HTF confirmation delay
        # ----------------------------------------------------

        expected_confirm = (
            zone_pivot_time
            + pd.Timedelta(
                hours=4 * PIVOT
            )
        )

        if (
            zone_confirm_time
            != expected_confirm
        ):

            errors.append(
                "HTF confirmation "
                "timing violation"
            )

        # ----------------------------------------------------
        # 3. Entry after retest
        # ----------------------------------------------------

        if not (
            entry_time
            > retest_time
        ):

            errors.append(
                "entry not after "
                "completed retest"
            )

        # ----------------------------------------------------
        # 4. RR
        # ----------------------------------------------------

        entry = float(
            trade["entry"]
        )

        sl = float(
            trade["sl"]
        )

        tp = float(
            trade["tp"]
        )

        risk = abs(
            entry - sl
        )

        if risk <= 0:

            errors.append(
                "invalid risk"
            )

        expected_distance = (
            RR * risk
        )

        actual_distance = abs(
            tp - entry
        )

        tolerance = max(
            1e-9,
            abs(entry)
            * 1e-8,
        )

        if (
            abs(
                actual_distance
                - expected_distance
            )
            > tolerance
        ):

            errors.append(
                "RR mismatch"
            )

        # ----------------------------------------------------
        # 5. Structural target viability
        # ----------------------------------------------------

        structural_target = float(
            trade[
                "structural_target"
            ]
        )

        if trade["side"] == "LONG":

            if tp > structural_target:

                errors.append(
                    "LONG TP beyond "
                    "structural target"
                )

        else:

            if tp < structural_target:

                errors.append(
                    "SHORT TP beyond "
                    "structural target"
                )

    # --------------------------------------------------------
    # 6. Same-symbol overlap
    # --------------------------------------------------------

    ordered_trades = sorted(
        closed_trades,
        key=lambda t: (
            t["symbol"],
            t["entry_time"],
        ),
    )

    for previous, current in zip(
        ordered_trades,
        ordered_trades[1:],
    ):

        if (
            previous["symbol"]
            != current["symbol"]
        ):
            continue

        previous_exit = pd.Timestamp(
            previous["exit_time"]
        )

        current_entry = pd.Timestamp(
            current["entry_time"]
        )

        if (
            current_entry
            <= previous_exit
        ):

            errors.append(
                "same-symbol overlap"
            )

        if (
            current_entry
            == previous_exit
        ):

            errors.append(
                "same-candle re-entry"
            )

    return sorted(
        set(errors)
    )


# ============================================================
# ENRICH TRADE TIMES
# ============================================================

def enrich_trade_times(
    trade,
    x,
):
    """
    Add exact 15m timestamps for all structural indices.

    This prevents mixing 4H and 15m integer indices during audit.
    """

    result = dict(
        trade
    )

    result["touch_time"] = (
        x.time.iloc[
            trade["touch_i"]
        ]
    )

    result["ob_time"] = (
        x.time.iloc[
            trade["ob_i"]
        ]
    )

    result["align1_time"] = (
        x.time.iloc[
            trade["align1_i"]
        ]
    )

    result["align2_time"] = (
        x.time.iloc[
            trade["align2_i"]
        ]
    )

    result["break_time"] = (
        x.time.iloc[
            trade["break_i"]
        ]
    )

    result["retest_time"] = (
        x.time.iloc[
            trade["retest_i"]
        ]
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    all_closed = []

    all_unresolved = []

    audit_rows = []

    diagnostic_all = []

    monthly_candidates = []

    # --------------------------------------------------------
    # SYMBOL LOOP
    # --------------------------------------------------------

    for symbol in SYMBOLS:

        print(
            f"[DATA] Fetching "
            f"{symbol}"
        )

        df = fetch_symbol(
            symbol
        )

        print(
            f"[DATA] {symbol}: "
            f"{len(df)} candles | "
            f"{df.time.iloc[0]} -> "
            f"{df.time.iloc[-1]}"
        )

        # ----------------------------------------------------
        # CRITICAL V2 FIX:
        #
        # 15m structure needs its own confirmed pivots.
        # ----------------------------------------------------

        x = confirmed_pivots(
            df
        )

        # ----------------------------------------------------
        # 4H context
        # ----------------------------------------------------

        htf = make_htf(
            df
        )

        zones = build_zones(
            htf
        )

        # ----------------------------------------------------
        # Candidate generation
        # ----------------------------------------------------

        candidates = make_candidates(
            symbol,
            x,
            htf,
            zones,
        )

        # ----------------------------------------------------
        # Simulation
        # ----------------------------------------------------

        closed, unresolved = simulate(
            symbol,
            x,
            candidates,
        )

        # ----------------------------------------------------
        # Enrich exact structural timestamps.
        # ----------------------------------------------------

        closed = [
            enrich_trade_times(
                trade,
                x,
            )
            for trade in closed
        ]

        unresolved = [
            enrich_trade_times(
                trade,
                x,
            )
            for trade in unresolved
        ]

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        diagnostic_all.extend(
            diagnostic_rows(
                symbol,
                x,
                zones,
                candidates,
                closed,
                unresolved,
            )
        )

        monthly_candidates.extend(
            monthly_candidate_rows(
                symbol,
                x,
                candidates,
            )
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        all_closed.extend(
            closed
        )

        all_unresolved.extend(
            unresolved
        )

        audit_rows.append(
            {
                "symbol":
                    symbol,

                "rows":
                    len(x),

                "first":
                    str(
                        x.time.iloc[0]
                    ),

                "last":
                    str(
                        x.time.iloc[-1]
                    ),

                "htf_zones":
                    len(zones),

                "candidates":
                    len(candidates),

                "closed":
                    len(closed),

                "unresolved":
                    len(unresolved),
            }
        )

        print(
            f"[RESULT] {symbol}: "
            f"4H_zones={len(zones)} "
            f"candidates={len(candidates)} "
            f"closed={len(closed)} "
            f"unresolved={len(unresolved)}"
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    all_closed.sort(
        key=lambda t: (
            t["entry_time"],
            t["symbol"],
        )
    )

    all_unresolved.sort(
        key=lambda t: (
            t["entry_time"],
            t["symbol"],
        )
    )

    # --------------------------------------------------------
    # FINAL INTEGRITY AUDIT
    # --------------------------------------------------------

    integrity_errors = audit(
        all_closed,
        all_unresolved,
    )

    if integrity_errors:

        raise RuntimeError(
            "FINAL INTEGRITY AUDIT FAILED: "
            + "; ".join(
                integrity_errors
            )
        )

    # --------------------------------------------------------
    # SPLIT REPORT
    # --------------------------------------------------------

    report = split_report(
        all_closed,
        all_unresolved,
    )

    # --------------------------------------------------------
    # OUTPUT CSVs
    # --------------------------------------------------------

    report.to_csv(
        OUT
        / "setup4_v2_split_report.csv",
        index=False,
    )

    pd.DataFrame(
        audit_rows
    ).to_csv(
        OUT
        / "setup4_v2_data_audit.csv",
        index=False,
    )

    pd.DataFrame(
        diagnostic_all
    ).to_csv(
        OUT
        / "setup4_v2_diagnostics.csv",
        index=False,
    )

    pd.DataFrame(
        monthly_candidates
    ).to_csv(
        OUT
        / "setup4_v2_candidates_by_month.csv",
        index=False,
    )

    if all_closed:

        pd.DataFrame(
            all_closed
        ).to_csv(
            OUT
            / "setup4_v2_trade_ledger.csv",
            index=False,
        )

    if all_unresolved:

        pd.DataFrame(
            all_unresolved
        ).to_csv(
            OUT
            / "setup4_v2_unresolved.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Overall stats
    # --------------------------------------------------------

    result = stats(
        all_closed
    )

    dd = max_drawdown(
        all_closed
    )

    # --------------------------------------------------------
    # Candidate counts
    # --------------------------------------------------------

    candidate_counts = {
        "Discovery": 0,
        "Development": 0,
        "Validation_OOS": 0,
    }

    for row in diagnostic_all:

        candidate_counts[
            row["split"]
        ] += int(
            row["candidates"]
        )

    # --------------------------------------------------------
    # TEXT REPORT
    # --------------------------------------------------------

    output = []

    output.append(
        "SETUP 4 V2 — BACKTEST REPORT"
    )

    output.append("")

    output.append(
        "RESEARCH WINDOW"
    )

    output.append(
        f"Research Start: "
        f"{RESEARCH_START}"
    )

    output.append(
        f"Research End: "
        f"{RESEARCH_END}"
    )

    output.append(
        f"Discovery End: "
        f"{RESEARCH_MID}"
    )

    output.append("")

    output.append(
        "FRESH OOS — LOCKED"
    )

    output.append(
        f"OOS Start: "
        f"{OOS_START}"
    )

    output.append(
        f"OOS End: "
        f"{OOS_END}"
    )

    output.append("")

    output.append(
        "CAPITAL"
    )

    output.append(
        f"Initial Capital: "
        f"${INITIAL_CAPITAL:.2f}"
    )

    output.append(
        f"Margin: "
        f"${MARGIN:.2f}"
    )

    output.append(
        f"Leverage: "
        f"{LEVERAGE:.1f}x"
    )

    output.append(
        f"Notional: "
        f"${NOTIONAL:.2f}"
    )

    output.append("")

    output.append(
        "CANDIDATES BY SPLIT"
    )

    output.append(
        f"Discovery: "
        f"{candidate_counts['Discovery']}"
    )

    output.append(
        f"Development: "
        f"{candidate_counts['Development']}"
    )

    output.append(
        f"Validation_OOS: "
        f"{candidate_counts['Validation_OOS']}"
    )

    output.append("")

    output.append(
        "OVERALL CLOSED-TRADE RESULT"
    )

    output.append(
        f"Trades: "
        f"{result['trades']}"
    )

    output.append(
        f"Wins: "
        f"{result['wins']}"
    )

    output.append(
        f"Losses: "
        f"{result['losses']}"
    )

    output.append(
        f"WR: "
        f"{result['wr']:.2f}%"
    )

    output.append(
        f"PF: "
        f"{result['pf']:.6f}"
    )

    output.append(
        f"net_R: "
        f"{result['net_R']:.6f}"
    )

    output.append(
        f"PnL: "
        f"${result['pnl']:.2f}"
    )

    output.append(
        f"Final Equity: "
        f"${INITIAL_CAPITAL + result['pnl']:.2f}"
    )

    output.append(
        f"Max losing streak: "
        f"{result['max_streak']}"
    )

    output.append(
        f"Max DD: "
        f"${dd:.2f}"
    )

    output.append(
        f"Unresolved: "
        f"{len(all_unresolved)}"
    )

    output.append("")

    output.append(
        "FINAL INTEGRITY AUDIT"
    )

    output.append(
        "Data gaps: PASSED"
    )

    output.append(
        "Structural causality: PASSED"
    )

    output.append(
        "Future target leak: PASSED"
    )

    output.append(
        "Same-symbol overlap: PASSED"
    )

    output.append(
        "Same-candle re-entry: PASSED"
    )

    output.append(
        "RR 1:2 consistency: PASSED"
    )

    output.append(
        "AUDIT STATUS: PASSED"
    )

    output.append("")

    if not report.empty:

        output.append(
            report.to_string(
                index=False
            )
        )

    else:

        output.append(
            "No closed trades."
        )

    report_text = "\n".join(
        output
    )

    (
        OUT
        / "setup4_v2_report.txt"
    ).write_text(
        report_text,
        encoding="utf-8",
    )

    print("")
    print(report_text)


if __name__ == "__main__":
    main()
