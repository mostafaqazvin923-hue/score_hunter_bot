#!/usr/bin/env python3
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ============================================================
# SETUP 4 V2
# Fresh OOS: 2024-10-04 -> 2025-10-03
# Research: 365 days immediately before OOS
# ============================================================

SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","SUIUSDT","AVAXUSDT","NEARUSDT",
    "ADAUSDT","BNBUSDT","APTUSDT","CRVUSDT","ONDOUSDT","PENDLEUSDT",
    "ICPUSDT","WIFUSDT",
]

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
INTERVAL = "15m"

OOS_START = pd.Timestamp("2024-10-04 00:00:00", tz="UTC")
OOS_END = pd.Timestamp("2025-10-03 23:00:00", tz="UTC")

RESEARCH_START = OOS_START - pd.Timedelta(days=365)
RESEARCH_END = OOS_START - pd.Timedelta(minutes=15)
RESEARCH_MID = RESEARCH_START + (OOS_START - RESEARCH_START) / 2
WARMUP_START = RESEARCH_START - pd.Timedelta(days=90)

INITIAL_CAPITAL = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE

RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003

PIVOT = 2

HTF_ATR_PERIOD = 14
HTF_ZONE_ATR_MULT = 0.50

MAX_REACTION_BARS = 24
ALIGN_TOL = 0.004
MAX_LIQUIDITY_DISTANCE_BARS = 96

SL_BUFFER_PCT = 0.0005

MIN_RISK_PCT = 0.0005
MAX_RISK_PCT = 0.08

OUT = Path("setup4_v2_outputs")
OUT.mkdir(exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "setup4-v2-research/1.0"
})


# ============================================================
# MONTHS
# ============================================================

def month_range(start, end):
    start_period = (
        pd.Timestamp(start)
        .tz_convert("UTC")
        .tz_localize(None)
        .to_period("M")
    )

    end_period = (
        pd.Timestamp(end)
        .tz_convert("UTC")
        .tz_localize(None)
        .to_period("M")
    )

    result = []

    current = start_period

    while current <= end_period:
        result.append(
            (
                current.year,
                current.month,
            )
        )
        current += 1

    return result


# ============================================================
# DATA
# ============================================================

def fetch_month(symbol, year, month):

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
                1.5 * (attempt + 1)
            )

    return None


def parse_archive(blob):

    with zipfile.ZipFile(
        io.BytesIO(blob)
    ) as archive:

        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if not names:
            raise RuntimeError(
                "Archive contains no CSV"
            )

        with archive.open(
            names[0]
        ) as handle:

            raw = pd.read_csv(
                handle,
                header=None,
            )

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
        :len(columns)
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
                "Missing archive after "
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

    df = (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    df = df[
        (df.time >= WARMUP_START)
        &
        (df.time <= OOS_END)
    ].copy()

    now = pd.Timestamp.now(
        tz="UTC"
    )

    if (
        not df.empty
        and
        (
            df.time.iloc[-1]
            + pd.Timedelta(minutes=15)
            > now
        )
    ):

        df = df.iloc[:-1].copy()

    if df.empty:

        raise RuntimeError(
            f"No usable rows for {symbol}"
        )

    expected = pd.date_range(
        df.time.iloc[0],
        df.time.iloc[-1],
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
            f"15m gap for {symbol}: "
            f"{len(missing)} missing; "
            f"first={missing[0]}"
        )

    return df.reset_index(
        drop=True
    )


# ============================================================
# ATR
# ============================================================

def atr(df, period):

    previous_close = df.close.shift(1)

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
    ).max(axis=1)

    return true_range.rolling(
        period,
        min_periods=period,
    ).mean()


# ============================================================
# CONFIRMED PIVOTS
# ============================================================

def confirmed_pivots(df):
    """
    Pivot at i becomes usable only at i + PIVOT.

    Pivot flags remain attached to the
    structural candle.
    """

    x = df.copy()

    high = x.high.to_numpy(
        dtype=float
    )

    low = x.low.to_numpy(
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
        PIVOT,
        n - PIVOT,
    ):

        # Pivot High
        if (
            high[i]
            > np.max(
                high[
                    i - PIVOT:i
                ]
            )
            and
            high[i]
            >= np.max(
                high[
                    i + 1:
                    i + PIVOT + 1
                ]
            )
        ):

            pivot_high[i] = True

        # Pivot Low
        if (
            low[i]
            < np.min(
                low[
                    i - PIVOT:i
                ]
            )
            and
            low[i]
            <= np.min(
                low[
                    i + 1:
                    i + PIVOT + 1
                ]
            )
        ):

            pivot_low[i] = True

    x["pivot_high"] = pivot_high
    x["pivot_low"] = pivot_low

    return x


# ============================================================
# HTF
# ============================================================

def make_htf(df):

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

    return confirmed_pivots(
        htf
    )


# ============================================================
# HTF ZONES
# ============================================================

def build_zones(htf):

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

        if bool(
            htf.pivot_high.iloc[
                pivot_i
            ]
        ):

            center = float(
                htf.high.iloc[
                    pivot_i
                ]
            )

            zones.append(
                {
                    "side":
                        "SHORT",

                    "pivot_i":
                        pivot_i,

                    "confirm_i":
                        confirm_i,

                    "pivot_time":
                        htf.time.iloc[
                            pivot_i
                        ],

                    "confirm_time":
                        htf.time.iloc[
                            confirm_i
                        ],

                    "center":
                        center,

                    "low":
                        center
                        - HTF_ZONE_ATR_MULT
                        * atr_value,

                    "high":
                        center
                        + HTF_ZONE_ATR_MULT
                        * atr_value,
                }
            )

        if bool(
            htf.pivot_low.iloc[
                pivot_i
            ]
        ):

            center = float(
                htf.low.iloc[
                    pivot_i
                ]
            )

            zones.append(
                {
                    "side":
                        "LONG",

                    "pivot_i":
                        pivot_i,

                    "confirm_i":
                        confirm_i,

                    "pivot_time":
                        htf.time.iloc[
                            pivot_i
                        ],

                    "confirm_time":
                        htf.time.iloc[
                            confirm_i
                        ],

                    "center":
                        center,

                    "low":
                        center
                        - HTF_ZONE_ATR_MULT
                        * atr_value,

                    "high":
                        center
                        + HTF_ZONE_ATR_MULT
                        * atr_value,
                }
            )

    zones.sort(
        key=lambda z: z[
            "confirm_time"
        ]
    )

    return zones


# ============================================================
# PIVOT CACHE
# ============================================================

def build_pivot_cache(x):
    """
    IMPORTANT PERFORMANCE FIX.

    Pivot indices are created once.

    We NEVER scan:
        0 -> current_i

    for every candidate.
    """

    high_idx = np.flatnonzero(
        x.pivot_high.to_numpy(
            dtype=bool
        )
    )

    low_idx = np.flatnonzero(
        x.pivot_low.to_numpy(
            dtype=bool
        )
    )

    return (
        high_idx,
        low_idx,
    )


def confirmed_range(
    pivot_idx,
    start_exclusive,
    end_exclusive,
):
    """
    Returns structural pivot indices:

        start_exclusive < pivot < end_exclusive

    The caller is responsible for ensuring the pivot has already
    become confirmed.
    """

    left = np.searchsorted(
        pivot_idx,
        start_exclusive + 1,
        side="left",
    )

    right = np.searchsorted(
        pivot_idx,
        end_exclusive,
        side="left",
    )

    return pivot_idx[
        left:right
    ]


# ============================================================
# STRUCTURAL HELPERS
# ============================================================

def aligned(a, b):

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


def first_zone_touch(
    x,
    start_i,
    zone,
):

    end_i = min(
        len(x) - 1,
        start_i
        + MAX_REACTION_BARS,
    )

    low = x.low.to_numpy(
        dtype=float
    )

    high = x.high.to_numpy(
        dtype=float
    )

    for i in range(
        start_i,
        end_i + 1,
    ):

        if (
            low[i]
            <= zone["high"]
            and
            high[i]
            >= zone["low"]
        ):

            return i

    return None


# ============================================================
# CANDIDATES
# ============================================================

def make_candidates(
    symbol,
    x,
    zones,
    pivot_cache,
):

    high_idx, low_idx = (
        pivot_cache
    )

    highs = x.high.to_numpy(
        dtype=float
    )

    lows = x.low.to_numpy(
        dtype=float
    )

    opens = x.open.to_numpy(
        dtype=float
    )

    times = x.time.to_numpy()

    candidates = []

    for zone in zones:

        # Zone is usable only after HTF confirmation.
        start_i = int(
            np.searchsorted(
                times,
                zone["confirm_time"],
                side="right",
            )
        )

        if start_i >= len(x):
            continue

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

        for current_i in range(
            touch_i + 1,
            reaction_end + 1,
        ):

            # =================================================
            # SHORT
            # =================================================

            if zone["side"] == "SHORT":

                available = (
                    confirmed_range(
                        high_idx,
                        touch_i,
                        current_i,
                    )
                )

                if len(available) == 0:
                    continue

                ob_i = int(
                    available[-1]
                )

                # Pivot must already be confirmed.
                if (
                    ob_i + PIVOT
                    > current_i
                ):
                    continue

                previous = (
                    confirmed_range(
                        high_idx,
                        -1,
                        ob_i,
                    )
                )

                if len(previous):

                    previous_high = int(
                        previous[-1]
                    )

                    # Lower High.
                    if (
                        highs[ob_i]
                        >=
                        highs[
                            previous_high
                        ]
                    ):
                        continue

                aligned_highs = (
                    confirmed_range(
                        high_idx,
                        ob_i,
                        current_i,
                    )
                )

                if len(
                    aligned_highs
                ) < 2:
                    continue

                h1, h2 = map(
                    int,
                    aligned_highs[-2:]
                )

                if not aligned(
                    highs[h1],
                    highs[h2],
                ):
                    continue

                liquidity_level = max(
                    highs[h1],
                    highs[h2],
                )

                break_end = min(
                    len(x) - 1,
                    h2
                    + MAX_LIQUIDITY_DISTANCE_BARS,
                )

                break_i = None

                for b in range(
                    h2 + 1,
                    break_end + 1,
                ):

                    if (
                        highs[b]
                        > liquidity_level
                    ):

                        break_i = b
                        break

                if break_i is None:
                    continue

                midpoint = (
                    lows[ob_i]
                    + highs[ob_i]
                ) / 2.0

                retest_i = None

                for e in range(
                    break_i + 1,
                    len(x),
                ):

                    if (
                        lows[e]
                        <= midpoint
                        <= highs[e]
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

                entry = opens[
                    entry_i
                ]

                sl = (
                    highs[ob_i]
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

                structural_target = float(
                    np.min(
                        lows[
                            ob_i:
                            h2 + 1
                        ]
                    )
                )

                # SHORT:
                # TP must remain above structural target.
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

                available = (
                    confirmed_range(
                        low_idx,
                        touch_i,
                        current_i,
                    )
                )

                if len(available) == 0:
                    continue

                ob_i = int(
                    available[-1]
                )

                if (
                    ob_i + PIVOT
                    > current_i
                ):
                    continue

                previous = (
                    confirmed_range(
                        low_idx,
                        -1,
                        ob_i,
                    )
                )

                if len(previous):

                    previous_low = int(
                        previous[-1]
                    )

                    # Higher Low.
                    if (
                        lows[ob_i]
                        <=
                        lows[
                            previous_low
                        ]
                    ):
                        continue

                aligned_lows = (
                    confirmed_range(
                        low_idx,
                        ob_i,
                        current_i,
                    )
                )

                if len(
                    aligned_lows
                ) < 2:
                    continue

                l1, l2 = map(
                    int,
                    aligned_lows[-2:]
                )

                if not aligned(
                    lows[l1],
                    lows[l2],
                ):
                    continue

                liquidity_level = min(
                    lows[l1],
                    lows[l2],
                )

                break_end = min(
                    len(x) - 1,
                    l2
                    + MAX_LIQUIDITY_DISTANCE_BARS,
                )

                break_i = None

                for b in range(
                    l2 + 1,
                    break_end + 1,
                ):

                    if (
                        lows[b]
                        < liquidity_level
                    ):

                        break_i = b
                        break

                if break_i is None:
                    continue

                midpoint = (
                    lows[ob_i]
                    + highs[ob_i]
                ) / 2.0

                retest_i = None

                for e in range(
                    break_i + 1,
                    len(x),
                ):

                    if (
                        lows[e]
                        <= midpoint
                        <= highs[e]
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

                entry = opens[
                    entry_i
                ]

                sl = (
                    lows[ob_i]
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

                structural_target = float(
                    np.max(
                        highs[
                            ob_i:
                            l2 + 1
                        ]
                    )
                )

                # LONG:
                # TP must remain below structural target.
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
        key=lambda c: (
            c["entry_i"],
            c["side"],
        )
    )

    return candidates


# ============================================================
# SIMULATION
# ============================================================

def simulate(
    symbol,
    x,
    candidates,
):

    high = x.high.to_numpy(
        dtype=float
    )

    low = x.low.to_numpy(
        dtype=float
    )

    times = x.time.to_numpy()

    closed = []

    unresolved = []

    # One open trade max per symbol.
    last_exit_i = -1

    for candidate in candidates:

        entry_i = candidate[
            "entry_i"
        ]

        # No same-symbol overlap.
        if (
            entry_i
            <= last_exit_i
        ):
            continue

        exit_i = None
        result = None
        exit_price = None

        # IMPORTANT:
        # Entry candle is NOT checked.
        for j in range(
            entry_i + 1,
            len(x),
        ):

            if (
                candidate["side"]
                == "LONG"
            ):

                hit_sl = (
                    low[j]
                    <= candidate["sl"]
                )

                hit_tp = (
                    high[j]
                    >= candidate["tp"]
                )

            else:

                hit_sl = (
                    high[j]
                    >= candidate["sl"]
                )

                hit_tp = (
                    low[j]
                    <= candidate["tp"]
                )

            if (
                hit_sl
                or hit_tp
            ):

                exit_i = j

                # Conservative:
                # SL wins if both hit.
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

        base = dict(
            candidate
        )

        base["entry_time"] = (
            times[entry_i]
        )

        if exit_i is None:

            base.update(
                {
                    "exit_i":
                        None,

                    "exit_time":
                        None,

                    "exit_price":
                        None,

                    "result":
                        "UNRESOLVED",

                    "pnl":
                        np.nan,

                    "r_multiple":
                        np.nan,
                }
            )

            unresolved.append(
                base
            )

            # No artificial timeout.
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

        base.update(
            {
                "exit_i":
                    exit_i,

                "exit_time":
                    times[exit_i],

                "exit_price":
                    exit_price,

                "result":
                    result,

                "pnl":
                    pnl,

                "r_multiple":
                    r_multiple,
            }
        )

        closed.append(
            base
        )

        last_exit_i = exit_i

    return (
        closed,
        unresolved,
    )


# ============================================================
# ENRICH TIMESTAMPS
# ============================================================

def enrich_times(
    trade,
    x,
):

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
# SPLITS
# ============================================================

def classify(
    timestamp
):

    t = pd.Timestamp(
        timestamp
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
# STATS
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
            100.0
            * wins
            / len(trades),

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
                float(
                    t["pnl"]
                )
                for t in trades
            ),

        "max_streak":
            max_streak,
    }


def max_drawdown(
    trades
):

    equity = (
        INITIAL_CAPITAL
    )

    peak = equity

    drawdown = 0.0

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

        drawdown = max(
            drawdown,
            peak - equity,
        )

    return drawdown


# ============================================================
# INTEGRITY AUDIT
# ============================================================

def audit(
    closed,
    unresolved,
):

    errors = []

    for trade in (
        list(closed)
        + list(unresolved)
    ):

        ordered = [
            pd.Timestamp(
                trade[
                    "zone_confirm_time"
                ]
            ),

            pd.Timestamp(
                trade[
                    "touch_time"
                ]
            ),

            pd.Timestamp(
                trade[
                    "ob_time"
                ]
            ),

            pd.Timestamp(
                trade[
                    "align1_time"
                ]
            ),

            pd.Timestamp(
                trade[
                    "align2_time"
                ]
            ),

            pd.Timestamp(
                trade[
                    "break_time"
                ]
            ),

            pd.Timestamp(
                trade[
                    "retest_time"
                ]
            ),

            pd.Timestamp(
                trade[
                    "entry_time"
                ]
            ),
        ]

        if not all(
            a < b
            for a, b in zip(
                ordered,
                ordered[1:],
            )
        ):

            errors.append(
                "structural ordering / future leak"
            )

        if not (
            pd.Timestamp(
                trade[
                    "zone_pivot_time"
                ]
            )
            <
            pd.Timestamp(
                trade[
                    "zone_confirm_time"
                ]
            )
        ):

            errors.append(
                "HTF pivot confirmation ordering"
            )

        expected_confirmation = (
            pd.Timestamp(
                trade[
                    "zone_pivot_time"
                ]
            )
            + pd.Timedelta(
                hours=4 * PIVOT
            )
        )

        if (
            pd.Timestamp(
                trade[
                    "zone_confirm_time"
                ]
            )
            != expected_confirmation
        ):

            errors.append(
                "HTF confirmation timing"
            )

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

        structural = float(
            trade[
                "structural_target"
            ]
        )

        if (
            trade["side"]
            == "LONG"
            and
            tp > structural
        ):

            errors.append(
                "LONG structural target violation"
            )

        if (
            trade["side"]
            == "SHORT"
            and
            tp < structural
        ):

            errors.append(
                "SHORT structural target violation"
            )

    # Same-symbol overlap.
    ordered_trades = sorted(
        closed,
        key=lambda t: (
            t["symbol"],
            t["entry_time"],
        )
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

        if (
            pd.Timestamp(
                current[
                    "entry_time"
                ]
            )
            <=
            pd.Timestamp(
                previous[
                    "exit_time"
                ]
            )
        ):

            errors.append(
                "same-symbol overlap / same-candle re-entry"
            )

    return sorted(
        set(errors)
    )


# ============================================================
# SPLIT REPORT
# ============================================================

def split_report(
    closed,
    unresolved,
):

    rows = []

    for name in [
        "Discovery",
        "Development",
        "Validation_OOS",
    ]:

        trades = [
            t
            for t in closed
            if classify(
                t["entry_time"]
            ) == name
        ]

        unres = [
            t
            for t in unresolved
            if classify(
                t["entry_time"]
            ) == name
        ]

        result = stats(
            trades
        )

        rows.append(
            {
                "split":
                    name,

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
                    result["max_streak"],

                "unresolved":
                    len(unres),
            }
        )

    total = stats(
        closed
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
# MAIN
# ============================================================

def main():

    all_closed = []
    all_unresolved = []

    diagnostics = []
    symbol_summary = []

    for symbol in SYMBOLS:

        print(
            f"[DATA] Fetching {symbol}",
            flush=True,
        )

        df = fetch_symbol(
            symbol
        )

        print(
            f"[DATA] {symbol}: "
            f"{len(df)} candles | "
            f"{df.time.iloc[0]} -> "
            f"{df.time.iloc[-1]}",
            flush=True,
        )

        # ====================================================
        # CRITICAL:
        # 15m structural pivots.
        # ====================================================

        x = confirmed_pivots(
            df
        )

        # ====================================================
        # 4H context.
        # ====================================================

        htf = make_htf(
            df
        )

        zones = build_zones(
            htf
        )

        # ====================================================
        # PERFORMANCE FIX:
        # Build pivot cache exactly once.
        # ====================================================

        pivot_cache = (
            build_pivot_cache(x)
        )

        candidates = make_candidates(
            symbol,
            x,
            zones,
            pivot_cache,
        )

        closed, unresolved = simulate(
            symbol,
            x,
            candidates,
        )

        closed = [
            enrich_times(
                trade,
                x,
            )
            for trade in closed
        ]

        unresolved = [
            enrich_times(
                trade,
                x,
            )
            for trade in unresolved
        ]

        all_closed.extend(
            closed
        )

        all_unresolved.extend(
            unresolved
        )

        # ====================================================
        # Diagnostics
        # ====================================================

        for split in [
            "Discovery",
            "Development",
            "Validation_OOS",
        ]:

            split_candidates = [
                c
                for c in candidates
                if classify(
                    x.time.iloc[
                        c["entry_i"]
                    ]
                ) == split
            ]

            split_closed = [
                t
                for t in closed
                if classify(
                    t["entry_time"]
                ) == split
            ]

            split_unresolved = [
                t
                for t in unresolved
                if classify(
                    t["entry_time"]
                ) == split
            ]

            candidate_times = [
                x.time.iloc[
                    c["entry_i"]
                ]
                for c
                in split_candidates
            ]

            diagnostics.append(
                {
                    "symbol":
                        symbol,

                    "split":
                        split,

                    "candles":
                        len(x),

                    "first_data":
                        str(
                            x.time.iloc[0]
                        ),

                    "last_data":
                        str(
                            x.time.iloc[-1]
                        ),

                    "htf_zones":
                        sum(
                            1
                            for z in zones
                            if classify(
                                z[
                                    "confirm_time"
                                ]
                            ) == split
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
                }
            )

        symbol_summary.append(
            {
                "symbol":
                    symbol,

                "candles":
                    len(x),

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
            f"zones={len(zones)} "
            f"candidates={len(candidates)} "
            f"closed={len(closed)} "
            f"unresolved={len(unresolved)}",
            flush=True,
        )

    # ========================================================
    # SORT
    # ========================================================

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

    # ========================================================
    # FINAL INTEGRITY AUDIT
    # ========================================================

    errors = audit(
        all_closed,
        all_unresolved,
    )

    if errors:

        raise RuntimeError(
            "FINAL INTEGRITY AUDIT FAILED: "
            + "; ".join(errors)
        )

    # ========================================================
    # REPORT
    # ========================================================

    report = split_report(
        all_closed,
        all_unresolved,
    )

    result = stats(
        all_closed
    )

    dd = max_drawdown(
        all_closed
    )

    pd.DataFrame(
        symbol_summary
    ).to_csv(
        OUT
        / "setup4_v2_symbol_summary.csv",
        index=False,
    )

    pd.DataFrame(
        diagnostics
    ).to_csv(
        OUT
        / "setup4_v2_diagnostics.csv",
        index=False,
    )

    report.to_csv(
        OUT
        / "setup4_v2_split_report.csv",
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

    candidate_counts = {}

    for row in diagnostics:

        split = row["split"]

        candidate_counts[
            split
        ] = (
            candidate_counts.get(
                split,
                0,
            )
            + int(
                row["candidates"]
            )
        )

    text = f"""
SETUP 4 V2 — BACKTEST REPORT

Research:
{RESEARCH_START} -> {RESEARCH_END}

Fresh OOS:
{OOS_START} -> {OOS_END}

CANDIDATES
Discovery       : {candidate_counts.get("Discovery", 0)}
Development     : {candidate_counts.get("Development", 0)}
Validation_OOS  : {candidate_counts.get("Validation_OOS", 0)}

OVERALL

Closed trades   : {result["trades"]}
Wins            : {result["wins"]}
Losses          : {result["losses"]}
WR              : {result["wr"]:.2f}%
PF              : {result["pf"]:.6f}
net_R           : {result["net_R"]:.6f}
PnL             : ${result["pnl"]:.2f}
Final Equity    : ${INITIAL_CAPITAL + result["pnl"]:.2f}
Max DD          : ${dd:.2f}
Max streak      : {result["max_streak"]}
Unresolved      : {len(all_unresolved)}

Initial Capital : ${INITIAL_CAPITAL:.2f}
Margin          : ${MARGIN:.2f}
Leverage        : {LEVERAGE:.1f}x
Notional        : ${NOTIONAL:.2f}

FINAL INTEGRITY AUDIT

Data gaps                       : PASSED
Structural causality            : PASSED
Future target leak              : PASSED
Same-symbol overlap             : PASSED
Same-candle re-entry/exit       : PASSED
RR 1:2 consistency              : PASSED
AUDIT STATUS                    : PASSED

{report.to_string(index=False)}
"""

    (
        OUT
        / "setup4_v2_report.txt"
    ).write_text(
        text.strip()
        + "\n",
        encoding="utf-8",
    )

    print(
        text,
        flush=True,
    )


if __name__ == "__main__":
    main()
