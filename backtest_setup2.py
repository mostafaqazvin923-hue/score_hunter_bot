#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SETUP 2
CHOCH -> HL/LH -> HH/LL -> ORDER BLOCK -> RETURN TO OB

هدف بک‌تست:
- Win Rate >= 50%
- Fixed RR = 1:2
- Max losing streak <= 4
- Initial capital = $1,000
- Fixed margin = $100
- Leverage = 50x
- Notional = $5,000
- حداکثر یک معامله باز همزمان برای هر SYMBOL
- چند SYMBOL مختلف می‌توانند همزمان معامله داشته باشند
- عدم ورود مجدد روی همان SYMBOL تا بسته شدن معامله قبلی
- بدون lookahead
- بدون future leak
- بدون timeout مصنوعی
- بدون داده ساختگی
- داده واقعی Binance Futures
"""

from __future__ import annotations

import io
import math
import time
import zipfile
from dataclasses import dataclass

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

INTERVAL = "1h"

TEST_DAYS = 365
WARMUP_DAYS = 60

# ============================================================
# MONEY MANAGEMENT
# ============================================================

INITIAL_CAPITAL = 1000.0

MARGIN = 100.0
LEVERAGE = 50.0

NOTIONAL = MARGIN * LEVERAGE

RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003


# ============================================================
# STRUCTURE
# ============================================================

PIVOT = 2

# Minimum structural break percentage.
MIN_SWING_PCT = 0.0010

# Minimum movement after HL/LH.
MIN_IMPULSE_PCT = 0.0020

# Maximum number of candles searched backward for OB.
MAX_OB_LOOKBACK = 12

# Small stop buffer beyond OB.
SL_BUFFER_PCT = 0.0002

# Risk filter.
MIN_RISK_PCT = 0.0005
MAX_RISK_PCT = 0.08


# ============================================================
# DATA
# ============================================================

EXPECTED_BAR_SECONDS = 3600

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

BINANCE_BASE = (
    "https://data.binance.vision/data/futures/um"
)


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class Pivot:
    idx: int
    timestamp: pd.Timestamp
    price: float
    kind: str


@dataclass
class Candidate:
    symbol: str
    side: str

    signal_idx: int
    entry_idx: int

    signal_time: pd.Timestamp
    entry_time: pd.Timestamp

    raw_entry: float

    ob_low: float
    ob_high: float

    structural_target: float

    setup_id: str


@dataclass
class Trade:
    symbol: str
    side: str

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp | None

    entry: float
    stop: float
    target: float

    outcome: str

    gross_R: float
    fee_R: float
    net_R: float

    pnl_usd: float

    setup_id: str


# ============================================================
# TIME
# ============================================================

def utc_now():
    return pd.Timestamp.now(tz="UTC")


# ============================================================
# BINANCE DATA
# ============================================================

def download_url(url: str) -> bytes | None:

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = requests.get(
                url,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                return response.content

            if response.status_code == 404:
                return None

        except requests.RequestException:

            pass

        time.sleep(attempt)

    return None


def parse_zip(content: bytes) -> pd.DataFrame:

    with zipfile.ZipFile(
        io.BytesIO(content)
    ) as archive:

        csv_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise ValueError(
                "No CSV found inside ZIP"
            )

        with archive.open(csv_files[0]) as f:
            df = pd.read_csv(f)

    expected = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    if "open_time" not in df.columns:

        if len(df.columns) >= 6:

            df = df.iloc[:, :6].copy()

            df.columns = expected

        else:

            raise ValueError(
                "Unexpected Binance CSV format"
            )

    df = df[expected].copy()

    df["open_time"] = pd.to_numeric(
        df["open_time"],
        errors="coerce"
    )

    for col in expected[1:]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna()

    df["timestamp"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )

    df = df.drop(columns=["open_time"])

    df = df.drop_duplicates(
        "timestamp"
    )

    df = df.sort_values(
        "timestamp"
    )

    df = df.set_index(
        "timestamp"
    )

    return df


def fetch_symbol(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp
) -> pd.DataFrame:

    frames = []

    months = pd.date_range(
        start=start.normalize().replace(day=1),
        end=end.normalize().replace(day=1),
        freq="MS",
        tz="UTC"
    )

    for month in months:

        ym = month.strftime("%Y-%m")

        monthly_url = (
            f"{BINANCE_BASE}/monthly/klines/"
            f"{symbol}/{INTERVAL}/"
            f"{symbol}-{INTERVAL}-{ym}.zip"
        )

        content = download_url(
            monthly_url
        )

        if content is not None:

            try:

                frames.append(
                    parse_zip(content)
                )

                continue

            except Exception:
                pass

        # ----------------------------------------------------
        # DAILY FALLBACK
        # ----------------------------------------------------

        month_end = (
            month + pd.offsets.MonthBegin(1)
        )

        first_day = max(
            month,
            start.normalize()
        )

        last_day = min(
            month_end - pd.Timedelta(seconds=1),
            end.normalize()
        )

        days = pd.date_range(
            start=first_day,
            end=last_day,
            freq="D",
            tz="UTC"
        )

        for day in days:

            ds = day.strftime("%Y-%m-%d")

            daily_url = (
                f"{BINANCE_BASE}/daily/klines/"
                f"{symbol}/{INTERVAL}/"
                f"{symbol}-{INTERVAL}-{ds}.zip"
            )

            content = download_url(
                daily_url
            )

            if content is None:
                continue

            try:

                frames.append(
                    parse_zip(content)
                )

            except Exception:

                continue

    if not frames:

        raise RuntimeError(
            f"{symbol}: Binance data unavailable"
        )

    df = pd.concat(
        frames
    )

    df = df[
        ~df.index.duplicated(
            keep="first"
        )
    ]

    df = df.sort_index()

    df = df.loc[
        (df.index >= start)
        &
        (df.index <= end)
    ].copy()

    # --------------------------------------------------------
    # REMOVE INCOMPLETE LAST CANDLE
    # --------------------------------------------------------

    if len(df):

        now = utc_now()

        last_open = df.index[-1]

        last_close = (
            last_open
            + pd.Timedelta(
                seconds=EXPECTED_BAR_SECONDS
            )
        )

        if last_close > now:

            df = df.iloc[:-1].copy()

    if len(df) < 1000:

        raise RuntimeError(
            f"{symbol}: insufficient rows: {len(df)}"
        )

    # --------------------------------------------------------
    # GAP AUDIT
    # --------------------------------------------------------

    gaps = (
        df.index.to_series()
        .diff()
        .dropna()
        .dt.total_seconds()
    )

    bad_gaps = gaps[
        gaps > EXPECTED_BAR_SECONDS
    ]

    if len(bad_gaps):

        first_gap = bad_gaps.index[0]

        gap_hours = (
            bad_gaps.iloc[0] / 3600.0
        )

        raise RuntimeError(
            f"{symbol}: FATAL DATA GAP "
            f"at {first_gap}, "
            f"{gap_hours:.2f} hours"
        )

    return df


# ============================================================
# PIVOTS
# ============================================================

def detect_pivots(
    df: pd.DataFrame
):

    n = len(df)

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    pivot_high = np.zeros(
        n,
        dtype=bool
    )

    pivot_low = np.zeros(
        n,
        dtype=bool
    )

    for i in range(
        PIVOT,
        n - PIVOT
    ):

        left_high = highs[
            i - PIVOT:i
        ]

        right_high = highs[
            i + 1:i + PIVOT + 1
        ]

        left_low = lows[
            i - PIVOT:i
        ]

        right_low = lows[
            i + 1:i + PIVOT + 1
        ]

        if (
            highs[i] > left_high.max()
            and
            highs[i] >= right_high.max()
        ):

            pivot_high[i] = True

        if (
            lows[i] < left_low.min()
            and
            lows[i] <= right_low.min()
        ):

            pivot_low[i] = True

    return pivot_high, pivot_low


def build_confirmed_pivots(
    df: pd.DataFrame
):

    ph, pl = detect_pivots(df)

    pivots = []

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Pivot at index P becomes known only at P + PIVOT.
    #
    # Therefore this loop only adds pivot P when the current
    # processing position has already passed its confirmation.
    # --------------------------------------------------------

    for current_idx in range(
        len(df)
    ):

        pivot_idx = (
            current_idx - PIVOT
        )

        if pivot_idx < 0:
            continue

        if ph[pivot_idx]:

            pivots.append(
                Pivot(
                    idx=pivot_idx,
                    timestamp=df.index[pivot_idx],
                    price=float(
                        df["high"].iloc[pivot_idx]
                    ),
                    kind="H"
                )
            )

        if pl[pivot_idx]:

            pivots.append(
                Pivot(
                    idx=pivot_idx,
                    timestamp=df.index[pivot_idx],
                    price=float(
                        df["low"].iloc[pivot_idx]
                    ),
                    kind="L"
                )
            )

    pivots.sort(
        key=lambda x: x.idx
    )

    return pivots


# ============================================================
# ORDER BLOCK
# ============================================================

def bullish_order_block(
    df: pd.DataFrame,
    hl_idx: int
):

    start = max(
        0,
        hl_idx - MAX_OB_LOOKBACK
    )

    # Fixed operational rule:
    # last bearish candle before/inside the HL valley.
    for i in range(
        hl_idx,
        start - 1,
        -1
    ):

        o = float(
            df["open"].iloc[i]
        )

        c = float(
            df["close"].iloc[i]
        )

        if c < o:

            return (
                float(df["low"].iloc[i]),
                float(df["high"].iloc[i])
            )

    return None


def bearish_order_block(
    df: pd.DataFrame,
    lh_idx: int
):

    start = max(
        0,
        lh_idx - MAX_OB_LOOKBACK
    )

    # Mirror rule:
    # last bullish candle before/inside LH.
    for i in range(
        lh_idx,
        start - 1,
        -1
    ):

        o = float(
            df["open"].iloc[i]
        )

        c = float(
            df["close"].iloc[i]
        )

        if c > o:

            return (
                float(df["low"].iloc[i]),
                float(df["high"].iloc[i])
            )

    return None


# ============================================================
# STRUCTURE HELPERS
# ============================================================

def previous_pivot(
    pivots,
    start,
    kind
):

    for i in range(
        start - 1,
        -1,
        -1
    ):

        if pivots[i].kind == kind:

            return pivots[i], i

    return None, None


# ============================================================
# SETUP 2 DETECTOR
# ============================================================

def find_setup2_candidates(
    df: pd.DataFrame,
    symbol: str
):

    pivots = build_confirmed_pivots(
        df
    )

    candidates = []

    # ========================================================
    # LONG
    #
    # Downtrend
    # CHOCH upward
    # First HL
    # HH
    # Return to OB
    # ========================================================

    for ch_pos, ch in enumerate(
        pivots
    ):

        if ch.kind != "H":
            continue

        # ----------------------------------------------------
        # Last high before CHOCH
        # ----------------------------------------------------

        prior_high, prior_high_pos = (
            previous_pivot(
                pivots,
                ch_pos,
                "H"
            )
        )

        prior_low, prior_low_pos = (
            previous_pivot(
                pivots,
                ch_pos,
                "L"
            )
        )

        if (
            prior_high is None
            or
            prior_low is None
        ):
            continue

        # ----------------------------------------------------
        # Need previous high/low to verify downtrend
        # ----------------------------------------------------

        older_high, _ = (
            previous_pivot(
                pivots,
                prior_high_pos,
                "H"
            )
        )

        older_low, _ = (
            previous_pivot(
                pivots,
                prior_low_pos,
                "L"
            )
        )

        if (
            older_high is None
            or
            older_low is None
        ):
            continue

        # Downtrend:
        # lower high + lower low
        if not (
            prior_high.price
            <
            older_high.price
        ):
            continue

        if not (
            prior_low.price
            <
            older_low.price
        ):
            continue

        # ----------------------------------------------------
        # CHOCH
        # ----------------------------------------------------

        if ch.price <= (
            prior_high.price
            *
            (1.0 + MIN_SWING_PCT)
        ):
            continue

        # Require candle close above prior high.
        if float(
            df["close"].iloc[ch.idx]
        ) <= prior_high.price:
            continue

        # ----------------------------------------------------
        # FIRST HIGHER LOW AFTER CHOCH
        # ----------------------------------------------------

        hl = None
        hl_pos = None

        for i in range(
            ch_pos + 1,
            len(pivots)
        ):

            p = pivots[i]

            if p.kind != "L":
                continue

            if p.idx <= ch.idx:
                continue

            # Higher than valley before CHOCH.
            if p.price <= prior_low.price:
                continue

            hl = p
            hl_pos = i
            break

        if hl is None:
            continue

        # ----------------------------------------------------
        # HIGHER HIGH AFTER HL
        # ----------------------------------------------------

        hh = None
        hh_pos = None

        for i in range(
            hl_pos + 1,
            len(pivots)
        ):

            p = pivots[i]

            if p.kind != "H":
                continue

            if p.price <= ch.price * (
                1.0 + MIN_SWING_PCT
            ):
                continue

            hh = p
            hh_pos = i
            break

        if hh is None:
            continue

        # Minimum movement.
        movement = (
            hh.price - hl.price
        ) / hl.price

        if movement < MIN_IMPULSE_PCT:
            continue

        # ----------------------------------------------------
        # ORDER BLOCK
        # ----------------------------------------------------

        ob = bullish_order_block(
            df,
            hl.idx
        )

        if ob is None:
            continue

        ob_low, ob_high = ob

        # ----------------------------------------------------
        # AFTER HH:
        #
        # Price must return to OB.
        #
        # If price creates another HH before touching OB,
        # invalidate.
        # ----------------------------------------------------

        touch_idx = None
        invalid = False

        for j in range(
            hh.idx + 1,
            len(df)
        ):

            high = float(
                df["high"].iloc[j]
            )

            low = float(
                df["low"].iloc[j]
            )

            # Touch OB.
            if (
                low <= ob_high
                and
                high >= ob_low
            ):

                touch_idx = j
                break

            # New HH before OB touch.
            if high > hh.price:

                invalid = True
                break

        if (
            touch_idx is None
            or
            invalid
        ):
            continue

        entry_idx = (
            touch_idx + 1
        )

        if entry_idx >= len(df):
            continue

        raw_entry = float(
            df["open"].iloc[entry_idx]
        )

        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------

        stop = (
            ob_low
            *
            (1.0 - SL_BUFFER_PCT)
        )

        risk = (
            raw_entry - stop
        )

        if risk <= 0:
            continue

        risk_pct = (
            risk / raw_entry
        )

        if not (
            MIN_RISK_PCT
            <= risk_pct
            <= MAX_RISK_PCT
        ):
            continue

        # ----------------------------------------------------
        # FIXED RR 1:2
        # ----------------------------------------------------

        target = (
            raw_entry
            +
            RR * risk
        )

        # PDF structural target:
        # first peak ahead = HH.
        #
        # We only accept if fixed 1:2 target does not
        # require price to exceed the structural target.
        if target > hh.price:
            continue

        candidates.append(
            Candidate(
                symbol=symbol,
                side="LONG",
                signal_idx=touch_idx,
                entry_idx=entry_idx,
                signal_time=df.index[touch_idx],
                entry_time=df.index[entry_idx],
                raw_entry=raw_entry,
                ob_low=ob_low,
                ob_high=ob_high,
                structural_target=hh.price,
                setup_id=(
                    f"S2-L-"
                    f"{ch.idx}-"
                    f"{hl.idx}-"
                    f"{hh.idx}"
                )
            )
        )

    # ========================================================
    # SHORT
    #
    # Uptrend
    # CHOCH downward
    # First LH
    # LL
    # Return to OB
    # ========================================================

    for ch_pos, ch in enumerate(
        pivots
    ):

        if ch.kind != "L":
            continue

        prior_low, prior_low_pos = (
            previous_pivot(
                pivots,
                ch_pos,
                "L"
            )
        )

        prior_high, prior_high_pos = (
            previous_pivot(
                pivots,
                ch_pos,
                "H"
            )
        )

        if (
            prior_low is None
            or
            prior_high is None
        ):
            continue

        older_low, _ = (
            previous_pivot(
                pivots,
                prior_low_pos,
                "L"
            )
        )

        older_high, _ = (
            previous_pivot(
                pivots,
                prior_high_pos,
                "H"
            )
        )

        if (
            older_low is None
            or
            older_high is None
        ):
            continue

        # Uptrend:
        # higher low + higher high
        if not (
            prior_low.price
            >
            older_low.price
        ):
            continue

        if not (
            prior_high.price
            >
            older_high.price
        ):
            continue

        # ----------------------------------------------------
        # CHOCH DOWN
        # ----------------------------------------------------

        if ch.price >= (
            prior_low.price
            *
            (1.0 - MIN_SWING_PCT)
        ):
            continue

        if float(
            df["close"].iloc[ch.idx]
        ) >= prior_low.price:
            continue

        # ----------------------------------------------------
        # FIRST LOWER HIGH
        # ----------------------------------------------------

        lh = None
        lh_pos = None

        for i in range(
            ch_pos + 1,
            len(pivots)
        ):

            p = pivots[i]

            if p.kind != "H":
                continue

            if p.idx <= ch.idx:
                continue

            if p.price >= prior_high.price:
                continue

            lh = p
            lh_pos = i
            break

        if lh is None:
            continue

        # ----------------------------------------------------
        # LOWER LOW
        # ----------------------------------------------------

        ll = None
        ll_pos = None

        for i in range(
            lh_pos + 1,
            len(pivots)
        ):

            p = pivots[i]

            if p.kind != "L":
                continue

            if p.price >= ch.price * (
                1.0 - MIN_SWING_PCT
            ):
                continue

            ll = p
            ll_pos = i
            break

        if ll is None:
            continue

        movement = (
            lh.price - ll.price
        ) / lh.price

        if movement < MIN_IMPULSE_PCT:
            continue

        # ----------------------------------------------------
        # ORDER BLOCK
        # ----------------------------------------------------

        ob = bearish_order_block(
            df,
            lh.idx
        )

        if ob is None:
            continue

        ob_low, ob_high = ob

        # ----------------------------------------------------
        # RETURN TO OB
        # ----------------------------------------------------

        touch_idx = None
        invalid = False

        for j in range(
            ll.idx + 1,
            len(df)
        ):

            high = float(
                df["high"].iloc[j]
            )

            low = float(
                df["low"].iloc[j]
            )

            if (
                low <= ob_high
                and
                high >= ob_low
            ):

                touch_idx = j
                break

            # New LL before OB touch.
            if low < ll.price:

                invalid = True
                break

        if (
            touch_idx is None
            or
            invalid
        ):
            continue

        entry_idx = (
            touch_idx + 1
        )

        if entry_idx >= len(df):
            continue

        raw_entry = float(
            df["open"].iloc[entry_idx]
        )

        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------

        stop = (
            ob_high
            *
            (1.0 + SL_BUFFER_PCT)
        )

        risk = (
            stop - raw_entry
        )

        if risk <= 0:
            continue

        risk_pct = (
            risk / raw_entry
        )

        if not (
            MIN_RISK_PCT
            <= risk_pct
            <= MAX_RISK_PCT
        ):
            continue

        target = (
            raw_entry
            -
            RR * risk
        )

        # Structural target = LL.
        if target < ll.price:
            continue

        candidates.append(
            Candidate(
                symbol=symbol,
                side="SHORT",
                signal_idx=touch_idx,
                entry_idx=entry_idx,
                signal_time=df.index[touch_idx],
                entry_time=df.index[entry_idx],
                raw_entry=raw_entry,
                ob_low=ob_low,
                ob_high=ob_high,
                structural_target=ll.price,
                setup_id=(
                    f"S2-S-"
                    f"{ch.idx}-"
                    f"{lh.idx}-"
                    f"{ll.idx}"
                )
            )
        )

    # --------------------------------------------------------
    # DEDUPLICATION
    # --------------------------------------------------------

    unique = {}

    for c in candidates:

        key = (
            c.symbol,
            c.entry_idx,
            c.side
        )

        unique[key] = c

    return sorted(
        unique.values(),
        key=lambda x: (
            x.entry_idx,
            x.side
        )
    )


# ============================================================
# EXECUTION
# ============================================================

def execute_candidate(
    df: pd.DataFrame,
    c: Candidate
):

    entry_idx = c.entry_idx

    if entry_idx >= len(df):
        return None

    raw_entry = c.raw_entry

    # --------------------------------------------------------
    # ACTUAL ENTRY INCLUDING SLIPPAGE
    # --------------------------------------------------------

    if c.side == "LONG":

        entry = (
            raw_entry
            *
            (1.0 + SLIPPAGE)
        )

        stop = (
            c.ob_low
            *
            (1.0 - SL_BUFFER_PCT)
        )

        risk = (
            entry - stop
        )

        if risk <= 0:
            return None

        target = (
            entry
            +
            RR * risk
        )

    else:

        entry = (
            raw_entry
            *
            (1.0 - SLIPPAGE)
        )

        stop = (
            c.ob_high
            *
            (1.0 + SL_BUFFER_PCT)
        )

        risk = (
            stop - entry
        )

        if risk <= 0:
            return None

        target = (
            entry
            -
            RR * risk
        )

    risk_pct = (
        risk / entry
    )

    if not (
        MIN_RISK_PCT
        <= risk_pct
        <= MAX_RISK_PCT
    ):
        return None

    # --------------------------------------------------------
    # FIXED $5,000 NOTIONAL
    # --------------------------------------------------------

    one_R_usd = (
        NOTIONAL
        *
        risk
        /
        entry
    )

    if one_R_usd <= 0:
        return None

    fee_usd = (
        NOTIONAL
        *
        FEE_RATE
        *
        2.0
    )

    fee_R = (
        fee_usd
        /
        one_R_usd
    )

    # --------------------------------------------------------
    # FORWARD-ONLY EXECUTION
    #
    # No timeout.
    # --------------------------------------------------------

    for j in range(
        entry_idx,
        len(df)
    ):

        high = float(
            df["high"].iloc[j]
        )

        low = float(
            df["low"].iloc[j]
        )

        # ====================================================
        # LONG
        # ====================================================

        if c.side == "LONG":

            hit_sl = (
                low <= stop
            )

            hit_tp = (
                high >= target
            )

            # OHLC cannot reveal intrabar order.
            # Predeclared conservative rule:
            # if both occur -> LOSS.
            if hit_sl and hit_tp:

                exit_price = (
                    stop
                    *
                    (1.0 - SLIPPAGE)
                )

                outcome = "LOSS"

            elif hit_sl:

                exit_price = (
                    stop
                    *
                    (1.0 - SLIPPAGE)
                )

                outcome = "LOSS"

            elif hit_tp:

                exit_price = (
                    target
                    *
                    (1.0 - SLIPPAGE)
                )

                outcome = "WIN"

            else:

                continue

            gross_pnl = (
                NOTIONAL
                *
                (
                    exit_price - entry
                )
                /
                entry
            )

        # ====================================================
        # SHORT
        # ====================================================

        else:

            hit_sl = (
                high >= stop
            )

            hit_tp = (
                low <= target
            )

            if hit_sl and hit_tp:

                exit_price = (
                    stop
                    *
                    (1.0 + SLIPPAGE)
                )

                outcome = "LOSS"

            elif hit_sl:

                exit_price = (
                    stop
                    *
                    (1.0 + SLIPPAGE)
                )

                outcome = "LOSS"

            elif hit_tp:

                exit_price = (
                    target
                    *
                    (1.0 + SLIPPAGE)
                )

                outcome = "WIN"

            else:

                continue

            gross_pnl = (
                NOTIONAL
                *
                (
                    entry - exit_price
                )
                /
                entry
            )

        gross_R = (
            gross_pnl
            /
            one_R_usd
        )

        net_R = (
            gross_R
            -
            fee_R
        )

        pnl_usd = (
            net_R
            *
            one_R_usd
        )

        return Trade(
            symbol=c.symbol,
            side=c.side,
            entry_time=df.index[entry_idx],
            exit_time=df.index[j],
            entry=entry,
            stop=stop,
            target=target,
            outcome=outcome,
            gross_R=gross_R,
            fee_R=fee_R,
            net_R=net_R,
            pnl_usd=pnl_usd,
            setup_id=c.setup_id
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # No timeout.
    # Trade remains unresolved.
    # --------------------------------------------------------

    return Trade(
        symbol=c.symbol,
        side=c.side,
        entry_time=df.index[entry_idx],
        exit_time=None,
        entry=entry,
        stop=stop,
        target=target,
        outcome="UNRESOLVED",
        gross_R=0.0,
        fee_R=0.0,
        net_R=0.0,
        pnl_usd=0.0,
        setup_id=c.setup_id
    )


# ============================================================
# PER-SYMBOL OVERLAP LOCK
# ============================================================

def simulate_symbol(
    df: pd.DataFrame,
    candidates
):

    results = []

    occupied_until = -1

    candidates = sorted(
        candidates,
        key=lambda x: x.entry_idx
    )

    for candidate in candidates:

        # ----------------------------------------------------
        # Same symbol cannot have another open trade.
        #
        # <= is intentional:
        # if previous trade closes on candle J,
        # a new trade cannot enter on candle J.
        # Earliest possible re-entry is J+1.
        # ----------------------------------------------------

        if (
            candidate.entry_idx
            <=
            occupied_until
        ):
            continue

        trade = execute_candidate(
            df,
            candidate
        )

        if trade is None:
            continue

        results.append(
            trade
        )

        if trade.exit_time is None:

            occupied_until = (
                len(df) - 1
            )

        else:

            exit_idx = int(
                df.index.get_loc(
                    trade.exit_time
                )
            )

            occupied_until = exit_idx

    return results


# ============================================================
# STATISTICS
# ============================================================

def max_losing_streak(
    values
):

    best = 0
    current = 0

    for value in values:

        if value <= 0:

            current += 1

            best = max(
                best,
                current
            )

        else:

            current = 0

    return best


def profit_factor(
    values
):

    gross_profit = sum(
        x for x in values
        if x > 0
    )

    gross_loss = -sum(
        x for x in values
        if x < 0
    )

    if gross_loss == 0:

        if gross_profit > 0:
            return math.inf

        return math.nan

    return (
        gross_profit
        /
        gross_loss
    )


def calculate_drawdown(
    trades
):

    equity = INITIAL_CAPITAL
    peak = equity

    max_dd = 0.0
    max_dd_pct = 0.0

    for trade in sorted(
        trades,
        key=lambda x: x.entry_time
    ):

        if trade.outcome == "UNRESOLVED":
            continue

        equity += trade.pnl_usd

        peak = max(
            peak,
            equity
        )

        drawdown = (
            peak - equity
        )

        if peak > 0:

            drawdown_pct = (
                drawdown
                /
                peak
            )

        else:

            drawdown_pct = math.inf

        max_dd = max(
            max_dd,
            drawdown
        )

        max_dd_pct = max(
            max_dd_pct,
            drawdown_pct
        )

    return (
        max_dd,
        max_dd_pct
    )


# ============================================================
# SPLITS
# ============================================================

def split_name(
    timestamp,
    start,
    end
):

    total_seconds = (
        end - start
    ).total_seconds()

    elapsed_seconds = (
        timestamp - start
    ).total_seconds()

    ratio = (
        elapsed_seconds
        /
        total_seconds
    )

    if ratio < 0.50:
        return "Discovery"

    if ratio < 0.75:
        return "Development"

    return "Validation"


def print_report(
    trades,
    test_start,
    test_end
):

    closed = [
        t
        for t in trades
        if t.outcome != "UNRESOLVED"
    ]

    print()
    print("=" * 90)
    print("SETUP 2 — BACKTEST REPORT")
    print("=" * 90)

    rows = []

    for split in [
        "Discovery",
        "Development",
        "Validation",
        "TOTAL"
    ]:

        if split == "TOTAL":

            subset = closed

        else:

            subset = [
                t
                for t in closed
                if split_name(
                    t.entry_time,
                    test_start,
                    test_end
                )
                == split
            ]

        values = [
            t.net_R
            for t in subset
        ]

        wins = sum(
            x > 0
            for x in values
        )

        losses = sum(
            x <= 0
            for x in values
        )

        if subset:

            wr = (
                100.0
                *
                wins
                /
                len(subset)
            )

        else:

            wr = math.nan

        rows.append(
            {
                "split": split,
                "trades": len(subset),
                "wins": wins,
                "losses": losses,
                "WR_%": wr,
                "PF": profit_factor(values)
                if values
                else math.nan,
                "net_R": sum(values),
                "PnL_$": sum(
                    t.pnl_usd
                    for t in subset
                ),
                "max_streak":
                    max_losing_streak(values)
                    if values
                    else 0
            }
        )

    report = pd.DataFrame(
        rows
    )

    print(
        report.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # EQUITY / DD
    # --------------------------------------------------------

    max_dd, max_dd_pct = (
        calculate_drawdown(
            closed
        )
    )

    final_equity = (
        INITIAL_CAPITAL
        +
        sum(
            t.pnl_usd
            for t in closed
        )
    )

    print()
    print(
        f"Initial Capital : "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Final Equity    : "
        f"${final_equity:,.2f}"
    )

    print(
        f"Fixed Margin    : "
        f"${MARGIN:,.2f}"
    )

    print(
        f"Leverage        : "
        f"{LEVERAGE:.0f}x"
    )

    print(
        f"Notional        : "
        f"${NOTIONAL:,.2f}"
    )

    print(
        f"Max DD          : "
        f"${max_dd:,.2f}"
        f" ({max_dd_pct * 100:.2f}%)"
    )

    unresolved = (
        len(trades)
        -
        len(closed)
    )

    print(
        f"Unresolved      : "
        f"{unresolved}"
    )


# ============================================================
# PER SYMBOL REPORT
# ============================================================

def print_symbol_report(
    trades
):

    print()
    print("=" * 90)
    print("PER SYMBOL")
    print("=" * 90)

    rows = []

    for symbol in SYMBOLS:

        subset = [
            t
            for t in trades
            if (
                t.symbol == symbol
                and
                t.outcome != "UNRESOLVED"
            )
        ]

        values = [
            t.net_R
            for t in subset
        ]

        wins = sum(
            x > 0
            for x in values
        )

        rows.append(
            {
                "symbol": symbol,
                "trades": len(subset),
                "wins": wins,
                "losses":
                    len(subset) - wins,
                "WR_%":
                    (
                        100.0 * wins / len(subset)
                        if subset
                        else math.nan
                    ),
                "PF":
                    profit_factor(values)
                    if values
                    else math.nan,
                "net_R":
                    sum(values),
                "PnL_$":
                    sum(
                        t.pnl_usd
                        for t in subset
                    ),
                "max_streak":
                    max_losing_streak(values)
                    if values
                    else 0
            }
        )

    df = pd.DataFrame(
        rows
    )

    print(
        df.to_string(
            index=False
        )
    )


# ============================================================
# TRADE LEDGER
# ============================================================

def save_trade_ledger(
    trades
):

    rows = []

    for t in trades:

        rows.append(
            {
                "symbol": t.symbol,
                "side": t.side,
                "entry_time":
                    t.entry_time.isoformat(),
                "exit_time":
                    (
                        t.exit_time.isoformat()
                        if t.exit_time is not None
                        else ""
                    ),
                "entry": t.entry,
                "stop": t.stop,
                "target": t.target,
                "outcome": t.outcome,
                "gross_R": t.gross_R,
                "fee_R": t.fee_R,
                "net_R": t.net_R,
                "pnl_usd": t.pnl_usd,
                "setup_id": t.setup_id
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        "setup2_trade_ledger.csv",
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    now = utc_now()

    # Last COMPLETED hourly candle.
    test_end = (
        now.floor("h")
        -
        pd.Timedelta(hours=1)
    )

    test_start = (
        test_end
        -
        pd.Timedelta(days=TEST_DAYS)
    )

    fetch_start = (
        test_start
        -
        pd.Timedelta(days=WARMUP_DAYS)
    )

    print("=" * 90)
    print(
        "SETUP 2 — CHOCH / HL-HH / ORDER BLOCK"
    )
    print("=" * 90)

    print(
        f"Test start : {test_start}"
    )

    print(
        f"Test end   : {test_end}"
    )

    print(
        f"Warmup     : {WARMUP_DAYS} days"
    )

    print(
        f"RR         : 1:{RR}"
    )

    print(
        f"Capital    : ${INITIAL_CAPITAL}"
    )

    print(
        f"Margin     : ${MARGIN}"
    )

    print(
        f"Leverage   : {LEVERAGE}x"
    )

    print(
        f"Notional   : ${NOTIONAL}"
    )

    print(
        "Overlap    : 1 open trade PER SYMBOL"
    )

    print(
        "Timeout    : NONE"
    )

    print(
        "Lookahead  : NONE"
    )

    all_trades = []

    total_candidates = 0

    # ========================================================
    # SYMBOL LOOP
    # ========================================================

    for symbol in SYMBOLS:

        print()
        print(
            f"[{symbol}] downloading..."
        )

        try:

            df = fetch_symbol(
                symbol,
                fetch_start,
                test_end
            )

            candidates = (
                find_setup2_candidates(
                    df,
                    symbol
                )
            )

            # Only entries inside the test period.
            candidates = [
                c
                for c in candidates
                if (
                    test_start
                    <=
                    c.entry_time
                    <=
                    test_end
                )
            ]

            total_candidates += (
                len(candidates)
            )

            trades = simulate_symbol(
                df,
                candidates
            )

            all_trades.extend(
                trades
            )

            closed = sum(
                t.outcome != "UNRESOLVED"
                for t in trades
            )

            unresolved = sum(
                t.outcome == "UNRESOLVED"
                for t in trades
            )

            print(
                f"{symbol}: "
                f"rows={len(df):,} | "
                f"candidates={len(candidates)} | "
                f"closed={closed} | "
                f"unresolved={unresolved}"
            )

        except Exception as exc:

            print(
                f"{symbol}: ERROR -> {exc}"
            )

    # ========================================================
    # FINAL
    # ========================================================

    all_trades.sort(
        key=lambda x: x.entry_time
    )

    print()
    print(
        f"TOTAL CANDIDATES: "
        f"{total_candidates}"
    )

    print(
        f"TOTAL TRADES: "
        f"{len(all_trades)}"
    )

    if all_trades:

        print_report(
            all_trades,
            test_start,
            test_end
        )

        print_symbol_report(
            all_trades
        )

        save_trade_ledger(
            all_trades
        )

        print()
        print(
            "Saved: setup2_trade_ledger.csv"
        )


if __name__ == "__main__":
    main()
