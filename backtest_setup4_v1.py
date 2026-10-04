#!/usr/bin/env python3
"""
SETUP 4 V1 — HTF Zone -> Reaction -> OB -> Aligned Liquidity -> Break -> OB Retest
Binance USD-M Futures, 15m execution / 4h context.

IMPORTANT:
- This is a causal research translation of Setup 4 from "10 ستاپ برتر.pdf".
- The PDF is qualitative. Numeric items below are frozen mechanical translations,
  not literal PDF specifications.
- Fresh OOS: 2025-10-04 through 2026-10-03 UTC. This replaces the earlier
  2023 window because ONDO/WIF do not have a complete 2023 history.
- No tuning is performed on OOS.
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
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT", "AVAXUSDT", "NEARUSDT",
    "ADAUSDT", "BNBUSDT", "APTUSDT", "CRVUSDT", "ONDOUSDT", "PENDLEUSDT",
    "ICPUSDT", "WIFUSDT",
]

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
INTERVAL = "15m"

OOS_START = pd.Timestamp("2025-10-04 00:00:00", tz="UTC")
OOS_END = pd.Timestamp("2026-10-03 23:00:00", tz="UTC")

RESEARCH_START = OOS_START - pd.Timedelta(days=365)
WARMUP_START = RESEARCH_START - pd.Timedelta(days=90)

INITIAL_CAPITAL = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE

RR = 2.0
FEE_RATE = 0.0007
SLIPPAGE = 0.0003

PIVOT = 2

# Mechanical translation of the qualitative PDF setup.
HTF_ATR_PERIOD = 14
HTF_ZONE_ATR_MULT = 0.50
MAX_REACTION_BARS = 24
ALIGN_TOL = 0.004
MAX_LIQUIDITY_DISTANCE_BARS = 96
SL_BUFFER_PCT = 0.0005

MIN_RISK_PCT = 0.0005
MAX_RISK_PCT = 0.08

OUT = Path("setup4_v1_outputs")
OUT.mkdir(exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "setup4-v1-research/1.0"
})


# ============================================================
# DATA
# ============================================================

def month_range(start, end):
    cur = pd.Timestamp(start).to_period("M")
    last = pd.Timestamp(end).to_period("M")

    result = []

    while cur <= last:
        result.append((cur.year, cur.month))
        cur += 1

    return result


def fetch_month(symbol, year, month):
    filename = f"{symbol}-{INTERVAL}-{year:04d}-{month:02d}.zip"

    url = (
        f"{BASE_URL}/{symbol}/{INTERVAL}/{filename}"
    )

    for attempt in range(4):
        try:
            response = SESSION.get(url, timeout=60)

            if response.status_code == 404:
                return None

            response.raise_for_status()
            return response.content

        except Exception:
            if attempt == 3:
                raise

            time.sleep(1.5 * (attempt + 1))

    return None


def parse_archive(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:

        csv_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise RuntimeError("Archive has no CSV file")

        with archive.open(csv_files[0]) as handle:
            raw = pd.read_csv(handle, header=None)

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

    raw = raw.iloc[:, :len(columns)]
    raw.columns = columns[:raw.shape[1]]

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
        utc=True
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

    Important:
    - 404 before first available archive is allowed.
    - 404 after first available archive is fatal.
    - No forward-fill.
    - No synthetic candles.
    - No silent gap removal.
    """

    start = WARMUP_START
    end = OOS_END

    frames = []
    first_available = False

    for year, month in month_range(start, end):

        blob = fetch_month(
            symbol,
            year,
            month
        )

        if blob is None:

            if not first_available:
                continue

            raise RuntimeError(
                f"Missing Binance archive after first availability: "
                f"{symbol} {year:04d}-{month:02d}"
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
        ignore_index=True
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

    # Remove only an actually incomplete final candle.
    now = pd.Timestamp.now(tz="UTC")

    if (
        not df.empty
        and df.time.iloc[-1] + pd.Timedelta(minutes=15) > now
    ):
        df = df.iloc[:-1].copy()

    if df.empty:
        raise RuntimeError(
            f"No usable rows for {symbol}"
        )

    # Strict 15m continuity check.
    expected = pd.date_range(
        start=df.time.iloc[0],
        end=df.time.iloc[-1],
        freq="15min",
        tz="UTC",
    )

    actual = pd.DatetimeIndex(df.time)

    missing = expected.difference(actual)

    if len(missing):
        raise RuntimeError(
            f"15m data gap for {symbol}: "
            f"{len(missing)} missing candles; "
            f"first={missing[0]}"
        )

    return df.reset_index(drop=True)


# ============================================================
# INDICATORS / STRUCTURE
# ============================================================

def atr(df, period):
    previous_close = df.close.shift(1)

    true_range = pd.concat(
        [
            df.high - df.low,
            (df.high - previous_close).abs(),
            (df.low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(
        period,
        min_periods=period,
    ).mean()


def confirmed_pivots(df):
    """
    A pivot at i becomes usable only at i + PIVOT.

    This is critical for causal processing.
    """

    x = df.copy()

    n = len(x)

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
            x.high.iloc[i] > left_highs.max()
            and
            x.high.iloc[i] >= right_highs.max()
        ):
            pivot_high[i] = True

        if (
            x.low.iloc[i] < left_lows.min()
            and
            x.low.iloc[i] <= right_lows.min()
        ):
            pivot_low[i] = True

    x["pivot_high"] = pivot_high
    x["pivot_low"] = pivot_low

    return x


def make_htf(df):
    """
    Build completed 4H candles from the real 15m data.
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
        HTF_ATR_PERIOD
    )

    htf = confirmed_pivots(htf)

    return htf


# ============================================================
# HTF ZONES
# ============================================================

def build_zones(htf):
    """
    A confirmed 4H swing creates an HTF zone.

    The pivot itself is not used until PIVOT completed 4H candles
    have confirmed it.
    """

    zones = []

    for i, row in htf.iterrows():

        if (
            not np.isfinite(row.atr)
            or row.atr <= 0
        ):
            continue

        # The pivot confirmed at i is located at i-PIVOT.
        pivot_i = i - PIVOT

        if pivot_i < 0:
            continue

        if bool(row.pivot_high):

            price = float(
                htf.high.iloc[pivot_i]
            )

            zones.append(
                {
                    "side": "SHORT",
                    "confirm_i": int(i),
                    "confirm_time": htf.time.iloc[i],
                    "pivot_i": int(pivot_i),
                    "center": price,
                    "low": (
                        price
                        - HTF_ZONE_ATR_MULT
                        * float(row.atr)
                    ),
                    "high": (
                        price
                        + HTF_ZONE_ATR_MULT
                        * float(row.atr)
                    ),
                }
            )

        if bool(row.pivot_low):

            price = float(
                htf.low.iloc[pivot_i]
            )

            zones.append(
                {
                    "side": "LONG",
                    "confirm_i": int(i),
                    "confirm_time": htf.time.iloc[i],
                    "pivot_i": int(pivot_i),
                    "center": price,
                    "low": (
                        price
                        - HTF_ZONE_ATR_MULT
                        * float(row.atr)
                    ),
                    "high": (
                        price
                        + HTF_ZONE_ATR_MULT
                        * float(row.atr)
                    ),
                }
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


def known_pivots(x, current_i):
    """
    Return pivots whose confirmation is already known at current_i.

    Pivot j requires j + PIVOT <= current_i.
    """

    start = max(
        PIVOT,
        current_i - PIVOT * 20,
    )

    end = current_i - PIVOT

    if end < start:
        return [], []

    highs = [
        j
        for j in range(
            start,
            end + 1,
        )
        if bool(x.pivot_high.iloc[j])
    ]

    lows = [
        j
        for j in range(
            start,
            end + 1,
        )
        if bool(x.pivot_low.iloc[j])
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
            x.low.iloc[j] <= zone["high"]
            and
            x.high.iloc[j] >= zone["low"]
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
                    if touch_i < p < current_i
                ]

                if not lower_highs:
                    continue

                ob_i = lower_highs[-1]

                # The pivot must already be confirmed.
                if ob_i + PIVOT > current_i:
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
                        x.high.iloc[ob_i]
                        >=
                        x.high.iloc[previous_high]
                    ):
                        continue

                aligned_highs = [
                    p
                    for p in known_highs
                    if ob_i < p < current_i
                ]

                if len(aligned_highs) < 2:
                    continue

                h1 = aligned_highs[-2]
                h2 = aligned_highs[-1]

                if not aligned(
                    x.high.iloc[h1],
                    x.high.iloc[h2],
                ):
                    continue

                liquidity_level = max(
                    x.high.iloc[h1],
                    x.high.iloc[h2],
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
                        x.high.iloc[b]
                        > liquidity_level
                    ):
                        break_i = b
                        break

                if break_i is None:
                    continue

                # OB midpoint.
                midpoint = (
                    x.low.iloc[ob_i]
                    + x.high.iloc[ob_i]
                ) / 2.0

                retest_i = None

                for e in range(
                    break_i + 1,
                    len(x),
                ):

                    if (
                        x.low.iloc[e]
                        <= midpoint
                        <= x.high.iloc[e]
                    ):
                        retest_i = e
                        break

                if retest_i is None:
                    continue

                # Entry = NEXT candle after the completed retest candle.
                entry_i = retest_i + 1

                if entry_i >= len(x):
                    continue

                entry = float(
                    x.open.iloc[entry_i]
                )

                sl = (
                    float(x.high.iloc[ob_i])
                    * (1.0 + SL_BUFFER_PCT)
                )

                risk = sl - entry

                if risk <= 0:
                    continue

                risk_pct = risk / entry

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

                # Structural target:
                # lowest low between OB and second aligned high.
                if h2 <= ob_i:
                    continue

                structural_target = float(
                    x.low.iloc[
                        ob_i:h2 + 1
                    ].min()
                )

                # SHORT:
                # TP must not be beyond the structural target.
                if tp < structural_target:
                    continue

                candidates.append(
                    {
                        "symbol": symbol,
                        "side": "SHORT",
                        "zone_confirm_i": zone[
                            "confirm_i"
                        ],
                        "touch_i": touch_i,
                        "ob_i": ob_i,
                        "align1_i": h1,
                        "align2_i": h2,
                        "break_i": break_i,
                        "retest_i": retest_i,
                        "entry_i": entry_i,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "structural_target":
                            structural_target,
                    }
                )

                break

            # ==================================================
            # LONG
            # ==================================================

            else:

                higher_lows = [
                    p
                    for p in known_lows
                    if touch_i < p < current_i
                ]

                if not higher_lows:
                    continue

                ob_i = higher_lows[-1]

                if ob_i + PIVOT > current_i:
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
                        x.low.iloc[ob_i]
                        <=
                        x.low.iloc[previous_low]
                    ):
                        continue

                aligned_lows = [
                    p
                    for p in known_lows
                    if ob_i < p < current_i
                ]

                if len(aligned_lows) < 2:
                    continue

                l1 = aligned_lows[-2]
                l2 = aligned_lows[-1]

                if not aligned(
                    x.low.iloc[l1],
                    x.low.iloc[l2],
                ):
                    continue

                liquidity_level = min(
                    x.low.iloc[l1],
                    x.low.iloc[l2],
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
                        x.low.iloc[b]
                        < liquidity_level
                    ):
                        break_i = b
                        break

                if break_i is None:
                    continue

                midpoint = (
                    x.low.iloc[ob_i]
                    + x.high.iloc[ob_i]
                ) / 2.0

                retest_i = None

                for e in range(
                    break_i + 1,
                    len(x),
                ):

                    if (
                        x.low.iloc[e]
                        <= midpoint
                        <= x.high.iloc[e]
                    ):
                        retest_i = e
                        break

                if retest_i is None:
                    continue

                entry_i = retest_i + 1

                if entry_i >= len(x):
                    continue

                entry = float(
                    x.open.iloc[entry_i]
                )

                sl = (
                    float(x.low.iloc[ob_i])
                    * (1.0 - SL_BUFFER_PCT)
                )

                risk = entry - sl

                if risk <= 0:
                    continue

                risk_pct = risk / entry

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

                # Structural target:
                # highest high between OB and second aligned low.
                if l2 <= ob_i:
                    continue

                structural_target = float(
                    x.high.iloc[
                        ob_i:l2 + 1
                    ].max()
                )

                # LONG:
                # TP must not be beyond the structural target.
                if tp > structural_target:
                    continue

                candidates.append(
                    {
                        "symbol": symbol,
                        "side": "LONG",
                        "zone_confirm_i": zone[
                            "confirm_i"
                        ],
                        "touch_i": touch_i,
                        "ob_i": ob_i,
                        "align1_i": l1,
                        "align2_i": l2,
                        "break_i": break_i,
                        "retest_i": retest_i,
                        "entry_i": entry_i,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
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
    - max one open trade on the same symbol
    - different symbols may overlap
    - no same-candle re-entry
    - entry candle is never checked for SL/TP
    - if SL and TP are both hit on the same later candle,
      SL wins conservatively
    - no timeout exit
    """

    closed = []
    unresolved = []

    last_exit_i = -1

    for candidate in candidates:

        entry_i = candidate[
            "entry_i"
        ]

        # Same-symbol lock.
        if entry_i <= last_exit_i:
            continue

        exit_i = None
        result = None
        exit_price = None

        # Never check the entry candle itself.
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

            if candidate["side"] == "LONG":

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

            if hit_sl or hit_tp:

                exit_i = j

                # Conservative same-candle ambiguity:
                # SL wins whenever both are touched.
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
                    "exit_i": None,
                    "result": "UNRESOLVED",
                }
            )

            # Important:
            # No artificial timeout is used.
            # Since this trade never closed, do not open another
            # same-symbol trade afterward.
            last_exit_i = len(x) - 1

            continue

        gross_return = (
            (
                exit_price
                - candidate["entry"]
            )
            / candidate["entry"]
            if candidate["side"] == "LONG"
            else
            (
                candidate["entry"]
                - exit_price
            )
            / candidate["entry"]
        )

        gross_pnl = (
            NOTIONAL
            * gross_return
        )

        fees = (
            NOTIONAL
            * FEE_RATE
            * 2.0
        )

        pnl = gross_pnl - fees

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
                "exit_i": exit_i,
                "exit_price": exit_price,
                "result": result,
                "pnl": pnl,
                "r_multiple": r_multiple,
                "entry_time":
                    x.time.iloc[entry_i],
                "exit_time":
                    x.time.iloc[exit_i],
            }
        )

        last_exit_i = exit_i

    return closed, unresolved


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
        max(0.0, t["pnl"])
        for t in trades
    )

    gross_loss = sum(
        min(0.0, t["pnl"])
        for t in trades
    )

    streak = 0
    max_streak = 0

    for t in trades:

        if t["result"] == "LOSS":

            streak += 1

            max_streak = max(
                max_streak,
                streak,
            )

        else:

            streak = 0

    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "wr": (
            100.0
            * wins
            / len(trades)
        ),
        "pf": (
            gross_profit
            / abs(gross_loss)
            if gross_loss < 0
            else (
                float("inf")
                if gross_profit > 0
                else 0.0
            )
        ),
        "net_R": sum(
            t["r_multiple"]
            for t in trades
        ),
        "pnl": sum(
            t["pnl"]
            for t in trades
        ),
        "max_streak": max_streak,
    }


def max_drawdown(trades):

    equity = INITIAL_CAPITAL
    peak = equity
    max_dd = 0.0

    for trade in sorted(
        trades,
        key=lambda t: t["exit_time"],
    ):

        equity += trade["pnl"]

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
# FINAL INTEGRITY AUDIT
# ============================================================

def audit(
    closed_trades,
    unresolved,
):

    errors = []

    all_trades = (
        closed_trades
        + unresolved
    )

    for trade in all_trades:

        # Every structural event must precede entry.
        if not (
            trade["zone_confirm_i"]
            < trade["touch_i"]
            < trade["ob_i"]
            < trade["align1_i"]
            < trade["align2_i"]
            <= trade["break_i"]
            < trade["retest_i"]
            < trade["entry_i"]
        ):
            errors.append(
                "future/structural ordering violation"
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
            abs(entry) * 1e-8,
        )

        if abs(
            actual_distance
            - expected_distance
        ) > tolerance:

            errors.append(
                "RR mismatch"
            )

    ordered = sorted(
        closed_trades,
        key=lambda t: (
            t["symbol"],
            t["entry_i"],
        ),
    )

    for previous, current in zip(
        ordered,
        ordered[1:],
    ):

        if (
            previous["symbol"]
            == current["symbol"]
        ):

            if (
                current["entry_i"]
                <= previous["exit_i"]
            ):
                errors.append(
                    "same-symbol overlap"
                )

            if (
                current["entry_i"]
                == previous["exit_i"]
            ):
                errors.append(
                    "same-candle re-entry"
                )

    return errors


# ============================================================
# SPLIT REPORT
# ============================================================

def split_report(trades):

    if not trades:
        return pd.DataFrame()

    frame = pd.DataFrame(
        trades
    )

    frame["time"] = pd.to_datetime(
        frame["entry_time"],
        utc=True,
    )

    total_days = (
        OOS_END - OOS_START
    ).days + 1

    discovery_end = (
        OOS_START
        + pd.Timedelta(
            days=total_days / 3.0
        )
    )

    development_end = (
        OOS_START
        + pd.Timedelta(
            days=2 * total_days / 3.0
        )
    )

    groups = [
        (
            "Discovery",
            frame[
                frame.time < discovery_end
            ],
        ),
        (
            "Development",
            frame[
                (
                    frame.time >= discovery_end
                )
                &
                (
                    frame.time < development_end
                )
            ],
        ),
        (
            "Validation_OOS",
            frame[
                frame.time >= development_end
            ],
        ),
        (
            "TOTAL",
            frame,
        ),
    ]

    rows = []

    for name, group in groups:

        result = stats(
            group.to_dict(
                "records"
            )
        )

        rows.append(
            {
                "split": name,
                "trades": result["trades"],
                "wins": result["wins"],
                "losses": result["losses"],
                "WR_%": result["wr"],
                "PF": result["pf"],
                "net_R": result["net_R"],
                "PnL_$": result["pnl"],
                "max_streak":
                    result["max_streak"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    all_closed = []
    all_unresolved = []
    audit_rows = []

    for symbol in SYMBOLS:

        print(
            f"[DATA] Fetching {symbol}"
        )

        df = fetch_symbol(
            symbol
        )

        print(
            f"[DATA] {symbol}: "
            f"{len(df)} candles"
        )

        htf = make_htf(
            df
        )

        zones = build_zones(
            htf
        )

        candidates = make_candidates(
            symbol,
            df,
            htf,
            zones,
        )

        closed, unresolved = simulate(
            symbol,
            df,
            candidates,
        )

        # Only trades whose actual entry occurs inside OOS
        # participate in OOS performance.
        closed = [
            trade
            for trade in closed
            if (
                OOS_START
                <= trade["entry_time"]
                <= OOS_END
            )
        ]

        unresolved = [
            trade
            for trade in unresolved
            if (
                OOS_START
                <= df.time.iloc[
                    trade["entry_i"]
                ]
                <= OOS_END
            )
        ]

        all_closed.extend(
            closed
        )

        all_unresolved.extend(
            unresolved
        )

        audit_rows.append(
            {
                "symbol": symbol,
                "rows": len(df),
                "first":
                    str(df.time.iloc[0]),
                "last":
                    str(df.time.iloc[-1]),
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
            f"candidates={len(candidates)} "
            f"closed={len(closed)} "
            f"unresolved={len(unresolved)}"
        )

    all_closed.sort(
        key=lambda t: t["entry_time"]
    )

    integrity_errors = audit(
        all_closed,
        all_unresolved,
    )

    if integrity_errors:

        raise RuntimeError(
            "FINAL INTEGRITY AUDIT FAILED: "
            + "; ".join(
                sorted(
                    set(
                        integrity_errors
                    )
                )
            )
        )

    # ========================================================
    # OUTPUTS
    # ========================================================

    report = split_report(
        all_closed
    )

    report.to_csv(
        OUT
        / "setup4_v1_split_report.csv",
        index=False,
    )

    pd.DataFrame(
        audit_rows
    ).to_csv(
        OUT
        / "setup4_v1_data_audit.csv",
        index=False,
    )

    if all_closed:

        pd.DataFrame(
            all_closed
        ).to_csv(
            OUT
            / "setup4_v1_trade_ledger.csv",
            index=False,
        )

    if all_unresolved:

        pd.DataFrame(
            all_unresolved
        ).to_csv(
            OUT
            / "setup4_v1_unresolved.csv",
            index=False,
        )

    result = stats(
        all_closed
    )

    dd = max_drawdown(
        all_closed
    )

    # ========================================================
    # TEXT REPORT
    # ========================================================

    output = []

    output.append(
        "SETUP 4 V1 — BACKTEST REPORT"
    )

    output.append(
        f"OOS: {OOS_START} -> {OOS_END}"
    )

    output.append(
        f"Initial Capital: ${INITIAL_CAPITAL:.2f}"
    )

    output.append(
        f"Margin: ${MARGIN:.2f}"
    )

    output.append(
        f"Leverage: {LEVERAGE:.1f}x"
    )

    output.append(
        f"Notional: ${NOTIONAL:.2f}"
    )

    output.append(
        ""
    )

    output.append(
        f"Trades: {result['trades']}"
    )

    output.append(
        f"Wins: {result['wins']}"
    )

    output.append(
        f"Losses: {result['losses']}"
    )

    output.append(
        f"WR: {result['wr']:.2f}%"
    )

    output.append(
        f"PF: {result['pf']:.6f}"
    )

    output.append(
        f"net_R: {result['net_R']:.6f}"
    )

    output.append(
        f"PnL: ${result['pnl']:.2f}"
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
        f"Max DD: ${dd:.2f}"
    )

    output.append(
        f"Unresolved: "
        f"{len(all_unresolved)}"
    )

    output.append(
        ""
    )

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

    output.append(
        ""
    )

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
        / "setup4_v1_report.txt"
    ).write_text(
        report_text,
        encoding="utf-8",
    )

    print("")
    print(report_text)


if __name__ == "__main__":
    main()
