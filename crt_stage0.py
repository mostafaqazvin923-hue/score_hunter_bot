#!/usr/bin/env python3

import io
import math
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
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT",
    "AVAXUSDT", "NEARUSDT", "ADAUSDT", "BNBUSDT",
    "APTUSDT", "CRVUSDT", "ONDOUSDT", "PENDLEUSDT",
    "ICPUSDT", "WIFUSDT",
]

RESEARCH_DAYS = 365
WARMUP_DAYS = 30

HTF_INTERVAL = "4h"
LTF_INTERVAL = "15m"

RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003

MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE

MAX_SIMULTANEOUS_POSITIONS = 10

# Pre-registered CRT constraint.
MAX_SWEEP_DEPTH = 0.35

MSS_LOOKBACK = 4
MSS_MAX_BARS = 8
RETEST_MAX_BARS = 6

MIN_RISK_PCT = 0.0005
MAX_RISK_PCT = 0.08

BASE_URL = "https://data.binance.vision/data/futures/um"

DATA_DIR = Path("crt_data")
DATA_DIR.mkdir(exist_ok=True)

OUT_TRADES = "crt_stage0_trades.csv"
OUT_SUMMARY = "crt_stage0_summary.csv"
OUT_VALIDATION = "crt_stage0_validation_symbols.csv"
OUT_AUDIT = "crt_stage0_audit.csv"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "CRT-Stage0-Research/1.0"
})


# ============================================================
# TIME HELPERS
# ============================================================

def ensure_utc(value):
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")


def month_range(start, end):
    start = (
        ensure_utc(start)
        .tz_localize(None)
        .to_period("M")
    )

    end = (
        ensure_utc(end)
        .tz_localize(None)
        .to_period("M")
    )

    return pd.period_range(
        start,
        end,
        freq="M"
    )


def day_range(start, end):
    start = (
        ensure_utc(start)
        .tz_localize(None)
        .to_period("D")
    )

    end = (
        ensure_utc(end)
        .tz_localize(None)
        .to_period("D")
    )

    return pd.period_range(
        start,
        end,
        freq="D"
    )


def interval_delta(interval):
    if interval.endswith("m"):
        return pd.Timedelta(
            minutes=int(interval[:-1])
        )

    if interval.endswith("h"):
        return pd.Timedelta(
            hours=int(interval[:-1])
        )

    if interval.endswith("d"):
        return pd.Timedelta(
            days=int(interval[:-1])
        )

    raise ValueError(interval)


def latest_completed_15m():
    now = pd.Timestamp.now(tz="UTC")

    return (
        now.floor("15min")
        - pd.Timedelta(minutes=15)
    )


def latest_completed_4h():
    now = pd.Timestamp.now(tz="UTC")

    return (
        now.floor("4h")
        - pd.Timedelta(hours=4)
    )


# ============================================================
# BINANCE VISION
# ============================================================

def archive_url(
    symbol,
    interval,
    period,
    monthly=True
):
    if monthly:
        return (
            f"{BASE_URL}/monthly/klines/"
            f"{symbol}/{interval}/"
            f"{symbol}-{interval}-{period}.zip"
        )

    return (
        f"{BASE_URL}/daily/klines/"
        f"{symbol}/{interval}/"
        f"{symbol}-{interval}-{period}.zip"
    )


def download_bytes(
    url,
    retries=3
):
    last_error = None

    for attempt in range(
        1,
        retries + 1
    ):
        try:
            response = SESSION.get(
                url,
                timeout=45
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
        f"Download failed: {url} | {last_error}"
    )


def parse_vision_zip(content):
    with zipfile.ZipFile(
        io.BytesIO(content)
    ) as z:

        names = z.namelist()

        if not names:
            raise RuntimeError(
                "Empty Binance archive"
            )

        with z.open(names[0]) as f:
            df = pd.read_csv(
                f,
                header=None
            )

    expected = [
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

    # Header-row protection
    if (
        str(df.iloc[0, 0]).lower()
        in {"open_time", "open time"}
    ):
        df = (
            df.iloc[1:]
            .reset_index(drop=True)
        )

    if df.shape[1] < 6:
        raise RuntimeError(
            f"Unexpected kline columns: "
            f"{df.shape[1]}"
        )

    df.columns = expected[:df.shape[1]]

    numeric_columns = [
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
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
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

    df["open_time"] = pd.to_datetime(
        df["open_time"].astype("int64"),
        unit="ms",
        utc=True
    )

    return df[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()


def fetch_binance_klines(
    symbol,
    interval,
    start,
    end
):
    start = ensure_utc(start)
    end = ensure_utc(end)

    cache_file = (
        DATA_DIR /
        f"{symbol}_{interval}.csv"
    )

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    if cache_file.exists():

        try:
            cached = pd.read_csv(
                cache_file
            )

            cached["open_time"] = pd.to_datetime(
                cached["open_time"],
                utc=True,
                errors="coerce"
            )

            cached = cached.dropna(
                subset=["open_time"]
            )

            cached = (
                cached
                .sort_values("open_time")
                .drop_duplicates(
                    "open_time",
                    keep="last"
                )
            )

            if (
                not cached.empty
                and cached["open_time"].min() <= start
                and cached["open_time"].max() >= end
            ):
                return cached[
                    (
                        cached["open_time"] >= start
                    )
                    &
                    (
                        cached["open_time"] <= end
                    )
                ].copy()

        except Exception:
            pass

    parts = []

    # --------------------------------------------------------
    # MONTHLY
    # --------------------------------------------------------

    for period in month_range(
        start,
        end
    ):

        ym = str(period)

        url = archive_url(
            symbol,
            interval,
            ym,
            monthly=True
        )

        content = download_bytes(url)

        if content is not None:

            parts.append(
                parse_vision_zip(
                    content
                )
            )

    # --------------------------------------------------------
    # DAILY FALLBACK
    # --------------------------------------------------------

    if parts:

        combined = pd.concat(
            parts,
            ignore_index=True
        )

        combined["open_time"] = pd.to_datetime(
            combined["open_time"],
            utc=True
        )

        have_min = (
            combined["open_time"].min()
        )

        have_max = (
            combined["open_time"].max()
        )

    else:

        combined = pd.DataFrame()

        have_min = None
        have_max = None

    need_daily = (
        combined.empty
        or have_min > start
        or have_max < end
    )

    if need_daily:

        for period in day_range(
            start,
            end
        ):

            ds = str(period)

            url = archive_url(
                symbol,
                interval,
                ds,
                monthly=False
            )

            content = download_bytes(url)

            if content is not None:

                parts.append(
                    parse_vision_zip(
                        content
                    )
                )

    if not parts:
        raise RuntimeError(
            f"{symbol} {interval}: "
            f"no Binance data downloaded"
        )

    df = pd.concat(
        parts,
        ignore_index=True
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True
    )

    df = (
        df
        .sort_values("open_time")
        .drop_duplicates(
            "open_time",
            keep="last"
        )
    )

    df = df[
        (
            df["open_time"] >= start
        )
        &
        (
            df["open_time"] <= end
        )
    ].copy()

    if df.empty:
        raise RuntimeError(
            f"{symbol} {interval}: "
            f"empty requested range"
        )

    df.to_csv(
        cache_file,
        index=False
    )

    return (
        df
        .reset_index(drop=True)
    )


# ============================================================
# DATA AUDIT
# ============================================================

def audit_continuity(
    df,
    interval,
    symbol
):
    if df.empty:
        raise RuntimeError(
            f"{symbol} {interval}: "
            f"empty dataframe"
        )

    expected = interval_delta(
        interval
    )

    ordered = (
        df
        .sort_values("open_time")
        .copy()
    )

    diffs = (
        ordered["open_time"]
        .diff()
        .dropna()
    )

    gaps = diffs[
        diffs > expected
    ]

    if not gaps.empty:

        raise RuntimeError(
            f"{symbol} {interval}: "
            f"gap detected. "
            f"largest={gaps.max()}"
        )


# ============================================================
# 4H RANGE
# ============================================================

def prepare_4h(h4):

    h4 = (
        h4
        .sort_values("open_time")
        .reset_index(drop=True)
        .copy()
    )

    # A 4H candle beginning at 08:00 closes at 12:00.
    h4["htf_close_time"] = (
        h4["open_time"]
        + pd.Timedelta(hours=4)
    )

    h4["parent_high"] = h4["high"]
    h4["parent_low"] = h4["low"]

    h4["parent_range"] = (
        h4["parent_high"]
        - h4["parent_low"]
    )

    h4["valid_parent_range"] = (
        h4["parent_range"] > 0
    )

    return h4


# ============================================================
# IMPORTANT CAUSAL ALIGNMENT FIX
# ============================================================

def attach_parent_range(
    ltf,
    h4
):
    """
    Causal mapping:

        4H 08:00 -> closes 12:00

        15M 12:00
        15M 12:15
        15M 12:30
        ...

    may see the completed 08:00-12:00 range.

    But:

        15M 11:45

    may NOT see the 08:00-12:00 range.

    This prevents lookahead.
    """

    ltf = (
        ltf
        .sort_values("open_time")
        .reset_index(drop=True)
        .copy()
    )

    h4 = (
        h4
        .sort_values("open_time")
        .reset_index(drop=True)
        .copy()
    )

    reference = h4[
        [
            "htf_close_time",
            "open_time",
            "parent_high",
            "parent_low",
            "parent_range",
            "valid_parent_range",
        ]
    ].copy()

    # --------------------------------------------------------
    # THE KEY FIX
    #
    # Do NOT merge datetime64[ms] with datetime64[us].
    # Convert BOTH sides to int64 nanoseconds.
    # --------------------------------------------------------

    ltf["_merge_key"] = (
        ltf["open_time"]
        .astype("int64")
    )

    reference["_merge_key"] = (
        reference["htf_close_time"]
        .astype("int64")
    )

    ltf = (
        ltf
        .sort_values("_merge_key")
        .reset_index(drop=True)
    )

    reference = (
        reference
        .sort_values("_merge_key")
        .reset_index(drop=True)
    )

    mapped = pd.merge_asof(
        ltf,
        reference,
        on="_merge_key",
        direction="backward",
        allow_exact_matches=True,
    )

    mapped["parent_close_time"] = pd.to_datetime(
        mapped["htf_close_time"],
        utc=True,
        errors="coerce"
    )

    # --------------------------------------------------------
    # EXPLICIT CAUSAL VALIDATION
    # --------------------------------------------------------

    mapped["parent_range_valid"] = (
        mapped["parent_high"].notna()
        &
        mapped["parent_low"].notna()
        &
        mapped["parent_range"].gt(0)
        &
        (
            mapped["parent_close_time"]
            <= mapped["open_time"]
        )
    )

    mapped = mapped.drop(
        columns=["_merge_key"]
    )

    return mapped


# ============================================================
# CRT SIGNAL ENGINE
# ============================================================

def find_crt_setups(
    df,
    symbol
):

    bars = (
        df
        .sort_values("open_time")
        .reset_index(drop=True)
        .copy()
    )

    stats = {
        "symbol": symbol,

        "parent_range_bars": 0,
        "valid_parent_range": 0,

        "long_sweep": 0,
        "short_sweep": 0,

        "long_reclaim": 0,
        "short_reclaim": 0,

        "sweep_depth_valid_long": 0,
        "sweep_depth_valid_short": 0,

        "mss_long": 0,
        "mss_short": 0,

        "retest_long": 0,
        "retest_short": 0,

        "entry_valid_long": 0,
        "entry_valid_short": 0,

        "risk_valid_long": 0,
        "risk_valid_short": 0,

        "target_valid_long": 0,
        "target_valid_short": 0,

        "final_long": 0,
        "final_short": 0,

        "final_candidates": 0,
    }

    setups = []

    n = len(bars)

    i = 0

    while i < n - 1:

        row = bars.iloc[i]

        if pd.notna(
            row["parent_high"]
        ):
            stats[
                "parent_range_bars"
            ] += 1

        if not bool(
            row["parent_range_valid"]
        ):
            i += 1
            continue

        stats[
            "valid_parent_range"
        ] += 1

        parent_high = float(
            row["parent_high"]
        )

        parent_low = float(
            row["parent_low"]
        )

        parent_range = float(
            row["parent_range"]
        )

        # ----------------------------------------------------
        # LONG SWEEP
        # ----------------------------------------------------

        long_sweep = (
            float(row["low"])
            < parent_low
        )

        long_reclaim = (
            long_sweep
            and
            float(row["close"])
            > parent_low
        )

        # ----------------------------------------------------
        # SHORT SWEEP
        # ----------------------------------------------------

        short_sweep = (
            float(row["high"])
            > parent_high
        )

        short_reclaim = (
            short_sweep
            and
            float(row["close"])
            < parent_high
        )

        if long_sweep:
            stats[
                "long_sweep"
            ] += 1

        if short_sweep:
            stats[
                "short_sweep"
            ] += 1

        candidates_here = []

        # ====================================================
        # LONG
        # ====================================================

        if long_reclaim:

            stats[
                "long_reclaim"
            ] += 1

            sweep_price = float(
                row["low"]
            )

            depth = (
                parent_low
                - sweep_price
            ) / parent_range

            if (
                depth > 0
                and
                depth <= MAX_SWEEP_DEPTH
            ):

                stats[
                    "sweep_depth_valid_long"
                ] += 1

                # --------------------------------------------
                # PRE-SWEEP STRUCTURE
                # --------------------------------------------

                structure_start = max(
                    0,
                    i - MSS_LOOKBACK
                )

                structure_slice = (
                    bars.iloc[
                        structure_start:i
                    ]
                )

                if not structure_slice.empty:

                    structure_high = float(
                        structure_slice[
                            "high"
                        ].max()
                    )

                    mss_index = None

                    max_j = min(
                        n - 1,
                        i + MSS_MAX_BARS
                    )

                    for j in range(
                        i + 1,
                        max_j + 1
                    ):

                        if (
                            float(
                                bars.iloc[j][
                                    "close"
                                ]
                            )
                            >
                            structure_high
                        ):
                            mss_index = j
                            break

                    if mss_index is not None:

                        stats[
                            "mss_long"
                        ] += 1

                        mss = (
                            bars.iloc[
                                mss_index
                            ]
                        )

                        body_low = min(
                            float(mss["open"]),
                            float(mss["close"])
                        )

                        body_high = max(
                            float(mss["open"]),
                            float(mss["close"])
                        )

                        retest_index = None

                        end_retest = min(
                            n - 2,
                            mss_index
                            + RETEST_MAX_BARS
                        )

                        for r in range(
                            mss_index + 1,
                            end_retest + 1
                        ):

                            rr = bars.iloc[r]

                            touches_body = (
                                float(rr["low"])
                                <= body_high
                                and
                                float(rr["high"])
                                >= body_low
                            )

                            holds_sweep = (
                                float(rr["close"])
                                > sweep_price
                            )

                            if (
                                touches_body
                                and
                                holds_sweep
                            ):
                                retest_index = r
                                break

                        if retest_index is not None:

                            stats[
                                "retest_long"
                            ] += 1

                            entry_index = (
                                retest_index + 1
                            )

                            if entry_index < n:

                                entry = float(
                                    bars.iloc[
                                        entry_index
                                    ]["open"]
                                )

                                # Entry must remain inside
                                # the completed CRT range.
                                if (
                                    entry > parent_low
                                    and
                                    entry < parent_high
                                ):

                                    stats[
                                        "entry_valid_long"
                                    ] += 1

                                    risk = (
                                        entry
                                        - sweep_price
                                    )

                                    if risk > 0:

                                        risk_pct = (
                                            risk / entry
                                        )

                                        if (
                                            MIN_RISK_PCT
                                            <=
                                            risk_pct
                                            <=
                                            MAX_RISK_PCT
                                        ):

                                            stats[
                                                "risk_valid_long"
                                            ] += 1

                                            target = (
                                                entry
                                                + RR * risk
                                            )

                                            if target > entry:

                                                stats[
                                                    "target_valid_long"
                                                ] += 1

                                                candidates_here.append({
                                                    "symbol": symbol,
                                                    "side": "LONG",
                                                    "signal_index": i,
                                                    "mss_index": mss_index,
                                                    "retest_index": retest_index,
                                                    "entry_index": entry_index,
                                                    "signal_time": row["open_time"],
                                                    "entry_time": bars.iloc[entry_index]["open_time"],
                                                    "parent_close_time": row["parent_close_time"],
                                                    "parent_high": parent_high,
                                                    "parent_low": parent_low,
                                                    "sweep_price": sweep_price,
                                                    "entry_price": entry,
                                                    "stop_price": sweep_price,
                                                    "target_price": target,
                                                    "risk_pct": risk_pct,
                                                })

        # ====================================================
        # SHORT
        # ====================================================

        if short_reclaim:

            stats[
                "short_reclaim"
            ] += 1

            sweep_price = float(
                row["high"]
            )

            depth = (
                sweep_price
                - parent_high
            ) / parent_range

            if (
                depth > 0
                and
                depth <= MAX_SWEEP_DEPTH
            ):

                stats[
                    "sweep_depth_valid_short"
                ] += 1

                structure_start = max(
                    0,
                    i - MSS_LOOKBACK
                )

                structure_slice = (
                    bars.iloc[
                        structure_start:i
                    ]
                )

                if not structure_slice.empty:

                    structure_low = float(
                        structure_slice[
                            "low"
                        ].min()
                    )

                    mss_index = None

                    max_j = min(
                        n - 1,
                        i + MSS_MAX_BARS
                    )

                    for j in range(
                        i + 1,
                        max_j + 1
                    ):

                        if (
                            float(
                                bars.iloc[j][
                                    "close"
                                ]
                            )
                            <
                            structure_low
                        ):
                            mss_index = j
                            break

                    if mss_index is not None:

                        stats[
                            "mss_short"
                        ] += 1

                        mss = (
                            bars.iloc[
                                mss_index
                            ]
                        )

                        body_low = min(
                            float(mss["open"]),
                            float(mss["close"])
                        )

                        body_high = max(
                            float(mss["open"]),
                            float(mss["close"])
                        )

                        retest_index = None

                        end_retest = min(
                            n - 2,
                            mss_index
                            + RETEST_MAX_BARS
                        )

                        for r in range(
                            mss_index + 1,
                            end_retest + 1
                        ):

                            rr = bars.iloc[r]

                            touches_body = (
                                float(rr["low"])
                                <= body_high
                                and
                                float(rr["high"])
                                >= body_low
                            )

                            holds_sweep = (
                                float(rr["close"])
                                < sweep_price
                            )

                            if (
                                touches_body
                                and
                                holds_sweep
                            ):
                                retest_index = r
                                break

                        if retest_index is not None:

                            stats[
                                "retest_short"
                            ] += 1

                            entry_index = (
                                retest_index + 1
                            )

                            if entry_index < n:

                                entry = float(
                                    bars.iloc[
                                        entry_index
                                    ]["open"]
                                )

                                if (
                                    entry > parent_low
                                    and
                                    entry < parent_high
                                ):

                                    stats[
                                        "entry_valid_short"
                                    ] += 1

                                    risk = (
                                        sweep_price
                                        - entry
                                    )

                                    if risk > 0:

                                        risk_pct = (
                                            risk / entry
                                        )

                                        if (
                                            MIN_RISK_PCT
                                            <=
                                            risk_pct
                                            <=
                                            MAX_RISK_PCT
                                        ):

                                            stats[
                                                "risk_valid_short"
                                            ] += 1

                                            target = (
                                                entry
                                                - RR * risk
                                            )

                                            if target < entry:

                                                stats[
                                                    "target_valid_short"
                                                ] += 1

                                                candidates_here.append({
                                                    "symbol": symbol,
                                                    "side": "SHORT",
                                                    "signal_index": i,
                                                    "mss_index": mss_index,
                                                    "retest_index": retest_index,
                                                    "entry_index": entry_index,
                                                    "signal_time": row["open_time"],
                                                    "entry_time": bars.iloc[entry_index]["open_time"],
                                                    "parent_close_time": row["parent_close_time"],
                                                    "parent_high": parent_high,
                                                    "parent_low": parent_low,
                                                    "sweep_price": sweep_price,
                                                    "entry_price": entry,
                                                    "stop_price": sweep_price,
                                                    "target_price": target,
                                                    "risk_pct": risk_pct,
                                                })

        # ----------------------------------------------------
        # Candidate accepted.
        # ----------------------------------------------------

        if candidates_here:

            candidates_here.sort(
                key=lambda x: x[
                    "entry_index"
                ]
            )

            selected = (
                candidates_here[0]
            )

            setups.append(
                selected
            )

            if selected["side"] == "LONG":
                stats[
                    "final_long"
                ] += 1
            else:
                stats[
                    "final_short"
                ] += 1

            stats[
                "final_candidates"
            ] += 1

            # Prevent same-symbol re-entry
            # while this setup is pending.
            i = (
                selected["entry_index"]
                + 1
            )

            continue

        i += 1

    return setups, stats


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    bars,
    setup
):

    entry_index = setup[
        "entry_index"
    ]

    side = setup[
        "side"
    ]

    if entry_index >= len(bars):
        return None

    raw_entry = float(
        bars.iloc[
            entry_index
        ]["open"]
    )

    if side == "LONG":

        entry = (
            raw_entry
            * (1.0 + SLIPPAGE)
        )

        stop = (
            setup["stop_price"]
            * (1.0 - SLIPPAGE)
        )

        target = (
            setup["target_price"]
            * (1.0 - SLIPPAGE)
        )

    else:

        entry = (
            raw_entry
            * (1.0 - SLIPPAGE)
        )

        stop = (
            setup["stop_price"]
            * (1.0 + SLIPPAGE)
        )

        target = (
            setup["target_price"]
            * (1.0 + SLIPPAGE)
        )

    initial_risk = (
        entry - stop
        if side == "LONG"
        else stop - entry
    )

    if initial_risk <= 0:
        return None

    exit_index = None
    exit_price = None
    outcome = None

    for j in range(
        entry_index,
        len(bars)
    ):

        bar = bars.iloc[j]

        high = float(
            bar["high"]
        )

        low = float(
            bar["low"]
        )

        if side == "LONG":

            hit_sl = (
                low <= stop
            )

            hit_tp = (
                high >= target
            )

            # Locked convention:
            # same candle SL+TP = LOSS
            if hit_sl and hit_tp:

                exit_index = j
                exit_price = stop
                outcome = "LOSS"

                break

            if hit_sl:

                exit_index = j
                exit_price = stop
                outcome = "LOSS"

                break

            if hit_tp:

                exit_index = j
                exit_price = target
                outcome = "WIN"

                break

        else:

            hit_sl = (
                high >= stop
            )

            hit_tp = (
                low <= target
            )

            if hit_sl and hit_tp:

                exit_index = j
                exit_price = stop
                outcome = "LOSS"

                break

            if hit_sl:

                exit_index = j
                exit_price = stop
                outcome = "LOSS"

                break

            if hit_tp:

                exit_index = j
                exit_price = target
                outcome = "WIN"

                break

    # Unresolved final-sample trade:
    # censored, NOT forced to loss/win.
    if exit_index is None:
        return None

    gross_r = (
        (
            exit_price - entry
        )
        / initial_risk
        if side == "LONG"
        else
        (
            entry - exit_price
        )
        / initial_risk
    )

    fee_dollars = (
        NOTIONAL
        * FEE_RATE
        * 2.0
    )

    risk_dollars = (
        NOTIONAL
        * initial_risk
        / entry
    )

    if risk_dollars <= 0:
        return None

    fee_r = (
        fee_dollars
        / risk_dollars
    )

    net_r = (
        gross_r
        - fee_r
    )

    pnl = (
        net_r
        * risk_dollars
    )

    return {
        **setup,

        "entry_price_effective": entry,
        "stop_price_effective": stop,
        "target_price_effective": target,

        "exit_time": bars.iloc[
            exit_index
        ]["open_time"],

        "exit_index": exit_index,
        "exit_price": exit_price,

        "outcome": outcome,

        "gross_r": gross_r,
        "fee_r": fee_r,
        "net_r": net_r,

        "risk_dollars": risk_dollars,
        "pnl": pnl,
    }


# ============================================================
# PORTFOLIO CONSTRAINTS
# ============================================================

def enforce_portfolio_constraints(
    trades
):

    if not trades:
        return []

    ordered = sorted(
        trades,
        key=lambda x: (
            pd.Timestamp(
                x["entry_time"]
            ),
            x["symbol"],
            x["side"],
        )
    )

    accepted = []

    last_exit_by_symbol = {}

    active = []

    for trade in ordered:

        symbol = trade[
            "symbol"
        ]

        entry_time = pd.Timestamp(
            trade["entry_time"]
        )

        exit_time = pd.Timestamp(
            trade["exit_time"]
        )

        # One simultaneous trade per symbol.
        if symbol in last_exit_by_symbol:

            if (
                entry_time
                <=
                last_exit_by_symbol[
                    symbol
                ]
            ):
                continue

        # Remove already closed positions.
        active = [
            t
            for t in active
            if pd.Timestamp(
                t["exit_time"]
            )
            > entry_time
        ]

        if (
            len(active)
            >= MAX_SIMULTANEOUS_POSITIONS
        ):
            continue

        accepted.append(
            trade
        )

        active.append(
            trade
        )

        last_exit_by_symbol[
            symbol
        ] = exit_time

    return accepted


# ============================================================
# METRICS
# ============================================================

def max_loss_streak(
    trades
):

    streak = 0
    maximum = 0

    ordered = sorted(
        trades,
        key=lambda x: pd.Timestamp(
            x["exit_time"]
        )
    )

    for trade in ordered:

        if trade["net_r"] < 0:

            streak += 1

            maximum = max(
                maximum,
                streak
            )

        else:

            streak = 0

    return maximum


def profit_factor(
    trades
):

    wins = sum(
        t["net_r"]
        for t in trades
        if t["net_r"] > 0
    )

    losses = -sum(
        t["net_r"]
        for t in trades
        if t["net_r"] < 0
    )

    if losses <= 0:

        if wins > 0:
            return math.inf

        return 0.0

    return (
        wins / losses
    )


def summarize(
    trades,
    label
):

    n = len(trades)

    if n == 0:

        return {
            "split": label,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "gross_r": 0.0,
            "net_r": 0.0,
            "pnl": 0.0,
            "max_loss_streak": 0,
        }

    wins = sum(
        1
        for t in trades
        if t["net_r"] > 0
    )

    losses = sum(
        1
        for t in trades
        if t["net_r"] < 0
    )

    return {
        "split": label,
        "trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate": (
            100.0
            * wins
            / n
        ),
        "profit_factor": profit_factor(
            trades
        ),
        "gross_r": sum(
            t["gross_r"]
            for t in trades
        ),
        "net_r": sum(
            t["net_r"]
            for t in trades
        ),
        "pnl": sum(
            t["pnl"]
            for t in trades
        ),
        "max_loss_streak":
            max_loss_streak(
                trades
            ),
    }


def split_trades(
    trades
):

    ordered = sorted(
        trades,
        key=lambda x: pd.Timestamp(
            x["exit_time"]
        )
    )

    n = len(ordered)

    a = int(
        n * 0.50
    )

    b = int(
        n * 0.75
    )

    return (
        ordered[:a],
        ordered[a:b],
        ordered[b:],
    )


# ============================================================
# AUDIT
# ============================================================

def audit_trades(
    trades
):

    if not trades:

        return {
            "trades": 0,
            "same_symbol_overlap": 0,
            "same_symbol_reentry": 0,
            "max_simultaneous": 0,
        }

    ordered = sorted(
        trades,
        key=lambda x: pd.Timestamp(
            x["entry_time"]
        )
    )

    same_symbol_overlap = 0
    same_symbol_reentry = 0
    max_active = 0

    by_symbol = {}

    for trade in ordered:

        symbol = trade[
            "symbol"
        ]

        entry = pd.Timestamp(
            trade["entry_time"]
        )

        exit_time = pd.Timestamp(
            trade["exit_time"]
        )

        if symbol in by_symbol:

            previous = by_symbol[
                symbol
            ]

            if (
                entry
                <=
                previous["exit_time"]
            ):
                same_symbol_overlap += 1

            if (
                entry
                ==
                previous["exit_time"]
            ):
                same_symbol_reentry += 1

        by_symbol[
            symbol
        ] = {
            "entry_time": entry,
            "exit_time": exit_time,
        }

        active = 0

        for other in ordered:

            other_entry = pd.Timestamp(
                other["entry_time"]
            )

            other_exit = pd.Timestamp(
                other["exit_time"]
            )

            if (
                other_entry
                <=
                entry
                <
                other_exit
            ):
                active += 1

        max_active = max(
            max_active,
            active
        )

    return {
        "trades": len(trades),
        "same_symbol_overlap":
            same_symbol_overlap,
        "same_symbol_reentry":
            same_symbol_reentry,
        "max_simultaneous":
            max_active,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    latest15 = (
        latest_completed_15m()
    )

    latest4 = (
        latest_completed_4h()
    )

    end_ts = min(
        latest15,
        latest4
    )

    research_start = (
        end_ts
        - pd.Timedelta(
            days=RESEARCH_DAYS
        )
    )

    fetch_start = (
        research_start
        - pd.Timedelta(
            days=WARMUP_DAYS
        )
    )

    print("=" * 72)
    print(
        "CRT STAGE-0"
    )
    print(
        "4H RANGE -> "
        "15M SWEEP -> "
        "RECLAIM -> "
        "MSS -> "
        "RETEST"
    )
    print("=" * 72)

    print(
        f"Fetch window : "
        f"{fetch_start}"
    )

    print(
        f"Research     : "
        f"{research_start}"
    )

    print(
        f"End          : "
        f"{end_ts}"
    )

    print(
        f"Symbols      : "
        f"{len(SYMBOLS)}"
    )

    print(
        f"RR           : "
        f"{RR}:1"
    )

    print(
        f"Max positions: "
        f"{MAX_SIMULTANEOUS_POSITIONS}"
    )

    print()
    print(
        "CAUSAL ALIGNMENT:"
    )
    print(
        "15M can only use a 4H range "
        "after that 4H candle is closed."
    )
    print()

    all_candidates = []

    diagnostics = []

    # ========================================================
    # DATA + SIGNAL GENERATION
    # ========================================================

    for index, symbol in enumerate(
        SYMBOLS,
        1
    ):

        print(
            f"[{index}/{len(SYMBOLS)}] "
            f"{symbol}"
        )

        h4 = fetch_binance_klines(
            symbol,
            HTF_INTERVAL,
            fetch_start,
            end_ts
        )

        m15 = fetch_binance_klines(
            symbol,
            LTF_INTERVAL,
            fetch_start,
            end_ts
        )

        audit_continuity(
            h4,
            HTF_INTERVAL,
            symbol
        )

        audit_continuity(
            m15,
            LTF_INTERVAL,
            symbol
        )

        h4 = h4[
            (
                h4["open_time"]
                >= fetch_start
            )
            &
            (
                h4["open_time"]
                <= end_ts
            )
        ].copy()

        m15 = m15[
            (
                m15["open_time"]
                >= fetch_start
            )
            &
            (
                m15["open_time"]
                <= end_ts
            )
        ].copy()

        h4 = prepare_4h(
            h4
        )

        m15 = attach_parent_range(
            m15,
            h4
        )

        causal_mappings = int(
            m15[
                "parent_range_valid"
            ].sum()
        )

        print(
            f"    4H rows="
            f"{len(h4):,} | "
            f"15M rows="
            f"{len(m15):,} | "
            f"causal mappings="
            f"{causal_mappings:,}"
        )

        # This is a hard data-integrity check.
        if causal_mappings == 0:

            raise RuntimeError(
                f"{symbol}: "
                f"ZERO causal 4H->15M mappings. "
                f"Data alignment is invalid."
            )

        setups, stats = (
            find_crt_setups(
                m15,
                symbol
            )
        )

        stats[
            "h4_rows"
        ] = len(h4)

        stats[
            "m15_rows"
        ] = len(m15)

        stats[
            "causal_mappings"
        ] = causal_mappings

        diagnostics.append(
            stats
        )

        all_candidates.extend(
            setups
        )

        print(
            "    "
            f"sweep L/S="
            f"{stats['long_sweep']}/"
            f"{stats['short_sweep']} | "
            f"reclaim L/S="
            f"{stats['long_reclaim']}/"
            f"{stats['short_reclaim']} | "
            f"MSS L/S="
            f"{stats['mss_long']}/"
            f"{stats['mss_short']} | "
            f"retest L/S="
            f"{stats['retest_long']}/"
            f"{stats['retest_short']} | "
            f"candidates="
            f"{stats['final_candidates']}"
        )

    diagnostics_df = pd.DataFrame(
        diagnostics
    )

    # Always save diagnostics.
    diagnostics_df.to_csv(
        OUT_AUDIT,
        index=False
    )

    # ========================================================
    # DIAGNOSTIC TOTALS
    # ========================================================

    print()
    print("=" * 72)
    print(
        "DIAGNOSTIC TOTALS"
    )
    print("=" * 72)

    diagnostic_columns = [
        "parent_range_bars",
        "valid_parent_range",
        "long_sweep",
        "short_sweep",
        "long_reclaim",
        "short_reclaim",
        "sweep_depth_valid_long",
        "sweep_depth_valid_short",
        "mss_long",
        "mss_short",
        "retest_long",
        "retest_short",
        "entry_valid_long",
        "entry_valid_short",
        "risk_valid_long",
        "risk_valid_short",
        "target_valid_long",
        "target_valid_short",
        "final_long",
        "final_short",
        "final_candidates",
    ]

    for column in diagnostic_columns:

        if column in diagnostics_df:

            print(
                f"{column:35s}: "
                f"{int(diagnostics_df[column].sum())}"
            )

    print()
    print(
        f"TOTAL CANDIDATE SETUPS: "
        f"{len(all_candidates)}"
    )

    # ========================================================
    # ZERO CANDIDATE
    # ========================================================

    if not all_candidates:

        print()
        print("=" * 72)
        print(
            "ZERO CRT CANDIDATES."
        )
        print(
            "NO PERFORMANCE RESULT "
            "IS VALID."
        )
        print(
            "DIAGNOSTIC FILE GENERATED."
        )
        print(
            "NO PARAMETERS WERE "
            "AUTOMATICALLY CHANGED."
        )
        print("=" * 72)

        return

    # ========================================================
    # SIMULATION
    # ========================================================

    simulated = []

    for symbol in SYMBOLS:

        candidates = [
            x
            for x in all_candidates
            if x["symbol"] == symbol
        ]

        if not candidates:
            continue

        bars = fetch_binance_klines(
            symbol,
            LTF_INTERVAL,
            fetch_start,
            end_ts
        )

        bars = (
            bars
            .sort_values("open_time")
            .reset_index(drop=True)
        )

        for setup in candidates:

            result = simulate_trade(
                bars,
                setup
            )

            if result is not None:

                simulated.append(
                    result
                )

    print()
    print(
        "Closed trades before "
        "portfolio constraints: "
        f"{len(simulated)}"
    )

    accepted = (
        enforce_portfolio_constraints(
            simulated
        )
    )

    print(
        "Accepted trades after "
        "portfolio constraints: "
        f"{len(accepted)}"
    )

    # ========================================================
    # SPLITS
    # ========================================================

    discovery, development, validation = (
        split_trades(
            accepted
        )
    )

    summaries = pd.DataFrame([
        summarize(
            accepted,
            "ALL"
        ),
        summarize(
            discovery,
            "DISCOVERY"
        ),
        summarize(
            development,
            "DEVELOPMENT"
        ),
        summarize(
            validation,
            "VALIDATION"
        ),
    ])

    print()
    print("=" * 72)
    print(
        "SUMMARY"
    )
    print("=" * 72)

    print(
        summaries.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

    # ========================================================
    # VALIDATION BY SYMBOL
    # ========================================================

    validation_rows = []

    for symbol in SYMBOLS:

        symbol_trades = [
            x
            for x in validation
            if x["symbol"] == symbol
        ]

        row = summarize(
            symbol_trades,
            f"VALIDATION_{symbol}"
        )

        row["symbol"] = symbol

        validation_rows.append(
            row
        )

    validation_df = pd.DataFrame(
        validation_rows
    )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    trades_df = pd.DataFrame(
        accepted
    )

    if not trades_df.empty:

        trades_df = (
            trades_df
            .sort_values("exit_time")
        )

    trades_df.to_csv(
        OUT_TRADES,
        index=False
    )

    summaries.to_csv(
        OUT_SUMMARY,
        index=False
    )

    validation_df.to_csv(
        OUT_VALIDATION,
        index=False
    )

    diagnostics_df.to_csv(
        OUT_AUDIT,
        index=False
    )

    # ========================================================
    # FINAL AUDIT
    # ========================================================

    audit = audit_trades(
        accepted
    )

    print()
    print("=" * 72)
    print(
        "FINAL AUDIT"
    )
    print("=" * 72)

    print(
        audit
    )

    if (
        audit[
            "same_symbol_overlap"
        ]
        != 0
    ):
        raise RuntimeError(
            "AUDIT FAIL: "
            "same-symbol overlap detected."
        )

    if (
        audit[
            "same_symbol_reentry"
        ]
        != 0
    ):
        raise RuntimeError(
            "AUDIT FAIL: "
            "same-symbol same-time "
            "re-entry detected."
        )

    if (
        audit[
            "max_simultaneous"
        ]
        > MAX_SIMULTANEOUS_POSITIONS
    ):
        raise RuntimeError(
            "AUDIT FAIL: "
            "maximum simultaneous "
            "positions exceeded."
        )

    print()
    print("=" * 72)
    print(
        "CRT STAGE-0 COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
