import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# CRT STAGE-0 DIAGNOSTIC
#
# PREVIOUS COMPLETED 4H RANGE
#          ↓
# 15M SWEEP
#          ↓
# RECLAIM
#          ↓
# 15M MSS
#          ↓
# RETEST
#          ↓
# NEXT 15M OPEN
#          ↓
# SL / TP = 1 : 2
#
# IMPORTANT:
# The CRT range is ALWAYS the PREVIOUS completed 4H candle.
#
# No future 4H candle information is used.
# ============================================================


# ============================================================
# RESEARCH RULES
# ============================================================
#
# - No lookahead
# - No future leak
# - No timeout
# - No breakeven
# - No trailing
# - No pyramiding
# - One open trade per symbol
# - Different symbols may overlap
# - Maximum 10 simultaneous positions
# - Same-candle SL + TP = LOSS
# - Unresolved final trades are censored
# - Entry = next 15M open after retest
# ============================================================


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

RESEARCH_DAYS = 365
WARMUP_DAYS = 30

HTF_INTERVAL = "4h"
LTF_INTERVAL = "15m"

RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE_RATE = 0.0003

MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE

MAX_SIMULTANEOUS_POSITIONS = 10

# ------------------------------------------------------------
# CRT STRUCTURE
# ------------------------------------------------------------

MAX_SWEEP_DEPTH = 0.35

MSS_LOOKBACK = 4
MSS_MAX_BARS = 8
RETEST_MAX_BARS = 6

MIN_RISK_PCT = 0.0005
MAX_RISK_PCT = 0.08

# ------------------------------------------------------------
# Diagnostic mode
# ------------------------------------------------------------

DIAGNOSTIC = True

DATA_DIR = Path("crt_data")

OUTPUT_FILES = [
    Path("crt_stage0_trades.csv"),
    Path("crt_stage0_summary.csv"),
    Path("crt_stage0_validation_symbols.csv"),
    Path("crt_stage0_audit.csv"),
    Path("crt_stage0_diagnostic.csv"),
]


# ============================================================
# HTTP
# ============================================================

session = requests.Session()

session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 CRT-Stage0-Research"
    }
)


# ============================================================
# TIME HELPERS
# ============================================================

def ensure_utc_timestamp(value):

    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts


def latest_completed_15m():

    now = pd.Timestamp.now(tz="UTC")

    return (
        now.floor("15min")
        - pd.Timedelta(minutes=15)
    )


def month_range(start, end):

    start_ts = (
        ensure_utc_timestamp(start)
        .tz_localize(None)
    )

    end_ts = (
        ensure_utc_timestamp(end)
        .tz_localize(None)
    )

    current = start_ts.to_period("M")
    last = end_ts.to_period("M")

    result = []

    while current <= last:

        result.append(str(current))
        current += 1

    return result


# ============================================================
# DOWNLOAD
# ============================================================

def download_archive(url, timeout=60):

    try:

        response = session.get(
            url,
            timeout=timeout,
        )

        if response.status_code == 200:

            return response.content

        if response.status_code != 404:

            print(
                f"HTTP {response.status_code}: {url}",
                flush=True,
            )

        return None

    except requests.RequestException as exc:

        print(
            f"Request error: {exc}",
            flush=True,
        )

        return None


# ============================================================
# BINANCE ZIP
# ============================================================

def read_binance_zip(blob):

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
                "No CSV inside Binance archive."
            )

        with archive.open(
            csv_files[0]
        ) as file:

            df = pd.read_csv(
                file,
                header=None,
            )

    if len(df) > 0:

        first = str(
            df.iloc[0, 0]
        ).strip().lower()

        if first in {
            "open_time",
            "open time",
        }:

            df = df.iloc[1:].copy()

    if df.shape[1] < 12:

        raise RuntimeError(
            f"Unexpected Binance columns: "
            f"{df.shape[1]}"
        )

    df = df.iloc[:, :12].copy()

    df.columns = [
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

    df["open_time"] = pd.to_numeric(
        df["open_time"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["open_time"]
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    return df[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
        ]
    ].copy()


# ============================================================
# FETCH
# ============================================================

def fetch_binance_klines(
    symbol,
    interval,
    start,
    end,
):

    start_ts = ensure_utc_timestamp(start)
    end_ts = ensure_utc_timestamp(end)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_file = (
        DATA_DIR
        / f"{symbol}_{interval}.csv"
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
            )

            cached = (
                cached
                .sort_values("open_time")
                .drop_duplicates("open_time")
                .reset_index(drop=True)
            )

            if (
                len(cached) >= 10
                and cached["open_time"].min() <= start_ts
                and cached["open_time"].max() >= end_ts
            ):

                result = cached[
                    (cached["open_time"] >= start_ts)
                    &
                    (cached["open_time"] < end_ts)
                ].copy()

                if len(result) >= 10:

                    return result

        except Exception as exc:

            print(
                f"Cache ignored: {exc}",
                flush=True,
            )

    frames = []

    # --------------------------------------------------------
    # MONTHLY
    # --------------------------------------------------------

    for ym in month_range(
        start_ts,
        end_ts,
    ):

        url = (
            "https://data.binance.vision/data/"
            "futures/um/monthly/klines/"
            f"{symbol}/{interval}/"
            f"{symbol}-{interval}-{ym}.zip"
        )

        blob = download_archive(url)

        if blob is not None:

            frames.append(
                read_binance_zip(blob)
            )

        time.sleep(0.03)

    # --------------------------------------------------------
    # EXISTING DAYS
    # --------------------------------------------------------

    if frames:

        monthly = pd.concat(
            frames,
            ignore_index=True,
        )

        existing_days = set(
            monthly["open_time"]
            .dt.strftime("%Y-%m-%d")
        )

    else:

        existing_days = set()

    # --------------------------------------------------------
    # DAILY FALLBACK
    # --------------------------------------------------------

    day = start_ts.floor("D")
    last_day = end_ts.floor("D")

    while day <= last_day:

        date_string = day.strftime("%Y-%m-%d")

        if date_string not in existing_days:

            url = (
                "https://data.binance.vision/data/"
                "futures/um/daily/klines/"
                f"{symbol}/{interval}/"
                f"{symbol}-{interval}-"
                f"{date_string}.zip"
            )

            blob = download_archive(url)

            if blob is not None:

                frames.append(
                    read_binance_zip(blob)
                )

            time.sleep(0.01)

        day += pd.Timedelta(days=1)

    if not frames:

        raise RuntimeError(
            f"{symbol} {interval}: "
            "NO BINANCE DATA DOWNLOADED."
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    df = (
        df
        .drop_duplicates("open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # REMOVE INCOMPLETE CURRENT CANDLE
    # --------------------------------------------------------

    if interval == "15m":

        completed_limit = (
            latest_completed_15m()
        )

    else:

        completed_limit = (
            end_ts
        )

    df = df[
        df["open_time"] <= completed_limit
    ].copy()

    # --------------------------------------------------------
    # WINDOW
    # --------------------------------------------------------

    df = df[
        (df["open_time"] >= start_ts)
        &
        (df["open_time"] < end_ts)
    ].copy()

    if len(df) < 10:

        raise RuntimeError(
            f"{symbol} {interval}: "
            "too few rows."
        )

    df.to_csv(
        cache_file,
        index=False,
    )

    return df


# ============================================================
# CONTINUITY
# ============================================================

def audit_continuity(
    df,
    expected_minutes,
    symbol,
    name,
):

    timestamps = (
        pd.to_datetime(
            df["open_time"],
            utc=True,
        )
        .sort_values()
        .drop_duplicates()
    )

    gaps = (
        timestamps.diff()
        .dropna()
        .dt.total_seconds()
        / 60.0
    )

    bad = gaps[
        gaps > expected_minutes * 1.01
    ]

    if len(bad) > 0:

        raise RuntimeError(
            f"{symbol} {name}: "
            f"DATA GAP detected. "
            f"Max gap={bad.max():.2f} minutes."
        )

    median_gap = float(
        gaps.median()
    )

    if abs(
        median_gap - expected_minutes
    ) > 0.01:

        raise RuntimeError(
            f"{symbol} {name}: "
            f"unexpected interval "
            f"{median_gap:.2f} minutes."
        )


# ============================================================
# 4H FEATURES
# ============================================================

def prepare_4h(df):

    x = df.copy()

    x["open_time"] = pd.to_datetime(
        x["open_time"],
        utc=True,
    )

    x["range"] = (
        x["high"] - x["low"]
    )

    previous_close = x["close"].shift(1)

    true_range = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - previous_close).abs(),
            (x["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["atr14"] = (
        true_range
        .rolling(
            14,
            min_periods=14,
        )
        .mean()
    )

    x["range_pct"] = (
        x["range"] / x["close"]
    )

    x["ema20"] = (
        x["close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    x["ema50"] = (
        x["close"]
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    x["bias"] = np.where(
        x["ema20"] > x["ema50"],
        1,
        -1,
    )

    return x


# ============================================================
# ATTACH PREVIOUS COMPLETED 4H RANGE
# ============================================================
#
# CRITICAL CHANGE:
#
# A 15M candle AFTER a completed 4H candle uses that
# completed 4H candle as the REFERENCE RANGE.
#
# Therefore:
#
# 4H candle A closes
#       ↓
# 4H candle A high/low becomes known
#       ↓
# subsequent 15M candles may sweep A high/low
#
# This is causal.
# ============================================================

def attach_parent_range(
    h4,
    m15,
):

    h = h4.copy()
    l = m15.copy()

    h["open_time"] = pd.to_datetime(
        h["open_time"],
        utc=True,
    )

    l["open_time"] = pd.to_datetime(
        l["open_time"],
        utc=True,
    )

    # --------------------------------------------------------
    # COMPLETION TIME
    # --------------------------------------------------------

    h["htf_close_time"] = (
        h["open_time"]
        + pd.Timedelta(hours=4)
    )

    # --------------------------------------------------------
    # USE COMPLETED 4H CANDLE
    # --------------------------------------------------------

    reference = h[
        [
            "htf_close_time",
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "range",
            "atr14",
            "range_pct",
            "bias",
        ]
    ].copy()

    reference.columns = [
        "htf_close_time",
        "reference_open_time",
        "parent_open",
        "parent_high",
        "parent_low",
        "parent_close",
        "parent_range",
        "parent_atr",
        "parent_range_pct",
        "parent_bias",
    ]

    # --------------------------------------------------------
    # INT64 MERGE KEYS
    # --------------------------------------------------------

    l["_merge_key"] = (
        l["open_time"]
        .astype("int64")
    )

    reference["_merge_key"] = (
        reference["htf_close_time"]
        .astype("int64")
    )

    l = (
        l
        .sort_values("_merge_key")
        .reset_index(drop=True)
    )

    reference = (
        reference
        .sort_values("_merge_key")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # CAUSAL ASOF
    # --------------------------------------------------------

    l = pd.merge_asof(
        l,
        reference,
        on="_merge_key",
        direction="backward",
        allow_exact_matches=True,
    )

    l = l.drop(
        columns=["_merge_key"]
    )

    l = l.dropna(
        subset=[
            "htf_close_time",
            "parent_high",
            "parent_low",
            "parent_bias",
        ]
    ).reset_index(drop=True)

    return l


# ============================================================
# DIAGNOSTIC COUNTER FACTORY
# ============================================================

def new_diagnostic():

    return {
        "parent_range_bars": 0,
        "valid_parent_range": 0,

        "long_sweep": 0,
        "short_sweep": 0,

        "long_reclaim": 0,
        "short_reclaim": 0,

        "sweep_depth_valid_long": 0,
        "sweep_depth_valid_short": 0,

        "bias_valid_long": 0,
        "bias_valid_short": 0,

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
    }


# ============================================================
# FIND CRT SETUPS
# ============================================================

def find_crt_setups(
    df,
    symbol,
):

    bars = (
        df
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    candidates = []

    diag = new_diagnostic()

    total = len(bars)

    i = MSS_LOOKBACK + 2

    while i < total - 20:

        current = bars.iloc[i]

        diag["parent_range_bars"] += 1

        parent_high = float(
            current.parent_high
        )

        parent_low = float(
            current.parent_low
        )

        parent_range = (
            parent_high - parent_low
        )

        if (
            not np.isfinite(parent_high)
            or not np.isfinite(parent_low)
            or parent_range <= 0
        ):

            i += 1
            continue

        diag["valid_parent_range"] += 1

        # ====================================================
        # SWEEP + RECLAIM
        # ====================================================

        side = None
        sweep_price = None

        # ----------------------------------------------------
        # LONG:
        # price sweeps previous 4H LOW
        # then closes back above it
        # ----------------------------------------------------

        if (
            float(current.low) < parent_low
            and float(current.close) > parent_low
        ):

            diag["long_sweep"] += 1

            # Must reclaim inside the range
            if float(current.close) < parent_high:

                diag["long_reclaim"] += 1

                side = "LONG"

                sweep_price = float(
                    current.low
                )

        # ----------------------------------------------------
        # SHORT:
        # price sweeps previous 4H HIGH
        # then closes back below it
        # ----------------------------------------------------

        elif (
            float(current.high) > parent_high
            and float(current.close) < parent_high
        ):

            diag["short_sweep"] += 1

            if float(current.close) > parent_low:

                diag["short_reclaim"] += 1

                side = "SHORT"

                sweep_price = float(
                    current.high
                )

        if side is None:

            i += 1
            continue

        # ====================================================
        # SWEEP DEPTH
        # ====================================================

        if side == "LONG":

            sweep_depth = (
                parent_low - sweep_price
            ) / parent_range

        else:

            sweep_depth = (
                sweep_price - parent_high
            ) / parent_range

        if not (
            0 <= sweep_depth <= MAX_SWEEP_DEPTH
        ):

            i += 1
            continue

        if side == "LONG":

            diag[
                "sweep_depth_valid_long"
            ] += 1

        else:

            diag[
                "sweep_depth_valid_short"
            ] += 1

        # ====================================================
        # HTF BIAS
        # ====================================================

        parent_bias = int(
            current.parent_bias
        )

        if side == "LONG":

            if parent_bias != 1:

                i += 1
                continue

            diag["bias_valid_long"] += 1

        else:

            if parent_bias != -1:

                i += 1
                continue

            diag["bias_valid_short"] += 1

        # ====================================================
        # MSS
        # ====================================================

        mss_index = None

        search_end = min(
            i + MSS_MAX_BARS + 1,
            total,
        )

        for j in range(
            i + 1,
            search_end,
        ):

            look_start = max(
                0,
                j - MSS_LOOKBACK,
            )

            previous = bars.iloc[
                look_start:j
            ]

            if len(previous) == 0:
                continue

            if side == "LONG":

                local_high = float(
                    previous["high"].max()
                )

                if (
                    float(
                        bars["close"].iloc[j]
                    )
                    > local_high
                ):

                    mss_index = j
                    break

            else:

                local_low = float(
                    previous["low"].min()
                )

                if (
                    float(
                        bars["close"].iloc[j]
                    )
                    < local_low
                ):

                    mss_index = j
                    break

        if mss_index is None:

            i += 1
            continue

        if side == "LONG":

            diag["mss_long"] += 1

        else:

            diag["mss_short"] += 1

        # ====================================================
        # MSS BODY
        # ====================================================

        mss_open = float(
            bars["open"].iloc[mss_index]
        )

        mss_close = float(
            bars["close"].iloc[mss_index]
        )

        zone_high = max(
            mss_open,
            mss_close,
        )

        zone_low = min(
            mss_open,
            mss_close,
        )

        # ====================================================
        # RETEST
        # ====================================================

        entry_index = None

        retest_end = min(
            mss_index + RETEST_MAX_BARS + 1,
            total,
        )

        for j in range(
            mss_index + 1,
            retest_end,
        ):

            touches_zone = (
                float(
                    bars["low"].iloc[j]
                ) <= zone_high
                and
                float(
                    bars["high"].iloc[j]
                ) >= zone_low
            )

            if touches_zone:

                # Entry is NEXT candle
                entry_index = j + 1

                break

        if (
            entry_index is None
            or entry_index >= total
        ):

            i = mss_index + 1
            continue

        if side == "LONG":

            diag["retest_long"] += 1

        else:

            diag["retest_short"] += 1

        # ====================================================
        # ENTRY
        # ====================================================

        entry_price = float(
            bars["open"].iloc[entry_index]
        )

        if not (
            parent_low
            < entry_price
            < parent_high
        ):

            i = entry_index
            continue

        if side == "LONG":

            diag["entry_valid_long"] += 1

        else:

            diag["entry_valid_short"] += 1

        # ====================================================
        # STOP
        # ====================================================

        stop_price = sweep_price

        # ====================================================
        # RISK / TARGET
        # ====================================================

        if side == "LONG":

            risk = (
                entry_price
                - stop_price
            )

            target_price = (
                entry_price
                + RR * risk
            )

        else:

            risk = (
                stop_price
                - entry_price
            )

            target_price = (
                entry_price
                - RR * risk
            )

        if risk <= 0:

            i = entry_index
            continue

        risk_pct = (
            risk / entry_price
        )

        if not (
            MIN_RISK_PCT
            <= risk_pct
            <= MAX_RISK_PCT
        ):

            i = entry_index
            continue

        if side == "LONG":

            diag["risk_valid_long"] += 1

        else:

            diag["risk_valid_short"] += 1

        # ====================================================
        # TARGET MUST BE INSIDE RANGE
        # ====================================================

        if side == "LONG":

            if target_price > parent_high:

                i = entry_index
                continue

        else:

            if target_price < parent_low:

                i = entry_index
                continue

        if side == "LONG":

            diag["target_valid_long"] += 1
            diag["final_long"] += 1

        else:

            diag["target_valid_short"] += 1
            diag["final_short"] += 1

        # ====================================================
        # FINAL CANDIDATE
        # ====================================================

        candidates.append(
            {
                "symbol": symbol,

                "side": side,

                "reference_4h_time":
                    current.reference_open_time,

                "parent_time":
                    current.htf_close_time,

                "sweep_time":
                    current.open_time,

                "mss_time":
                    bars[
                        "open_time"
                    ].iloc[mss_index],

                "retest_time":
                    bars[
                        "open_time"
                    ].iloc[
                        entry_index - 1
                    ],

                "entry_time":
                    bars[
                        "open_time"
                    ].iloc[entry_index],

                "entry_index":
                    int(entry_index),

                "entry":
                    entry_price,

                "stop":
                    stop_price,

                "target":
                    target_price,

                "risk":
                    risk,

                "risk_pct":
                    risk_pct,

                "parent_high":
                    parent_high,

                "parent_low":
                    parent_low,
            }
        )

        # ----------------------------------------------------
        # No same-symbol re-entry before this candidate
        # ----------------------------------------------------

        i = entry_index + 1

    return candidates, diag


# ============================================================
# SIMULATE
# ============================================================

def simulate_trade(
    candidate,
    bars,
):

    entry_index = int(
        candidate["entry_index"]
    )

    side = candidate["side"]

    entry = float(
        candidate["entry"]
    )

    stop = float(
        candidate["stop"]
    )

    target = float(
        candidate["target"]
    )

    if side == "LONG":

        effective_entry = (
            entry * (1 + SLIPPAGE_RATE)
        )

        effective_stop = (
            stop * (1 - SLIPPAGE_RATE)
        )

        effective_target = (
            target * (1 - SLIPPAGE_RATE)
        )

    else:

        effective_entry = (
            entry * (1 - SLIPPAGE_RATE)
        )

        effective_stop = (
            stop * (1 + SLIPPAGE_RATE)
        )

        effective_target = (
            target * (1 + SLIPPAGE_RATE)
        )

    if side == "LONG":

        risk_eff = (
            effective_entry
            - effective_stop
        )

    else:

        risk_eff = (
            effective_stop
            - effective_entry
        )

    if risk_eff <= 0:
        return None

    for k in range(
        entry_index,
        len(bars),
    ):

        high = float(
            bars["high"].iloc[k]
        )

        low = float(
            bars["low"].iloc[k]
        )

        timestamp = bars[
            "open_time"
        ].iloc[k]

        if side == "LONG":

            hit_sl = (
                low <= effective_stop
            )

            hit_tp = (
                high >= effective_target
            )

        else:

            hit_sl = (
                high >= effective_stop
            )

            hit_tp = (
                low <= effective_target
            )

        if not (
            hit_sl or hit_tp
        ):
            continue

        if hit_sl and hit_tp:

            outcome = "LOSS"
            exit_reason = (
                "SL_AND_TP_SAME_CANDLE_LOSS"
            )
            gross_r = -1.0

        elif hit_sl:

            outcome = "LOSS"
            exit_reason = "SL"
            gross_r = -1.0

        else:

            outcome = "WIN"
            exit_reason = "TP"
            gross_r = RR

        risk_pct_effective = (
            risk_eff / effective_entry
        )

        dollar_risk = (
            NOTIONAL
            * risk_pct_effective
        )

        fee_dollars = (
            2
            * FEE_RATE
            * NOTIONAL
        )

        fee_r = (
            fee_dollars / dollar_risk
            if dollar_risk > 0
            else 0.0
        )

        net_r = gross_r - fee_r

        pnl = net_r * dollar_risk

        return {
            **candidate,

            "exit_time":
                timestamp,

            "outcome":
                outcome,

            "exit_reason":
                exit_reason,

            "effective_entry":
                effective_entry,

            "effective_stop":
                effective_stop,

            "effective_target":
                effective_target,

            "gross_r":
                gross_r,

            "fee_r":
                fee_r,

            "net_r":
                net_r,

            "dollar_risk":
                dollar_risk,

            "pnl":
                pnl,

            "exit_index":
                int(k),
        }

    # Censored
    return None


# ============================================================
# PORTFOLIO CONSTRAINTS
# ============================================================

def enforce_portfolio_constraints(
    raw_candidates,
    symbol_bars,
):

    resolved = []

    for symbol, candidates in (
        raw_candidates.items()
    ):

        bars = symbol_bars[symbol]

        for candidate in candidates:

            trade = simulate_trade(
                candidate,
                bars,
            )

            if trade is not None:

                resolved.append(trade)

    resolved.sort(
        key=lambda x: (
            x["entry_time"],
            x["symbol"],
        )
    )

    accepted = []

    symbol_last_exit = {}

    active = []

    for trade in resolved:

        entry_time = pd.Timestamp(
            trade["entry_time"]
        )

        exit_time = pd.Timestamp(
            trade["exit_time"]
        )

        symbol = trade["symbol"]

        active = [
            item
            for item in active
            if pd.Timestamp(
                item["exit_time"]
            ) > entry_time
        ]

        if symbol in symbol_last_exit:

            if (
                entry_time
                <= symbol_last_exit[symbol]
            ):

                continue

        if (
            len(active)
            >= MAX_SIMULTANEOUS_POSITIONS
        ):

            continue

        accepted.append(trade)

        symbol_last_exit[symbol] = exit_time

        active.append(trade)

    return accepted


# ============================================================
# METRICS
# ============================================================

def max_loss_streak(values):

    current = 0
    best = 0

    for value in values:

        if value < 0:

            current += 1
            best = max(best, current)

        else:

            current = 0

    return best


def summarize(
    trades,
    split_name,
):

    if not trades:

        return {
            "split": split_name,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": np.nan,
            "profit_factor": np.nan,
            "gross_r": 0.0,
            "net_r": 0.0,
            "pnl": 0.0,
            "max_loss_streak": 0,
            "max_drawdown_r": 0.0,
        }

    ordered = sorted(
        trades,
        key=lambda x: x["exit_time"],
    )

    values = np.array(
        [
            float(t["net_r"])
            for t in ordered
        ],
        dtype=float,
    )

    wins = int(
        (values > 0).sum()
    )

    losses = int(
        (values < 0).sum()
    )

    gross_profit = (
        float(
            values[values > 0].sum()
        )
        if wins
        else 0.0
    )

    gross_loss = (
        float(
            -values[values < 0].sum()
        )
        if losses
        else 0.0
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    equity = np.cumsum(values)

    running_peak = np.maximum.accumulate(
        np.r_[0.0, equity]
    )

    drawdown = (
        np.r_[0.0, equity]
        - running_peak
    )

    return {
        "split": split_name,
        "trades": len(values),
        "wins": wins,
        "losses": losses,

        "win_rate_pct":
            100.0 * wins / len(values),

        "profit_factor":
            profit_factor,

        "gross_r":
            float(
                sum(
                    t["gross_r"]
                    for t in ordered
                )
            ),

        "net_r":
            float(values.sum()),

        "pnl":
            float(
                sum(
                    t["pnl"]
                    for t in ordered
                )
            ),

        "max_loss_streak":
            max_loss_streak(values),

        "max_drawdown_r":
            float(-drawdown.min()),
    }


def split_trades(trades):

    ordered = sorted(
        trades,
        key=lambda x: x["exit_time"],
    )

    n = len(ordered)

    first = int(n * 0.50)
    second = int(n * 0.75)

    return (
        ordered[:first],
        ordered[first:second],
        ordered[second:],
    )


# ============================================================
# VALIDATION SYMBOLS
# ============================================================

def validation_symbol_rows(
    validation
):

    rows = []

    for symbol in SYMBOLS:

        symbol_trades = [
            trade
            for trade in validation
            if trade["symbol"] == symbol
        ]

        row = summarize(
            symbol_trades,
            "VALIDATION",
        )

        row["symbol"] = symbol

        rows.append(row)

    return rows


# ============================================================
# FINAL AUDIT
# ============================================================

def audit_trades(trades):

    ordered = sorted(
        trades,
        key=lambda x: x["entry_time"],
    )

    max_active = 0
    max_active_time = None

    for trade in ordered:

        timestamp = pd.Timestamp(
            trade["entry_time"]
        )

        active_count = sum(
            (
                pd.Timestamp(
                    other["entry_time"]
                )
                <= timestamp
                <
                pd.Timestamp(
                    other["exit_time"]
                )
            )
            for other in ordered
        )

        if active_count > max_active:

            max_active = active_count
            max_active_time = timestamp

    same_symbol_overlap = 0

    by_symbol = {}

    for trade in ordered:

        symbol = trade["symbol"]

        previous_trades = by_symbol.get(
            symbol,
            [],
        )

        for previous in previous_trades:

            if (
                pd.Timestamp(
                    trade["entry_time"]
                )
                <=
                pd.Timestamp(
                    previous["exit_time"]
                )
            ):

                same_symbol_overlap += 1

        by_symbol.setdefault(
            symbol,
            [],
        ).append(trade)

    status = (
        "PASS"
        if (
            max_active
            <= MAX_SIMULTANEOUS_POSITIONS
            and same_symbol_overlap == 0
        )
        else
        "FAIL"
    )

    return {
        "total_accepted_trades":
            len(ordered),

        "max_simultaneous_positions":
            max_active,

        "max_simultaneous_limit":
            MAX_SIMULTANEOUS_POSITIONS,

        "portfolio_overlap_violation":
            int(
                max_active
                > MAX_SIMULTANEOUS_POSITIONS
            ),

        "same_symbol_overlap_violations":
            same_symbol_overlap,

        "parent_range_lookahead":
            "NONE_PREVIOUS_COMPLETED_4H_ONLY",

        "entry_execution":
            "NEXT_15M_OPEN_AFTER_RETEST",

        "same_candle_sl_tp":
            "LOSS",

        "unresolved_final_trades":
            "CENSORED",

        "status":
            status,

        "max_active_time":
            max_active_time,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    end_ts = latest_completed_15m()

    research_start = (
        end_ts
        - pd.Timedelta(days=RESEARCH_DAYS)
    )

    fetch_start = (
        research_start
        - pd.Timedelta(days=WARMUP_DAYS)
    )

    print("=" * 78)
    print("CRT STAGE-0 DIAGNOSTIC")
    print(
        "PREVIOUS 4H RANGE -> "
        "15M SWEEP -> RECLAIM -> MSS -> RETEST"
    )
    print("=" * 78)

    print(
        f"Fetch window : {fetch_start} -> {end_ts}"
    )

    print(
        f"Research     : {research_start} -> {end_ts}"
    )

    print(
        f"Symbols      : {len(SYMBOLS)}"
    )

    print(
        f"RR           : {RR}:1"
    )

    print(
        f"Max positions: {MAX_SIMULTANEOUS_POSITIONS}"
    )

    print()

    # --------------------------------------------------------
    # REMOVE OLD OUTPUTS
    # --------------------------------------------------------

    for output_file in OUTPUT_FILES:

        if output_file.exists():

            output_file.unlink()

    all_candidates = {}

    symbol_bars = {}

    data_audit_rows = []

    diagnostic_rows = []

    # ========================================================
    # SYMBOL LOOP
    # ========================================================

    for number, symbol in enumerate(
        SYMBOLS,
        1,
    ):

        print(
            f"[{number}/{len(SYMBOLS)}] {symbol}",
            flush=True,
        )

        h4 = fetch_binance_klines(
            symbol,
            HTF_INTERVAL,
            fetch_start,
            end_ts,
        )

        m15 = fetch_binance_klines(
            symbol,
            LTF_INTERVAL,
            fetch_start,
            end_ts,
        )

        audit_continuity(
            h4,
            240,
            symbol,
            "4H",
        )

        audit_continuity(
            m15,
            15,
            symbol,
            "15M",
        )

        h4 = prepare_4h(h4)

        combined = attach_parent_range(
            h4,
            m15,
        )

        candidates, diag = find_crt_setups(
            combined,
            symbol,
        )

        all_candidates[symbol] = candidates

        symbol_bars[symbol] = combined

        # ----------------------------------------------------
        # DIAGNOSTIC ROW
        # ----------------------------------------------------

        diagnostic_row = {
            "symbol": symbol,
            **diag,
            "final_candidates":
                len(candidates),
        }

        diagnostic_rows.append(
            diagnostic_row
        )

        data_audit_rows.append(
            {
                "symbol": symbol,
                "h4_rows": len(h4),
                "m15_rows": len(m15),
                "combined_rows": len(combined),

                "candidate_setups":
                    len(candidates),

                "h4_start":
                    h4["open_time"].min(),

                "h4_end":
                    h4["open_time"].max(),

                "m15_start":
                    m15["open_time"].min(),

                "m15_end":
                    m15["open_time"].max(),
            }
        )

        print(
            f"    4H={len(h4):,} "
            f"15M={len(m15):,} "
            f"parent-bars={diag['valid_parent_range']:,}"
        )

        print(
            f"    "
            f"L sweep={diag['long_sweep']:,} "
            f"-> reclaim={diag['long_reclaim']:,} "
            f"-> depth={diag['sweep_depth_valid_long']:,} "
            f"-> bias={diag['bias_valid_long']:,} "
            f"-> MSS={diag['mss_long']:,} "
            f"-> retest={diag['retest_long']:,} "
            f"-> risk={diag['risk_valid_long']:,} "
            f"-> 2R={diag['target_valid_long']:,}"
        )

        print(
            f"    "
            f"S sweep={diag['short_sweep']:,} "
            f"-> reclaim={diag['short_reclaim']:,} "
            f"-> depth={diag['sweep_depth_valid_short']:,} "
            f"-> bias={diag['bias_valid_short']:,} "
            f"-> MSS={diag['mss_short']:,} "
            f"-> retest={diag['retest_short']:,} "
            f"-> risk={diag['risk_valid_short']:,} "
            f"-> 2R={diag['target_valid_short']:,}"
        )

        print(
            f"    FINAL CANDIDATES="
            f"{len(candidates):,}",
            flush=True,
        )

    # ========================================================
    # TOTAL DIAGNOSTIC
    # ========================================================

    diagnostic_df = pd.DataFrame(
        diagnostic_rows
    )

    diagnostic_df.to_csv(
        "crt_stage0_diagnostic.csv",
        index=False,
    )

    print()
    print("=" * 78)
    print("DIAGNOSTIC TOTALS")
    print("=" * 78)

    numeric_columns = [
        column
        for column in diagnostic_df.columns
        if column != "symbol"
    ]

    totals = diagnostic_df[
        numeric_columns
    ].sum()

    for column in numeric_columns:

        print(
            f"{column:35s}: "
            f"{int(totals[column]):,}"
        )

    total_candidates = int(
        totals["final_candidates"]
    )

    print()
    print(
        f"TOTAL CANDIDATE SETUPS: "
        f"{total_candidates:,}"
    )

    # ========================================================
    # DO NOT HIDE ZERO CANDIDATE RESULT
    # ========================================================
    #
    # If zero, we still save diagnostic files and STOP.
    #
    # This is intentional.
    #
    # We do NOT randomly loosen thresholds.
    # ========================================================

    if total_candidates == 0:

        pd.DataFrame(
            data_audit_rows
        ).to_csv(
            "crt_stage0_audit.csv",
            index=False,
        )

        pd.DataFrame(
            [
                {
                    "metric":
                        "status",

                    "value":
                        "ZERO_CANDIDATES_DIAGNOSTIC_ONLY",
                },

                {
                    "metric":
                        "note",

                    "value":
                        (
                            "No trade simulation performed. "
                            "No parameters changed automatically."
                        ),
                },
            ]
        ).to_csv(
            "crt_stage0_summary.csv",
            index=False,
        )

        pd.DataFrame(
            columns=[
                "symbol",
                "split",
                "trades",
                "wins",
                "losses",
                "win_rate_pct",
                "profit_factor",
                "gross_r",
                "net_r",
                "pnl",
                "max_loss_streak",
                "max_drawdown_r",
            ]
        ).to_csv(
            "crt_stage0_validation_symbols.csv",
            index=False,
        )

        pd.DataFrame(
            columns=[
                "symbol",
                "side",
                "entry_time",
                "exit_time",
                "entry",
                "stop",
                "target",
                "net_r",
                "pnl",
            ]
        ).to_csv(
            "crt_stage0_trades.csv",
            index=False,
        )

        print()
        print("=" * 78)
        print(
            "ZERO CANDIDATES."
        )
        print(
            "DIAGNOSTIC FILES WERE GENERATED."
        )
        print(
            "NO PARAMETERS WERE AUTOMATICALLY CHANGED."
        )
        print("=" * 78)

        return

    # ========================================================
    # SIMULATION
    # ========================================================

    trades = enforce_portfolio_constraints(
        all_candidates,
        symbol_bars,
    )

    print(
        f"RESOLVED/ACCEPTED TRADES: "
        f"{len(trades):,}"
    )

    if not trades:

        raise RuntimeError(
            "ZERO RESOLVED TRADES after "
            "execution/portfolio constraints."
        )

    # ========================================================
    # SPLITS
    # ========================================================

    (
        discovery,
        development,
        validation,
    ) = split_trades(trades)

    summary_rows = [
        summarize(trades, "ALL"),
        summarize(discovery, "DISCOVERY"),
        summarize(development, "DEVELOPMENT"),
        summarize(validation, "VALIDATION"),
    ]

    # ========================================================
    # AUDIT
    # ========================================================

    audit = audit_trades(trades)

    # ========================================================
    # TRADES
    # ========================================================

    trade_columns = [
        "symbol",
        "side",
        "reference_4h_time",
        "parent_time",
        "sweep_time",
        "mss_time",
        "retest_time",
        "entry_time",
        "exit_time",
        "entry",
        "stop",
        "target",
        "effective_entry",
        "effective_stop",
        "effective_target",
        "risk",
        "risk_pct",
        "gross_r",
        "fee_r",
        "net_r",
        "dollar_risk",
        "pnl",
        "outcome",
        "exit_reason",
        "parent_high",
        "parent_low",
    ]

    trades_df = pd.DataFrame(trades)

    trades_df[
        trade_columns
    ].to_csv(
        "crt_stage0_trades.csv",
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    pd.DataFrame(
        summary_rows
    ).to_csv(
        "crt_stage0_summary.csv",
        index=False,
    )

    # ========================================================
    # VALIDATION SYMBOLS
    # ========================================================

    pd.DataFrame(
        validation_symbol_rows(
            validation
        )
    ).to_csv(
        "crt_stage0_validation_symbols.csv",
        index=False,
    )

    # ========================================================
    # AUDIT
    # ========================================================

    audit_rows = [
        {
            "metric": key,
            "value": value,
        }
        for key, value in audit.items()
    ]

    for row in data_audit_rows:

        audit_rows.append(
            {
                "metric":
                    f'DATA_{row["symbol"]}',

                "value":
                    (
                        f'4H={row["h4_rows"]}; '
                        f'15M={row["m15_rows"]}; '
                        f'combined={row["combined_rows"]}; '
                        f'candidates='
                        f'{row["candidate_setups"]}'
                    ),
            }
        )

    pd.DataFrame(
        audit_rows
    ).to_csv(
        "crt_stage0_audit.csv",
        index=False,
    )

    # ========================================================
    # OUTPUT VALIDATION
    # ========================================================

    missing = []

    for output_file in OUTPUT_FILES:

        if (
            not output_file.exists()
            or output_file.stat().st_size == 0
        ):

            missing.append(
                str(output_file)
            )

    if missing:

        raise RuntimeError(
            "OUTPUT VALIDATION FAILED: "
            f"{missing}"
        )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)

    for row in summary_rows:

        pf = row["profit_factor"]

        if np.isinf(pf):

            pf_text = "INF"

        elif np.isnan(pf):

            pf_text = "NA"

        else:

            pf_text = f"{pf:.3f}"

        print(
            f'{row["split"]:12s} '
            f'trades={row["trades"]:5d} '
            f'WR={row["win_rate_pct"]:.2f}% '
            f'PF={pf_text} '
            f'NetR={row["net_r"]:.2f} '
            f'Streak={row["max_loss_streak"]}'
        )

    # ========================================================
    # FINAL AUDIT
    # ========================================================

    print()
    print("=" * 78)
    print("FINAL AUDIT")
    print("=" * 78)

    for key, value in audit.items():

        print(
            f"{key}: {value}"
        )

    if audit["status"] != "PASS":

        raise RuntimeError(
            "TRADE AUDIT FAILED: "
            f"{audit}"
        )

    print()
    print("=" * 78)
    print(
        "CRT STAGE-0 COMPLETED SUCCESSFULLY"
    )
    print("=" * 78)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
