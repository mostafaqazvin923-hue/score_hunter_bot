import io
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# CRT STAGE-0
# 4H RANGE -> SWEEP -> RECLAIM -> 15M MSS -> RETEST
#
# Research venue:
# Binance USD-M Futures public historical data
#
# Rules:
# - 365 days + 30 days warmup
# - Fixed 14-symbol universe
# - 4H parent range
# - 15M execution
# - Completed candles only
# - Entry at next 15M open after retest
# - RR = 1:2
# - No timeout
# - No BE
# - No trailing
# - No pyramiding
# - One open trade per symbol
# - Different symbols may overlap
# - No same-candle re-entry
# - Same-candle SL + TP = LOSS
# - Unresolved final trades are censored
# - Chronological 50/25/25 split
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

MAX_SWEEP_DEPTH = 0.35

MSS_LOOKBACK = 4
MSS_MAX_BARS = 8

RETEST_MAX_BARS = 6

MIN_RISK_PCT = 0.0005
MAX_RISK_PCT = 0.08

DATA_DIR = Path("crt_data")


session = requests.Session()

session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 CRT-Stage0-Research"
    }
)


# ============================================================
# TIMEZONE HELPERS
# ============================================================

def ensure_utc_timestamp(value):
    """
    Convert any timestamp to UTC-aware Timestamp.

    Handles both:
    - timezone-naive timestamps
    - timezone-aware timestamps

    This avoids:
    ValueError:
    Cannot pass a datetime or Timestamp with tzinfo
    with the tz parameter.
    """

    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts


# ============================================================
# MONTH RANGE
# ============================================================

def month_range(start, end):
    """
    Return YYYY-MM strings between start and end.

    Period objects are deliberately made timezone-naive
    to avoid pandas timezone warnings.
    """

    start_ts = ensure_utc_timestamp(start)
    end_ts = ensure_utc_timestamp(end)

    start_naive = start_ts.tz_localize(None)
    end_naive = end_ts.tz_localize(None)

    current = start_naive.to_period("M")
    last = end_naive.to_period("M")

    months = []

    while current <= last:
        months.append(str(current))
        current += 1

    return months


# ============================================================
# DOWNLOAD
# ============================================================

def download_archive(url):
    try:

        response = session.get(
            url,
            timeout=60,
        )

        if response.status_code == 200:
            return response.content

        return None

    except requests.RequestException:
        return None


# ============================================================
# READ BINANCE ZIP
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
                "No CSV file inside Binance archive."
            )

        with archive.open(
            csv_files[0]
        ) as file:

            df = pd.read_csv(
                file,
                header=None,
            )

    # --------------------------------------------------------
    # Binance sometimes includes a header row.
    # --------------------------------------------------------

    if len(df) > 0:

        first_value = (
            str(df.iloc[0, 0])
            .strip()
            .lower()
        )

        if first_value in (
            "open_time",
            "open time",
        ):

            df = df.iloc[1:].copy()

    if df.shape[1] < 12:

        raise RuntimeError(
            f"Unexpected Binance CSV columns: "
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
# FETCH BINANCE KLINES
# ============================================================

def fetch_binance_klines(
    symbol,
    interval,
    start,
    end,
):

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_ts = ensure_utc_timestamp(
        start
    )

    end_ts = ensure_utc_timestamp(
        end
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

            cached["open_time"] = (
                pd.to_datetime(
                    cached["open_time"],
                    utc=True,
                )
            )

            cached = (
                cached
                .sort_values("open_time")
                .reset_index(drop=True)
            )

            if (
                len(cached) > 0
                and cached["open_time"].min()
                <= start_ts
                and cached["open_time"].max()
                >= end_ts
            ):

                return cached

        except Exception:

            pass

    frames = []

    # ========================================================
    # MONTHLY ARCHIVES
    # ========================================================

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

        blob = download_archive(
            url
        )

        if blob is not None:

            try:

                frame = read_binance_zip(
                    blob
                )

                frames.append(
                    frame
                )

            except Exception as exc:

                print(
                    f"  Monthly parse failed "
                    f"{symbol} {ym}: {exc}"
                )

        time.sleep(0.05)

    # ========================================================
    # DAILY FALLBACK
    # ========================================================

    if frames:

        existing = pd.concat(
            frames,
            ignore_index=True,
        )

        existing_days = set(
            existing["open_time"]
            .dt.strftime("%Y-%m-%d")
        )

    else:

        existing = pd.DataFrame()

        existing_days = set()

    daily_start = (
        start_ts
        .floor("D")
    )

    daily_end = (
        end_ts
        .floor("D")
    )

    for day in pd.date_range(
        daily_start,
        daily_end,
        freq="D",
    ):

        date_string = (
            day.strftime(
                "%Y-%m-%d"
            )
        )

        if date_string in existing_days:
            continue

        url = (
            "https://data.binance.vision/data/"
            "futures/um/daily/klines/"
            f"{symbol}/{interval}/"
            f"{symbol}-{interval}-{date_string}.zip"
        )

        blob = download_archive(
            url
        )

        if blob is not None:

            try:

                frame = read_binance_zip(
                    blob
                )

                frames.append(
                    frame
                )

            except Exception as exc:

                print(
                    f"  Daily parse failed "
                    f"{symbol} {date_string}: {exc}"
                )

        time.sleep(0.02)

    # ========================================================
    # NO DATA
    # ========================================================

    if not frames:

        raise RuntimeError(
            f"{symbol} {interval}: "
            "no Binance historical data downloaded."
        )

    # ========================================================
    # MERGE
    # ========================================================

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df = (
        df
        .drop_duplicates(
            subset=["open_time"]
        )
        .sort_values(
            "open_time"
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # DATE FILTER
    # ========================================================

    df = df[
        (
            df["open_time"]
            >= start_ts
        )
        &
        (
            df["open_time"]
            < end_ts
        )
    ].copy()

    if len(df) == 0:

        raise RuntimeError(
            f"{symbol} {interval}: "
            "empty dataframe after date filtering."
        )

    # ========================================================
    # SAVE CACHE
    # ========================================================

    df.to_csv(
        cache_file,
        index=False,
    )

    return df


# ============================================================
# DATA CONTINUITY AUDIT
# ============================================================

def audit_continuity(
    df,
    interval_minutes,
    symbol,
    interval_name,
):

    if len(df) < 10:

        raise RuntimeError(
            f"{symbol} {interval_name}: "
            "too few rows."
        )

    timestamps = (
        df["open_time"]
        .sort_values()
        .drop_duplicates()
    )

    deltas = (
        timestamps
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60.0
    )

    bad = deltas[
        deltas
        > interval_minutes * 1.10
    ]

    if len(bad) > 0:

        maximum_gap = float(
            bad.max()
        )

        raise RuntimeError(
            f"{symbol} {interval_name}: "
            f"data gap detected. "
            f"Max gap={maximum_gap:.2f} minutes."
        )


# ============================================================
# 4H FEATURES
# ============================================================

def prepare_4h(df):

    x = df.copy()

    x["range"] = (
        x["high"]
        - x["low"]
    )

    previous_close = (
        x["close"]
        .shift(1)
    )

    true_range = pd.concat(
        [
            x["high"] - x["low"],
            (
                x["high"]
                - previous_close
            ).abs(),
            (
                x["low"]
                - previous_close
            ).abs(),
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
        x["range"]
        / x["close"]
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
        x["ema20"]
        > x["ema50"],
        1,
        -1,
    )

    return x


# ============================================================
# ATTACH CLOSED 4H RANGE TO 15M
# ============================================================

def attach_parent_range(
    h4,
    m15,
):

    reference = h4.copy()

    reference[
        "htf_close_time"
    ] = (
        reference["open_time"]
        + pd.Timedelta(
            hours=4
        )
    )

    reference = reference[
        [
            "htf_close_time",
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
        "parent_open",
        "parent_high",
        "parent_low",
        "parent_close",
        "parent_range",
        "parent_atr",
        "parent_range_pct",
        "parent_bias",
    ]

    ltf = (
        m15
        .copy()
        .sort_values(
            "open_time"
        )
    )

    # IMPORTANT:
    #
    # A 4H candle becomes available ONLY after it closes.
    #
    # Therefore:
    #
    # 08:00 -> 12:00
    #
    # can only be used by 15M candles
    # starting at 12:00.

    ltf = pd.merge_asof(
        ltf,
        reference.sort_values(
            "htf_close_time"
        ),
        left_on="open_time",
        right_on="htf_close_time",
        direction="backward",
    )

    ltf = ltf.dropna(
        subset=[
            "parent_high",
            "parent_low",
            "parent_bias",
        ]
    ).reset_index(
        drop=True
    )

    return ltf


# ============================================================
# CRT SETUP DETECTION
# ============================================================

def find_crt_setups(
    df,
    symbol,
):

    bars = (
        df
        .sort_values(
            "open_time"
        )
        .reset_index(
            drop=True
        )
    )

    trades = []

    i = (
        MSS_LOOKBACK
        + 2
    )

    total = len(bars)

    while i < total - 20:

        current = bars.iloc[i]

        parent_high = float(
            current.parent_high
        )

        parent_low = float(
            current.parent_low
        )

        parent_range = (
            parent_high
            - parent_low
        )

        if (
            not np.isfinite(
                parent_high
            )
            or not np.isfinite(
                parent_low
            )
            or parent_range <= 0
        ):

            i += 1
            continue

        parent_time = (
            current.htf_close_time
        )

        # ====================================================
        # CRT SWEEP
        # ====================================================

        side = None

        # LONG:
        # sweep parent low,
        # close back inside range.

        if (
            current.low
            < parent_low
            and
            current.close
            > parent_low
            and
            current.close
            < parent_high
        ):

            side = "LONG"

            sweep_price = float(
                current.low
            )

        # SHORT:
        # sweep parent high,
        # close back inside range.

        elif (
            current.high
            > parent_high
            and
            current.close
            < parent_high
            and
            current.close
            > parent_low
        ):

            side = "SHORT"

            sweep_price = float(
                current.high
            )

        else:

            i += 1
            continue

        # ====================================================
        # SWEEP DEPTH
        # ====================================================

        if side == "LONG":

            sweep_depth = (
                parent_low
                - sweep_price
            ) / parent_range

        else:

            sweep_depth = (
                sweep_price
                - parent_high
            ) / parent_range

        if (
            sweep_depth < 0
            or sweep_depth
            > MAX_SWEEP_DEPTH
        ):

            i += 1
            continue

        # ====================================================
        # HTF BIAS
        # ====================================================

        parent_bias = int(
            current.parent_bias
        )

        if (
            side == "LONG"
            and parent_bias != 1
        ):

            i += 1
            continue

        if (
            side == "SHORT"
            and parent_bias != -1
        ):

            i += 1
            continue

        # ====================================================
        # 15M MSS
        # ====================================================

        mss_index = None

        search_end = min(
            i
            + MSS_MAX_BARS
            + 1,
            total,
        )

        for j in range(
            i + 1,
            search_end,
        ):

            if side == "LONG":

                local_high = (
                    bars["high"]
                    .iloc[
                        max(
                            0,
                            j
                            - MSS_LOOKBACK,
                        ):j
                    ]
                    .max()
                )

                if (
                    bars["close"]
                    .iloc[j]
                    > local_high
                ):

                    mss_index = j
                    break

            else:

                local_low = (
                    bars["low"]
                    .iloc[
                        max(
                            0,
                            j
                            - MSS_LOOKBACK,
                        ):j
                    ]
                    .min()
                )

                if (
                    bars["close"]
                    .iloc[j]
                    < local_low
                ):

                    mss_index = j
                    break

        if mss_index is None:

            i += 1
            continue

        # ====================================================
        # RETEST ZONE
        # ====================================================

        mss_open = float(
            bars["open"]
            .iloc[mss_index]
        )

        mss_close = float(
            bars["close"]
            .iloc[mss_index]
        )

        zone_high = max(
            mss_open,
            mss_close,
        )

        zone_low = min(
            mss_open,
            mss_close,
        )

        entry_index = None

        retest_end = min(
            mss_index
            + RETEST_MAX_BARS
            + 1,
            total,
        )

        for j in range(
            mss_index + 1,
            retest_end,
        ):

            touches_zone = (
                bars["low"].iloc[j]
                <= zone_high
                and
                bars["high"].iloc[j]
                >= zone_low
            )

            if touches_zone:

                # Retest candle is confirmation.
                #
                # Actual execution happens
                # at the NEXT 15M OPEN.

                entry_index = j + 1

                break

        if (
            entry_index is None
            or entry_index >= total
        ):

            i = (
                mss_index
                + 1
            )

            continue

        entry_price = float(
            bars["open"]
            .iloc[entry_index]
        )

        # Entry must remain inside original range.

        if not (
            parent_low
            < entry_price
            < parent_high
        ):

            i = entry_index
            continue

        # ====================================================
        # STOP / TARGET
        # ====================================================

        stop_price = sweep_price

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
            risk
            / entry_price
        )

        if (
            risk_pct
            < MIN_RISK_PCT
            or
            risk_pct
            > MAX_RISK_PCT
        ):

            i = entry_index
            continue

        # ====================================================
        # 1
