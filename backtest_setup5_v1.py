# ============================================================
# SETUP 5 V1
# Liquidity -> CHOCH -> Order Block -> Sweep -> Return
#
# Execution : 15m
# HTF       : 4H
# RR        : 1:2
# Capital   : $1,000
# Margin    : $100
# Leverage  : 50x
#
# IMPORTANT:
# - No lookahead
# - No future leak
# - Confirmed pivots only
# - No same-candle re-entry
# - Max 1 simultaneous trade per symbol
# - Different symbols may overlap
# - SL wins if SL and TP are both touched on same candle
# - No timeout / BE / trailing / partial exit
# ============================================================

import io
import os
import sys
import time
import math
import zipfile
import warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DATA_BASE = "https://data.binance.vision/data/futures/um"

INTERVAL = "15m"
HTF_INTERVAL = "4h"

TEST_DAYS = 365
WARMUP_DAYS = 90

PIVOT = 2

RR = 2.0

INITIAL_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE = 50.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003

ALIGN_TOL = 0.004
MAX_LIQ_TO_OB_BARS = 24

SL_BUFFER = 0.0005

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

REQUEST_TIMEOUT = 30
MAX_RETRIES = 4

# Parallel downloading.
# 4 is intentionally conservative for GitHub Actions.
DOWNLOAD_WORKERS = 4

CACHE_DIR = Path(".binance_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TIME
# ============================================================

UTC = timezone.utc


def utc_now():
    return datetime.now(UTC)


def month_start(dt):
    return pd.Timestamp(
        year=dt.year,
        month=dt.month,
        day=1,
        tz="UTC",
    )


def next_month(dt):
    if dt.month == 12:
        return pd.Timestamp(
            year=dt.year + 1,
            month=1,
            day=1,
            tz="UTC",
        )

    return pd.Timestamp(
        year=dt.year,
        month=dt.month + 1,
        day=1,
        tz="UTC",
    )


def iter_months(start, end):
    cur = month_start(start)
    end_m = month_start(end)

    while cur <= end_m:
        yield cur
        cur = next_month(cur)


# ============================================================
# GLOBAL DATES
# ============================================================

TODAY = pd.Timestamp(utc_now())

OOS_END = TODAY.floor("15min") - pd.Timedelta(minutes=15)

OOS_START = OOS_END - pd.Timedelta(days=TEST_DAYS) + pd.Timedelta(minutes=15)

DATA_START = OOS_START - pd.Timedelta(days=WARMUP_DAYS)

RESEARCH_START = OOS_START

# Development / discovery period.
DISCOVERY_DAYS = 255
DISCOVERY_END = RESEARCH_START + pd.Timedelta(days=DISCOVERY_DAYS)


# ============================================================
# HTTP SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 Setup5Backtest/1.0",
        "Accept": "*/*",
    }
)


# ============================================================
# DOWNLOAD HELPERS
# ============================================================

def cache_path(symbol, year, month, day=None):
    if day is None:
        name = f"{symbol}-{INTERVAL}-{year:04d}-{month:02d}.zip"
    else:
        name = (
            f"{symbol}-{INTERVAL}-"
            f"{year:04d}-{month:02d}-{day:02d}.zip"
        )

    return CACHE_DIR / name


def download_bytes(url):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            r = SESSION.get(
                url,
                timeout=REQUEST_TIMEOUT,
            )

            if r.status_code == 200:
                return r.content

            if r.status_code == 404:
                return None

            last_error = (
                f"HTTP {r.status_code}: {url}"
            )

        except Exception as e:
            last_error = repr(e)

        if attempt < MAX_RETRIES:
            time.sleep(1.5 * attempt)

    raise RuntimeError(
        f"Download failed after {MAX_RETRIES} attempts:\n"
        f"{url}\n"
        f"{last_error}"
    )


def load_zip_csv(raw_bytes):
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:

        csv_names = [
            n for n in z.namelist()
            if n.lower().endswith(".csv")
        ]

        if not csv_names:
            raise RuntimeError(
                "ZIP contains no CSV file"
            )

        with z.open(csv_names[0]) as f:
            df = pd.read_csv(
                f,
                header=None,
            )

    return df


def normalize_kline_df(df):
    if df.empty:
        return df

    expected_cols = [
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

    if len(df.columns) < 12:
        raise RuntimeError(
            f"Unexpected Binance kline columns: {len(df.columns)}"
        )

    df = df.iloc[:, :12].copy()
    df.columns = expected_cols

    # Robust timestamp conversion.
    for col in ["open_time", "close_time"]:
        raw = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        # Binance archives normally use milliseconds.
        # Keep this defensive for possible microsecond files.
        sample = raw.dropna()

        if not sample.empty and sample.iloc[0] > 10**14:
            unit = "us"
        else:
            unit = "ms"

        df[col] = pd.to_datetime(
            raw,
            unit=unit,
            utc=True,
            errors="coerce",
        )

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "open_time",
            "close_time",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    df = (
        df.sort_values("open_time")
        .drop_duplicates(
            subset=["open_time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# ARCHIVE DOWNLOAD
# ============================================================

def fetch_monthly(symbol, ym):
    year = ym.year
    month = ym.month

    path = cache_path(
        symbol,
        year,
        month,
    )

    if path.exists() and path.stat().st_size > 100:
        try:
            return load_zip_csv(
                path.read_bytes()
            )
        except Exception:
            path.unlink(missing_ok=True)

    url = (
        f"{DATA_BASE}/monthly/klines/"
        f"{symbol}/{INTERVAL}/"
        f"{symbol}-{INTERVAL}-"
        f"{year:04d}-{month:02d}.zip"
    )

    raw = download_bytes(url)

    if raw is None:
        return None

    path.write_bytes(raw)

    return load_zip_csv(raw)


def fetch_daily(symbol, day):
    year = day.year
    month = day.month
    date_str = (
        f"{year:04d}-{month:02d}-{day.day:02d}"
    )

    path = cache_path(
        symbol,
        year,
        month,
        day.day,
    )

    if path.exists() and path.stat().st_size > 100:
        try:
            return load_zip_csv(
                path.read_bytes()
            )
        except Exception:
            path.unlink(missing_ok=True)

    url = (
        f"{DATA_BASE}/daily/klines/"
        f"{symbol}/{INTERVAL}/"
        f"{symbol}-{INTERVAL}-"
        f"{date_str}.zip"
    )

    raw = download_bytes(url)

    if raw is None:
        return None

    path.write_bytes(raw)

    return load_zip_csv(raw)


# ============================================================
# SYMBOL LOADER
# ============================================================

def load_symbol(symbol):
    """
    Correct archive strategy:

    - First requested month  -> daily
    - Last requested month   -> daily
    - Complete middle months -> monthly

    This prevents the current/final month from requiring a
    monthly archive that may not exist yet.
    """

    frames = []

    first_month = month_start(DATA_START)
    last_month = month_start(OOS_END)

    months = list(
        iter_months(
            DATA_START,
            OOS_END,
        )
    )

    for ym in months:

        use_daily = (
            ym == first_month
            or ym == last_month
        )

        if use_daily:

            day = ym

            while day < next_month(ym):

                if (
                    day >= DATA_START.normalize()
                    and day <= OOS_END.normalize()
                ):
                    df = fetch_daily(
                        symbol,
                        day,
                    )

                    if df is not None:
                        frames.append(df)

                day += pd.Timedelta(days=1)

        else:

            df = fetch_monthly(
                symbol,
                ym,
            )

            if df is None:
                raise RuntimeError(
                    "Historical monthly archive missing:\n"
                    f"{symbol} {ym.strftime('%Y-%m')}"
                )

            frames.append(df)

    if not frames:
        raise RuntimeError(
            f"No data downloaded for {symbol}"
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df = normalize_kline_df(df)

    df = df[
        (df["open_time"] >= DATA_START)
        &
        (df["open_time"] <= OOS_END)
    ].copy()

    df = (
        df.sort_values("open_time")
        .drop_duplicates(
            "open_time"
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise RuntimeError(
            f"No rows after trimming for {symbol}"
        )

    # --------------------------------------------------------
    # Remove incomplete candle only if it is genuinely ahead
    # of the requested historical endpoint.
    # --------------------------------------------------------

    now = utc_now()

    df = df[
        df["close_time"] <= pd.Timestamp(
            now,
            tz="UTC",
        )
    ].copy()

    df = df.reset_index(drop=True)

    # --------------------------------------------------------
    # Strict continuity.
    # --------------------------------------------------------

    diffs = (
        df["open_time"]
        .diff()
        .dropna()
    )

    bad = diffs[
        diffs != pd.Timedelta(minutes=15)
    ]

    if not bad.empty:
        first_bad = bad.index[0]

        prev_t = df.loc[
            first_bad - 1,
            "open_time",
        ]

        curr_t = df.loc[
            first_bad,
            "open_time",
        ]

        raise RuntimeError(
            f"{symbol}: 15m data gap detected:\n"
            f"{prev_t} -> {curr_t}"
        )

    # Expected count approximately:
    expected = int(
        (
            OOS_END - DATA_START
        ).total_seconds()
        / 900
    ) + 1

    if len(df) < expected * 0.995:
        raise RuntimeError(
            f"{symbol}: insufficient candles. "
            f"got={len(df)}, expected≈{expected}"
        )

    return df


# ============================================================
# PIVOTS
# ============================================================

def confirmed_pivots(df):
    """
    A pivot at i becomes known only at i + PIVOT.

    Therefore pivot information is NEVER used before
    its confirmation candle.
    """

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    n = len(df)

    ph = np.zeros(n, dtype=bool)
    pl = np.zeros(n, dtype=bool)

    for i in range(
        PIVOT,
        n - PIVOT,
    ):

        h = highs[i]
        l = lows[i]

        left_h = highs[
            i - PIVOT:i
        ]

        right_h = highs[
            i + 1:i + PIVOT + 1
        ]

        left_l = lows[
            i - PIVOT:i
        ]

        right_l = lows[
            i + 1:i + PIVOT + 1
        ]

        if (
            h > left_h.max()
            and h >= right_h.max()
        ):
            ph[i] = True

        if (
            l < left_l.min()
            and l <= right_l.min()
        ):
            pl[i] = True

    return ph, pl


def confirmed_pivot_indices(
    df,
    pivot_type,
):
    ph, pl = confirmed_pivots(df)

    if pivot_type == "high":
        return np.flatnonzero(ph)

    return np.flatnonzero(pl)


def pivots_known_before(
    pivot_indices,
    candle_index,
):
    """
    Return pivot centers whose confirmation candle has already
    closed before candle_index.

    Pivot center i is known at i + PIVOT.
    """

    cutoff = candle_index - PIVOT

    if cutoff < 0:
        return np.array([], dtype=int)

    return pivot_indices[
        pivot_indices <= cutoff
    ]


# ============================================================
# TREND
# ============================================================

def last_two_confirmed(
    indices,
    end_index,
):
    known = pivots_known_before(
        indices,
        end_index,
    )

    if len(known) < 2:
        return None

    return int(known[-2]), int(known[-1])


def downtrend(
    df,
    pivot_highs,
    pivot_lows,
    at_index,
):
    hs = last_two_confirmed(
        pivot_highs,
        at_index,
    )

    ls = last_two_confirmed(
        pivot_lows,
        at_index,
    )

    if hs is None or ls is None:
        return False

    h1, h2 = hs
    l1, l2 = ls

    return (
        df.iloc[h2]["high"]
        < df.iloc[h1]["high"]
        and
        df.iloc[l2]["low"]
        < df.iloc[l1]["low"]
    )


def uptrend(
    df,
    pivot_highs,
    pivot_lows,
    at_index,
):
    hs = last_two_confirmed(
        pivot_highs,
        at_index,
    )

    ls = last_two_confirmed(
        pivot_lows,
        at_index,
    )

    if hs is None or ls is None:
        return False

    h1, h2 = hs
    l1, l2 = ls

    return (
        df.iloc[h2]["high"]
        > df.iloc[h1]["high"]
        and
        df.iloc[l2]["low"]
        > df.iloc[l1]["low"]
    )


# ============================================================
# HTF ZONES
# ============================================================

def build_htf_zones(htf):
    """
    HTF demand/supply zones are generated only from confirmed
    4H pivots.

    Long:
        pivot low -> demand
        zone_low  = candle low
        zone_high = max(open, close)

    Short:
        pivot high -> supply
        zone_low  = min(open, close)
        zone_high = candle high
    """

    ph, pl = confirmed_pivots(
        htf
    )

    zones = []

    for i in np.flatnonzero(pl):

        known_at = i + PIVOT

        if known_at >= len(htf):
            continue

        row = htf.iloc[i]

        zones.append(
            {
                "side": "LONG",
                "pivot_index": int(i),
                "known_index": int(known_at),
                "known_time": htf.iloc[
                    known_at
                ]["open_time"],
                "zone_low": float(row["low"]),
                "zone_high": float(
                    max(
                        row["open"],
                        row["close"],
                    )
                ),
            }
        )

    for i in np.flatnonzero(ph):

        known_at = i + PIVOT

        if known_at >= len(htf):
            continue

        row = htf.iloc[i]

        zones.append(
            {
                "side": "SHORT",
                "pivot_index": int(i),
                "known_index": int(known_at),
                "known_time": htf.iloc[
                    known_at
                ]["open_time"],
                "zone_low": float(
                    min(
                        row["open"],
                        row["close"],
                    )
                ),
                "zone_high": float(row["high"]),
            }
        )

    zones.sort(
        key=lambda x: x["known_time"]
    )

    return zones


def map_htf_zones_to_15m(
    df,
    htf,
    zones,
):
    """
    HTF zone becomes usable only after the 4H confirmation
    candle is closed.

    Mapping is timestamp based and therefore causal.
    """

    result = []

    times = df["open_time"].to_numpy()

    for z in zones:

        idx = np.searchsorted(
            times,
            np.datetime64(
                z["known_time"]
            ),
            side="left",
        )

        if idx >= len(df):
            continue

        x = dict(z)
        x["execution_known_index"] = int(idx)

        result.append(x)

    return result


# ============================================================
# ZONE REACTION
# ============================================================

def find_zone_reaction(
    df,
    zone,
    start_index,
    end_index,
):
    """
    First valid touch.

    If the zone is broken through before reaction,
    the zone is invalidated.
    """

    zl = zone["zone_low"]
    zh = zone["zone_high"]

    side = zone["side"]

    for i in range(
        max(start_index, zone["execution_known_index"]),
        min(end_index, len(df) - 1),
    ):

        row = df.iloc[i]

        if side == "LONG":

            # invalidation before reaction
            if row["close"] < zl:
                return None

            touched = (
                row["low"] <= zh
                and row["high"] >= zl
            )

        else:

            if row["close"] > zh:
                return None

            touched = (
                row["high"] >= zl
                and row["low"] <= zh
            )

        if touched:
            return i

    return None


# ============================================================
# CHOCH
# ============================================================

def find_choch_long(
    df,
    pivot_highs,
    reaction_idx,
    end_index,
):
    """
    Bullish CHOCH:
    close > latest confirmed lower high.

    The CHOCH candle itself is NOT an entry candle.
    """

    known = pivots_known_before(
        pivot_highs,
        reaction_idx + 1,
    )

    if len(known) == 0:
        return None

    lh_idx = int(known[-1])
    lh_price = float(
        df.iloc[lh_idx]["high"]
    )

    for i in range(
        reaction_idx + 1,
        min(end_index, len(df)),
    ):

        if (
            df.iloc[i]["close"]
            > lh_price
        ):
            return i

    return None


def find_choch_short(
    df,
    pivot_lows,
    reaction_idx,
    end_index,
):
    """
    Bearish CHOCH:
    close < latest confirmed higher low.
    """

    known = pivots_known_before(
        pivot_lows,
        reaction_idx + 1,
    )

    if len(known) == 0:
        return None

    hl_idx = int(known[-1])
    hl_price = float(
        df.iloc[hl_idx]["low"]
    )

    for i in range(
        reaction_idx + 1,
        min(end_index, len(df)),
    ):

        if (
            df.iloc[i]["close"]
            < hl_price
        ):
            return i

    return None


# ============================================================
# ORDER BLOCK
# ============================================================

def find_ob_long(
    df,
    pivot_lows,
    reaction_idx,
    choch_idx,
):
    """
    Lowest confirmed swing low between reaction and CHOCH.
    """

    known = pivots_known_before(
        pivot_lows,
        choch_idx,
    )

    known = known[
        known >= reaction_idx
    ]

    if len(known) == 0:
        return None

    best = min(
        known,
        key=lambda i: df.iloc[i]["low"],
    )

    row = df.iloc[best]

    return {
        "index": int(best),
        "low": float(row["low"]),
        "high": float(
            max(
                row["open"],
                row["close"],
            )
        ),
    }


def find_ob_short(
    df,
    pivot_highs,
    reaction_idx,
    choch_idx,
):
    """
    Highest confirmed swing high between reaction and CHOCH.
    """

    known = pivots_known_before(
        pivot_highs,
        choch_idx,
    )

    known = known[
        known >= reaction_idx
    ]

    if len(known) == 0:
        return None

    best = max(
        known,
        key=lambda i: df.iloc[i]["high"],
    )

    row = df.iloc[best]

    return {
        "index": int(best),
        "low": float(
            min(
                row["open"],
                row["close"],
            )
        ),
        "high": float(row["high"]),
    }


# ============================================================
# LIQUIDITY
# ============================================================

def find_aligned_liquidity_long(
    df,
    pivot_lows,
    choch_idx,
    ob_touch_idx,
):
    known = pivots_known_before(
        pivot_lows,
        ob_touch_idx,
    )

    known = known[
        (known > choch_idx)
        &
        (known < ob_touch_idx)
    ]

    if len(known) < 2:
        return None

    # Search pairs from most recent backwards.
    # This keeps liquidity close to the OB and avoids using
    # an arbitrary old pair.
    for a in range(
        len(known) - 2,
        -1,
        -1,
    ):

        i1 = int(known[a])

        for b in range(
            len(known) - 1,
            a,
            -1,
        ):

            i2 = int(known[b])

            p1 = float(
                df.iloc[i1]["low"]
            )

            p2 = float(
                df.iloc[i2]["low"]
            )

            reference = max(
                abs(p1),
                abs(p2),
                1e-12,
            )

            if (
                abs(p1 - p2)
                / reference
                <= ALIGN_TOL
            ):
                return {
                    "indices": [i1, i2],
                    "level": (p1 + p2) / 2.0,
                }

    return None


def find_aligned_liquidity_short(
    df,
    pivot_highs,
    choch_idx,
    ob_touch_idx,
):
    known = pivots_known_before(
        pivot_highs,
        ob_touch_idx,
    )

    known = known[
        (known > choch_idx)
        &
        (known < ob_touch_idx)
    ]

    if len(known) < 2:
        return None

    for a in range(
        len(known) - 2,
        -1,
        -1,
    ):

        i1 = int(known[a])

        for b in range(
            len(known) - 1,
            a,
            -1,
        ):

            i2 = int(known[b])

            p1 = float(
                df.iloc[i1]["high"]
            )

            p2 = float(
                df.iloc[i2]["high"]
            )

            reference = max(
                abs(p1),
                abs(p2),
                1e-12,
            )

            if (
                abs(p1 - p2)
                / reference
                <= ALIGN_TOL
            ):
                return {
                    "indices": [i1, i2],
                    "level": (p1 + p2) / 2.0,
                }

    return None


# ============================================================
# OB TOUCH
# ============================================================

def find_ob_touch(
    df,
    ob,
    start_index,
    end_index,
):
    for i in range(
        start_index,
        min(end_index, len(df)),
    ):

        row = df.iloc[i]

        touched = (
            row["low"] <= ob["high"]
            and row["high"] >= ob["low"]
        )

        if touched:
            return i

    return None


# ============================================================
# LIQUIDITY SWEEP
# ============================================================

def find_sweep_long(
    df,
    level,
    start_index,
    end_index,
):
    for i in range(
        start_index,
        min(end_index, len(df)),
    ):

        row = df.iloc[i]

        if (
            row["low"] < level
            and row["close"] >= level
        ):
            return i

    return None


def find_sweep_short(
    df,
    level,
    start_index,
    end_index,
):
    for i in range(
        start_index,
        min(end_index, len(df)),
    ):

        row = df.iloc[i]

        if (
            row["high"] > level
            and row["close"] <= level
        ):
            return i

    return None


# ============================================================
# STRUCTURAL TARGET
# ============================================================

def structural_target_long(
    df,
    pivot_highs,
    choch_idx,
    entry_signal_idx,
):
    known = pivots_known_before(
        pivot_highs,
        entry_signal_idx,
    )

    known = known[
        known > choch_idx
    ]

    if len(known) == 0:
        return None, None

    best = max(
        known,
        key=lambda i: df.iloc[i]["high"],
    )

    return (
        float(df.iloc[best]["high"]),
        int(best),
    )


def structural_target_short(
    df,
    pivot_lows,
    choch_idx,
    entry_signal_idx,
):
    known = pivots_known_before(
        pivot_lows,
        entry_signal_idx,
    )

    known = known[
        known > choch_idx
    ]

    if len(known) == 0:
        return None, None

    best = min(
        known,
        key=lambda i: df.iloc[i]["low"],
    )

    return (
        float(df.iloc[best]["low"]),
        int(best),
    )


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(
    symbol,
    df,
    htf,
):
    ph, pl = confirmed_pivots(
        df
    )

    pivot_highs = np.flatnonzero(ph)
    pivot_lows = np.flatnonzero(pl)

    htf_zones = build_htf_zones(
        htf
    )

    zones = map_htf_zones_to_15m(
        df,
        htf,
        htf_zones,
    )

    candidates = []

    # --------------------------------------------------------
    # Process zones in chronological order.
    # --------------------------------------------------------

    for zone in zones:

        known_idx = zone[
            "execution_known_index"
        ]

        if known_idx < 0:
            continue

        # Search only inside the actual research + OOS area.
        if known_idx >= len(df) - 20:
            continue

        side = zone["side"]

        # ----------------------------------------------------
        # Need trend before HTF reaction.
        # ----------------------------------------------------

        reaction_idx = find_zone_reaction(
            df,
            zone,
            known_idx,
            len(df) - 10,
        )

        if reaction_idx is None:
            continue

        if side == "LONG":

            if not downtrend(
                df,
                pivot_highs,
                pivot_lows,
                reaction_idx,
            ):
                continue

            choch_idx = find_choch_long(
                df,
                pivot_highs,
                reaction_idx,
                len(df) - 10,
            )

            if choch_idx is None:
                continue

            ob = find_ob_long(
                df,
                pivot_lows,
                reaction_idx,
                choch_idx,
            )

            if ob is None:
                continue

            # OB must be formed before CHOCH confirmation.
            if ob["index"] >= choch_idx:
                continue

            # ------------------------------------------------
            # Search for first OB touch after CHOCH.
            # This is NOT entry yet.
            # ------------------------------------------------

            ob_touch = find_ob_touch(
                df,
                ob,
                choch_idx + 1,
                len(df) - 2,
            )

            if ob_touch is None:
                continue

            liquidity = find_aligned_liquidity_long(
                df,
                pivot_lows,
                choch_idx,
                ob_touch,
            )

            if liquidity is None:
                continue

            liq2 = max(
                liquidity["indices"]
            )

            if (
                ob_touch - liq2
                > MAX_LIQ_TO_OB_BARS
            ):
                continue

            sweep_idx = find_sweep_long(
                df,
                liquidity["level"],
                liq2 + 1,
                ob_touch + 1,
            )

            if sweep_idx is None:
                continue

            # Sweep must occur before OB return.
            if sweep_idx >= ob_touch:
                continue

            # ------------------------------------------------
            # Entry signal = first return to OB after sweep.
            # ------------------------------------------------

            entry_signal = find_ob_touch(
                df,
                ob,
                sweep_idx + 1,
                len(df) - 2,
            )

            if entry_signal is None:
                continue

            entry_idx = entry_signal + 1

            if entry_idx >= len(df):
                continue

            entry_price = (
                float(
                    df.iloc[entry_idx]["open"]
                )
                * (1.0 + SLIPPAGE)
            )

            stop = (
                ob["low"]
                * (1.0 - SL_BUFFER)
            )

            risk = entry_price - stop

            if risk <= 0:
                continue

            tp = entry_price + RR * risk

            structural, structural_idx = (
                structural_target_long(
                    df,
                    pivot_highs,
                    choch_idx,
                    entry_signal,
                )
            )

            if structural is None:
                continue

            # Structural target itself must already be known
            # before entry.
            if structural_idx >= entry_idx:
                continue

            if tp > structural:
                continue

            candidates.append(
                {
                    "symbol": symbol,
                    "side": "LONG",
                    "zone_index": known_idx,
                    "reaction_index": reaction_idx,
                    "choch_index": choch_idx,
                    "ob_index": ob["index"],
                    "liq_indices": liquidity[
                        "indices"
                    ],
                    "liq_level": liquidity[
                        "level"
                    ],
                    "sweep_index": sweep_idx,
                    "entry_signal_index": entry_signal,
                    "entry_index": entry_idx,
                    "entry_price": entry_price,
                    "stop": stop,
                    "target": tp,
                    "structural_target": structural,
                    "structural_target_index": structural_idx,
                }
            )

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        else:

            if not uptrend(
                df,
                pivot_highs,
                pivot_lows,
                reaction_idx,
            ):
                continue

            choch_idx = find_choch_short(
                df,
                pivot_lows,
                reaction_idx,
                len(df) - 10,
            )

            if choch_idx is None:
                continue

            ob = find_ob_short(
                df,
                pivot_highs,
                reaction_idx,
                choch_idx,
            )

            if ob is None:
                continue

            if ob["index"] >= choch_idx:
                continue

            ob_touch = find_ob_touch(
                df,
                ob,
                choch_idx + 1,
                len(df) - 2,
            )

            if ob_touch is None:
                continue

            liquidity = find_aligned_liquidity_short(
                df,
                pivot_highs,
                choch_idx,
                ob_touch,
            )

            if liquidity is None:
                continue

            liq2 = max(
                liquidity["indices"]
            )

            if (
                ob_touch - liq2
                > MAX_LIQ_TO_OB_BARS
            ):
                continue

            sweep_idx = find_sweep_short(
                df,
                liquidity["level"],
                liq2 + 1,
                ob_touch + 1,
            )

            if sweep_idx is None:
                continue

            if sweep_idx >= ob_touch:
                continue

            entry_signal = find_ob_touch(
                df,
                ob,
                sweep_idx + 1,
                len(df) - 2,
            )

            if entry_signal is None:
                continue

            entry_idx = entry_signal + 1

            if entry_idx >= len(df):
                continue

            entry_price = (
                float(
                    df.iloc[entry_idx]["open"]
                )
                * (1.0 - SLIPPAGE)
            )

            stop = (
                ob["high"]
                * (1.0 + SL_BUFFER)
            )

            risk = stop - entry_price

            if risk <= 0:
                continue

            tp = entry_price - RR * risk

            structural, structural_idx = (
                structural_target_short(
                    df,
                    pivot_lows,
                    choch_idx,
                    entry_signal,
                )
            )

            if structural is None:
                continue

            if structural_idx >= entry_idx:
                continue

            if tp < structural:
                continue

            candidates.append(
                {
                    "symbol": symbol,
                    "side": "SHORT",
                    "zone_index": known_idx,
                    "reaction_index": reaction_idx,
                    "choch_index": choch_idx,
                    "ob_index": ob["index"],
                    "liq_indices": liquidity[
                        "indices"
                    ],
                    "liq_level": liquidity[
                        "level"
                    ],
                    "sweep_index": sweep_idx,
                    "entry_signal_index": entry_signal,
                    "entry_index": entry_idx,
                    "entry_price": entry_price,
                    "stop": stop,
                    "target": tp,
                    "structural_target": structural,
                    "structural_target_index": structural_idx,
                }
            )

    # Remove exact duplicate entries.
    unique = {}

    for c in candidates:

        key = (
            c["symbol"],
            c["side"],
            c["entry_index"],
        )

        unique[key] = c

    return sorted(
        unique.values(),
        key=lambda x: x["entry_index"],
    )


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_symbol(
    symbol,
    df,
    candidates,
):
    """
    One open trade maximum per symbol.

    Different symbols are simulated independently and may
    overlap.

    Re-entry on same candle as exit is forbidden.
    """

    trades = []

    occupied_until = -1

    for c in candidates:

        entry_idx = c["entry_index"]

        # Same-symbol overlap lock.
        if entry_idx <= occupied_until:
            continue

        if entry_idx >= len(df):
            continue

        side = c["side"]

        entry = c["entry_price"]
        stop = c["stop"]
        target = c["target"]

        exit_idx = None
        exit_price = None
        result = None

        # ----------------------------------------------------
        # Important:
        # Entry candle is NOT checked for SL/TP.
        # ----------------------------------------------------

        for j in range(
            entry_idx + 1,
            len(df),
        ):

            row = df.iloc[j]

            high = float(row["high"])
            low = float(row["low"])

            if side == "LONG":

                hit_sl = low <= stop
                hit_tp = high >= target

                if hit_sl and hit_tp:
                    # Conservative assumption.
                    exit_idx = j
                    exit_price = stop
                    result = "LOSS"
                    break

                if hit_sl:
                    exit_idx = j
                    exit_price = stop
                    result = "LOSS"
                    break

                if hit_tp:
                    exit_idx = j
                    exit_price = target
                    result = "WIN"
                    break

            else:

                hit_sl = high >= stop
                hit_tp = low <= target

                if hit_sl and hit_tp:
                    exit_idx = j
                    exit_price = stop
                    result = "LOSS"
                    break

                if hit_sl:
                    exit_idx = j
                    exit_price = stop
                    result = "LOSS"
                    break

                if hit_tp:
                    exit_idx = j
                    exit_price = target
                    result = "WIN"
                    break

        # ----------------------------------------------------
        # Unresolved trade.
        # ----------------------------------------------------

        if exit_idx is None:

            trades.append(
                {
                    **c,
                    "exit_index": None,
                    "exit_price": None,
                    "result": "UNRESOLVED",
                    "pnl": None,
                    "r_multiple": None,
                }
            )

            # Unresolved trade occupies the symbol until end.
            occupied_until = len(df) - 1
            continue

        # ----------------------------------------------------
        # PnL.
        # ----------------------------------------------------

        notional = (
            MARGIN_PER_TRADE
            * LEVERAGE
        )

        if side == "LONG":
            gross = (
                (
                    exit_price - entry
                )
                / entry
            ) * notional
        else:
            gross = (
                (
                    entry - exit_price
                )
                / entry
            ) * notional

        # Entry + exit fees.
        entry_fee = (
            entry
            * (
                notional / entry
            )
            * FEE_RATE
        )

        exit_fee = (
            exit_price
            * (
                notional / entry
            )
            * FEE_RATE
        )

        # More directly:
        fees = (
            notional * FEE_RATE
            + notional * FEE_RATE
        )

        pnl = gross - fees

        # Fixed 1R dollar risk approximation from actual
        # entry/stop distance.
        risk_fraction = (
            abs(entry - stop)
            / entry
        )

        risk_dollars = (
            notional
            * risk_fraction
        )

        if risk_dollars > 0:
            r_multiple = (
                pnl / risk_dollars
            )
        else:
            r_multiple = (
                1.0
                if result == "WIN"
                else -1.0
            )

        trades.append(
            {
                **c,
                "exit_index": exit_idx,
                "exit_price": exit_price,
                "result": result,
                "pnl": pnl,
                "r_multiple": r_multiple,
            }
        )

        # IMPORTANT:
        # Exit candle is occupied.
        # Earliest next trade = exit_idx + 1.
        occupied_until = exit_idx

    return trades


# ============================================================
# AUDIT
# ============================================================

def audit_trades(
    trades,
    df_by_symbol,
):
    errors = []

    for t in trades:

        symbol = t["symbol"]
        df = df_by_symbol[symbol]

        z = t["zone_index"]
        ch = t["choch_index"]
        ob = t["ob_index"]
        sw = t["sweep_index"]
        en = t["entry_index"]
        st = t["structural_target_index"]

        liq = t["liq_indices"]

        if not (
            z < ch
        ):
            errors.append(
                f"{symbol}: zone >= CHOCH"
            )

        if not (
            ob < ch
        ):
            errors.append(
                f"{symbol}: OB >= CHOCH"
            )

        if not all(
            x < sw
            for x in liq
        ):
            errors.append(
                f"{symbol}: liquidity after sweep"
            )

        if not (
            sw < en
        ):
            errors.append(
                f"{symbol}: sweep >= entry"
            )

        if not (
            st < en
        ):
            errors.append(
                f"{symbol}: structural target not known"
            )

        # Entry must be next candle after signal.
        if (
            en
            != t["entry_signal_index"] + 1
        ):
            errors.append(
                f"{symbol}: invalid next-open entry"
            )

        # RR check.
        entry = t["entry_price"]
        stop = t["stop"]
        target = t["target"]

        if t["side"] == "LONG":

            actual_rr = (
                target - entry
            ) / (
                entry - stop
            )

        else:

            actual_rr = (
                entry - target
            ) / (
                stop - entry
            )

        if not math.isclose(
            actual_rr,
            RR,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            errors.append(
                f"{symbol}: RR violation {actual_rr}"
            )

        # No same-candle entry/exit.
        if (
            t["exit_index"] is not None
            and t["exit_index"] <= en
        ):
            errors.append(
                f"{symbol}: same-candle entry/exit"
            )

    # --------------------------------------------------------
    # Per-symbol overlap audit.
    # --------------------------------------------------------

    by_symbol = {}

    for t in trades:
        by_symbol.setdefault(
            t["symbol"],
            [],
        ).append(t)

    for symbol, arr in by_symbol.items():

        arr = sorted(
            arr,
            key=lambda x: x["entry_index"],
        )

        for a, b in zip(
            arr,
            arr[1:],
        ):

            if (
                a["exit_index"] is not None
                and b["entry_index"]
                <= a["exit_index"]
            ):
                errors.append(
                    f"{symbol}: overlapping trades"
                )

    if errors:
        raise RuntimeError(
            "INTEGRITY AUDIT FAILED:\n"
            + "\n".join(errors[:100])
        )

    return True


# ============================================================
# METRICS
# ============================================================

def max_losing_streak(trades):
    best = 0
    cur = 0

    for t in trades:

        if t["result"] == "LOSS":
            cur += 1
            best = max(
                best,
                cur,
            )

        elif t["result"] == "WIN":
            cur = 0

    return best


def metrics(
    trades,
):
    resolved = [
        t for t in trades
        if t["result"] in {
            "WIN",
            "LOSS",
        }
    ]

    if not resolved:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "wr": 0.0,
            "pf": 0.0,
            "pnl": 0.0,
            "max_streak": 0,
            "max_dd": 0.0,
            "max_dd_pct": 0.0,
        }

    wins = [
        t["pnl"]
        for t in resolved
        if t["result"] == "WIN"
    ]

    losses = [
        t["pnl"]
        for t in resolved
        if t["result"] == "LOSS"
    ]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    if gross_loss > 0:
        pf = (
            gross_profit
            / gross_loss
        )
    else:
        pf = float("inf")

    pnl = sum(
        t["pnl"]
        for t in resolved
    )

    # Sequential trade equity for reporting.
    equity = INITIAL_CAPITAL
    peak = equity
    max_dd = 0.0

    for t in resolved:

        equity += t["pnl"]

        peak = max(
            peak,
            equity,
        )

        dd = peak - equity

        max_dd = max(
            max_dd,
            dd,
        )

    max_dd_pct = (
        max_dd / peak * 100
        if peak > 0
        else 0.0
    )

    return {
        "trades": len(resolved),
        "wins": len(wins),
        "losses": len(losses),
        "wr": (
            len(wins)
            / len(resolved)
            * 100
        ),
        "pf": pf,
        "pnl": pnl,
        "max_streak": max_losing_streak(
            resolved
        ),
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
    }


# ============================================================
# DATAFRAME TIME MAPPING
# ============================================================

def add_times(
    trades,
    df_by_symbol,
):
    out = []

    for t in trades:

        x = dict(t)

        df = df_by_symbol[
            t["symbol"]
        ]

        for key, idx_key in [
            (
                "entry_time",
                "entry_index",
            ),
            (
                "exit_time",
                "exit_index",
            ),
            (
                "choch_time",
                "choch_index",
            ),
            (
                "sweep_time",
                "sweep_index",
            ),
        ]:

            idx = x.get(idx_key)

            if idx is None:
                x[key] = None
            else:
                x[key] = df.iloc[
                    idx
                ]["open_time"]

        out.append(x)

    return out


# ============================================================
# PARALLEL SYMBOL WORKER
# ============================================================

def process_symbol(symbol):
    started = time.time()

    print(
        f"\n[{symbol}] downloading...",
        flush=True,
    )

    df = load_symbol(symbol)

    # Build 4H directly from 15m data.
    # This avoids a second network download.
    #
    # Resampling is causal because each 4H candle consists
    # only of already available 15m candles.
    htf = (
        df.set_index("open_time")
        .resample("4h", label="left",
