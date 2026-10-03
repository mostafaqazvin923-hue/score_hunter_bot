import io
import math
import time
import zipfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


# ============================================================
# SETUP 2 V2
# Lowest Valley -> LH Break -> HH -> Higher Low -> HH Break
# -> Return to Higher-Low Order Block
#
# SOURCE:
# "10 ستاپ برتر.pdf" — pages 12-16
#
# IMPORTANT:
# This is a mechanical research translation of the PDF.
# The PDF describes the structure but does not numerically define:
#   - pivot confirmation distance
#   - exact OB boundaries
#   - exact meaning of "near" for the second SL
#
# Those definitions are frozen here BEFORE the new test.
#
# CORE PDF STRUCTURE:
# LONG:
#   1) Downtrend: LH + LL
#   2) Lowest valley forms
#   3) Price breaks the last LH
#   4) Price forms HH
#   5) Price falls but does NOT reach previous valley
#   6) This creates first Higher Low after CHOCH
#   7) Price rises and breaks the prior peak
#   8) Price forms another HH
#   9) Price returns to the HL Order Block
#  10) Buy
#
# SHORT = exact mirror image.
#
# BACKTEST INTEGRITY:
# - No lookahead
# - Confirmed pivots only
# - Entry only after structural information exists
# - Real Binance USD-M futures OHLCV
# - No synthetic candles
# - No timeout
# - No same-candle re-entry
# - One open trade per symbol
# - Different symbols may overlap
# - Fixed RR = 1:2
# - $1,000 starting equity
# - $100 margin
# - 50x leverage
# - $5,000 notional
# - Fees + slippage
# - Same-candle SL/TP = LOSS
# - Unresolved final trades excluded
#
# RESEARCH PROTOCOL:
# Discovery  = 50%
# Development = 25%
# Validation  = 25%
#
# Any future parameter selection must use Discovery/Development.
# The final Validation segment must remain untouched.
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

INTERVAL = "1h"

TEST_DAYS = 365
WARMUP_DAYS = 90

INITIAL_CAPITAL = 1000.0

MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE

RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003

# ------------------------------------------------------------
# STRUCTURE DEFINITIONS
# ------------------------------------------------------------

# Confirmed pivot:
# P bars on left + P bars on right.
#
# A pivot at candle i is only usable after i+P candles exist.
PIVOT = 2

# Minimum structural displacement.
# These are deliberately modest and fixed.
MIN_SWING_PCT = 0.0010

# The initial CHOCH break should have meaningful displacement.
MIN_CHOCH_BREAK_PCT = 0.0010

# Maximum bars allowed between structural legs.
MAX_CHOCH_TO_HH_BARS = 24
MAX_HH_TO_HL_BARS = 24
MAX_HL_TO_SECOND_HH_BARS = 30

# After the second HH, the setup has a limited structural lifetime.
# This is NOT a trade timeout.
# It only prevents a centuries-old structural pattern from remaining valid.
MAX_RETURN_BARS = 48

# OB definition:
# The complete range of the HL pivot candle.
#
# Long:
#   OB_LOW  = HL candle low
#   OB_HIGH = HL candle high
#
# Short:
#   OB_LOW  = LH candle low
#   OB_HIGH = LH candle high
#
# Entry uses the proximal edge.
#
# Long entry = OB_HIGH
# Short entry = OB_LOW

# Protective buffer behind OB.
SL_BUFFER_PCT = 0.0005

# Maximum accepted initial risk.
MIN_RISK_PCT = 0.0005
MAX_RISK_PCT = 0.08

# If the second structural valley/peak is extremely close to entry,
# the PDF allows using the CHOCH valley/peak as an alternative SL.
# We define "near" mechanically.
SECOND_SL_NEAR_RATIO = 0.50

# Data source.
DATA_BASE = "https://data.binance.vision/data/futures/um"

CACHE_DIR = Path("data_cache_setup2_v2")
OUTPUT_DIR = Path("reports/setup2_v2")

TRADE_LEDGER = Path("setup2_v2_trade_ledger.csv")


# ============================================================
# HTTP
# ============================================================

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 Setup2-V2-Research"
    }
)


# ============================================================
# TIME
# ============================================================

def utc_now():
    return pd.Timestamp.now(tz="UTC")


def month_starts(start_ts, end_ts):
    start = pd.Timestamp(start_ts).tz_convert("UTC").normalize().replace(day=1)
    end = pd.Timestamp(end_ts).tz_convert("UTC").normalize().replace(day=1)

    out = []
    cur = start

    while cur <= end:
        out.append(cur)
        cur = cur + pd.offsets.MonthBegin(1)

    return out


# ============================================================
# BINANCE ARCHIVE
# ============================================================

def download_month(symbol, month, interval):
    """
    Download one Binance futures monthly archive.

    No synthetic fallback.
    Failure is explicit.
    """

    ym = month.strftime("%Y-%m")

    url = (
        f"{DATA_BASE}/monthly/klines/"
        f"{symbol}/{interval}/"
        f"{symbol}-{interval}-{ym}.zip"
    )

    for attempt in range(4):
        try:
            r = SESSION.get(url, timeout=60)

            if r.status_code == 404:
                return None

            r.raise_for_status()

            if not r.content:
                raise RuntimeError("Empty archive")

            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                names = z.namelist()

                csv_names = [
                    n for n in names
                    if n.lower().endswith(".csv")
                ]

                if not csv_names:
                    raise RuntimeError("Archive contains no CSV")

                with z.open(csv_names[0]) as f:
                    return pd.read_csv(f, header=None)

        except Exception as exc:
            if attempt == 3:
                raise RuntimeError(
                    f"{symbol} {ym}: monthly archive failed: {exc}"
                )

            time.sleep(1.0 + attempt)

    return None


def normalize_klines(raw):
    """
    Binance archive kline normalization.

    Standard futures kline fields:
      0 open time
      1 open
      2 high
      3 low
      4 close
      5 volume
    """

    if raw is None or raw.empty:
        return pd.DataFrame()

    if raw.shape[1] < 6:
        raise RuntimeError("Kline CSV has fewer than 6 columns")

    x = raw.iloc[:, :6].copy()

    x.columns = [
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    x["ts"] = pd.to_numeric(x["ts"], errors="coerce")
    x["open"] = pd.to_numeric(x["open"], errors="coerce")
    x["high"] = pd.to_numeric(x["high"], errors="coerce")
    x["low"] = pd.to_numeric(x["low"], errors="coerce")
    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    x["volume"] = pd.to_numeric(x["volume"], errors="coerce")

    x = x.dropna().copy()

    x["ts"] = x["ts"].astype(np.int64)

    x["Date"] = pd.to_datetime(
        x["ts"],
        unit="ms",
        utc=True,
    )

    x = (
        x[
            [
                "Date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]
        .drop_duplicates("Date")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return x


def validate_ohlcv(df, symbol):
    if df.empty:
        raise RuntimeError(f"{symbol}: empty OHLCV")

    if df["Date"].duplicated().any():
        raise RuntimeError(f"{symbol}: duplicate timestamps")

    if not df["Date"].is_monotonic_increasing:
        raise RuntimeError(f"{symbol}: timestamps not increasing")

    if not (
        (df["high"] >= df["open"]).all()
        and
        (df["high"] >= df["close"]).all()
        and
        (df["low"] <= df["open"]).all()
        and
        (df["low"] <= df["close"]).all()
    ):
        raise RuntimeError(f"{symbol}: invalid OHLC relationship")

    expected = pd.Timedelta(hours=1)

    gaps = df["Date"].diff().dropna()

    bad = gaps[gaps > expected]

    if not bad.empty:
        first_bad = bad.index[0]

        raise RuntimeError(
            f"{symbol}: data gap detected at "
            f"{df.loc[first_bad, 'Date']} "
            f"gap={bad.iloc[0]}"
        )


def fetch_symbol(symbol, start_ts, end_ts):
    cache_path = (
        CACHE_DIR
        / f"{symbol}_{INTERVAL}_{start_ts.strftime('%Y%m%d')}_"
          f"{end_ts.strftime('%Y%m%d')}.csv"
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        df = pd.read_csv(cache_path)
        df["Date"] = pd.to_datetime(
            df["Date"],
            utc=True,
        )

        df = df[
            (df["Date"] >= start_ts)
            &
            (df["Date"] <= end_ts)
        ].copy()

        validate_ohlcv(df, symbol)

        return df.reset_index(drop=True)

    frames = []

    for month in month_starts(start_ts, end_ts):

        raw = download_month(
            symbol,
            month,
            INTERVAL,
        )

        if raw is None:
            continue

        x = normalize_klines(raw)

        if not x.empty:
            frames.append(x)

    if not frames:
        raise RuntimeError(
            f"{symbol}: no Binance archive data"
        )

    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("Date")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Remove incomplete latest candle.
    now = utc_now()
    current_bucket = now.floor("1h")

    df = df[
        df["Date"] < current_bucket
    ].copy()

    df = df[
        (df["Date"] >= start_ts)
        &
        (df["Date"] <= end_ts)
    ].copy()

    df = df.reset_index(drop=True)

    validate_ohlcv(df, symbol)

    df.to_csv(
        cache_path,
        index=False,
    )

    return df


# ============================================================
# INDICATORS
# ============================================================

def add_atr(df, period=14):
    x = df.copy()

    prev_close = x["close"].shift(1)

    tr = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["ATR"] = tr.rolling(
        period,
        min_periods=period,
    ).mean()

    return x


# ============================================================
# CONFIRMED PIVOTS
# ============================================================

def build_confirmed_pivots(df):
    """
    Returns confirmed pivot flags.

    IMPORTANT:
    Pivot at i becomes known only at i + PIVOT.

    We therefore NEVER use a pivot before its confirmation candle.
    """

    x = df.copy().reset_index(drop=True)

    x["pivot_high"] = False
    x["pivot_low"] = False

    n = len(x)

    for i in range(PIVOT, n - PIVOT):

        h = float(x.at[i, "high"])
        l = float(x.at[i, "low"])

        left_highs = x.loc[
            i - PIVOT:i - 1,
            "high",
        ]

        right_highs = x.loc[
            i + 1:i + PIVOT,
            "high",
        ]

        left_lows = x.loc[
            i - PIVOT:i - 1,
            "low",
        ]

        right_lows = x.loc[
            i + 1:i + PIVOT,
            "low",
        ]

        if (
            h > left_highs.max()
            and
            h >= right_highs.max()
        ):
            x.at[i, "pivot_high"] = True

        if (
            l < left_lows.min()
            and
            l <= right_lows.min()
        ):
            x.at[i, "pivot_low"] = True

    return x


# ============================================================
# STRUCTURAL HELPERS
# ============================================================

def pct_change(a, b):
    if b == 0:
        return 0.0

    return abs(a - b) / abs(b)


def valid_downtrend(df, h1, h2, l1, l2):
    """
    Confirmed structural downtrend:
      lower high
      lower low
    """

    if h2 <= h1 or l2 <= l1:
        return False

    return (
        df.at[h2, "high"]
        <
        df.at[h1, "high"]
        *
        (1.0 - MIN_SWING_PCT)
        and
        df.at[l2, "low"]
        <
        df.at[l1, "low"]
        *
        (1.0 - MIN_SWING_PCT)
    )


def valid_uptrend(df, h1, h2, l1, l2):
    return (
        df.at[h2, "high"]
        >
        df.at[h1, "high"]
        *
        (1.0 + MIN_SWING_PCT)
        and
        df.at[l2, "low"]
        >
        df.at[l1, "low"]
        *
        (1.0 + MIN_SWING_PCT)
    )


def get_pivots_before(x, i):
    highs = [
        j for j in range(i)
        if bool(x.at[j, "pivot_high"])
    ]

    lows = [
        j for j in range(i)
        if bool(x.at[j, "pivot_low"])
    ]

    return highs, lows


# ============================================================
# SETUP 2 LONG
# ============================================================

def build_long_candidates(df, symbol):
    """
    Long structure:

    Downtrend
        |
        v
    lowest valley
        |
        v
    break last LH
        |
        v
       HH
        |
        v
       HL  <-- OB
        |
        v
    break previous HH
        |
        v
      HH2
        |
        v
    return to HL OB
        |
        v
       BUY
    """

    x = df.reset_index(drop=True)

    candidates = []

    pivot_highs = [
        i for i in range(len(x))
        if bool(x.at[i, "pivot_high"])
    ]

    pivot_lows = [
        i for i in range(len(x))
        if bool(x.at[i, "pivot_low"])
    ]

    for valley_idx in pivot_lows:

        # Need enough structure before the lowest valley.
        previous_highs = [
            h for h in pivot_highs
            if h < valley_idx
        ]

        previous_lows = [
            l for l in pivot_lows
            if l < valley_idx
        ]

        if len(previous_highs) < 2:
            continue

        if len(previous_lows) < 2:
            continue

        # Last two confirmed highs/lows before the valley.
        h1, h2 = previous_highs[-2:]
        l1, l2 = previous_lows[-2:]

        if not valid_downtrend(
            x,
            h1,
            h2,
            l1,
            l2,
        ):
            continue

        # Valley must be lower than the preceding LL.
        if x.at[valley_idx, "low"] >= x.at[l2, "low"]:
            continue

        # Last LH that must be broken.
        choch_level = float(x.at[h2, "high"])

        # Search for first break after valley.
        break_candidates = [
            h for h in pivot_highs
            if h > valley_idx
            and h <= valley_idx + MAX_CHOCH_TO_HH_BARS
        ]

        if not break_candidates:
            continue

        choch_break = None

        for b in range(
            valley_idx + 1,
            min(
                len(x),
                valley_idx + MAX_CHOCH_TO_HH_BARS + 1,
            ),
        ):

            high = float(x.at[b, "high"])
            close = float(x.at[b, "close"])

            # Wick break OR close break is accepted by PDF.
            if high <= choch_level:
                continue

            if pct_change(high, choch_level) < MIN_CHOCH_BREAK_PCT:
                continue

            choch_break = b
            break

        if choch_break is None:
            continue

        # Need a confirmed HH after CHOCH.
        hh_candidates = [
            h for h in pivot_highs
            if h > choch_break
            and h <= choch_break + MAX_CHOCH_TO_HH_BARS
        ]

        if not hh_candidates:
            continue

        hh1 = hh_candidates[0]

        # HH must actually be above broken LH.
        if x.at[hh1, "high"] <= choch_level:
            continue

        # ----------------------------------------------------
        # FIRST HIGHER LOW AFTER CHOCH
        # ----------------------------------------------------

        hl_candidates = [
            l for l in pivot_lows
            if l > hh1
            and l <= hh1 + MAX_HH_TO_HL_BARS
        ]

        if not hl_candidates:
            continue

        hl_idx = None

        for l in hl_candidates:

            # The HL must remain ABOVE the old valley.
            if x.at[l, "low"] <= x.at[valley_idx, "low"]:
                continue

            # It must also remain structurally higher than
            # the preceding valley/LL.
            if x.at[l, "low"] <= x.at[l2, "low"]:
                continue

            hl_idx = l
            break

        if hl_idx is None:
            continue

        # ----------------------------------------------------
        # SECOND HH
        # ----------------------------------------------------

        hh2_candidates = [
            h for h in pivot_highs
            if h > hl_idx
            and h <= hl_idx + MAX_HL_TO_SECOND_HH_BARS
        ]

        if not hh2_candidates:
            continue

        hh2 = None

        for h in hh2_candidates:

            # Must break previous HH.
            if x.at[h, "high"] <= x.at[hh1, "high"]:
                continue

            hh2 = h
            break

        if hh2 is None:
            continue

        # ----------------------------------------------------
        # ORDER BLOCK
        # ----------------------------------------------------
        #
        # Frozen mechanical definition:
        # The OB is the full range of the first HL pivot candle.
        #
        # This directly follows the PDF's instruction to draw
        # the OB at the first higher valley after CHOCH.
        # ----------------------------------------------------

        ob_low = float(x.at[hl_idx, "low"])
        ob_high = float(x.at[hl_idx, "high"])

        if ob_high <= ob_low:
            continue

        # ----------------------------------------------------
        # RETURN TO OB
        # ----------------------------------------------------

        search_end = min(
            len(x) - 1,
            hh2 + MAX_RETURN_BARS,
        )

        entry_idx = None
        entry_price = None

        invalid = False

        for j in range(hh2 + 1, search_end + 1):

            # If price expands upward again before touching OB,
            # the PDF says the setup loses validity.
            if (
                float(x.at[j, "high"])
                >
                float(x.at[hh2, "high"])
            ):
                invalid = True
                break

            candle_low = float(x.at[j, "low"])
            candle_open = float(x.at[j, "open"])

            if candle_low > ob_high:
                continue

            # ------------------------------------------------
            # LIMIT ENTRY
            #
            # Buy limit at proximal OB edge.
            #
            # If market opens below that price, a real resting
            # buy limit would be filled at the better open price.
            # ------------------------------------------------

            if candle_open <= ob_high:
                fill = candle_open
            else:
                fill = ob_high

            entry_idx = j
            entry_price = fill
            break

        if invalid or entry_idx is None:
            continue

        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------

        stop_ob = ob_low * (
            1.0 - SL_BUFFER_PCT
        )

        # Alternative structural stop:
        # below the CHOCH valley.
        stop_valley = float(
            x.at[valley_idx, "low"]
        ) * (1.0 - SL_BUFFER_PCT)

        # PDF says if the CHOCH valley is close to entry,
        # it can be used as stronger SL reference.
        distance_ob = entry_price - stop_ob
        distance_valley = entry_pr
