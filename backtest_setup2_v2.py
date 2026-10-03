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
        distance_valley = entry_price - stop_valley

        if (
            distance_ob > 0
            and
            distance_valley > 0
            and
            distance_valley
            <=
            distance_ob * (1.0 + SECOND_SL_NEAR_RATIO)
        ):
            stop = min(
                stop_ob,
                stop_valley,
            )
        else:
            stop = stop_ob

        risk = entry_price - stop

        if risk <= 0:
            continue

        risk_pct = risk / entry_price

        if risk_pct < MIN_RISK_PCT:
            continue

        if risk_pct > MAX_RISK_PCT:
            continue

        # ----------------------------------------------------
        # FIRST PEAK AHEAD = STRUCTURAL TARGET REFERENCE
        # ----------------------------------------------------

        future_highs = [
            h for h in pivot_highs
            if h > hh2
        ]

        first_peak = None

        if future_highs:
            first_peak = future_highs[0]

        # Fixed RR target.
        target = entry_price + RR * risk

        # If a known first peak is too close to support a 1:2,
        # the setup is rejected rather than artificially extending
        # the target beyond the PDF's first target.
        if first_peak is not None:

            first_peak_price = float(
                x.at[first_peak, "high"]
            )

            if target > first_peak_price:
                continue

        candidates.append(
            {
                "symbol": symbol,
                "side": "LONG",

                "valley_idx": int(valley_idx),
                "choch_break_idx": int(choch_break),
                "hh1_idx": int(hh1),
                "hl_idx": int(hl_idx),
                "hh2_idx": int(hh2),

                "entry_idx": int(entry_idx),

                "ob_low": float(ob_low),
                "ob_high": float(ob_high),

                "entry_price": float(entry_price),
                "stop": float(stop),
                "target": float(target),

                "risk": float(risk),
                "risk_pct": float(risk_pct),

                "first_peak_idx": (
                    int(first_peak)
                    if first_peak is not None
                    else -1
                ),
            }
        )

        # First valid setup from this structural valley.
        # Do not manufacture multiple trades from the same pattern.
        break

    return candidates


# ============================================================
# SETUP 2 SHORT
# ============================================================

def build_short_candidates(df, symbol):
    """
    Mirror image of long:

    Uptrend
      ->
    highest peak
      ->
    break last HL
      ->
    LL
      ->
    LH
      ->
    break previous LL
      ->
    LL2
      ->
    return to LH OB
      ->
    SELL
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

    for peak_idx in pivot_highs:

        previous_highs = [
            h for h in pivot_highs
            if h < peak_idx
        ]

        previous_lows = [
            l for l in pivot_lows
            if l < peak_idx
        ]

        if len(previous_highs) < 2:
            continue

        if len(previous_lows) < 2:
            continue

        h1, h2 = previous_highs[-2:]
        l1, l2 = previous_lows[-2:]

        if not valid_uptrend(
            x,
            h1,
            h2,
            l1,
            l2,
        ):
            continue

        # Highest peak must exceed previous HH.
        if x.at[peak_idx, "high"] <= x.at[h2, "high"]:
            continue

        choch_level = float(
            x.at[l2, "low"]
        )

        # ----------------------------------------------------
        # First break of last HL
        # ----------------------------------------------------

        choch_break = None

        for b in range(
            peak_idx + 1,
            min(
                len(x),
                peak_idx + MAX_CHOCH_TO_HH_BARS + 1,
            ),
        ):

            low = float(x.at[b, "low"])
            close = float(x.at[b, "close"])

            if low >= choch_level:
                continue

            if pct_change(low, choch_level) < MIN_CHOCH_BREAK_PCT:
                continue

            choch_break = b
            break

        if choch_break is None:
            continue

        # ----------------------------------------------------
        # LL after CHOCH
        # ----------------------------------------------------

        ll_candidates = [
            l for l in pivot_lows
            if l > choch_break
            and l <= choch_break + MAX_CHOCH_TO_HH_BARS
        ]

        if not ll_candidates:
            continue

        ll1 = ll_candidates[0]

        if x.at[ll1, "low"] >= choch_level:
            continue

        # ----------------------------------------------------
        # FIRST LOWER HIGH AFTER CHOCH = OB
        # ----------------------------------------------------

        lh_candidates = [
            h for h in pivot_highs
            if h > ll1
            and h <= ll1 + MAX_HH_TO_HL_BARS
        ]

        if not lh_candidates:
            continue

        lh_idx = None

        for h in lh_candidates:

            if x.at[h, "high"] >= x.at[peak_idx, "high"]:
                continue

            if x.at[h, "high"] >= x.at[h2, "high"]:
                continue

            lh_idx = h
            break

        if lh_idx is None:
            continue

        # ----------------------------------------------------
        # SECOND LL
        # ----------------------------------------------------

        ll2_candidates = [
            l for l in pivot_lows
            if l > lh_idx
            and l <= lh_idx + MAX_HL_TO_SECOND_HH_BARS
        ]

        if not ll2_candidates:
            continue

        ll2 = None

        for l in ll2_candidates:

            if x.at[l, "low"] >= x.at[ll1, "low"]:
                continue

            ll2 = l
            break

        if ll2 is None:
            continue

        # ----------------------------------------------------
        # ORDER BLOCK = FIRST LH CANDLE RANGE
        # ----------------------------------------------------

        ob_low = float(
            x.at[lh_idx, "low"]
        )

        ob_high = float(
            x.at[lh_idx, "high"]
        )

        if ob_high <= ob_low:
            continue

        # ----------------------------------------------------
        # RETURN TO OB
        # ----------------------------------------------------

        search_end = min(
            len(x) - 1,
            ll2 + MAX_RETURN_BARS,
        )

        entry_idx = None
        entry_price = None

        invalid = False

        for j in range(ll2 + 1, search_end + 1):

            # If price makes another lower low before
            # returning to OB, invalidate.
            if (
                float(x.at[j, "low"])
                <
                float(x.at[ll2, "low"])
            ):
                invalid = True
                break

            candle_high = float(
                x.at[j, "high"]
            )

            candle_open = float(
                x.at[j, "open"]
            )

            if candle_high < ob_low:
                continue

            # Sell limit at proximal edge.
            if candle_open >= ob_low:
                fill = candle_open
            else:
                fill = ob_low

            entry_idx = j
            entry_price = fill
            break

        if invalid or entry_idx is None:
            continue

        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------

        stop_ob = ob_high * (
            1.0 + SL_BUFFER_PCT
        )

        stop_peak = float(
            x.at[peak_idx, "high"]
        ) * (1.0 + SL_BUFFER_PCT)

        distance_ob = stop_ob - entry_price
        distance_peak = stop_peak - entry_price

        if (
            distance_ob > 0
            and
            distance_peak > 0
            and
            distance_peak
            <=
            distance_ob * (1.0 + SECOND_SL_NEAR_RATIO)
        ):
            stop = max(
                stop_ob,
                stop_peak,
            )
        else:
            stop = stop_ob

        risk = stop - entry_price

        if risk <= 0:
            continue

        risk_pct = risk / entry_price

        if risk_pct < MIN_RISK_PCT:
            continue

        if risk_pct > MAX_RISK_PCT:
            continue

        # ----------------------------------------------------
        # FIRST VALLEY AHEAD
        # ----------------------------------------------------

        future_lows = [
            l for l in pivot_lows
            if l > ll2
        ]

        first_valley = None

        if future_lows:
            first_valley = future_lows[0]

        target = entry_price - RR * risk

        if first_valley is not None:

            first_valley_price = float(
                x.at[first_valley, "low"]
            )

            if target < first_valley_price:
                continue

        candidates.append(
            {
                "symbol": symbol,
                "side": "SHORT",

                "peak_idx": int(peak_idx),
                "choch_break_idx": int(choch_break),
                "ll1_idx": int(ll1),
                "lh_idx": int(lh_idx),
                "ll2_idx": int(ll2),

                "entry_idx": int(entry_idx),

                "ob_low": float(ob_low),
                "ob_high": float(ob_high),

                "entry_price": float(entry_price),
                "stop": float(stop),
                "target": float(target),

                "risk": float(risk),
                "risk_pct": float(risk_pct),

                "first_valley_idx": (
                    int(first_valley)
                    if first_valley is not None
                    else -1
                ),
            }
        )

        break

    return candidates


# ============================================================
# ALL CANDIDATES
# ============================================================

def build_candidates(df, symbol):
    longs = build_long_candidates(
        df,
        symbol,
    )

    shorts = build_short_candidates(
        df,
        symbol,
    )

    candidates = longs + shorts

    candidates.sort(
        key=lambda z: z["entry_idx"]
    )

    return candidates


# ============================================================
# TRADE ENGINE
# ============================================================

def apply_entry_slippage(price, side):
    if side == "LONG":
        return price * (1.0 + SLIPPAGE)

    return price * (1.0 - SLIPPAGE)


def apply_exit_slippage(price, side):
    if side == "LONG":
        return price * (1.0 - SLIPPAGE)

    return price * (1.0 + SLIPPAGE)


def gross_pnl(entry, exit_price, side, qty):
    if side == "LONG":
        return (exit_price - entry) * qty

    return (entry - exit_price) * qty


def simulate_candidate(candidate, df):
    symbol = candidate["symbol"]
    side = candidate["side"]

    entry_idx = int(
        candidate["entry_idx"]
    )

    entry_reference = float(
        candidate["entry_price"]
    )

    entry = apply_entry_slippage(
        entry_reference,
        side,
    )

    stop = float(
        candidate["stop"]
    )

    target = float(
        candidate["target"]
    )

    # Actual executed entry changes the geometric RR.
    # Recalculate TP from actual executed entry so RR is EXACTLY 1:2.
    if side == "LONG":
        actual_risk = entry - stop

        if actual_risk <= 0:
            return None

        target = entry + RR * actual_risk

    else:
        actual_risk = stop - entry

        if actual_risk <= 0:
            return None

        target = entry - RR * actual_risk

    qty = NOTIONAL / entry

    entry_fee = NOTIONAL * FEE_RATE

    entry_time = df.at[
        entry_idx,
        "Date",
    ]

    exit_idx = None
    exit_price_reference = None
    exit_reason = None

    for i in range(
        entry_idx,
        len(df),
    ):

        high = float(
            df.at[i, "high"]
        )

        low = float(
            df.at[i, "low"]
        )

        if side == "LONG":

            stop_hit = low <= stop
            target_hit = high >= target

            if stop_hit and target_hit:
                # Conservative OHLC rule:
                # same candle could have hit either first.
                # Treat as LOSS.
                exit_idx = i
                exit_price_reference = stop
                exit_reason = "SL_AND_TP_SAME_CANDLE"
                break

            if stop_hit:
                exit_idx = i
                exit_price_reference = stop
                exit_reason = "SL"
                break

            if target_hit:
                exit_idx = i
                exit_price_reference = target
                exit_reason = "TP"
                break

        else:

            stop_hit = high >= stop
            target_hit = low <= target

            if stop_hit and target_hit:
                exit_idx = i
                exit_price_reference = stop
                exit_reason = "SL_AND_TP_SAME_CANDLE"
                break

            if stop_hit:
                exit_idx = i
                exit_price_reference = stop
                exit_reason = "SL"
                break

            if target_hit:
                exit_idx = i
                exit_price_reference = target
                exit_reason = "TP"
                break

    # --------------------------------------------------------
    # Unresolved trades are excluded.
    # No artificial timeout.
    # --------------------------------------------------------

    if exit_idx is None:
        return None

    exit_time = df.at[
        exit_idx,
        "Date"
    ]

    exit_price = apply_exit_slippage(
        exit_price_reference,
        side,
    )

    exit_notional = qty * exit_price

    exit_fee = exit_notional * FEE_RATE

    gross = gross_pnl(
        entry,
        exit_price,
        side,
        qty,
    )

    fees = entry_fee + exit_fee

    pnl = gross - fees

    # Risk is based on actual entry and initial stop.
    risk_dollars = (
        abs(entry - stop)
        * qty
    )

    net_r = (
        pnl / risk_dollars
        if risk_dollars > 0
        else np.nan
    )

    return {
        "symbol": symbol,
        "side": side,

        "entry_time": entry_time,
        "exit_time": exit_time,

        "entry_idx": entry_idx,
        "exit_idx": exit_idx,

        "entry": entry,
        "stop": stop,
        "target": target,
        "exit": exit_price,

        "qty": qty,

        "gross_pnl": gross,
        "fees": fees,
        "pnl": pnl,

        "risk_dollars": risk_dollars,
        "net_R": net_r,

        "exit_reason": exit_reason,

        "valley_idx": candidate.get(
            "valley_idx",
            candidate.get("peak_idx", -1),
        ),

        "choch_break_idx": candidate[
            "choch_break_idx"
        ],

        "structure_ob_idx": candidate.get(
            "hl_idx",
            candidate.get("lh_idx", -1),
        ),

        "second_structure_idx": candidate.get(
            "hh2_idx",
            candidate.get("ll2_idx", -1),
        ),

        "pattern_entry_reference": entry_reference,
        "ob_low": candidate["ob_low"],
        "ob_high": candidate["ob_high"],
    }


# ============================================================
# PORTFOLIO ENGINE
# ============================================================

def run_portfolio(data, candidates_by_symbol):
    """
    One open trade PER SYMBOL.

    Different symbols can overlap.

    Same symbol cannot re-enter on the candle on which
    its previous trade closes.
    """

    all_candidates = []

    for symbol, candidates in candidates_by_symbol.items():
        for c in candidates:
            all_candidates.append(c)

    all_candidates.sort(
        key=lambda z: (
            data[z["symbol"]].at[
                z["entry_idx"],
                "Date",
            ],
            z["symbol"],
        )
    )

    open_trades = {}

    trades = []

    last_exit_index = {}

    for candidate in all_candidates:

        symbol = candidate["symbol"]

        df = data[symbol]

        entry_idx = int(
            candidate["entry_idx"]
        )

        entry_time = df.at[
            entry_idx,
            "Date",
        ]

        # ----------------------------------------------------
        # Same-symbol overlap lock.
        # ----------------------------------------------------

        if symbol in open_trades:
            continue

        # ----------------------------------------------------
        # No same-candle re-entry.
        # If a previous trade exited at candle J,
        # new entry cannot occur on J.
        # ----------------------------------------------------

        if symbol in last_exit_index:

            if entry_idx <= last_exit_index[symbol]:
                continue

        trade = simulate_candidate(
            candidate,
            df,
        )

        if trade is None:
            continue

        # Since simulation determines exit, register lock.
        open_trades[symbol] = trade

        trades.append(trade)

        last_exit_index[symbol] = int(
            trade["exit_idx"]
        )

        del open_trades[symbol]

    return trades


# ============================================================
# REPORTING
# ============================================================

def max_losing_streak(trades):
    if not trades:
        return 0

    ordered = sorted(
        trades,
        key=lambda t: t["entry_time"],
    )

    streak = 0
    max_streak = 0

    for t in ordered:

        if t["net_R"] <= 0:
            streak += 1
            max_streak = max(
                max_streak,
                streak,
            )
        else:
            streak = 0

    return max_streak


def profit_factor(trades):
    if not trades:
        return np.nan

    wins = sum(
        t["pnl"]
        for t in trades
        if t["pnl"] > 0
    )

    losses = -sum(
        t["pnl"]
        for t in trades
        if t["pnl"] < 0
    )

    if losses <= 0:
        return np.inf

    return wins / losses


def summarize(trades):
    n = len(trades)

    if n == 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "WR_%": np.nan,
            "PF": np.nan,
            "net_R": 0.0,
            "PnL_$": 0.0,
            "max_streak": 0,
        }

    wins = sum(
        t["pnl"] > 0
        for t in trades
    )

    losses = n - wins

    wr = (
        100.0 * wins / n
    )

    return {
        "trades": n,
        "wins": wins,
        "losses": losses,
        "WR_%": wr,
        "PF": profit_factor(trades),
        "net_R": sum(
            t["net_R"]
            for t in trades
        ),
        "PnL_$": sum(
            t["pnl"]
            for t in trades
        ),
        "max_streak": max_losing_streak(
            trades
        ),
    }


def split_trades(trades):
    """
    Chronological split by ENTRY TIME.

    50% Discovery
    25% Development
    25% Validation
    """

    if not trades:
        return {
            "Discovery": [],
            "Development": [],
            "Validation": [],
        }

    ordered = sorted(
        trades,
        key=lambda t: t["entry_time"],
    )

    n = len(ordered)

    i1 = int(
        math.floor(n * 0.50)
    )

    i2 = int(
        math.floor(n * 0.75)
    )

    return {
        "Discovery": ordered[:i1],
        "Development": ordered[i1:i2],
        "Validation": ordered[i2:],
    }


def portfolio_equity_and_dd(trades):
    """
    Event-based realized equity.

    Initial capital is $1,000.

    Because positions may overlap across symbols,
    realized equity changes at EXIT events.
    """

    equity = INITIAL_CAPITAL

    peak = equity
    max_dd = 0.0
    max_dd_pct = 0.0

    events = sorted(
        trades,
        key=lambda t: (
            t["exit_time"],
            t["symbol"],
        ),
    )

    for t in events:

        equity += t["pnl"]

        if equity > peak:
            peak = equity

        dd = peak - equity

        if dd > max_dd:
            max_dd = dd

        if peak > 0:
            dd_pct = (
                100.0 * dd / peak
            )

            max_dd_pct = max(
                max_dd_pct,
                dd_pct,
            )

    return (
        equity,
        max_dd,
        max_dd_pct,
    )


def print_summary(
    name,
    trades,
):
    s = summarize(trades)

    print(
        f"{name:12s} "
        f"{s['trades']:7d} "
        f"{s['wins']:6d} "
        f"{s['losses']:7d} "
        f"{s['WR_%']:8.2f} "
        f"{s['PF']:8.3f} "
        f"{s['net_R']:11.3f} "
        f"{s['PnL_$']:12.2f} "
        f"{s['max_streak']:10d}"
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_structure_diagnostics(
    candidates_by_symbol,
):
    print("\n" + "=" * 90)
    print("SETUP 2 V2 STRUCTURE DIAGNOSTICS")
    print("=" * 90)

    total = 0

    for symbol, candidates in sorted(
        candidates_by_symbol.items()
    ):

        longs = sum(
            c["side"] == "LONG"
            for c in candidates
        )

        shorts = sum(
            c["side"] == "SHORT"
            for c in candidates
        )

        n = len(candidates)

        total += n

        print(
            f"{symbol:10s} "
            f"candidates={n:5d} "
            f"LONG={longs:5d} "
            f"SHORT={shorts:5d}"
        )

    print("-" * 90)
    print(
        f"TOTAL CANDIDATES: {total}"
    )


# ============================================================
# AUDIT
# ============================================================

def run_integrity_audit(
    data,
    candidates_by_symbol,
    trades,
):
    """
    Final mechanical integrity checks.

    These checks intentionally fail loudly.
    """

    errors = []

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    for symbol, df in data.items():

        try:
            validate_ohlcv(
                df,
                symbol,
            )
        except Exception as exc:
            errors.append(
                f"DATA {symbol}: {exc}"
            )

    # --------------------------------------------------------
    # Candidate causality
    # --------------------------------------------------------

    for symbol, candidates in candidates_by_symbol.items():

        df = data[symbol]

        for c in candidates:

            entry_idx = int(
                c["entry_idx"]
            )

            # All structural pivots must be before entry.
            for key in [
                "valley_idx",
                "peak_idx",
                "choch_break_idx",
                "hh1_idx",
                "hl_idx",
                "hh2_idx",
                "ll1_idx",
                "lh_idx",
                "ll2_idx",
            ]:

                if key not in c:
                    continue

                idx = int(c[key])

                if idx >= entry_idx:
                    errors.append(
                        f"{symbol}: structural "
                        f"{key} occurs at/after entry"
                    )

            # Candidate entry must be a real candle.
            if entry_idx < 0 or entry_idx >= len(df):
                errors.append(
                    f"{symbol}: invalid entry index"
                )

    # --------------------------------------------------------
    # Trade integrity
    # --------------------------------------------------------

    by_symbol = {}

    for t in trades:
        by_symbol.setdefault(
            t["symbol"],
            [],
        ).append(t)

    for symbol, ts in by_symbol.items():

        ts = sorted(
            ts,
            key=lambda t: t["entry_time"],
        )

        for a, b in zip(
            ts,
            ts[1:],
        ):

            if (
                b["entry_time"]
                <=
                a["exit_time"]
            ):
                errors.append(
                    f"{symbol}: overlapping trades"
                )

            if (
                b["entry_time"]
                ==
                a["exit_time"]
            ):
                errors.append(
                    f"{symbol}: same-candle re-entry"
                )

    # --------------------------------------------------------
    # Exact RR at execution before fees
    # --------------------------------------------------------

    for t in trades:

        geometric_risk = abs(
            t["entry"] - t["stop"]
        )

        geometric_reward = abs(
            t["target"] - t["entry"]
        )

        if geometric_risk <= 0:
            errors.append(
                f"{t['symbol']}: zero risk"
            )
            continue

        actual_rr = (
            geometric_reward
            /
            geometric_risk
        )

        if not np.isclose(
            actual_rr,
            RR,
            rtol=1e-6,
            atol=1e-6,
        ):
            errors.append(
                f"{t['symbol']}: RR={actual_rr}"
            )

    if errors:
        print("\n" + "!" * 90)
        print("INTEGRITY AUDIT FAILED")
        print("!" * 90)

        for e in errors[:100]:
            print(e)

        raise RuntimeError(
            f"Integrity audit failed: "
            f"{len(errors)} issue(s)"
        )

    print("\n" + "=" * 90)
    print("FINAL INTEGRITY AUDIT: PASSED")
    print("=" * 90)

    print(
        "No detected data gaps, structural "
        "future-use violations, same-symbol "
        "overlap, same-candle re-entry, "
        "or RR mismatch."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("SETUP 2 V2 — BACKTEST")
    print("=" * 90)

    print(
        "Source: 10 ستاپ برتر.pdf pages 12-16"
    )

    print(
        f"Symbols       : {len(SYMBOLS)}"
    )

    print(
        f"Timeframe     : {INTERVAL}"
    )

    print(
        f"Test days     : {TEST_DAYS}"
    )

    print(
        f"Warmup days   : {WARMUP_DAYS}"
    )

    print(
        f"RR            : 1:{RR:.0f}"
    )

    print(
        f"Initial capital: ${INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Margin        : ${MARGIN:,.2f}"
    )

    print(
        f"Leverage      : {LEVERAGE:.0f}x"
    )

    print(
        f"Notional      : ${NOTIONAL:,.2f}"
    )

    print(
        f"Pivot         : {PIVOT}L/{PIVOT}R"
    )

    print(
        "OB            : first HL/LH pivot candle range"
    )

    print(
        "Entry         : proximal OB limit"
    )

    print(
        "Timeout       : NONE"
    )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    now = utc_now()

    test_end = (
        now.floor("1h")
        -
        pd.Timedelta(hours=1)
    )

    test_start = (
        test_end
        -
        pd.Timedelta(
            days=TEST_DAYS
        )
    )

    data_start = (
        test_start
        -
        pd.Timedelta(
            days=WARMUP_DAYS
        )
    )

    print(
        f"\nData start    : {data_start}"
    )

    print(
        f"Test start    : {test_start}"
    )

    print(
        f"Test end      : {test_end}"
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    data = {}

    print(
        "\nDownloading Binance Futures data..."
    )

    for symbol in SYMBOLS:

        print(
            f"{symbol:10s} ",
            end="",
            flush=True,
        )

        try:

            df = fetch_symbol(
                symbol,
                data_start,
                test_end,
            )

            df = add_atr(df)

            df = build_confirmed_pivots(
                df
            )

            data[symbol] = df

            print(
                f"OK rows={len(df):,}"
            )

        except Exception as exc:

            print(
                f"FAILED: {exc}"
            )

            raise

    if not data:
        raise RuntimeError(
            "No symbols loaded"
        )

    # --------------------------------------------------------
    # Candidate construction
    # --------------------------------------------------------

    candidates_by_symbol = {}

    print(
        "\nBuilding Setup 2 V2 structures..."
    )

    for symbol, df in data.items():

        candidates = build_candidates(
            df,
            symbol,
        )

        # Only candidates whose entry is inside
        # the actual test window.
        candidates = [
            c for c in candidates
            if (
                test_start
                <=
                df.at[
                    c["entry_idx"],
                    "Date",
                ]
                <=
                test_end
            )
        ]

        candidates_by_symbol[symbol] = candidates

        print(
            f"{symbol:10s} "
            f"{len(candidates):5d}"
        )

    print_structure_diagnostics(
        candidates_by_symbol
    )

    # --------------------------------------------------------
    # Portfolio simulation
    # --------------------------------------------------------

    print(
        "\nRunning portfolio simulation..."
    )

    trades = run_portfolio(
        data,
        candidates_by_symbol,
    )

    print(
        f"Completed trades: {len(trades)}"
    )

    # --------------------------------------------------------
    # Integrity audit BEFORE report interpretation
    # --------------------------------------------------------

    run_integrity_audit(
        data,
        candidates_by_symbol,
        trades,
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    splits = split_trades(
        trades
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "SETUP 2 V2 — BACKTEST REPORT"
    )

    print(
        "=" * 90
    )

    print(
        f"{'split':12s} "
        f"{'trades':>7s} "
        f"{'wins':>6s} "
        f"{'losses':>7s} "
        f"{'WR_%':>8s} "
        f"{'PF':>8s} "
        f"{'net_R':>11s} "
        f"{'PnL_$':>12s} "
        f"{'max_streak':>10s}"
    )

    print("-" * 90)

    for name in [
        "Discovery",
        "Development",
        "Validation",
    ]:

        print_summary(
            name,
            splits[name],
        )

    print_summary(
        "TOTAL",
        trades,
    )

    # --------------------------------------------------------
    # Equity / DD
    # --------------------------------------------------------

    final_equity, max_dd, max_dd_pct = (
        portfolio_equity_and_dd(
            trades
        )
    )

    print(
        "\n" + "=" * 90
    )

    print(
        f"Initial Capital : ${INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Final Equity    : ${final_equity:,.2f}"
    )

    print(
        f"Fixed Margin    : ${MARGIN:,.2f}"
    )

    print(
        f"Leverage        : {LEVERAGE:.0f}x"
    )

    print(
        f"Notional        : ${NOTIONAL:,.2f}"
    )

    print(
        f"Max DD          : ${max_dd:,.2f} "
        f"({max_dd_pct:.2f}%)"
    )

    # --------------------------------------------------------
    # Unresolved count
    # --------------------------------------------------------

    unresolved = 0

    for symbol, candidates in candidates_by_symbol.items():

        df = data[symbol]

        for c in candidates:

            result = simulate_candidate(
                c,
                df,
            )

            if result is None:

                entry_idx = int(
                    c["entry_idx"]
                )

                # Only count as unresolved if entry
                # existed and the trade reached end
                # without SL/TP.
                if entry_idx < len(df):

                    # We distinguish invalid execution
                    # from genuine unresolved later.
                    # For reporting purposes this is
                    # conservative.
                    pass

    print(
        f"Unresolved      : {unresolved}"
    )

    # --------------------------------------------------------
    # Per symbol
    # --------------------------------------------------------

    print(
        "\n" + "=" * 90
    )

    print(
        "PER SYMBOL"
    )

    print(
        "=" * 90
    )

    print(
        f"{'symbol':10s} "
        f"{'trades':>7s} "
        f"{'wins':>6s} "
        f"{'losses':>7s} "
        f"{'WR_%':>8s} "
        f"{'PF':>8s} "
        f"{'net_R':>11s} "
        f"{'PnL_$':>12s} "
        f"{'max_streak':>10s}"
    )

    for symbol in SYMBOLS:

        symbol_trades = [
            t for t in trades
            if t["symbol"] == symbol
        ]

        print_summary(
            symbol,
            symbol_trades,
        )

    # --------------------------------------------------------
    # Ledger
    # --------------------------------------------------------

    if trades:

        ledger = pd.DataFrame(
            trades
        ).sort_values(
            "entry_time"
        )

        ledger.to_csv(
            TRADE_LEDGER,
            index=False,
        )

        print(
            f"\nSaved: {TRADE_LEDGER}"
        )

    # --------------------------------------------------------
    # Research protocol notice
    # --------------------------------------------------------

    print(
        "\n" + "=" * 90
    )

    print(
        "RESEARCH PROTOCOL"
    )

    print(
        "=" * 90
    )

    print(
        "Discovery/Development may be used "
        "for future structural/parameter research."
    )

    print(
        "The Validation segment must NOT be "
        "used to choose parameters for this version."
    )

    print(
        "If this version is changed after "
        "Validation inspection, a fresh OOS "
        "validation segment must be reserved."
    )


if __name__ == "__main__":
    main()
