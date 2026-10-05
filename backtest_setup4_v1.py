import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# SETUP 4 V1
# HTF ZONE -> LOWER-TF STRUCTURE -> ORDER BLOCK
# -> ALIGNED LIQUIDITY -> LIQUIDITY BREAK -> OB RETEST
#
# Source basis:
#   10 ستاپ برتر.pdf
#
# IMPORTANT:
# The PDF describes the setup structurally/qualitatively.
# Numerical values below are frozen mechanical research
# translations and are NOT claimed to be literal PDF values.
#
# Research protocol:
#   - Binance USD-M Futures historical klines
#   - 15m execution timeframe
#   - 4h higher-timeframe context
#   - 365d research + 90d warmup
#   - untouched Validation OOS:
#       2025-10-04 00:00 UTC
#       through
#       2026-10-03 23:00 UTC
#   - RR = 1:2
#   - Initial capital = $1,000
#   - Margin = $100
#   - Leverage = 50x
#   - Notional = $5,000
#   - Fee = 0.07% per side
#   - Slippage = 0.03% per side
#   - One simultaneous trade per symbol
#   - Different symbols may overlap
#   - No same-candle re-entry
#   - No timeout
#   - No BE
#   - No trailing stop
#   - No synthetic/fabricated OHLCV
# ============================================================


# ============================================================
# UNIVERSE
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


# ============================================================
# TIME / CAPITAL
# ============================================================

INTERVAL = "15m"
HTF_INTERVAL = "4h"

PIVOT = 2

INITIAL_CAPITAL = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE

RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003


# ============================================================
# RESEARCH WINDOWS
# ============================================================

RESEARCH_DAYS = 365
WARMUP_DAYS = 90

OOS_START = pd.Timestamp(
    "2025-10-04 00:00:00",
    tz="UTC",
)

OOS_END = pd.Timestamp(
    "2026-10-03 23:00:00",
    tz="UTC",
)

RESEARCH_START = OOS_START - pd.Timedelta(
    days=RESEARCH_DAYS
)

WARMUP_START = RESEARCH_START - pd.Timedelta(
    days=WARMUP_DAYS
)


# ============================================================
# FROZEN MECHANICAL TRANSLATIONS
# ============================================================

# HTF zone = confirmed 4h swing high/low +/- 0.5 ATR.
HTF_ZONE_ATR = 0.50

# First reaction/touch after HTF zone confirmation.
REACTION_MAX_BARS = 96
# 96 x 15m = 24 hours.

# Aligned liquidity tolerance.
ALIGN_TOL = 0.004
# 0.4%

# Liquidity formation cannot be excessively far from OB.
MAX_LIQUIDITY_DISTANCE = 96
# 96 x 15m = 24 hours.

# SL buffer behind OB.
SL_BUFFER_PCT = 0.0005
# 0.05%

# Minimum / maximum geometric risk.
MIN_RISK_PCT = 0.0005
MAX_RISK_PCT = 0.08


# ============================================================
# DATA
# ============================================================

BASE = (
    "https://data.binance.vision/data/futures/um"
)

CACHE_DIR = Path("data_cache_setup4_v1")
OUT_DIR = Path("setup4_v1_outputs")

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "setup4-v1-backtest/1.0"
    }
)


# ============================================================
# TIME HELPERS
# ============================================================

def utc_now():
    return pd.Timestamp.now(tz="UTC")


def month_range(start, end):
    start = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")

    cur = (
        start
        .tz_localize(None)
        .to_period("M")
        .to_timestamp()
    )

    last = (
        end
        .tz_localize(None)
        .to_period("M")
        .to_timestamp()
    )

    while cur <= last:
        yield cur
        cur += pd.offsets.MonthBegin(1)


def day_range(start, end):
    start = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")

    cur = start.normalize()
    last = end.normalize()

    while cur <= last:
        yield cur
        cur += pd.Timedelta(days=1)


# ============================================================
# HTTP
# ============================================================

def get_bytes(url, retries=4):
    last_error = None

    for attempt in range(retries):
        try:
            response = SESSION.get(
                url,
                timeout=60,
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()

            return response.content

        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(
        f"Download failed after {retries} attempts:\n"
        f"{url}\n"
        f"Last error: {last_error}"
    )


# ============================================================
# ZIP / CSV PARSER
# ============================================================

def parse_zip(blob):
    with zipfile.ZipFile(
        io.BytesIO(blob)
    ) as zf:

        csv_names = [
            name
            for name in zf.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_names:
            raise RuntimeError(
                "Archive contains no CSV file."
            )

        with zf.open(csv_names[0]) as fh:
            raw = pd.read_csv(
                fh,
                header=None,
            )

    if raw.shape[1] < 12:
        raise RuntimeError(
            f"Unexpected kline column count: "
            f"{raw.shape[1]}"
        )

    raw = raw.iloc[:, :12].copy()

    raw.columns = [
        "open_time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
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
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:
        raw[column] = pd.to_numeric(
            raw[column],
            errors="coerce",
        )

    raw = raw.dropna(
        subset=[
            "open_time",
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    raw["Date"] = pd.to_datetime(
        raw["open_time"],
        unit="ms",
        utc=True,
    )

    raw["CloseTime"] = pd.to_datetime(
        raw["close_time"],
        unit="ms",
        utc=True,
    )

    return raw[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "CloseTime",
        ]
    ]


# ============================================================
# SYMBOL DATA FETCH
# ============================================================

def fetch_symbol(symbol, start, end):
    """
    Fetch historical 15m Binance Futures data.

    Important:
    A symbol may not exist during the beginning of the common
    warmup period.

    Therefore:
      - missing monthly archives BEFORE first availability
        are allowed;
      - after the first available archive is found, any later
        missing archive is fatal;
      - no synthetic candles;
      - no forward-fill;
      - no silent gap removal.
    """

    start = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")

    pieces = []

    first_available_month = None

    now = utc_now()

    current_month_start = (
        now
        .tz_convert("UTC")
        .tz_localize(None)
        .to_period("M")
        .to_timestamp()
    )

    for month in month_range(start, end):

        month_utc = month.tz_localize("UTC")

        next_month_utc = (
            month + pd.offsets.MonthBegin(1)
        ).tz_localize("UTC")

        month_end = (
            next_month_utc
            - pd.Timedelta(milliseconds=1)
        )

        # Current month may not have a monthly archive yet.
        incomplete_month = (
            month >= current_month_start
        )

        if incomplete_month:

            day_pieces = []

            day_start = max(
                start,
                month_utc,
            )

            day_end = min(
                end,
                month_end,
            )

            for day in day_range(
                day_start,
                day_end,
            ):

                day_end_ts = (
                    day
                    + pd.Timedelta(days=1)
                    - pd.Timedelta(milliseconds=1)
                )

                url = (
                    f"{BASE}/daily/klines/"
                    f"{symbol}/{INTERVAL}/"
                    f"{symbol}-{INTERVAL}-"
                    f"{day.strftime('%Y-%m-%d')}.zip"
                )

                blob = get_bytes(url)

                if blob is None:

                    # Future / not-yet-published day.
                    if day_end_ts > utc_now():
                        continue

                    raise RuntimeError(
                        "Missing required daily archive:\n"
                        f"{url}"
                    )

                day_pieces.append(
                    parse_zip(blob)
                )

            if day_pieces:

                pieces.extend(day_pieces)

                if first_available_month is None:
                    first_available_month = month

            elif first_available_month is not None:

                raise RuntimeError(
                    f"No daily data after first availability: "
                    f"{symbol} {month:%Y-%m}"
                )

            continue

        # ----------------------------------------------------
        # Completed month
        # ----------------------------------------------------

        url = (
            f"{BASE}/monthly/klines/"
            f"{symbol}/{INTERVAL}/"
            f"{symbol}-{INTERVAL}-"
            f"{month.strftime('%Y-%m')}.zip"
        )

        blob = get_bytes(url)

        if blob is None:

            # Symbol had not launched yet.
            if first_available_month is None:
                continue

            # Once history exists, a missing month is a
            # genuine data-integrity failure.
            raise RuntimeError(
                "Missing monthly archive after "
                "first availability:\n"
                f"{url}"
            )

        pieces.append(
            parse_zip(blob)
        )

        if first_available_month is None:
            first_available_month = month

    if not pieces:
        raise RuntimeError(
            f"No historical data found for {symbol}"
        )

    df = pd.concat(
        pieces,
        ignore_index=True,
    )

    df = (
        df.sort_values("Date")
        .drop_duplicates(
            subset="Date",
            keep="first",
        )
        .reset_index(drop=True)
    )

    # Never use an incomplete candle.
    now = utc_now()

    df = df[
        df["CloseTime"] <= now
    ].copy()

    # Exact requested range.
    df = df[
        (df["Date"] >= start)
        & (df["Date"] <= end)
    ].copy()

    if df.empty:
        raise RuntimeError(
            f"No completed candles remain for {symbol}"
        )

    # --------------------------------------------------------
    # STRICT 15m CONTINUITY
    #
    # We allow the symbol to start later than WARMUP_START.
    # From its first available candle onward, every 15m bar
    # must exist.
    # --------------------------------------------------------

    dates = pd.DatetimeIndex(
        df["Date"]
    )

    diffs = (
        dates.to_series()
        .diff()
        .dropna()
    )

    bad = diffs[
        diffs != pd.Timedelta(minutes=15)
    ]

    if not bad.empty:

        sample = bad.head(10)

        raise RuntimeError(
            f"{symbol}: 15m data gaps detected:\n"
            f"{sample.to_dict()}"
        )

    return df.reset_index(drop=True)


# ============================================================
# ATR
# ============================================================

def add_atr(df, period=14):
    x = df.copy()

    previous_close = x["Close"].shift(1)

    true_range = pd.concat(
        [
            x["High"] - x["Low"],
            (x["High"] - previous_close).abs(),
            (x["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["ATR"] = (
        true_range
        .rolling(
            period,
            min_periods=period,
        )
        .mean()
    )

    return x


# ============================================================
# CONFIRMED PIVOTS
# ============================================================

def add_confirmed_pivots(df):
    """
    A pivot at p becomes known only at p + PIVOT.

    Therefore the pivot flag is stored on the confirmation
    candle, while pivot_*_idx stores the actual pivot candle.
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

    pivot_high_idx = np.full(
        n,
        -1,
        dtype=int,
    )

    pivot_low_idx = np.full(
        n,
        -1,
        dtype=int,
    )

    highs = x["High"].to_numpy()
    lows = x["Low"].to_numpy()

    for pivot_index in range(
        PIVOT,
        n - PIVOT,
    ):

        left_highs = highs[
            pivot_index - PIVOT:
            pivot_index
        ]

        right_highs = highs[
            pivot_index + 1:
            pivot_index + PIVOT + 1
        ]

        left_lows = lows[
            pivot_index - PIVOT:
            pivot_index
        ]

        right_lows = lows[
            pivot_index + 1:
            pivot_index + PIVOT + 1
        ]

        is_high = (
            highs[pivot_index]
            > left_highs.max()
            and
            highs[pivot_index]
            >= right_highs.max()
        )

        is_low = (
            lows[pivot_index]
            < left_lows.min()
            and
            lows[pivot_index]
            <= right_lows.min()
        )

        confirmation_index = (
            pivot_index + PIVOT
        )

        if is_high:

            pivot_high[
                confirmation_index
            ] = True

            pivot_high_idx[
                confirmation_index
            ] = pivot_index

        if is_low:

            pivot_low[
                confirmation_index
            ] = True

            pivot_low_idx[
                confirmation_index
            ] = pivot_index

    x["pivot_high"] = pivot_high
    x["pivot_low"] = pivot_low

    x["pivot_high_idx"] = (
        pivot_high_idx
    )

    x["pivot_low_idx"] = (
        pivot_low_idx
    )

    return x


# ============================================================
# 4H RESAMPLING
# ============================================================

def make_htf(df15):
    x = (
        df15
        .set_index("Date")[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        ]
    )

    htf = (
        x.resample(
            HTF_INTERVAL,
            label="left",
            closed="left",
        )
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )

    htf = add_atr(htf)
    htf = add_confirmed_pivots(htf)

    # 4h candle becomes usable only after its close.
    htf["AvailableAt"] = (
        htf["Date"]
        + pd.Timedelta(hours=4)
    )

    return htf


# ============================================================
# HTF ZONES
# ============================================================

def build_htf_zones(htf):
    zones = []

    for i in range(len(htf)):

        atr = htf["ATR"].iloc[i]

        if not np.isfinite(atr):
            continue

        # ----------------------------------------------------
        # LONG = confirmed HTF swing low / demand area
        # ----------------------------------------------------

        if bool(
            htf["pivot_low"].iloc[i]
        ):

            pivot_index = int(
                htf["pivot_low_idx"].iloc[i]
            )

            price = float(
                htf["Low"].iloc[pivot_index]
            )

            pivot_atr = float(
                htf["ATR"].iloc[pivot_index]
            )

            if not np.isfinite(pivot_atr):
                continue

            zones.append(
                {
                    "side": "LONG",
                    "zone_idx": i,
                    "available_at": htf[
                        "AvailableAt"
                    ].iloc[i],
                    "center": price,
                    "low": (
                        price
                        - HTF_ZONE_ATR
                        * pivot_atr
                    ),
                    "high": (
                        price
                        + HTF_ZONE_ATR
                        * pivot_atr
                    ),
                    "pivot_idx": pivot_index,
                }
            )

        # ----------------------------------------------------
        # SHORT = confirmed HTF swing high / supply area
        # ----------------------------------------------------

        if bool(
            htf["pivot_high"].iloc[i]
        ):

            pivot_index = int(
                htf["pivot_high_idx"].iloc[i]
            )

            price = float(
                htf["High"].iloc[pivot_index]
            )

            pivot_atr = float(
                htf["ATR"].iloc[pivot_index]
            )

            if not np.isfinite(pivot_atr):
                continue

            zones.append(
                {
                    "side": "SHORT",
                    "zone_idx": i,
                    "available_at": htf[
                        "AvailableAt"
                    ].iloc[i],
                    "center": price,
                    "low": (
                        price
                        - HTF_ZONE_ATR
                        * pivot_atr
                    ),
                    "high": (
                        price
                        + HTF_ZONE_ATR
                        * pivot_atr
                    ),
                    "pivot_idx": pivot_index,
                }
            )

    zones.sort(
        key=lambda z: z["available_at"]
    )

    return zones


# ============================================================
# CONFIRMED PIVOTS BEFORE A POINT
# ============================================================

def confirmed_pivots_before(
    x,
    end_i,
    kind,
):
    if kind == "HIGH":

        confirmation_col = (
            "pivot_high"
        )

        pivot_col = (
            "pivot_high_idx"
        )

        price_col = "High"

    else:

        confirmation_col = (
            "pivot_low"
        )

        pivot_col = (
            "pivot_low_idx"
        )

        price_col = "Low"

    result = []

    start = max(
        0,
        end_i - 3000,
    )

    for confirmation_i in range(
        start,
        end_i + 1,
    ):

        if not bool(
            x[confirmation_col].iloc[
                confirmation_i
            ]
        ):
            continue

        pivot_i = int(
            x[pivot_col].iloc[
                confirmation_i
            ]
        )

        if pivot_i < 0:
            continue

        if pivot_i >= end_i:
            continue

        price = float(
            x[price_col].iloc[pivot_i]
        )

        result.append(
            (
                pivot_i,
                confirmation_i,
                price,
            )
        )

    return result


# ============================================================
# FIRST HTF ZONE TOUCH
# ============================================================

def first_zone_touch(
    x,
    zone,
    start_i,
):
    last_i = min(
        len(x) - 1,
        start_i + REACTION_MAX_BARS,
    )

    for i in range(
        start_i,
        last_i + 1,
    ):

        candle_high = float(
            x["High"].iloc[i]
        )

        candle_low = float(
            x["Low"].iloc[i]
        )

        touches = (
            candle_high >= zone["low"]
            and
            candle_low <= zone["high"]
        )

        if touches:
            return i

    return None


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def make_candidates(
    x,
    symbol,
):
    """
    Setup structure:

    1. HTF zone is confirmed.
    2. Price first reacts to the HTF zone.
    3. Lower-TF structure forms.
    4. First confirmed LH/HL creates the OB.
    5. Aligned liquidity forms after OB.
    6. Liquidity is broken.
    7. Price returns to OB.
    8. Entry is next candle open.

    Every structural index must be strictly before entry.
    """

    htf = make_htf(x)

    zones = build_htf_zones(htf)

    candidates = []

    used_entries = set()

    dates = x["Date"]

    for zone in zones:

        # ----------------------------------------------------
        # HTF confirmation -> first available 15m candle
        # ----------------------------------------------------

        start_i = int(
            dates.searchsorted(
                zone["available_at"],
                side="left",
            )
        )

        if start_i >= len(x) - 20:
            continue

        touch_i = first_zone_touch(
            x,
            zone,
            start_i,
        )

        if touch_i is None:
            continue

        side = zone["side"]

        # ====================================================
        # SHORT
        # ====================================================

        if side == "SHORT":

            all_highs = confirmed_pivots_before(
                x,
                len(x) - 1,
                "HIGH",
            )

            highs_after_reaction = [
                p
                for p in all_highs
                if p[1] > touch_i
            ]

            if len(highs_after_reaction) < 2:
                continue

            # ------------------------------------------------
            # FIRST confirmed lower high after reaction.
            # It must be lower than the immediately previous
            # confirmed swing high.
            # ------------------------------------------------

            lh = None

            for j in range(
                1,
                len(highs_after_reaction),
            ):

                previous = (
                    highs_after_reaction[j - 1]
                )

                current = (
                    highs_after_reaction[j]
                )

                if current[2] < previous[2]:

                    lh = current
                    break

            if lh is None:
                continue

            (
                ob_pivot_i,
                ob_confirmed_i,
                ob_price,
            ) = lh

            if ob_confirmed_i <= touch_i:
                continue

            # ------------------------------------------------
            # Bearish OB:
            # high -> body low
            # ------------------------------------------------

            ob_high = float(
                x["High"].iloc[ob_pivot_i]
            )

            ob_low = float(
                min(
                    x["Open"].iloc[ob_pivot_i],
                    x["Close"].iloc[ob_pivot_i],
                )
            )

            if ob_low >= ob_high:
                continue

            # ------------------------------------------------
            # Aligned liquidity highs after OB.
            # ------------------------------------------------

            liquidity_highs = [
                p
                for p in highs_after_reaction
                if p[1] > ob_confirmed_i
            ]

            if len(liquidity_highs) < 2:
                continue

            liq_1 = liquidity_highs[0]
            liq_2 = None

            for candidate in liquidity_highs[1:]:

                relative_difference = (
                    abs(
                        candidate[2]
                        - liq_1[2]
                    )
                    / max(
                        abs(liq_1[2]),
                        1e-12,
                    )
                )

                if (
                    relative_difference
                    <= ALIGN_TOL
                ):
                    liq_2 = candidate
                    break

            if liq_2 is None:
                continue

            if (
                liq_2[1]
                - ob_pivot_i
                > MAX_LIQUIDITY_DISTANCE
            ):
                continue

            liquidity_level = (
                liq_1[2]
                + liq_2[2]
            ) / 2.0

            # ------------------------------------------------
            # Lowest valley between the two liquidity highs.
            # ------------------------------------------------

            valley_i = None
            lowest = float("inf")

            for k in range(
                liq_1[0] + 1,
                liq_2[0],
            ):

                low = float(
                    x["Low"].iloc[k]
                )

                if low < lowest:

                    lowest = low
                    valley_i = k

            if valley_i is None:
                continue

            # ------------------------------------------------
            # Liquidity break:
            # wick above liquidity,
            # close back below.
            # ------------------------------------------------

            break_i = None

            break_end = min(
                len(x),
                liq_2[1]
                + MAX_LIQUIDITY_DISTANCE
                + 1,
            )

            for b in range(
                liq_2[1],
                break_end,
            ):

                high = float(
                    x["High"].iloc[b]
                )

                close = float(
                    x["Close"].iloc[b]
                )

                if (
                    high > liquidity_level
                    and
                    close < liquidity_level
                ):
                    break_i = b
                    break

            if break_i is None:
                continue

            # ------------------------------------------------
            # Return to OB.
            #
            # Entry happens on the NEXT candle, so the return
            # candle itself is never used as an execution
            # candle.
            # ------------------------------------------------

            entry_i = None

            search_end = min(
                len(x),
                break_i + 501,
            )

            for e in range(
                break_i + 1,
                search_end,
            ):

                high = float(
                    x["High"].iloc[e]
                )

                low = float(
                    x["Low"].iloc[e]
                )

                close = float(
                    x["Close"].iloc[e]
                )

                touched_ob = (
                    high >= ob_low
                    and
                    low <= ob_high
                )

                valid_retest = (
                    touched_ob
                    and
                    close <= ob_high
                )

                if valid_retest:

                    entry_i = e + 1
                    break

            if entry_i is None:
                continue

            if entry_i >= len(x):
                continue

            # ------------------------------------------------
            # Execution
            # ------------------------------------------------

            raw_entry = float(
                x["Open"].iloc[entry_i]
            )

            entry = (
                raw_entry
                * (1.0 - SLIPPAGE)
            )

            stop = (
                ob_high
                * (1.0 + SL_BUFFER_PCT)
            )

            risk = stop - entry

            if risk <= 0:
                continue

            risk_pct = (
                risk
                / max(entry, 1e-12)
            )

            if (
                risk_pct < MIN_RISK_PCT
                or
                risk_pct > MAX_RISK_PCT
            ):
                continue

            target = (
                entry
                - RR * risk
            )

            structural_target = (
                lowest
            )

            # Short target must not extend beyond the
            # structural target.
            if target < structural_target:
                continue

            structural_indices = [
                zone["zone_idx"],
                touch_i,
                ob_confirmed_i,
                liq_1[1],
                liq_2[1],
                break_i,
            ]

            if not all(
                idx < entry_i
                for idx in structural_indices
            ):
                continue

            key = (
                symbol,
                entry_i,
                side,
            )

            if key in used_entries:
                continue

            used_entries.add(key)

            candidates.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "entry_i": entry_i,
                    "entry": entry,
                    "stop": stop,
                    "target": target,

                    "zone_idx": zone["zone_idx"],
                    "touch_i": touch_i,

                    "ob_i": ob_pivot_i,
                    "ob_confirmed": ob_confirmed_i,

                    "liq1_i": liq_1[1],
                    "liq2_i": liq_2[1],

                    "break_i": break_i,

                    "structural_target":
                        structural_target,
                }
            )

        # ====================================================
        # LONG
        # ====================================================

        else:

            all_lows = confirmed_pivots_before(
                x,
                len(x) - 1,
                "LOW",
            )

            lows_after_reaction = [
                p
                for p in all_lows
                if p[1] > touch_i
            ]

            if len(lows_after_reaction) < 2:
                continue

            # ------------------------------------------------
            # FIRST confirmed higher low after reaction.
            # ------------------------------------------------

            hl = None

            for j in range(
                1,
                len(lows_after_reaction),
            ):

                previous = (
                    lows_after_reaction[j - 1]
                )

                current = (
                    lows_after_reaction[j]
                )

                if current[2] > previous[2]:

                    hl = current
                    break

            if hl is None:
                continue

            (
                ob_pivot_i,
                ob_confirmed_i,
                ob_price,
            ) = hl

            if ob_confirmed_i <= touch_i:
                continue

            # ------------------------------------------------
            # Bullish OB:
            # body high -> low
            # ------------------------------------------------

            ob_low = float(
                x["Low"].iloc[ob_pivot_i]
            )

            ob_high = float(
                max(
                    x["Open"].iloc[ob_pivot_i],
                    x["Close"].iloc[ob_pivot_i],
                )
            )

            if ob_high <= ob_low:
                continue

            # ------------------------------------------------
            # Aligned liquidity lows.
            # ------------------------------------------------

            liquidity_lows = [
                p
                for p in lows_after_reaction
                if p[1] > ob_confirmed_i
            ]

            if len(liquidity_lows) < 2:
                continue

            liq_1 = liquidity_lows[0]
            liq_2 = None

            for candidate in liquidity_lows[1:]:

                relative_difference = (
                    abs(
                        candidate[2]
                        - liq_1[2]
                    )
                    / max(
                        abs(liq_1[2]),
                        1e-12,
                    )
                )

                if (
                    relative_difference
                    <= ALIGN_TOL
                ):
                    liq_2 = candidate
                    break

            if liq_2 is None:
                continue

            if (
                liq_2[1]
                - ob_pivot_i
                > MAX_LIQUIDITY_DISTANCE
            ):
                continue

            liquidity_level = (
                liq_1[2]
                + liq_2[2]
            ) / 2.0

            # ------------------------------------------------
            # Highest peak between the liquidity lows.
            # ------------------------------------------------

            peak_i = None
            highest = -float("inf")

            for k in range(
                liq_1[0] + 1,
                liq_2[0],
            ):

                high = float(
                    x["High"].iloc[k]
                )

                if high > highest:

                    highest = high
                    peak_i = k

            if peak_i is None:
                continue

            # ------------------------------------------------
            # Liquidity break:
            # wick below liquidity,
            # close back above.
            # ------------------------------------------------

            break_i = None

            break_end = min(
                len(x),
                liq_2[1]
                + MAX_LIQUIDITY_DISTANCE
                + 1,
            )

            for b in range(
                liq_2[1],
                break_end,
            ):

                low = float(
                    x["Low"].iloc[b]
                )

                close = float(
                    x["Close"].iloc[b]
                )

                if (
                    low < liquidity_level
                    and
                    close > liquidity_level
                ):
                    break_i = b
                    break

            if break_i is None:
                continue

            # ------------------------------------------------
            # Return to OB.
            # ------------------------------------------------

            entry_i = None

            search_end = min(
                len(x),
                break_i + 501,
            )

            for e in range(
                break_i + 1,
                search_end,
            ):

                high = float(
                    x["High"].iloc[e]
                )

                low = float(
                    x["Low"].iloc[e]
                )

                close = float(
                    x["Close"].iloc[e]
                )

                touched_ob = (
                    high >= ob_low
                    and
                    low <= ob_high
                )

                valid_retest = (
                    touched_ob
                    and
                    close >= ob_low
                )

                if valid_retest:

                    entry_i = e + 1
                    break

            if entry_i is None:
                continue

            if entry_i >= len(x):
                continue

            # ------------------------------------------------
            # Execution
            # ------------------------------------------------

            raw_entry = float(
                x["Open"].iloc[entry_i]
            )

            entry = (
                raw_entry
                * (1.0 + SLIPPAGE)
            )

            stop = (
                ob_low
                * (1.0 - SL_BUFFER_PCT)
            )

            risk = entry - stop

            if risk <= 0:
                continue

            risk_pct = (
                risk
                / max(entry, 1e-12)
            )

            if (
                risk_pct < MIN_RISK_PCT
                or
                risk_pct > MAX_RISK_PCT
            ):
                continue

            target = (
                entry
                + RR * risk
            )

            structural_target = (
                highest
            )

            # Long target must not exceed structural target.
            if target > structural_target:
                continue

            structural_indices = [
                zone["zone_idx"],
                touch_i,
                ob_confirmed_i,
                liq_1[1],
                liq_2[1],
                break_i,
            ]

            if not all(
                idx < entry_i
                for idx in structural_indices
            ):
                continue

            key = (
                symbol,
                entry_i,
                side,
            )

            if key in used_entries:
                continue

            used_entries.add(key)

            candidates.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "entry_i": entry_i,
                    "entry": entry,
                    "stop": stop,
                    "target": target,

                    "zone_idx": zone["zone_idx"],
                    "touch_i": touch_i,

                    "ob_i": ob_pivot_i,
                    "ob_confirmed": ob_confirmed_i,

                    "liq1_i": liq_1[1],
                    "liq2_i": liq_2[1],

                    "break_i": break_i,

                    "structural_target":
                        structural_target,
                }
            )

    candidates.sort(
        key=lambda c: (
            c["entry_i"],
            c["side"],
            c["ob_i"],
        )
    )

    return candidates


# ============================================================
# PNL
# ============================================================

def pnl_for_trade(
    side,
    entry,
    exit_price,
):
    if side == "LONG":

        gross = (
            NOTIONAL
            * (exit_price - entry)
            / entry
        )

    else:

        gross = (
            NOTIONAL
            * (entry - exit_price)
            / entry
        )

    fees = (
        NOTIONAL
        * FEE_RATE
        * 2.0
    )

    return gross - fees


# ============================================================
# SIMULATION
# ============================================================

def simulate_symbol(
    x,
    candidates,
    symbol,
):
    trades = []

    # Important:
    # This lock is PER SYMBOL.
    #
    # Other symbols are simulated independently and therefore
    # may have trades open at the same time.
    last_exit_i = -1

    n = len(x)

    for candidate in candidates:

        entry_i = int(
            candidate["entry_i"]
        )

        # No same-symbol overlap.
        # Entry must be strictly after previous exit.
        if entry_i <= last_exit_i:
            continue

        if entry_i >= n:
            continue

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

        exit_i = None
        exit_price = None
        outcome = None

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Entry happens at the OPEN of entry candle.
        #
        # We intentionally do NOT inspect that candle for
        # SL/TP because OHLC cannot tell us the intrabar
        # order after the entry.
        # ----------------------------------------------------

        for i in range(
            entry_i + 1,
            n,
        ):

            high = float(
                x["High"].iloc[i]
            )

            low = float(
                x["Low"].iloc[i]
            )

            if side == "LONG":

                hit_sl = (
                    low <= stop
                )

                hit_tp = (
                    high >= target
                )

            else:

                hit_sl = (
                    high >= stop
                )

                hit_tp = (
                    low <= target
                )

            # Conservative ambiguity rule:
            # if both are hit on same candle,
            # SL wins.
            if hit_sl and hit_tp:

                exit_i = i
                exit_price = stop
                outcome = "LOSS"

                break

            if hit_sl:

                exit_i = i
                exit_price = stop
                outcome = "LOSS"

                break

            if hit_tp:

                exit_i = i
                exit_price = target
                outcome = "WIN"

                break

        # ----------------------------------------------------
        # Unresolved at end of sample.
        # It is NOT silently removed.
        # ----------------------------------------------------

        if exit_i is None:

            trades.append(
                {
                    **candidate,
                    "exit_i": None,
                    "exit": None,
                    "outcome": "UNRESOLVED",
                    "pnl": 0.0,
                    "r_multiple": np.nan,
                    "entry_time": x[
                        "Date"
                    ].iloc[entry_i],
                    "exit_time": None,
                }
            )

            # Conservative:
            # unresolved trade blocks any later same-symbol
            # entry because there is no known closing candle.
            last_exit_i = n - 1

            continue

        pnl = pnl_for_trade(
            side,
            entry,
            exit_price,
        )

        geometric_risk_dollars = (
            NOTIONAL
            * abs(entry - stop)
            / entry
        )

        r_multiple = (
            pnl
            / geometric_risk_dollars
            if geometric_risk_dollars > 0
            else np.nan
        )

        trades.append(
            {
                **candidate,
                "exit_i": exit_i,
                "exit": exit_price,
                "outcome": outcome,
                "pnl": pnl,
                "r_multiple": r_multiple,
                "entry_time": x[
                    "Date"
                ].iloc[entry_i],
                "exit_time": x[
                    "Date"
                ].iloc[exit_i],
            }
        )

        last_exit_i = exit_i

    return trades


# ============================================================
# STATS
# ============================================================

def max_streak(trades):

    streak = 0
    best = 0

    for trade in trades:

        if trade["outcome"] == "LOSS":

            streak += 1
            best = max(
                best,
                streak,
            )

        elif trade["outcome"] == "WIN":

            streak = 0

    return best


def profit_factor(trades):

    resolved = [
        t
        for t in trades
        if t["outcome"]
        in ("WIN", "LOSS")
    ]

    gross_profit = sum(
        max(0.0, t["pnl"])
        for t in resolved
    )

    gross_loss = sum(
        -min(0.0, t["pnl"])
        for t in resolved
    )

    if gross_loss > 0:
        return (
            gross_profit
            / gross_loss
        )

    if gross_profit > 0:
        return float("inf")

    return 0.0


def max_drawdown(trades):

    resolved = [
        t
        for t in trades
        if t["outcome"]
        in ("WIN", "LOSS")
    ]

    resolved.sort(
        key=lambda t: (
            t["exit_time"],
            t["symbol"],
        )
    )

    equity = INITIAL_CAPITAL
    peak = equity
    max_dd = 0.0

    for trade in resolved:

        equity += trade["pnl"]

        if equity > peak:
            peak = equity

        drawdown = (
            peak - equity
        )

        if drawdown > max_dd:
            max_dd = drawdown

    return max_dd


def report(
    trades,
    name,
):
    resolved = [
        t
        for t in trades
        if t["outcome"]
        in ("WIN", "LOSS")
    ]

    wins = sum(
        t["outcome"] == "WIN"
        for t in resolved
    )

    losses = sum(
        t["outcome"] == "LOSS"
        for t in resolved
    )

    unresolved = sum(
        t["outcome"] == "UNRESOLVED"
        for t in trades
    )

    pnl = sum(
        t["pnl"]
        for t in resolved
    )

    net_r = sum(
        t["r_multiple"]
        for t in resolved
        if np.isfinite(
            t["r_multiple"]
        )
    )

    wr = (
        100.0 * wins / len(resolved)
        if resolved
        else 0.0
    )

    pf = profit_factor(
        trades
    )

    dd = max_drawdown(
        trades
    )

    streak = max_streak(
        resolved
    )

    print(
        f"{name:20s} "
        f"{len(resolved):6d} "
        f"{wins:6d} "
        f"{losses:7d} "
        f"{wr:8.2f} "
        f"{pf:9.3f} "
        f"{net_r:12.3f} "
        f"{pnl:12.2f} "
        f"{streak:10d} "
        f"{unresolved:6d}"
    )

    return {
        "split": name,
        "trades": len(resolved),
        "wins": wins,
        "losses": losses,
        "WR_%": wr,
        "PF": pf,
        "net_R": net_r,
        "PnL_$": pnl,
        "max_streak": streak,
        "unresolved": unresolved,
        "max_dd_$": dd,
    }


# ============================================================
# INTEGRITY AUDIT
# ============================================================

def audit(
    all_candidates,
    all_trades,
    frames,
):
    issues = []

    # --------------------------------------------------------
    # Candidate causality
    # --------------------------------------------------------

    for candidate in all_candidates:

        entry_i = int(
            candidate["entry_i"]
        )

        structural_keys = [
            "zone_idx",
            "touch_i",
            "ob_confirmed",
            "liq1_i",
            "liq2_i",
            "break_i",
        ]

        for key in structural_keys:

            index_value = int(
                candidate[key]
            )

            if index_value >= entry_i:

                issues.append(
                    "future structural index: "
                    f"{candidate['symbol']} "
                    f"{key}={index_value} "
                    f"entry={entry_i}"
                )

        # ----------------------------------------------------
        # RR audit
        # ----------------------------------------------------

        entry = float(
            candidate["entry"]
        )

        stop = float(
            candidate["stop"]
        )

        target = float(
            candidate["target"]
        )

        risk = abs(
            entry - stop
        )

        expected_target_distance = (
            RR * risk
        )

        actual_target_distance = abs(
            target - entry
        )

        if not np.isclose(
            actual_target_distance,
            expected_target_distance,
            rtol=1e-8,
            atol=1e-10,
        ):

            issues.append(
                "RR mismatch: "
                f"{candidate['symbol']} "
                f"entry={entry_i}"
            )

    # --------------------------------------------------------
    # Per-symbol overlap
    # --------------------------------------------------------

    for symbol, frame in frames.items():

        symbol_trades = [
            t
            for t in all_trades
            if (
                t["symbol"] == symbol
                and
                t["outcome"]
                != "UNRESOLVED"
            )
        ]

        symbol_trades.sort(
            key=lambda t: t["entry_i"]
        )

        for previous, current in zip(
            symbol_trades,
            symbol_trades[1:],
        ):

            if (
                current["entry_i"]
                <= previous["exit_i"]
            ):

                issues.append(
                    "same-symbol overlap: "
                    f"{symbol} "
                    f"{previous['entry_i']}"
                    f"->{previous['exit_i']} "
                    f"then "
                    f"{current['entry_i']}"
                )

            if (
                current["entry_i"]
                == previous["exit_i"]
            ):

                issues.append(
                    "same-candle re-entry: "
                    f"{symbol} "
                    f"candle={current['entry_i']}"
                )

    # --------------------------------------------------------
    # Exit ordering
    # --------------------------------------------------------

    for trade in all_trades:

        if (
            trade["outcome"]
            == "UNRESOLVED"
        ):
            continue

        if (
            int(trade["exit_i"])
            <= int(trade["entry_i"])
        ):

            issues.append(
                "invalid exit ordering: "
                f"{trade['symbol']} "
                f"{trade['entry_i']}"
                f"->{trade['exit_i']}"
            )

    # --------------------------------------------------------
    # Print audit
    # --------------------------------------------------------

    print()
    print(
        "FINAL INTEGRITY AUDIT"
    )

    checks = {
        "Data gaps":
            not any(
                "data gaps"
                in issue.lower()
                for issue in issues
            ),

        "Structural causality":
            not any(
                "future structural"
                in issue
                for issue in issues
            ),

        "Future target leak":
            not any(
                "future structural"
                in issue
                for issue in issues
            ),

        "Same-symbol overlap":
            not any(
                "same-symbol overlap"
                in issue
                for issue in issues
            ),

        "Same-candle re-entry/exit":
            not any(
                (
                    "same-candle"
                    in issue
                    or
                    "invalid exit"
                    in issue
                )
                for issue in issues
            ),

        "RR 1:2 consistency":
            not any(
                "RR mismatch"
                in issue
                for issue in issues
            ),
    }

    for name, passed in checks.items():

        print(
            f"{name:30s}: "
            f"{'PASSED' if passed else 'FAILED'}"
        )

    if issues:

        print()
        print(
            "Audit issues:"
        )

        for issue in issues[:30]:
            print(
                " -",
                issue,
            )

    audit_ok = (
        len(issues) == 0
    )

    print(
        "AUDIT STATUS                    :",
        "PASSED"
        if audit_ok
        else "FAILED",
    )

    return audit_ok


# ============================================================
# SAVE RESULTS
# ============================================================

def save_trades(
    trades,
):
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not trades:
        return

    df = pd.DataFrame(
        trades
    )

    df.to_csv(
        OUT_DIR
        / "setup4_v1_trade_ledger.csv",
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "============================================================"
    )
    print(
        "SETUP 4 V1 — HTF ZONE / OB / LIQUIDITY"
    )
    print(
        "============================================================"
    )

    print(
        f"Research start : {RESEARCH_START}"
    )

    print(
        f"Research end   : "
        f"{OOS_START - pd.Timedelta(minutes=15)}"
    )

    print(
        f"Validation OOS : {OOS_START}"
    )

    print(
        f"OOS end        : {OOS_END}"
    )

    print(
        f"RR             : 1:{RR:.0f}"
    )

    print(
        f"Initial capital: ${INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Margin         : ${MARGIN:,.2f}"
    )

    print(
        f"Leverage       : {LEVERAGE:.0f}x"
    )

    print(
        f"Notional       : ${NOTIONAL:,.2f}"
    )

    print()

    frames = {}

    all_candidates = []
    all_trades = []

    fetch_end = (
        OOS_END
        + pd.Timedelta(minutes=15)
    )

    # ========================================================
    # DATA + SIGNALS + SIMULATION
    # ========================================================

    for number, symbol in enumerate(
        SYMBOLS,
        1,
    ):

        print(
            f"[DATA {number}/{len(SYMBOLS)}] "
            f"Fetching {symbol}"
        )

        df = fetch_symbol(
            symbol,
            WARMUP_START,
            fetch_end,
        )

        print(
            f"[DATA] {symbol}: "
            f"{len(df)} candles | "
            f"{df['Date'].iloc[0]} -> "
            f"{df['Date'].iloc[-1]}"
        )

        df = add_atr(df)

        df = add_confirmed_pivots(
            df
        )

        frames[symbol] = df

        candidates = make_candidates(
            df,
            symbol,
        )

        all_candidates.extend(
            candidates
        )

        print(
            f"[SIGNALS] {symbol}: "
            f"{len(candidates)} candidates"
        )

        trades = simulate_symbol(
            df,
            candidates,
            symbol,
        )

        all_trades.extend(
            trades
        )

        print(
            f"[TRADES] {symbol}: "
            f"{len(trades)} records"
        )

    # ========================================================
    # GLOBAL TRADE ORDER
    # ========================================================

    all_trades.sort(
        key=lambda t: (
            t.get(
                "entry_time",
                pd.Timestamp.max.tz_localize(
                    "UTC"
                ),
            ),
            t["symbol"],
        )
    )

    # ========================================================
    # RESEARCH SPLITS
    #
    # 365d research:
    #   first ~50% = Discovery
    #   remaining ~50% = Development
    #
    # Validation OOS is kept completely separate and untouched.
    # ========================================================

    discovery_start = (
        RESEARCH_START
    )

    discovery_end = (
        RESEARCH_START
        + pd.Timedelta(days=182)
    )

    development_start = (
        discovery_end
        + pd.Timedelta(minutes=15)
    )

    development_end = (
        OOS_START
        - pd.Timedelta(minutes=15)
    )

    def in_window(
        trade,
        start,
        end,
    ):

        entry_time = trade.get(
            "entry_time"
        )

        if entry_time is None:
            return False

        return (
            start
            <= entry_time
            <= end
        )

    discovery = [
        t
        for t in all_trades
        if in_window(
            t,
            discovery_start,
            discovery_end,
        )
    ]

    development = [
        t
        for t in all_trades
        if in_window(
            t,
            development_start,
            development_end,
        )
    ]

    validation_oos = [
        t
        for t in all_trades
        if in_window(
            t,
            OOS_START,
            OOS_END,
        )
    ]

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print(
        "SETUP 4 V1 — BACKTEST REPORT"
    )

    print(
        f"{'split':20s} "
        f"{'trades':>6s} "
        f"{'wins':>6s} "
        f"{'losses':>7s} "
        f"{'WR_%':>8s} "
        f"{'PF':>9s} "
        f"{'net_R':>12s} "
        f"{'PnL_$':>12s} "
        f"{'max_streak':>10s} "
        f"{'unres':>6s}"
    )

    reports = []

    reports.append(
        report(
            discovery,
            "Discovery",
        )
    )

    reports.append(
        report(
            development,
            "Development",
        )
    )

    reports.append(
        report(
            validation_oos,
            "Validation_OOS",
        )
    )

    resolved_total = [
        t
        for t in all_trades
        if t["outcome"]
        in ("WIN", "LOSS")
    ]

    report(
        resolved_total,
        "TOTAL",
    )

    total_pnl = sum(
        t["pnl"]
        for t in resolved_total
    )

    unresolved_count = sum(
        t["outcome"]
        == "UNRESOLVED"
        for t in all_trades
    )

    print()

    print(
        f"Initial Capital : "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Final Equity    : "
        f"${INITIAL_CAPITAL + total_pnl:,.2f}"
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
        f"Unresolved      : "
        f"{unresolved_count}"
    )

    # ========================================================
    # FINAL INTEGRITY AUDIT
    # ========================================================

    audit_ok = audit(
        all_candidates,
        all_trades,
        frames,
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_trades(
        all_trades
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        reports
    ).to_csv(
        OUT_DIR
        / "setup4_v1_split_report.csv",
        index=False,
    )

    # ========================================================
    # HARD FAIL ON AUDIT
    # ========================================================

    if not audit_ok:

        raise RuntimeError(
            "FINAL INTEGRITY AUDIT FAILED"
        )

    print()
    print(
        "Backtest completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
