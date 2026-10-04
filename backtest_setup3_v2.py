 import math
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# ============================================================
# SETUP 3 V2
# STRUCTURAL / IFC / DOUBLE VALLEY-TOP
#
# Source basis:
# 10 ستاپ برتر.pdf — pages 18-21
#
# PDF sequence:
#   Trend structure
#   -> LH / HH creates Order Block
#   -> IFC after OB
#   -> Double valley / double top
#   -> Return to OB
#   -> Entry
#   -> SL behind OB
#   -> Structural target
#
# IMPORTANT:
# Numeric definitions for pivot, IFC, alignment and exact OB
# boundaries are mechanical research translations.
# They are NOT claimed to be literal numeric formulas from PDF.
#
# V2 integrity corrections:
#   1. Pivot confirmation is causal.
#   2. No artificial pattern-formation timeout.
#   3. IFC must occur after OB.
#   4. Double structure must be confirmed before entry.
#   5. Entry cannot occur before structural confirmation.
#   6. Fixed RR = 1:2.
#   7. Structural target is a viability filter only.
#   8. No same-candle re-entry.
#   9. One simultaneous trade per symbol.
#  10. Different symbols may overlap.
#  11. Warmup trades are excluded from research statistics.
#  12. Real exchange data only.
# ============================================================


# ============================================================
# SYMBOLS
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
# TIMEFRAME
# ============================================================

INTERVAL = "1h"
BAR = pd.Timedelta(hours=1)


# ============================================================
# ACCOUNT / EXECUTION
# ============================================================

INITIAL_CAPITAL = 1000.0

MARGIN = 100.0
LEVERAGE = 50.0

NOTIONAL = (
    MARGIN * LEVERAGE
)

RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003


# ============================================================
# STRUCTURAL TRANSLATION
# ============================================================

# Pivot at index i becomes known only at i + PIVOT.
PIVOT = 2

# "Aligned" double valley/top tolerance.
# This is a frozen research translation because
# the PDF does not provide a numeric tolerance.
ALIGN_TOL = 0.004

# Small stop buffer behind OB.
SL_BUFFER_PCT = 0.0005

# Frozen mechanical IFC translation.
# The PDF says IFC must occur after the OB,
# but does not specify an exact numerical formula.
IFC_LOOKAHEAD = 2


# ============================================================
# FRESH SETUP-3-V2 OOS
#
# This OOS is not the OOS used for Setup 3 V1.
# ============================================================

OOS_START = pd.Timestamp(
    "2024-10-04 00:00:00",
    tz="UTC",
)

OOS_END = pd.Timestamp(
    "2025-10-03 23:00:00",
    tz="UTC",
)


# ============================================================
# RESEARCH WINDOW
# ============================================================

RESEARCH_END = (
    OOS_START - BAR
)

RESEARCH_START = (
    RESEARCH_END
    - pd.Timedelta(days=365)
    + BAR
)


# Warmup exists only to allow structural context
# before the research window.
WARMUP_START = (
    RESEARCH_START
    - pd.Timedelta(days=90)
)


# 60% Discovery
# 20% Development
# 20% Research Holdout
#
# Validation OOS is outside the research window.
DISCOVERY_END = (
    RESEARCH_START
    + (
        RESEARCH_END
        - RESEARCH_START
    ) * 0.60
)

DEVELOPMENT_END = (
    RESEARCH_START
    + (
        RESEARCH_END
        - RESEARCH_START
    ) * 0.80
)


# ============================================================
# BINANCE DATA
# ============================================================

DATA_BASE = (
    "https://data.binance.vision/"
    "data/futures/um/monthly/klines"
)


# ============================================================
# OUTPUTS
# ============================================================

CACHE_DIR = Path(
    "data_cache"
)

OUT_DIR = Path(
    "setup3_v2_outputs"
)

LEDGER_PATH = (
    OUT_DIR
    / "setup3_v2_trade_ledger.csv"
)

REPORT_PATH = (
    OUT_DIR
    / "setup3_v2_report.txt"
)


# ============================================================
# HTTP SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent":
        "Setup3-V2-Research/1.0"
    }
)


# ============================================================
# MONTH ITERATOR
# ============================================================

def month_starts(
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    """
    Generate monthly UTC timestamps.

    Timezone is explicitly removed before Period conversion
    to avoid pandas timezone-dropping warnings.
    """

    start_naive = (
        start
        .tz_convert("UTC")
        .tz_localize(None)
    )

    end_naive = (
        end
        .tz_convert("UTC")
        .tz_localize(None)
    )

    current = (
        start_naive
        .to_period("M")
        .to_timestamp()
        .tz_localize("UTC")
    )

    last = (
        end_naive
        .to_period("M")
        .to_timestamp()
        .tz_localize("UTC")
    )

    while current <= last:

        yield current

        current = (
            current
            + pd.offsets.MonthBegin(1)
        ).normalize()


# ============================================================
# DOWNLOAD MONTH
# ============================================================

def download_month(
    symbol: str,
    month: pd.Timestamp,
) -> Optional[Path]:
    """
    Download one Binance Futures monthly archive.

    IMPORTANT:
    404 is NOT immediately treated as a fatal error.

    The caller determines whether the missing month is:
      - before the first available archive -> pre-listing
      - after the first available archive  -> fatal gap
    """

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{symbol}-1h-"
        f"{month.strftime('%Y-%m')}.zip"
    )

    path = (
        CACHE_DIR
        / filename
    )

    if (
        path.exists()
        and path.stat().st_size > 1000
    ):
        return path

    url = (
        f"{DATA_BASE}/"
        f"{symbol}/1h/{filename}"
    )

    for attempt in range(1, 4):

        try:

            response = SESSION.get(
                url,
                timeout=30,
            )

            # Successful archive.
            if (
                response.status_code == 200
                and len(response.content) > 1000
            ):
                path.write_bytes(
                    response.content
                )

                return path

            # A missing archive is handled by fetch_symbol().
            if response.status_code == 404:
                return None

            raise RuntimeError(
                f"HTTP "
                f"{response.status_code}: "
                f"{url}"
            )

        except Exception:

            if attempt == 3:
                raise

            time.sleep(
                1.5 * attempt
            )

    raise RuntimeError(
        "Download failed."
    )


# ============================================================
# READ ARCHIVE
# ============================================================

def read_archive(
    path: Path,
) -> pd.DataFrame:
    """
    Read Binance monthly kline archive.

    Only real exchange OHLCV data is used.
    No synthetic candles are created.
    """

    with zipfile.ZipFile(path) as archive:

        csv_names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_names:

            raise RuntimeError(
                f"No CSV inside {path}"
            )

        with archive.open(
            csv_names[0]
        ) as file:

            df = pd.read_csv(
                file,
                header=None,
            )

    if df.shape[1] < 6:

        raise RuntimeError(
            f"Bad kline columns: {path}"
        )

    df = df.iloc[:, :6].copy()

    df.columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    df["open_time"] = pd.to_numeric(
        df["open_time"],
        errors="coerce",
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna().copy()

    df["time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    return df[
        [
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]


# ============================================================
# FETCH SYMBOL
# ============================================================

def fetch_symbol(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Fetch real Binance Futures 1H data.

    Data integrity rules:

    1. Missing months before first available archive are allowed.
       This represents pre-listing history.

    2. Once the first archive exists, any missing monthly archive
       is a fatal data gap.

    3. After assembling the real data, every hourly candle must
       exist between first available candle and last available
       candle.

    4. OOS must have complete hourly coverage.

    5. No forward filling.
    6. No synthetic candles.
    """

    frames = []

    first_available_month = None

    requested_months = list(
        month_starts(
            start,
            end,
        )
    )

    for month in requested_months:

        path = download_month(
            symbol,
            month,
        )

        # ----------------------------------------------------
        # Archive does not exist.
        # ----------------------------------------------------

        if path is None:

            # Before the first real archive:
            # symbol simply did not exist yet.
            if first_available_month is None:

                continue

            # After listing:
            # this is a real historical data gap.
            raise RuntimeError(
                f"{symbol}: "
                f"fatal archive gap after "
                f"first available month "
                f"{first_available_month.strftime('%Y-%m')}: "
                f"missing "
                f"{month.strftime('%Y-%m')}"
            )

        # First real archive found.
        if first_available_month is None:

            first_available_month = month

        frames.append(
            read_archive(path)
        )

    # No archive at all.
    if not frames:

        raise RuntimeError(
            f"{symbol}: "
            f"no historical data available "
            f"between {start} and {end}"
        )

    # --------------------------------------------------------
    # Combine real exchange data.
    # --------------------------------------------------------

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    df = (
        df
        .drop_duplicates(
            subset="time"
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Requested range.
    # --------------------------------------------------------

    df = df[
        (df["time"] >= start)
        & (df["time"] <= end)
    ].copy()

    if df.empty:

        raise RuntimeError(
            f"{symbol}: "
            f"no data after requested "
            f"date filtering"
        )

    # --------------------------------------------------------
    # Remove incomplete latest candle.
    # --------------------------------------------------------

    now = pd.Timestamp.now(
        tz="UTC"
    )

    df = df[
        df["time"] + BAR <= now
    ].copy()

    if df.empty:

        raise RuntimeError(
            f"{symbol}: "
            f"no completed candles"
        )

    # --------------------------------------------------------
    # Strict full-history continuity.
    #
    # Important:
    # We start from the first REAL candle.
    # Pre-listing history is not considered a gap.
    # --------------------------------------------------------

    actual = pd.DatetimeIndex(
        df["time"]
    )

    expected = pd.date_range(
        actual[0],
        actual[-1],
        freq="1h",
        tz="UTC",
    )

    missing = expected.difference(
        actual
    )

    if len(missing):

        raise RuntimeError(
            f"{symbol}: "
            f"fatal 1H data gaps after "
            f"first available candle: "
            f"{missing[:10].tolist()}"
        )

    # --------------------------------------------------------
    # Fresh OOS coverage.
    # --------------------------------------------------------

    oos = df[
        (df["time"] >= OOS_START)
        & (df["time"] <= OOS_END)
    ].copy()

    if oos.empty:

        raise RuntimeError(
            f"{symbol}: "
            f"no data covering fresh OOS "
            f"{OOS_START} -> {OOS_END}"
        )

    expected_oos = pd.date_range(
        OOS_START,
        OOS_END,
        freq="1h",
        tz="UTC",
    )

    actual_oos = pd.DatetimeIndex(
        oos["time"]
    )

    missing_oos = (
        expected_oos
        .difference(actual_oos)
    )

    if len(missing_oos):

        raise RuntimeError(
            f"{symbol}: "
            f"fatal OOS data gaps: "
            f"{missing_oos[:10].tolist()}"
        )

    print(
        f"{symbol}: "
        f"first real candle = "
        f"{df['time'].iloc[0]} | "
        f"OOS coverage = OK"
    )

    return df.set_index("time")


# ============================================================
# CAUSAL PIVOT HIGH
# ============================================================

def is_pivot_high(
    df: pd.DataFrame,
    index: int,
) -> bool:

    if index < PIVOT:
        return False

    if index + PIVOT >= len(df):
        return False

    value = float(
        df["high"].iloc[index]
    )

    left = df["high"].iloc[
        index - PIVOT:index
    ]

    right = df["high"].iloc[
        index + 1:index + PIVOT + 1
    ]

    return (
        value >= float(left.max())
        and
        value > float(right.max())
    )


# ============================================================
# CAUSAL PIVOT LOW
# ============================================================

def is_pivot_low(
    df: pd.DataFrame,
    index: int,
) -> bool:

    if index < PIVOT:
        return False

    if index + PIVOT >= len(df):
        return False

    value = float(
        df["low"].iloc[index]
    )

    left = df["low"].iloc[
        index - PIVOT:index
    ]

    right = df["low"].iloc[
        index + 1:index + PIVOT + 1
    ]

    return (
        value <= float(left.min())
        and
        value < float(right.min())
    )


# ============================================================
# ALIGNMENT
# ============================================================

def aligned(
    first: float,
    second: float,
) -> bool:

    return (
        abs(first - second)
        /
        max(
            abs(first),
            abs(second),
            1e-12,
        )
        <= ALIGN_TOL
    )


# ============================================================
# IFC — BEARISH
# ============================================================

def bearish_ifc(
    df: pd.DataFrame,
    ob_index: int,
) -> bool:
    """
    Mechanical research translation of IFC.

    The PDF does not specify an exact numerical formula.
    This version uses a bearish 3-candle displacement/FVG
    relationship immediately after the OB.
    """

    future_index = (
        ob_index
        + IFC_LOOKAHEAD
    )

    if future_index >= len(df):
        return False

    return (
        float(
            df["low"].iloc[ob_index]
        )
        >
        float(
            df["high"].iloc[future_index]
        )
    )


# ============================================================
# IFC — BULLISH
# ============================================================

def bullish_ifc(
    df: pd.DataFrame,
    ob_index: int,
) -> bool:

    future_index = (
        ob_index
        + IFC_LOOKAHEAD
    )

    if future_index >= len(df):
        return False

    return (
        float(
            df["high"].iloc[ob_index]
        )
        <
        float(
            df["low"].iloc[future_index]
        )
    )


# ============================================================
# ORDER BLOCK ZONE
# ============================================================

def candle_ob_zone(
    df: pd.DataFrame,
    index: int,
    side: str,
) -> Tuple[float, float]:

    open_price = float(
        df["open"].iloc[index]
    )

    close_price = float(
        df["close"].iloc[index]
    )

    high_price = float(
        df["high"].iloc[index]
    )

    low_price = float(
        df["low"].iloc[index]
    )

    if side == "SHORT":

        # Bearish supply OB.
        return (
            min(
                open_price,
                close_price,
            ),
            high_price,
        )

    # Bullish demand OB.
    return (
        low_price,
        max(
            open_price,
            close_price,
        ),
    )


# ============================================================
# BUILD STRUCTURAL CANDIDATES
# ============================================================

def build_candidates(
    df: pd.DataFrame,
    symbol: str,
) -> List[dict]:

    highs = []
    lows = []

    candidates = []

    # --------------------------------------------------------
    # confirm_index represents CURRENTLY AVAILABLE information.
    #
    # A pivot at pivot_index is confirmed only when:
    #
    #     confirm_index = pivot_index + PIVOT
    #
    # Therefore the pivot is never used before confirmation.
    # --------------------------------------------------------

    for confirm_index in range(
        PIVOT,
        len(df) - PIVOT,
    ):

        pivot_index = (
            confirm_index - PIVOT
        )

        if is_pivot_high(
            df,
            pivot_index,
        ):

            highs.append(
                (
                    pivot_index,
                    confirm_index,
                    float(
                        df["high"].iloc[
                            pivot_index
                        ]
                    ),
                )
            )

        if is_pivot_low(
            df,
            pivot_index,
        ):

            lows.append(
                (
                    pivot_index,
                    confirm_index,
                    float(
                        df["low"].iloc[
                            pivot_index
                        ]
                    ),
                )
            )

        # ====================================================
        # BEARISH SETUP
        #
        # Trend down
        # -> LH creates OB
        # -> IFC after OB
        # -> double valley
        # -> confirmed
        # ====================================================

        if (
            len(highs) >= 2
            and
            len(lows) >= 2
        ):

            previous_high = highs[-2]
            ob_high = highs[-1]

            # Lower High.
            lower_high = (
                ob_high[0]
                > previous_high[0]
                and
                ob_high[2]
                < previous_high[2]
            )

            if lower_high:

                prior_lows = [
                    item
                    for item in lows
                    if item[0] < ob_high[0]
                ]

                if len(prior_lows) >= 2:

                    prior_low_1 = prior_lows[-2]
                    prior_low_2 = prior_lows[-1]

                    downtrend = (
                        prior_low_2[2]
                        <
                        prior_low_1[2]
                    )

                    if downtrend:

                        ob_index = (
                            ob_high[0]
                        )

                        if (
                            ob_index
                            + IFC_LOOKAHEAD
                            < len(df)
                            and
                            bearish_ifc(
                                df,
                                ob_index,
                            )
                        ):

                            post_ob_lows = [
                                item
                                for item in lows
                                if item[0] > ob_index
                                and item[1] <= confirm_index
                            ]

                            if len(
                                post_ob_lows
                            ) >= 2:

                                valley_1 = (
                                    post_ob_lows[-2]
                                )

                                valley_2 = (
                                    post_ob_lows[-1]
                                )

                                structure_ok = (
                                    valley_2[0]
                                    > valley_1[0]
                                    and
                                    valley_1[2]
                                    <
                                    prior_low_2[2]
                                    and
                                    valley_2[2]
                                    <
                                    prior_low_2[2]
                                )

                                alignment_ok = (
                                    aligned(
                                        valley_1[2],
                                        valley_2[2],
                                    )
                                    and
                                    valley_2[2]
                                    >=
                                    valley_1[2]
                                )

                                if (
                                    structure_ok
                                    and
                                    alignment_ok
                                ):

                                    ready_index = max(
                                        valley_2[1],
                                        ob_index
                                        + IFC_LOOKAHEAD,
                                    )

                                    # Candidate is created exactly
                                    # when all required information
                                    # is confirmed.
                                    if (
                                        ready_index
                                        == confirm_index
                                    ):

                                        (
                                            ob_low,
                                            ob_high_price,
                                        ) = candle_ob_zone(
                                            df,
                                            ob_index,
                                            "SHORT",
                                        )

                                        candidates.append(
                                            {
                                                "symbol": symbol,
                                                "side": "SHORT",
                                                "ob_idx": ob_index,
                                                "ob_confirm": ob_high[1],
                                                "double_1": valley_1[0],
                                                "double_2": valley_2[0],
                                                "ready_index": ready_index,
                                                "ob_low": ob_low,
                                                "ob_high": ob_high_price,
                                                "structural_target": min(
                                                    valley_1[2],
                                                    valley_2[2],
                                                ),
                                                "pattern_confirm_time":
                                                    df.index[
                                                        ready_index
                                                    ],
                                            }
                                        )

        # ====================================================
        # BULLISH SETUP
        #
        # Trend up
        # -> HL / HH structural OB
        # -> IFC after OB
        # -> double top
        # -> confirmed
        # ====================================================

        if (
            len(highs) >= 2
            and
            len(lows) >= 2
        ):

            previous_low = lows[-2]
            ob_low = lows[-1]

            # Higher Low.
            higher_low = (
                ob_low[0]
                > previous_low[0]
                and
                ob_low[2]
                > previous_low[2]
            )

            if higher_low:

                prior_highs = [
                    item
                    for item in highs
                    if item[0] < ob_low[0]
                ]

                if len(prior_highs) >= 2:

                    prior_high_1 = prior_highs[-2]
                    prior_high_2 = prior_highs[-1]

                    uptrend = (
                        prior_high_2[2]
                        >
                        prior_high_1[2]
                    )

                    if uptrend:

                        ob_index = (
                            ob_low[0]
                        )

                        if (
                            ob_index
                            + IFC_LOOKAHEAD
                            < len(df)
                            and
                            bullish_ifc(
                                df,
                                ob_index,
                            )
                        ):

                            post_ob_highs = [
                                item
                                for item in highs
                                if item[0] > ob_index
                                and item[1] <= confirm_index
                            ]

                            if len(
                                post_ob_highs
                            ) >= 2:

                                top_1 = (
                                    post_ob_highs[-2]
                                )

                                top_2 = (
                                    post_ob_highs[-1]
                                )

                                structure_ok = (
                                    top_2[0]
                                    > top_1[0]
                                    and
                                    top_1[2]
                                    >
                                    prior_high_2[2]
                                    and
                                    top_2[2]
                                    >
                                    prior_high_2[2]
                                )

                                alignment_ok = (
                                    aligned(
                                        top_1[2],
                                        top_2[2],
                                    )
                                    and
                                    top_2[2]
                                    <=
                                    top_1[2]
                                )

                                if (
                                    structure_ok
                                    and
                                    alignment_ok
                                ):

                                    ready_index = max(
                                        top_2[1],
                                        ob_index
                                        + IFC_LOOKAHEAD,
                                    )

                                    if (
                                        ready_index
                                        == confirm_index
                                    ):

                                        (
                                            ob_low_price,
                                            ob_high_price,
                                        ) = candle_ob_zone(
                                            df,
                                            ob_index,
                                            "LONG",
                                        )

                                        candidates.append(
                                            {
                                                "symbol": symbol,
                                                "side": "LONG",
                                                "ob_idx": ob_index,
                                                "ob_confirm": ob_low[1],
                                                "double_1": top_1[0],
                                                "double_2": top_2[0],
                                                "ready_index": ready_index,
                                                "ob_low": ob_low_price,
                                                "ob_high": ob_high_price,
                                                "structural_target": max(
                                                    top_1[2],
                                                    top_2[2],
                                                ),
                                                "pattern_confirm_time":
                                                    df.index[
                                                        ready_index
                                                    ],
                                            }
                                        )

    # ========================================================
    # DEDUPLICATION
    # ========================================================

    seen = set()

    output = []

    for candidate in sorted(
        candidates,
        key=lambda item: (
            item["pattern_confirm_time"],
            item["symbol"],
            item["side"],
            item["ob_idx"],
            item["double_2"],
        ),
    ):

        key = (
            candidate["symbol"],
            candidate["side"],
            candidate["ob_idx"],
            candidate["double_1"],
            candidate["double_2"],
        )

        if key in seen:
            continue

        seen.add(key)

        output.append(
            candidate
        )

    return output


# ============================================================
# SLIPPAGE
# ============================================================

def slipped_entry(
    price: float,
    side: str,
) -> float:

    if side == "LONG":

        return price * (
            1.0 + SLIPPAGE
        )

    return price * (
        1.0 - SLIPPAGE
    )


def slipped_exit(
    price: float,
    side: str,
) -> float:

    if side == "LONG":

        return price * (
            1.0 - SLIPPAGE
        )

    return price * (
        1.0 + SLIPPAGE
    )


# ============================================================
# SIMULATE ONE CANDIDATE
# ============================================================

def simulate_candidate(
    df: pd.DataFrame,
    candidate: dict,
) -> Optional[dict]:

    side = candidate["side"]

    # --------------------------------------------------------
    # Entry cannot occur on the confirmation candle.
    #
    # Earliest possible entry:
    #
    #     ready_index + 1
    # --------------------------------------------------------

    start_index = (
        candidate["ready_index"]
        + 1
    )

    if start_index >= len(df):

        return None

    # --------------------------------------------------------
    # Order Block zone and stop.
    # --------------------------------------------------------

    if side == "LONG":

        raw_limit = (
            candidate["ob_high"]
        )

        stop = (
            candidate["ob_low"]
            * (
                1.0
                - SL_BUFFER_PCT
            )
        )

    else:

        raw_limit = (
            candidate["ob_low"]
        )

        stop = (
            candidate["ob_high"]
            * (
                1.0
                + SL_BUFFER_PCT
            )
        )

    # --------------------------------------------------------
    # Search for return to OB.
    #
    # NO artificial timeout.
    #
    # The setup remains valid until:
    #   - OB is touched -> entry
    #   - OB is structurally invalidated -> reject
    #   - dataset ends
    # --------------------------------------------------------

    entry_index = None

    for index in range(
        start_index,
        len(df),
    ):

        row = df.iloc[index]

        low = float(
            row["low"]
        )

        high = float(
            row["high"]
        )

        touched = (
            low
            <= raw_limit
            <= high
        )

        if touched:

            entry_index = index

            break

        # ----------------------------------------------------
        # Invalidation before entry.
        # ----------------------------------------------------

        if side == "LONG":

            if low < stop:

                return None

        else:

            if high > stop:

                return None

    if entry_index is None:

        return None

    # --------------------------------------------------------
    # Entry
    # --------------------------------------------------------

    entry = slipped_entry(
        float(raw_limit),
        side,
    )

    if side == "LONG":

        risk_price = (
            entry
            - stop
        )

    else:

        risk_price = (
            stop
            - entry
        )

    if risk_price <= 0:

        return None

    # --------------------------------------------------------
    # FIXED RR 1:2
    # --------------------------------------------------------

    if side == "LONG":

        target = (
            entry
            + RR * risk_price
        )

    else:

        target = (
            entry
            - RR * risk_price
        )

    structural_target = float(
        candidate[
            "structural_target"
        ]
    )

    # --------------------------------------------------------
    # Structural target viability.
    #
    # We DO NOT replace the TP with the structural target.
    #
    # User requires fixed 1:2.
    #
    # LONG:
    #   2R must be at or before structural target.
    #
    # SHORT:
    #   2R must be at or before structural target.
    # --------------------------------------------------------

    if (
        side == "LONG"
        and
        target > structural_target
    ):

        return None

    if (
        side == "SHORT"
        and
        target < structural_target
    ):

        return None

    # --------------------------------------------------------
    # Position sizing.
    # --------------------------------------------------------

    quantity = (
        NOTIONAL
        / entry
    )

    entry_fee = (
        NOTIONAL
        * FEE_RATE
    )

    # --------------------------------------------------------
    # Exit.
    #
    # Exit checking begins on the candle AFTER entry.
    #
    # If SL and TP both occur on the same later candle,
    # SL wins conservatively.
    #
    # No timeout.
    # --------------------------------------------------------

    exit_index = None
    outcome = None
    raw_exit = None

    for index in range(
        entry_index + 1,
        len(df),
    ):

        row = df.iloc[index]

        low = float(
            row["low"]
        )

        high = float(
            row["high"]
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

        # Conservative same-candle resolution.
        if (
            hit_sl
            and
            hit_tp
        ):

            exit_index = index
            outcome = "LOSS"
            raw_exit = stop

            break

        if hit_sl:

            exit_index = index
            outcome = "LOSS"
            raw_exit = stop

            break

        if hit_tp:

            exit_index = index
            outcome = "WIN"
            raw_exit = target

            break

    # --------------------------------------------------------
    # Unresolved position.
    # --------------------------------------------------------

    if exit_index is None:

        return {
            **candidate,

            "entry_index":
                entry_index,

            "entry_time":
                df.index[entry_index],

            "entry":
                entry,

            "stop":
                stop,

            "target":
                target,

            "risk_price":
                risk_price,

            "structural_target":
                structural_target,

            "exit_index":
                np.nan,

            "exit_time":
                pd.NaT,

            "exit":
                np.nan,

            "outcome":
                "UNRESOLVED",

            "gross_pnl":
                np.nan,

            "fees":
                np.nan,

            "pnl":
                np.nan,

            "r_multiple":
                np.nan,
        }

    # --------------------------------------------------------
    # Exit execution.
    # --------------------------------------------------------

    exit_price = slipped_exit(
        float(raw_exit),
        side,
    )

    if side == "LONG":

        gross_pnl = (
            exit_price
            - entry
        ) * quantity

    else:

        gross_pnl = (
            entry
            - exit_price
        ) * quantity

    exit_fee = (
        abs(
            exit_price
            * quantity
        )
        * FEE_RATE
    )

    fees = (
        entry_fee
        + exit_fee
    )

    pnl = (
        gross_pnl
        - fees
    )

    risk_cash = (
        risk_price
        * quantity
    )

    r_multiple = (
        pnl
        / risk_cash
    )

    return {
        **candidate,

        "entry_index":
            entry_index,

        "entry_time":
            df.index[entry_index],

        "entry":
            entry,

        "stop":
            stop,

        "target":
            target,

        "risk_price":
            risk_price,

        "structural_target":
            structural_target,

        "exit_index":
            exit_index,

        "exit_time":
            df.index[exit_index],

        "exit":
            exit_price,

        "outcome":
            outcome,

        "gross_pnl":
            gross_pnl,

        "fees":
            fees,

        "pnl":
            pnl,

        "r_multiple":
            r_multiple,
    }


# ============================================================
# SPLIT
# ============================================================

def split_name(
    timestamp: pd.Timestamp,
) -> str:

    # Warmup is explicitly excluded from statistics.
    if timestamp < RESEARCH_START:

        return "WARMUP"

    if timestamp < DISCOVERY_END:

        return "Discovery"

    if timestamp < DEVELOPMENT_END:

        return "Development"

    if timestamp < OOS_START:

        return "Research_Holdout"

    return "Validation_OOS"


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================

def portfolio_simulation(
    candidates: List[dict],
    data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:

    simulated = []

    for candidate in candidates:

        trade = simulate_candidate(
            data[
                candidate["symbol"]
            ],
            candidate,
        )

        if trade is None:

            continue

        # ----------------------------------------------------
        # Never allow warmup trades into the research sample.
        # They may use warmup candles for structure, but their
        # actual entry must occur inside the research period.
        # ----------------------------------------------------

        if (
            trade["entry_time"]
            < RESEARCH_START
        ):

            continue

        simulated.append(
            trade
        )

    if not simulated:

        return pd.DataFrame()

    raw = pd.DataFrame(
        simulated
    )

    raw = (
        raw
        .sort_values(
            [
                "entry_time",
                "symbol",
                "side",
            ]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # USER'S CURRENT OVERLAP RULE
    #
    # One simultaneous position per symbol.
    #
    # Different symbols may overlap.
    #
    # If previous trade exits on candle J,
    # new trade may enter on J+1.
    # ========================================================

    last_exit = {}

    accepted = []

    for _, trade in raw.iterrows():

        symbol = trade["symbol"]

        if symbol in last_exit:

            previous_exit = (
                last_exit[symbol]
            )

            if (
                trade["entry_time"]
                <= previous_exit
            ):

                continue

        accepted.append(
            trade.to_dict()
        )

        if pd.notna(
            trade["exit_time"]
        ):

            last_exit[symbol] = (
                trade["exit_time"]
            )

        else:

            # An unresolved position remains open
            # through the dataset.
            last_exit[symbol] = (
                pd.Timestamp.max
                .tz_localize("UTC")
            )

    return pd.DataFrame(
        accepted
    )


# ============================================================
# METRICS
# ============================================================

def metrics(
    trades: pd.DataFrame,
) -> dict:

    closed = trades[
        trades["outcome"].isin(
            [
                "WIN",
                "LOSS",
            ]
        )
    ].copy()

    if closed.empty:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "wr": 0.0,
            "pf": 0.0,
            "net_r": 0.0,
            "pnl": 0.0,
            "max_streak": 0,
        }

    wins = int(
        (
            closed["outcome"]
            == "WIN"
        ).sum()
    )

    losses = int(
        (
            closed["outcome"]
            == "LOSS"
        ).sum()
    )

    gross_profit = float(
        closed.loc[
            closed["pnl"] > 0,
            "pnl",
        ].sum()
    )

    gross_loss = float(
        -closed.loc[
            closed["pnl"] < 0,
            "pnl",
        ].sum()
    )

    if gross_loss > 0:

        pf = (
            gross_profit
            / gross_loss
        )

    elif gross_profit > 0:

        pf = math.inf

    else:

        pf = 0.0

    streak = 0
    max_streak = 0

    ordered = (
        closed
        .sort_values(
            "entry_time"
        )
    )

    for outcome in (
        ordered[
            "outcome"
        ].tolist()
    ):

        if outcome == "LOSS":

            streak += 1

            max_streak = max(
                max_streak,
                streak,
            )

        else:

            streak = 0

    return {
        "trades":
            len(closed),

        "wins":
            wins,

        "losses":
            losses,

        "wr":
            (
                100.0
                * wins
                / len(closed)
            ),

        "pf":
            pf,

        "net_r":
            float(
                closed[
                    "r_multiple"
                ].sum()
            ),

        "pnl":
            float(
                closed[
                    "pnl"
                ].sum()
            ),

        "max_streak":
            max_streak,
    }


# ============================================================
# EQUITY / DRAWDOWN
# ============================================================

def equity_and_dd(
    trades: pd.DataFrame,
) -> Tuple[
    float,
    float,
    float,
]:

    equity = (
        INITIAL_CAPITAL
    )

    peak = equity
    max_dd = 0.0

    closed = trades[
        trades["outcome"].isin(
            [
                "WIN",
                "LOSS",
            ]
        )
        &
        trades["exit_time"].notna()
    ].copy()

    closed = (
        closed
        .sort_values(
            "exit_time"
        )
    )

    for _, trade in (
        closed.iterrows()
    ):

        equity += float(
            trade["pnl"]
        )

        peak = max(
            peak,
            equity,
        )

        max_dd = max(
            max_dd,
            peak - equity,
        )

    if peak > 0:

        dd_pct = (
            max_dd
            / peak
            * 100.0
        )

    else:

        dd_pct = math.inf

    return (
        equity,
        max_dd,
        dd_pct,
    )


# ============================================================
# FINAL INTEGRITY AUDIT
# ============================================================

def integrity_audit(
    trades: pd.DataFrame,
) -> List[Tuple[str, str]]:

    causal_ok = True
    target_ok = True
    same_symbol_ok = True
    same_candle_ok = True
    rr_ok = True

    # ========================================================
    # TRADE-LEVEL CHECKS
    # ========================================================

    for _, trade in (
        trades.iterrows()
    ):

        # ----------------------------------------------------
        # Entry cannot occur on/before confirmation.
        # ----------------------------------------------------

        if (
            trade["entry_index"]
            <=
            trade["ready_index"]
        ):

            causal_ok = False

        # ----------------------------------------------------
        # Explicit timestamp check.
        # ----------------------------------------------------

        if (
            trade["entry_time"]
            <=
            trade["pattern_confirm_time"]
        ):

            causal_ok = False

        # ----------------------------------------------------
        # Exit cannot occur on same candle as entry.
        # ----------------------------------------------------

        if pd.notna(
            trade["exit_index"]
        ):

            if (
                trade["exit_index"]
                <=
                trade["entry_index"]
            ):

                same_candle_ok = False

        # ----------------------------------------------------
        # RR geometric check.
        # ----------------------------------------------------

        if trade["side"] == "LONG":

            risk = (
                trade["entry"]
                -
                trade["stop"]
            )

            if risk <= 0:

                rr_ok = False

            else:

                actual_rr = (
                    trade["target"]
                    -
                    trade["entry"]
                ) / risk

                if abs(
                    actual_rr
                    - RR
                ) > 1e-9:

                    rr_ok = False

            # TP must be reachable before structural target.
            if (
                trade["target"]
                >
                trade["structural_target"]
                + 1e-12
            ):

                target_ok = False

        else:

            risk = (
                trade["stop"]
                -
                trade["entry"]
            )

            if risk <= 0:

                rr_ok = False

            else:

                actual_rr = (
                    trade["entry"]
                    -
                    trade["target"]
                ) / risk

                if abs(
                    actual_rr
                    - RR
                ) > 1e-9:

                    rr_ok = False

            if (
                trade["target"]
                <
                trade["structural_target"]
                - 1e-12
            ):

                target_ok = False

        # ----------------------------------------------------
        # Structural confirmation consistency.
        #
        # Double-2 pivot itself must have been confirmed by
        # ready_index.
        # ----------------------------------------------------

        if (
            trade["double_2"]
            + PIVOT
            >
            trade["ready_index"]
        ):

            causal_ok = False

        # ----------------------------------------------------
        # OB confirmation must precede final readiness.
        # ----------------------------------------------------

        if (
            trade["ob_confirm"]
            >
            trade["ready_index"]
        ):

            causal_ok = False

    # ========================================================
    # SAME-SYMBOL OVERLAP
    # ========================================================

    ordered = (
        trades
        .sort_values(
            "entry_time"
        )
    )

    for _, group in (
        ordered.groupby("symbol")
    ):

        previous_exit = None

        for _, trade in (
            group.iterrows()
        ):

            if (
                previous_exit
                is not None
                and
                trade["entry_time"]
                <= previous_exit
            ):

                same_symbol_ok = False

            if pd.notna(
                trade["exit_time"]
            ):

                previous_exit = (
                    trade["exit_time"]
                )

    return [
        (
            "Data gaps",
            "PASSED",
        ),
        (
            "Structural causality",
            "PASSED"
            if causal_ok
            else "FAILED",
        ),
        (
            "Future target leak",
            "PASSED"
            if target_ok
            else "FAILED",
        ),
        (
            "Same-symbol overlap",
            "PASSED"
            if same_symbol_ok
            else "FAILED",
        ),
        (
            "Same-candle re-entry/exit",
            "PASSED"
            if same_candle_ok
            else "FAILED",
        ),
        (
            "RR 1:2 consistency",
            "PASSED"
            if rr_ok
            else "FAILED",
        ),
    ]


# ============================================================
# REPORT FORMAT
# ============================================================

def format_metrics(
    name: str,
    result: dict,
) -> str:

    if math.isinf(
        result["pf"]
    ):

        pf_text = "inf"

    else:

        pf_text = (
            f'{result["pf"]:.3f}'
        )

    return (
        f"{name:<18} "
        f'{result["trades"]:>7} '
        f'{result["wins"]:>7} '
        f'{result["losses"]:>7} '
        f'{result["wr"]:>8.2f} '
        f"{pf_text:>8} "
        f'{result["net_r"]:>12.3f} '
        f'{result["pnl"]:>12.2f} '
        f'{result["max_streak"]:>10}'
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {}

    # ========================================================
    # DOWNLOAD
    # ========================================================

    print(
        "Downloading strict Binance Futures 1H data..."
    )

    for symbol in SYMBOLS:

        print(
            f"Loading {symbol}...",
            flush=True,
        )

        data[symbol] = (
            fetch_symbol(
                symbol,
                WARMUP_START,
                OOS_END,
            )
        )

        print(
            f"{symbol} OK "
            f"{len(data[symbol])}"
        )

    # ========================================================
    # BUILD CANDIDATES
    # ========================================================

    all_candidates = []

    for symbol, df in data.items():

        candidates = (
            build_candidates(
                df,
                symbol,
            )
        )

        usable = [
            candidate
            for candidate in candidates
            if (
                candidate[
                    "pattern_confirm_time"
                ]
                <= OOS_END
            )
        ]

        all_candidates.extend(
            usable
        )

        print(
            f"{symbol}: "
            f"{len(candidates)} "
            f"structural candidates"
        )

    if not all_candidates:

        raise RuntimeError(
            "No structural candidates produced."
        )

    print(
        f"TOTAL STRUCTURAL CANDIDATES: "
        f"{len(all_candidates)}"
    )

    # ========================================================
    # PORTFOLIO SIMULATION
    # ========================================================

    print(
        "Running portfolio simulation..."
    )

    trades = (
        portfolio_simulation(
            all_candidates,
            data,
        )
    )

    if trades.empty:

        raise RuntimeError(
            "No trades produced."
        )

    trades["split"] = (
        trades["entry_time"]
        .map(split_name)
    )

    # ========================================================
    # AUDIT
    # ========================================================

    checks = integrity_audit(
        trades
    )

    audit_passed = all(
        status == "PASSED"
        for _, status in checks
    )

    # ========================================================
    # REPORT
    # ========================================================

    lines = []

    lines.append(
        "=" * 110
    )

    lines.append(
        "SETUP 3 V2 — BACKTEST REPORT"
    )

    lines.append(
        "=" * 110
    )

    lines.append(
        f"{'split':<18} "
        f"{'trades':>7} "
        f"{'wins':>7} "
        f"{'losses':>7} "
        f"{'WR_%':>8} "
        f"{'PF':>8} "
        f"{'net_R':>12} "
        f"{'PnL_$':>12} "
        f"{'max_streak':>10}"
    )

    lines.append(
        "-" * 110
    )

    for split in [
        "Discovery",
        "Development",
        "Research_Holdout",
        "Validation_OOS",
    ]:

        split_trades = trades[
            trades["split"]
            == split
        ]

        lines.append(
            format_metrics(
                split,
                metrics(
                    split_trades
                ),
            )
        )

    lines.append(
        format_metrics(
            "TOTAL",
            metrics(trades),
        )
    )

    # ========================================================
    # EQUITY
    # ========================================================

    (
        final_equity,
        max_dd,
        dd_pct,
    ) = equity_and_dd(
        trades
    )

    unresolved = int(
        (
            trades["outcome"]
            == "UNRESOLVED"
        ).sum()
    )

    lines.extend(
        [
            "",
            f"Initial Capital : "
            f"${INITIAL_CAPITAL:,.2f}",

            f"Final Equity    : "
            f"${final_equity:,.2f}",

            f"Fixed Margin    : "
            f"${MARGIN:,.2f}",

            f"Leverage        : "
            f"{LEVERAGE:.0f}x",

            f"Notional        : "
            f"${NOTIONAL:,.2f}",

            f"Max DD          : "
            f"${max_dd:,.2f} "
            f"({dd_pct:.2f}%)",

            f"Unresolved      : "
            f"{unresolved}",

            "",
            "=" * 110,
            "INTEGRITY AUDIT",
            "=" * 110,
        ]
    )

    for name, status in checks:

        lines.append(
            f"{name:<32}: "
            f"{status}"
        )

    lines.append(
        f"{'AUDIT STATUS':<32}: "
        f"{'PASSED' if audit_passed else 'FAILED'}"
    )

    # ========================================================
    # RESEARCH PROTOCOL
    # ========================================================

    lines.extend(
        [
            "",
            "RESEARCH PROTOCOL",
            "=" * 110,

            "Source basis: "
            "10 ستاپ برتر.pdf pages 18-21.",

            "V2 is a frozen mechanical translation.",

            "Numeric pivot / IFC / alignment / "
            "OB formulas are not claimed to be "
            "literal PDF formulas.",

            "Pivot confirmation is causal: "
            "pivot i is usable only at i + PIVOT.",

            "No artificial pattern-formation "
            "timeout is used.",

            "Entry starts only after full "
            "structural confirmation.",

            "Fixed RR=1:2 is calculated from "
            "actual entry and stop.",

            "Double valley/top is used only as "
            "structural target viability.",

            "No BE, trailing, or artificial "
            "exit timeout.",

            "One simultaneous trade per symbol.",

            "Different symbols may overlap.",

            "Same-symbol re-entry requires "
            "previous exit candle + 1.",

            "Warmup trades are excluded from "
            "research statistics.",

            f"Fresh OOS: "
            f"{OOS_START.isoformat()} "
            f"-> "
            f"{OOS_END.isoformat()}",

            "OOS is not used for parameter selection.",

            "Any V2 modification after OOS inspection "
            "requires another fresh OOS.",

            "",
        ]
    )

    report = "\n".join(
        lines
    )

    print(
        report
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    # ========================================================
    # TRADE LEDGER
    # ========================================================

    ledger_columns = [
        "split",
        "symbol",
        "side",
        "ob_idx",
        "ob_confirm",
        "double_1",
        "double_2",
        "ready_index",
        "pattern_confirm_time",
        "entry_index",
        "entry_time",
        "entry",
        "stop",
        "target",
        "structural_target",
        "exit_index",
        "exit_time",
        "exit",
        "outcome",
        "gross_pnl",
        "fees",
        "pnl",
        "r_multiple",
    ]

    for column in ledger_columns:

        if column not in trades.columns:

            trades[column] = np.nan

    trades[
        ledger_columns
    ].to_csv(
        LEDGER_PATH,
        index=False,
    )

    print(
        f"Saved: "
        f"{LEDGER_PATH}"
    )

    print(
        f"Saved: "
        f"{REPORT_PATH}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()                                   (
                                            ob_lo,
                                            ob_hi,
                                        ) = candle_ob_zone(
                                            df,
                                            ob_idx,
                                            "LONG",
                                        )

                                        candidates.append(
                                            {
                                                "symbol": symbol,
                                                "side": "LONG",
                                                "ob_idx": ob_idx,
                                                "ob_confirm": l_ob[1],
                                                "double_1": t1[0],
                                                "double_2": t2[0],
                                                "double_confirm": t2[1],
                                                "ready_index": ready,
                                                "ob_low": ob_lo,
                                                "ob_high": ob_hi,
                                                "structural_target": max(
                                                    t1[2],
                                                    t2[2],
                                                ),
                                                "prior_peak": ph2[2],
                                                "pattern_confirm_time":
                                                    df.index[ready],
                                            }
                                        )

    # ========================================================
    # DEDUPLICATION
    # ========================================================

    seen = set()
    output = []

    for c in sorted(
        candidates,
        key=lambda x: (
            x["ready_index"],
            x["symbol"],
            x["side"],
        ),
    ):

        key = (
            c["symbol"],
            c["side"],
            c["ob_idx"],
            c["double_1"],
            c["double_2"],
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(c)

    return output


# ============================================================
# EXECUTION
# ============================================================

def slipped_entry(
    price: float,
    side: str,
) -> float:

    if side == "LONG":
        return price * (
            1.0 + SLIPPAGE
        )

    return price * (
        1.0 - SLIPPAGE
    )


def slipped_exit(
    price: float,
    side: str,
) -> float:

    if side == "LONG":
        return price * (
            1.0 - SLIPPAGE
        )

    return price * (
        1.0 + SLIPPAGE
    )


# ============================================================
# CANDIDATE SIMULATION
# ============================================================

def simulate_candidate(
    df: pd.DataFrame,
    c: dict,
) -> Optional[dict]:

    side = c["side"]

    # CRITICAL:
    # never enter on the confirmation candle itself.
    start = (
        c["ready_index"] + 1
    )

    if start >= len(df):
        return None

    # ========================================================
    # LIMIT ENTRY AT OB
    # ========================================================

    if side == "LONG":
        raw_limit = c["ob_high"]
        stop = (
            c["ob_low"]
            * (1.0 - SL_BUFFER_PCT)
        )
    else:
        raw_limit = c["ob_low"]
        stop = (
            c["ob_high"]
            * (1.0 + SL_BUFFER_PCT)
        )

    entry_i = None
    raw_entry = None

    # No timeout.
    # Search until the OB is touched or invalidated.
    for i in range(
        start,
        len(df),
    ):

        row = df.iloc[i]

        touched = (
            float(row.low)
            <= raw_limit
            <= float(row.high)
        )

        if touched:

            entry_i = i
            raw_entry = raw_limit
            break

        # OB invalidation.
        if side == "LONG":

            if float(row.low) < stop:
                return None

        else:

            if float(row.high) > stop:
                return None

    if (
        entry_i is None
        or raw_entry is None
    ):
        return None

    entry = slipped_entry(
        float(raw_entry),
        side,
    )

    if side == "LONG":
        risk = entry - stop
    else:
        risk = stop - entry

    if risk <= 0:
        return None

    # ========================================================
    # FIXED RR 1:2
    # ========================================================

    if side == "LONG":
        target = (
            entry
            + RR * risk
        )
    else:
        target = (
            entry
            - RR * risk
        )

    structural = float(
        c["structural_target"]
    )

    # ========================================================
    # STRUCTURAL TARGET VIABILITY
    #
    # The PDF uses the double valley/top as the target.
    # User requires fixed RR 1:2.
    #
    # Therefore:
    #
    # LONG:
    # 2R must not be beyond the double-top liquidity level.
    #
    # SHORT:
    # 2R must not be beyond the double-bottom liquidity level.
    #
    # We do NOT replace 2R with the structural target.
    # ========================================================

    if (
        side == "LONG"
        and target > structural
    ):
        return None

    if (
        side == "SHORT"
        and target < structural
    ):
        return None

    qty = (
        NOTIONAL / entry
    )

    entry_fee = (
        NOTIONAL
        * FEE_RATE
    )

    # ========================================================
    # EXIT
    #
    # Start checking exits on the candle AFTER entry.
    # Same-candle SL/TP ambiguity is impossible here.
    # If both are hit on a later candle, SL wins conservatively.
    # ========================================================

    exit_i = None
    outcome = None
    raw_exit = None

    for i in range(
        entry_i + 1,
        len(df),
    ):

        row = df.iloc[i]

        if side == "LONG":

            hit_sl = (
                float(row.low)
                <= stop
            )

            hit_tp = (
                float(row.high)
                >= target
            )

        else:

            hit_sl = (
                float(row.high)
                >= stop
            )

            hit_tp = (
                float(row.low)
                <= target
            )

        if (
            hit_sl
            and hit_tp
        ):

            exit_i = i
            outcome = "LOSS"
            raw_exit = stop
            break

        if hit_sl:

            exit_i = i
            outcome = "LOSS"
            raw_exit = stop
            break

        if hit_tp:

            exit_i = i
            outcome = "WIN"
            raw_exit = target
            break

    # ========================================================
    # UNRESOLVED
    # ========================================================

    if exit_i is None:

        return {
            **c,
            "entry_index": entry_i,
            "entry_time": df.index[entry_i],
            "entry": entry,
            "stop": stop,
            "target": target,
            "risk_price": risk,
            "structural_target": structural,
            "exit_index": np.nan,
            "exit_time": pd.NaT,
            "exit": np.nan,
            "outcome": "UNRESOLVED",
            "gross_pnl": np.nan,
            "fees": np.nan,
            "pnl": np.nan,
            "r_multiple": np.nan,
        }

    # ========================================================
    # PNL
    # ========================================================

    exit_price = slipped_exit(
        float(raw_exit),
        side,
    )

    if side == "LONG":

        gross = (
            exit_price - entry
        ) * qty

    else:

        gross = (
            entry - exit_price
        ) * qty

    exit_fee = (
        abs(exit_price * qty)
        * FEE_RATE
    )

    fees = (
        entry_fee
        + exit_fee
    )

    pnl = (
        gross
        - fees
    )

    risk_cash = (
        risk * qty
    )

    return {
        **c,
        "entry_index": entry_i,
        "entry_time": df.index[entry_i],
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk_price": risk,
        "structural_target": structural,
        "exit_index": exit_i,
        "exit_time": df.index[exit_i],
        "exit": exit_price,
        "outcome": outcome,
        "gross_pnl": gross,
        "fees": fees,
        "pnl": pnl,
        "r_multiple": (
            pnl / risk_cash
            if risk_cash > 0
            else np.nan
        ),
    }


# ============================================================
# SPLITS
# ============================================================

def split_name(
    ts: pd.Timestamp,
) -> str:

    if ts < DISCOVERY_END:
        return "Discovery"

    if ts < DEVELOPMENT_END:
        return "Development"

    if ts < OOS_START:
        return "Research_Holdout"

    return "Validation_OOS"


# ============================================================
# METRICS
# ============================================================

def metrics(
    trades: pd.DataFrame,
) -> dict:

    if trades.empty:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "wr": 0.0,
            "pf": 0.0,
            "net_r": 0.0,
            "pnl": 0.0,
            "max_streak": 0,
        }

    x = trades[
        trades.outcome.isin(
            ["WIN", "LOSS"]
        )
    ].copy()

    wins = int(
        (x.outcome == "WIN").sum()
    )

    losses = int(
        (x.outcome == "LOSS").sum()
    )

    gross_win = float(
        x.loc[
            x.pnl > 0,
            "pnl",
        ].sum()
    )

    gross_loss = float(
        -x.loc[
            x.pnl < 0,
            "pnl",
        ].sum()
    )

    if gross_loss > 0:

        pf = (
            gross_win
            / gross_loss
        )

    elif gross_win > 0:

        pf = math.inf

    else:

        pf = 0.0

    streak = 0
    best = 0

    for outcome in (
        x.sort_values(
            "entry_time"
        ).outcome.tolist()
    ):

        if outcome == "LOSS":

            streak += 1
            best = max(
                best,
                streak,
            )

        else:

            streak = 0

    return {
        "trades": len(x),
        "wins": wins,
        "losses": losses,
        "wr": (
            100.0 * wins / len(x)
            if len(x)
            else 0.0
        ),
        "pf": pf,
        "net_r": float(
            x.r_multiple.sum()
        ),
        "pnl": float(
            x.pnl.sum()
        ),
        "max_streak": best,
    }


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================

def portfolio_simulation(
    all_candidates: List[dict],
    data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:

    simulated = []

    for c in all_candidates:

        trade = simulate_candidate(
            data[c["symbol"]],
            c,
        )

        if trade is not None:
            simulated.append(trade)

    if not simulated:
        return pd.DataFrame()

    raw = pd.DataFrame(
        simulated
    )

    raw = (
        raw
        .sort_values(
            [
                "entry_time",
                "symbol",
                "side",
            ]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # CURRENT USER OVERLAP RULE
    #
    # One simultaneous trade per symbol.
    # Different symbols may overlap.
    #
    # Re-entry is allowed only AFTER previous trade closes.
    # If previous exit is candle J,
    # next entry must be candle J+1 or later.
    # ========================================================

    last_exit: Dict[
        str,
        pd.Timestamp,
    ] = {}

    accepted = []

    for _, t in raw.iterrows():

        symbol = t.symbol

        if symbol in last_exit:

            if (
                pd.notna(t.exit_time)
                and
                t.entry_time
                <= last_exit[symbol]
            ):
                continue

            if pd.isna(t.exit_time):
                continue

        accepted.append(
            t.to_dict()
        )

        if pd.notna(t.exit_time):

            last_exit[symbol] = (
                t.exit_time
            )

        else:

            last_exit[symbol] = (
                pd.Timestamp.max
                .tz_localize("UTC")
            )

    return pd.DataFrame(
        accepted
    )


# ============================================================
# EQUITY / DRAW DOWN
# ============================================================

def equity_and_dd(
    trades: pd.DataFrame,
) -> Tuple[
    float,
    float,
    float,
]:

    equity = INITIAL_CAPITAL
    peak = equity
    max_dd = 0.0

    closed = trades[
        trades.outcome.isin(
            ["WIN", "LOSS"]
        )
        & trades.exit_time.notna()
    ].copy()

    for _, t in (
        closed
        .sort_values("exit_time")
        .iterrows()
    ):

        equity += float(
            t.pnl
        )

        peak = max(
            peak,
            equity,
        )

        max_dd = max(
            max_dd,
            peak - equity,
        )

    if peak > 0:

        dd_pct = (
            max_dd
            / peak
            * 100.0
        )

    else:

        dd_pct = math.inf

    return (
        equity,
        max_dd,
        dd_pct,
    )


# ============================================================
# FINAL INTEGRITY AUDIT
# ============================================================

def audit(
    trades: pd.DataFrame,
    data: Dict[str, pd.DataFrame],
) -> List[Tuple[str, str]]:

    checks = []

    # Data gaps are fatal in fetch_symbol().
    checks.append(
        (
            "Data gaps",
            "PASSED",
        )
    )

    causal = True
    same_symbol = True
    same_candle = True
    rr_ok = True
    target_ok = True

    for _, t in trades.iterrows():

        # Entry MUST be after ready_index.
        if (
            t.entry_index
            <= t.ready_index
        ):
            causal = False

        # No same-candle exit.
        if (
            pd.notna(t.exit_index)
            and
            t.exit_index
            <= t.entry_index
        ):
            same_candle = False

        # Geometric RR must be exactly 1:2.
        if t.side == "LONG":

            risk = (
                t.entry
                - t.stop
            )

            if (
                risk <= 0
                or
                abs(
                    (
                        t.target
                        - t.entry
                    ) / risk
                    - RR
                ) > 1e-9
            ):
                rr_ok = False

            if (
                t.target
                >
                t.structural_target
                + 1e-12
            ):
                target_ok = False

        else:

            risk = (
                t.stop
                - t.entry
            )

            if (
                risk <= 0
                or
                abs(
                    (
                        t.entry
                        - t.target
                    ) / risk
                    - RR
                ) > 1e-9
            ):
                rr_ok = False

            if (
                t.target
                <
                t.structural_target
                - 1e-12
            ):
                target_ok = False

    # Same-symbol overlap.
    z = (
        trades
        .sort_values(
            "entry_time"
        )
    )

    for symbol, g in (
        z.groupby("symbol")
    ):

        previous_exit = None

        for _, t in g.iterrows():

            if (
                previous_exit
                is not None
                and
                t.entry_time
                <= previous_exit
            ):
                same_symbol = False

            if pd.notna(
                t.exit_time
            ):
                previous_exit = (
                    t.exit_time
                )

    checks.append(
        (
            "Structural causality",
            "PASSED"
            if causal
            else "FAILED",
        )
    )

    checks.append(
        (
            "Future target leak",
            "PASSED"
            if target_ok
            else "FAILED",
        )
    )

    checks.append(
        (
            "Same-symbol overlap",
            "PASSED"
            if same_symbol
            else "FAILED",
        )
    )

    checks.append(
        (
            "Same-candle re-entry/exit",
            "PASSED"
            if same_candle
            else "FAILED",
        )
    )

    checks.append(
        (
            "RR 1:2 consistency",
            "PASSED"
            if rr_ok
            else "FAILED",
        )
    )

    return checks


# ============================================================
# REPORT
# ============================================================

def fmt_metrics(
    name: str,
    m: dict,
) -> str:

    if math.isinf(
        m["pf"]
    ):
        pf = "inf"
    else:
        pf = f"{m['pf']:.3f}"

    return (
        f"{name:<18} "
        f"{m['trades']:>7} "
        f"{m['wins']:>7} "
        f"{m['losses']:>7} "
        f"{m['wr']:>8.2f} "
        f"{pf:>8} "
        f"{m['net_r']:>12.3f} "
        f"{m['pnl']:>12.2f} "
        f"{m['max_streak']:>10}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {}

    print(
        "Downloading strict Binance Futures 1h data..."
    )

    for symbol in SYMBOLS:

        print(
            symbol,
            end=" ",
            flush=True,
        )

        data[symbol] = fetch_symbol(
            symbol,
            WARMUP_START,
            OOS_END,
        )

        print(
            f"OK {len(data[symbol])}"
        )

    # ========================================================
    # CANDIDATES
    # ========================================================

    all_candidates = []

    for symbol, df in data.items():

        candidates = build_candidates(
            df,
            symbol,
        )

        for c in candidates:

            if (
                c["pattern_confirm_time"]
                <= OOS_END
            ):
                all_candidates.append(c)

        print(
            f"{symbol}: "
            f"{len(candidates)} "
            f"structural candidates"
        )

    if not all_candidates:
        raise RuntimeError(
            "No structural candidates produced."
        )

    # ========================================================
    # SIMULATE
    # ========================================================

    trades = portfolio_simulation(
        all_candidates,
        data,
    )

    if trades.empty:
        raise RuntimeError(
            "No trades produced."
        )

    trades["split"] = (
        trades.entry_time.map(
            split_name
        )
    )

    # ========================================================
    # AUDIT
    # ========================================================

    checks = audit(
        trades,
        data,
    )

    audit_status = all(
        status == "PASSED"
        for _, status in checks
    )

    # ========================================================
    # REPORT
    # ========================================================

    order = [
        "Discovery",
        "Development",
        "Research_Holdout",
        "Validation_OOS",
    ]

    lines = []

    lines.append(
        "=" * 110
    )

    lines.append(
        "SETUP 3 V2 — BACKTEST REPORT"
    )

    lines.append(
        "=" * 110
    )

    lines.append(
        f"{'split':<18} "
        f"{'trades':>7} "
        f"{'wins':>7} "
        f"{'losses':>7} "
        f"{'WR_%':>8} "
        f"{'PF':>8} "
        f"{'net_R':>12} "
        f"{'PnL_$':>12} "
        f"{'max_streak':>10}"
    )

    lines.append(
        "-" * 110
    )

    for split in order:

        lines.append(
            fmt_metrics(
                split,
                metrics(
                    trades[
                        trades.split
                        == split
                    ]
                ),
            )
        )

    lines.append(
        fmt_metrics(
            "TOTAL",
            metrics(trades),
        )
    )

    final_equity, max_dd, dd_pct = (
        equity_and_dd(trades)
    )

    unresolved = int(
        (
            trades.outcome
            == "UNRESOLVED"
        ).sum()
    )

    lines += [
        "",
        f"Initial Capital : ${INITIAL_CAPITAL:,.2f}",
        f"Final Equity    : ${final_equity:,.2f}",
        f"Fixed Margin    : ${MARGIN:,.2f}",
        f"Leverage        : {LEVERAGE:.0f}x",
        f"Notional        : ${NOTIONAL:,.2f}",
        f"Max DD          : ${max_dd:,.2f} ({dd_pct:.2f}%)",
        f"Unresolved      : {unresolved}",
        "",
        "=" * 110,
        "INTEGRITY AUDIT",
        "=" * 110,
    ]

    for key, status in checks:

        lines.append(
            f"{key:<32}: {status}"
        )

    lines.append(
        "AUDIT STATUS"
        f"{'':<18}: "
        f"{'PASSED' if audit_status else 'FAILED'}"
    )

    lines += [
        "",
        "RESEARCH PROTOCOL",
        "=" * 110,
        "Source basis: 10 ستاپ برتر.pdf pages 18-21.",
        "V2 is a frozen mechanical translation.",
        "Numeric pivot / IFC / alignment / OB formulas are not claimed to be literal PDF formulas.",
        "A pivot at i becomes usable only at i + PIVOT.",
        "No artificial pattern-formation timeout is used.",
        "Entry starts only after full structural confirmation.",
        "Fixed RR=1:2 is calculated from actual entry and stop.",
        "Double valley/top is used only as structural target viability.",
        "No BE, trailing, or artificial exit timeout.",
        "One simultaneous trade per symbol.",
        "Different symbols may overlap.",
        "Same-symbol re-entry requires previous exit candle + 1.",
        f"Fresh OOS: {OOS_START.isoformat()} -> {OOS_END.isoformat()}",
        "OOS must not be used for parameter selection.",
        "Any V2 modification after OOS inspection requires another fresh OOS.",
        "",
    ]

    report = "\n".join(
        lines
    )

    print(report)

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    # ========================================================
    # LEDGER
    # ========================================================

    cols = [
        "split",
        "symbol",
        "side",
        "ob_idx",
        "ob_confirm",
        "double_1",
        "double_2",
        "ready_index",
        "entry_index",
        "entry_time",
        "entry",
        "stop",
        "target",
        "structural_target",
        "exit_index",
        "exit_time",
        "exit",
        "outcome",
        "gross_pnl",
        "fees",
        "pnl",
        "r_multiple",
    ]

    for c in cols:

        if c not in trades.columns:
            trades[c] = np.nan

    trades[cols].to_csv(
        LEDGER_PATH,
        index=False,
    )

    print(
        f"Saved: {LEDGER_PATH}"
    )

    print(
        f"Saved: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
