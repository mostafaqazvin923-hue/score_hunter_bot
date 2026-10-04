import math
import time
import zipfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd


# ============================================================
# SETUP 3 V2
# STRUCTURAL / IFC / DOUBLE VALLEY-TOP
#
# Source basis:
# 10 ستاپ برتر.pdf — pages 18-21
#
# PDF sequence:
#   trend
#   -> LH / HH creates OB
#   -> IFC after OB
#   -> double valley / double top
#   -> return to OB
#   -> entry
#   -> SL behind OB
#   -> structural target
#
# IMPORTANT:
# Numeric definitions for pivot, IFC, alignment and exact OB
# boundaries are not explicitly specified in the PDF.
# They are frozen mechanical research translations.
#
# V2 fixes:
#   1. Pivot confirmation is causal.
#   2. No artificial pattern-formation timeout.
#   3. IFC must occur after OB.
#   4. Double structure must be confirmed before entry.
#   5. Entry cannot occur before pattern confirmation.
#   6. RR is exactly 1:2 geometrically.
#   7. Structural target is only a viability filter.
#   8. No same-candle re-entry.
#   9. One simultaneous position per symbol.
#  10. Different symbols may overlap.
#
# Fresh Setup-3-V2 OOS:
#   2024-10-04 -> 2025-10-03
#
# If this V2 is changed after inspecting OOS,
# a NEW OOS must be reserved.
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
BAR = pd.Timedelta(hours=1)


# ============================================================
# ACCOUNT / EXECUTION
# ============================================================

INITIAL_CAPITAL = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE

RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003


# ============================================================
# FROZEN STRUCTURAL TRANSLATIONS
# ============================================================

# A pivot at i is not usable until i + PIVOT.
PIVOT = 2

# PDF says valleys/tops only need to be aligned,
# not exactly equal.
ALIGN_TOL = 0.004

# Small buffer behind the OB for the stop.
SL_BUFFER_PCT = 0.0005

# IFC translation:
# the first two candles after OB create the 3-candle
# displacement/FVG structure.
IFC_LOOKAHEAD = 2


# ============================================================
# RESEARCH / FRESH OOS
# ============================================================

OOS_START = pd.Timestamp(
    "2024-10-04 00:00:00",
    tz="UTC",
)

OOS_END = pd.Timestamp(
    "2025-10-03 23:00:00",
    tz="UTC",
)

RESEARCH_END = OOS_START - BAR

RESEARCH_START = (
    RESEARCH_END
    - pd.Timedelta(days=365)
    + BAR
)

WARMUP_START = (
    RESEARCH_START
    - pd.Timedelta(days=90)
)

DISCOVERY_END = (
    RESEARCH_START
    + (RESEARCH_END - RESEARCH_START) * 0.60
)

DEVELOPMENT_END = (
    RESEARCH_START
    + (RESEARCH_END - RESEARCH_START) * 0.80
)


# ============================================================
# DATA
# ============================================================

DATA_BASE = (
    "https://data.binance.vision/"
    "data/futures/um/monthly/klines"
)

CACHE_DIR = Path("data_cache")
OUT_DIR = Path("setup3_v2_outputs")

LEDGER_PATH = (
    OUT_DIR / "setup3_v2_trade_ledger.csv"
)

REPORT_PATH = (
    OUT_DIR / "setup3_v2_report.txt"
)

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "Setup3-V2-Research/1.0"
    }
)


# ============================================================
# DATA DOWNLOAD
# ============================================================

def month_starts(
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    cur = (
        start
        .to_period("M")
        .to_timestamp()
        .tz_localize("UTC")
    )

    last = (
        end
        .to_period("M")
        .to_timestamp()
        .tz_localize("UTC")
    )

    while cur <= last:
        yield cur

        cur = (
            cur
            + pd.offsets.MonthBegin(1)
        ).normalize()

        if cur.tzinfo is None:
            cur = cur.tz_localize("UTC")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def download_month(
    symbol: str,
    month: pd.Timestamp,
) -> Path:

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    name = (
        f"{symbol}-1h-"
        f"{month.strftime('%Y-%m')}.zip"
    )

    path = CACHE_DIR / name

    if (
        path.exists()
        and path.stat().st_size > 1000
    ):
        return path

    url = (
        f"{DATA_BASE}/"
        f"{symbol}/1h/{name}"
    )

    for attempt in range(1, 4):

        try:
            r = SESSION.get(
                url,
                timeout=30,
            )

            if (
                r.status_code == 200
                and len(r.content) > 1000
            ):
                path.write_bytes(
                    r.content
                )
                return path

            if r.status_code == 404:
                raise FileNotFoundError(
                    f"Archive missing: {url}"
                )

            raise RuntimeError(
                f"HTTP {r.status_code}: {url}"
            )

        except Exception:

            if attempt == 3:
                raise

            time.sleep(
                1.5 * attempt
            )

    raise RuntimeError(
        "Download failed"
    )


def read_archive(
    path: Path,
) -> pd.DataFrame:

    with zipfile.ZipFile(path) as z:

        names = [
            n
            for n in z.namelist()
            if n.lower().endswith(".csv")
        ]

        if not names:
            raise RuntimeError(
                f"No CSV inside {path}"
            )

        with z.open(names[0]) as f:
            df = pd.read_csv(
                f,
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

    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[c] = pd.to_numeric(
            df[c],
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


def fetch_symbol(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:

    frames = []

    for month in month_starts(
        start,
        end,
    ):
        path = download_month(
            symbol,
            month,
        )

        frames.append(
            read_archive(path)
        )

    if not frames:
        raise RuntimeError(
            f"No archives for {symbol}"
        )

    df = pd.concat(
        frames,
        ignore_index=True,
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

    if df.empty:
        raise RuntimeError(
            f"No data for {symbol}"
        )

    # Remove incomplete current candle.
    now = pd.Timestamp.now(
        tz="UTC"
    )

    df = df[
        df.time + BAR <= now
    ].copy()

    if df.empty:
        raise RuntimeError(
            f"No completed candles for {symbol}"
        )

    expected = pd.date_range(
        df.time.iloc[0],
        df.time.iloc[-1],
        freq="1h",
        tz="UTC",
    )

    actual = pd.DatetimeIndex(
        df.time
    )

    missing = expected.difference(
        actual
    )

    if len(missing):
        raise RuntimeError(
            f"{symbol}: fatal 1h data gaps: "
            f"{missing[:5].tolist()}"
        )

    return df.set_index("time")


# ============================================================
# CAUSAL PIVOTS
# ============================================================

def is_pivot_high(
    df: pd.DataFrame,
    i: int,
) -> bool:

    if i < PIVOT:
        return False

    if i + PIVOT >= len(df):
        return False

    h = float(
        df.high.iloc[i]
    )

    left = df.high.iloc[
        i - PIVOT:i
    ]

    right = df.high.iloc[
        i + 1:i + PIVOT + 1
    ]

    return (
        h >= float(left.max())
        and h > float(right.max())
    )


def is_pivot_low(
    df: pd.DataFrame,
    i: int,
) -> bool:

    if i < PIVOT:
        return False

    if i + PIVOT >= len(df):
        return False

    lo = float(
        df.low.iloc[i]
    )

    left = df.low.iloc[
        i - PIVOT:i
    ]

    right = df.low.iloc[
        i + 1:i + PIVOT + 1
    ]

    return (
        lo <= float(left.min())
        and lo < float(right.min())
    )


def aligned(
    a: float,
    b: float,
) -> bool:

    return (
        abs(a - b)
        / max(abs(a), abs(b), 1e-12)
        <= ALIGN_TOL
    )


# ============================================================
# IFC
# ============================================================

def bearish_ifc(
    df: pd.DataFrame,
    ob_idx: int,
) -> bool:

    j = ob_idx + IFC_LOOKAHEAD

    if j >= len(df):
        return False

    # Mechanical research translation:
    # bearish FVG immediately after OB.
    return (
        float(df.low.iloc[ob_idx])
        >
        float(df.high.iloc[j])
    )


def bullish_ifc(
    df: pd.DataFrame,
    ob_idx: int,
) -> bool:

    j = ob_idx + IFC_LOOKAHEAD

    if j >= len(df):
        return False

    return (
        float(df.high.iloc[ob_idx])
        <
        float(df.low.iloc[j])
    )


# ============================================================
# ORDER BLOCK
# ============================================================

def candle_ob_zone(
    df: pd.DataFrame,
    idx: int,
    side: str,
) -> Tuple[float, float]:

    o = float(
        df.open.iloc[idx]
    )

    c = float(
        df.close.iloc[idx]
    )

    h = float(
        df.high.iloc[idx]
    )

    l = float(
        df.low.iloc[idx]
    )

    if side == "SHORT":

        # Bearish supply OB:
        # candle body is the executable zone.
        return (
            min(o, c),
            h,
        )

    # Bullish demand OB.
    return (
        l,
        max(o, c),
    )


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def build_candidates(
    df: pd.DataFrame,
    symbol: str,
) -> List[dict]:

    highs = []
    lows = []

    candidates = []

    for confirm_i in range(
        PIVOT,
        len(df) - PIVOT,
    ):

        pivot_i = (
            confirm_i - PIVOT
        )

        # IMPORTANT:
        # pivot_i is usable ONLY NOW.
        if is_pivot_high(
            df,
            pivot_i,
        ):
            highs.append(
                (
                    pivot_i,
                    confirm_i,
                    float(
                        df.high.iloc[
                            pivot_i
                        ]
                    ),
                )
            )

        if is_pivot_low(
            df,
            pivot_i,
        ):
            lows.append(
                (
                    pivot_i,
                    confirm_i,
                    float(
                        df.low.iloc[
                            pivot_i
                        ]
                    ),
                )
            )

        # ====================================================
        # BEARISH
        # Trend down
        # LH -> OB
        # IFC
        # double valley
        # ====================================================

        if (
            len(highs) >= 2
            and len(lows) >= 2
        ):

            h_prev = highs[-2]
            h_ob = highs[-1]

            # Lower high.
            if (
                h_ob[0] > h_prev[0]
                and h_ob[2] < h_prev[2]
            ):

                prior_lows = [
                    x
                    for x in lows
                    if x[0] < h_ob[0]
                ]

                if len(prior_lows) >= 2:

                    pl1 = prior_lows[-2]
                    pl2 = prior_lows[-1]

                    downtrend = (
                        pl2[2] < pl1[2]
                    )

                    if downtrend:

                        ob_idx = h_ob[0]

                        if (
                            ob_idx
                            + IFC_LOOKAHEAD
                            < len(df)
                            and bearish_ifc(
                                df,
                                ob_idx,
                            )
                        ):

                            post_lows = [
                                x
                                for x in lows
                                if x[0] > ob_idx
                            ]

                            eligible = [
                                x
                                for x in post_lows
                                if x[1] <= confirm_i
                            ]

                            if len(eligible) >= 2:

                                v1 = eligible[-2]
                                v2 = eligible[-1]

                                valid_structure = (
                                    v2[0] > v1[0]
                                    and v1[2] < pl2[2]
                                    and v2[2] < pl2[2]
                                )

                                # Source says the two valleys
                                # are aligned and second valley
                                # should not be below the first.
                                valid_alignment = (
                                    aligned(
                                        v1[2],
                                        v2[2],
                                    )
                                    and
                                    v2[2] >= v1[2]
                                )

                                if (
                                    valid_structure
                                    and valid_alignment
                                ):

                                    ready = max(
                                        v2[1],
                                        ob_idx
                                        + IFC_LOOKAHEAD,
                                    )

                                    # Candidate becomes known
                                    # exactly at confirmation.
                                    if ready == confirm_i:

                                        (
                                            ob_lo,
                                            ob_hi,
                                        ) = candle_ob_zone(
                                            df,
                                            ob_idx,
                                            "SHORT",
                                        )

                                        candidates.append(
                                            {
                                                "symbol": symbol,
                                                "side": "SHORT",
                                                "ob_idx": ob_idx,
                                                "ob_confirm": h_ob[1],
                                                "double_1": v1[0],
                                                "double_2": v2[0],
                                                "double_confirm": v2[1],
                                                "ready_index": ready,
                                                "ob_low": ob_lo,
                                                "ob_high": ob_hi,
                                                "structural_target": min(
                                                    v1[2],
                                                    v2[2],
                                                ),
                                                "prior_valley": pl2[2],
                                                "pattern_confirm_time":
                                                    df.index[ready],
                                            }
                                        )

        # ====================================================
        # BULLISH
        # Trend up
        # HH -> OB
        # IFC
        # double top
        # ====================================================

        if (
            len(highs) >= 2
            and len(lows) >= 2
        ):

            l_prev = lows[-2]
            l_ob = lows[-1]

            # Higher low.
            if (
                l_ob[0] > l_prev[0]
                and l_ob[2] > l_prev[2]
            ):

                prior_highs = [
                    x
                    for x in highs
                    if x[0] < l_ob[0]
                ]

                if len(prior_highs) >= 2:

                    ph1 = prior_highs[-2]
                    ph2 = prior_highs[-1]

                    uptrend = (
                        ph2[2] > ph1[2]
                    )

                    if uptrend:

                        ob_idx = l_ob[0]

                        if (
                            ob_idx
                            + IFC_LOOKAHEAD
                            < len(df)
                            and bullish_ifc(
                                df,
                                ob_idx,
                            )
                        ):

                            post_highs = [
                                x
                                for x in highs
                                if x[0] > ob_idx
                            ]

                            eligible = [
                                x
                                for x in post_highs
                                if x[1] <= confirm_i
                            ]

                            if len(eligible) >= 2:

                                t1 = eligible[-2]
                                t2 = eligible[-1]

                                valid_structure = (
                                    t2[0] > t1[0]
                                    and t1[2] > ph2[2]
                                    and t2[2] > ph2[2]
                                )

                                valid_alignment = (
                                    aligned(
                                        t1[2],
                                        t2[2],
                                    )
                                    and
                                    t2[2] <= t1[2]
                                )

                                if (
                                    valid_structure
                                    and valid_alignment
                                ):

                                    ready = max(
                                        t2[1],
                                        ob_idx
                                        + IFC_LOOKAHEAD,
                                    )

                                    if ready == confirm_i:

                                        (
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
