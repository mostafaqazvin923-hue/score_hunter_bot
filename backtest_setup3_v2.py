import math
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


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

INITIAL_CAPITAL = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE

RR = 2.0
FEE_RATE = 0.0007
SLIPPAGE = 0.0003

PIVOT = 2
ALIGN_TOL = 0.004
SL_BUFFER_PCT = 0.0005
IFC_LOOKAHEAD = 2

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


def month_starts(
    start: pd.Timestamp,
    end: pd.Timestamp,
):
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


def download_month(
    symbol: str,
    month: pd.Timestamp,
) -> Optional[Path]:

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{symbol}-1h-"
        f"{month.strftime('%Y-%m')}.zip"
    )

    path = CACHE_DIR / filename

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

            if (
                response.status_code == 200
                and len(response.content) > 1000
            ):
                path.write_bytes(
                    response.content
                )
                return path

            if response.status_code == 404:
                return None

            raise RuntimeError(
                f"HTTP {response.status_code}: {url}"
            )

        except Exception:
            if attempt == 3:
                raise

            time.sleep(1.5 * attempt)

    return None


def read_archive(
    path: Path,
) -> pd.DataFrame:

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


def fetch_symbol(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:

    frames = []
    first_available_month = None

    for month in month_starts(
        start,
        end,
    ):

        path = download_month(
            symbol,
            month,
        )

        if path is None:

            if first_available_month is None:
                continue

            raise RuntimeError(
                f"{symbol}: fatal archive gap "
                f"after first available month "
                f"{first_available_month.strftime('%Y-%m')}: "
                f"missing {month.strftime('%Y-%m')}"
            )

        if first_available_month is None:
            first_available_month = month

        frames.append(
            read_archive(path)
        )

    if not frames:
        raise RuntimeError(
            f"{symbol}: no historical data "
            f"available in requested range"
        )

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

    # Remove incomplete final candle.
    now = pd.Timestamp.now(
        tz="UTC"
    )

    df = df[
        df["time"] + BAR <= now
    ].copy()

    if df.empty:
        raise RuntimeError(
            f"{symbol}: no completed candles"
        )

    # --------------------------------------------------------
    # Check continuity BEFORE requested-range filtering.
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
            f"{symbol}: fatal 1H data gap: "
            f"{missing[:10].tolist()}"
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
            f"{symbol}: empty requested range"
        )

    # --------------------------------------------------------
    # Strict OOS coverage.
    # --------------------------------------------------------

    oos = df[
        (df["time"] >= OOS_START)
        & (df["time"] <= OOS_END)
    ].copy()

    if oos.empty:
        raise RuntimeError(
            f"{symbol}: no OOS coverage"
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
            f"{symbol}: fatal OOS gap: "
            f"{missing_oos[:10].tolist()}"
        )

    print(
        f"{symbol} OK {len(df)}"
    )

    return df.set_index("time")


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
        and value > float(right.max())
    )


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
        and value < float(right.min())
    )


def aligned(
    first: float,
    second: float,
) -> bool:

    return (
        abs(first - second)
        / max(
            abs(first),
            abs(second),
            1e-12,
        )
        <= ALIGN_TOL
    )


def bearish_ifc(
    df: pd.DataFrame,
    ob_index: int,
) -> bool:

    future_index = (
        ob_index + IFC_LOOKAHEAD
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


def bullish_ifc(
    df: pd.DataFrame,
    ob_index: int,
) -> bool:

    future_index = (
        ob_index + IFC_LOOKAHEAD
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
        return (
            min(
                open_price,
                close_price,
            ),
            high_price,
        )

    return (
        low_price,
        max(
            open_price,
            close_price,
        ),
    )


def build_candidates(
    df: pd.DataFrame,
    symbol: str,
) -> List[dict]:

    highs = []
    lows = []
    candidates = []

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
        # SHORT
        # ====================================================

        if (
            len(highs) >= 2
            and len(lows) >= 2
        ):

            previous_high = highs[-2]
            ob_high = highs[-1]

            lower_high = (
                ob_high[0] > previous_high[0]
                and ob_high[2] < previous_high[2]
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
                        < prior_low_1[2]
                    )

                    if downtrend:

                        ob_index = ob_high[0]

                        if bearish_ifc(
                            df,
                            ob_index,
                        ):

                            post_ifc_lows = [
                                item
                                for item in lows
                                if (
                                    item[0]
                                    > ob_index
                                    + IFC_LOOKAHEAD
                                    and item[1]
                                    <= confirm_index
                                )
                            ]

                            if len(
                                post_ifc_lows
                            ) >= 2:

                                valley_1 = (
                                    post_ifc_lows[-2]
                                )

                                valley_2 = (
                                    post_ifc_lows[-1]
                                )

                                structure_ok = (
                                    valley_2[0]
                                    > valley_1[0]
                                    and valley_1[2]
                                    < prior_low_2[2]
                                    and valley_2[2]
                                    < prior_low_2[2]
                                )

                                alignment_ok = (
                                    aligned(
                                        valley_1[2],
                                        valley_2[2],
                                    )
                                    and valley_2[2]
                                    >= valley_1[2]
                                )

                                if (
                                    structure_ok
                                    and alignment_ok
                                ):

                                    ready_index = max(
                                        valley_2[1],
                                        ob_index
                                        + IFC_LOOKAHEAD,
                                    )

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
        # LONG
        # ====================================================

        if (
            len(highs) >= 2
            and len(lows) >= 2
        ):

            previous_low = lows[-2]
            ob_low = lows[-1]

            higher_low = (
                ob_low[0] > previous_low[0]
                and ob_low[2] > previous_low[2]
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
                        > prior_high_1[2]
                    )

                    if uptrend:

                        ob_index = ob_low[0]

                        if bullish_ifc(
                            df,
                            ob_index,
                        ):

                            post_ifc_highs = [
                                item
                                for item in highs
                                if (
                                    item[0]
                                    > ob_index
                                    + IFC_LOOKAHEAD
                                    and item[1]
                                    <= confirm_index
                                )
                            ]

                            if len(
                                post_ifc_highs
                            ) >= 2:

                                top_1 = (
                                    post_ifc_highs[-2]
                                )

                                top_2 = (
                                    post_ifc_highs[-1]
                                )

                                structure_ok = (
                                    top_2[0]
                                    > top_1[0]
                                    and top_1[2]
                                    > prior_high_2[2]
                                    and top_2[2]
                                    > prior_high_2[2]
                                )

                                alignment_ok = (
                                    aligned(
                                        top_1[2],
                                        top_2[2],
                                    )
                                    and top_2[2]
                                    <= top_1[2]
                                )

                                if (
                                    structure_ok
                                    and alignment_ok
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

    seen = set()
    output = []

    for candidate in sorted(
        candidates,
        key=lambda x: (
            x["pattern_confirm_time"],
            x["symbol"],
            x["side"],
            x["ob_idx"],
            x["double_2"],
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
        output.append(candidate)

    return output


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


def simulate_candidate(
    df: pd.DataFrame,
    candidate: dict,
) -> Optional[dict]:

    side = candidate["side"]

    start_index = (
        candidate["ready_index"] + 1
    )

    if start_index >= len(df):
        return None

    if side == "LONG":

        raw_limit = candidate["ob_high"]

        stop = (
            candidate["ob_low"]
            * (1.0 - SL_BUFFER_PCT)
        )

    else:

        raw_limit = candidate["ob_low"]

        stop = (
            candidate["ob_high"]
            * (1.0 + SL_BUFFER_PCT)
        )

    entry_index = None

    for index in range(
        start_index,
        len(df),
    ):

        row = df.iloc[index]

        low = float(row["low"])
        high = float(row["high"])

        touched = (
            low <= raw_limit <= high
        )

        if touched:
            entry_index = index
            break

        if side == "LONG":
            if low < stop:
                return None
        else:
            if high > stop:
                return None

    if entry_index is None:
        return None

    entry = slipped_entry(
        float(raw_limit),
        side,
    )

    if side == "LONG":
        risk_price = entry - stop
    else:
        risk_price = stop - entry

    if risk_price <= 0:
        return None

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
        candidate["structural_target"]
    )

    if (
        side == "LONG"
        and target > structural_target
    ):
        return None

    if (
        side == "SHORT"
        and target < structural_target
    ):
        return None

    quantity = NOTIONAL / entry

    entry_fee = (
        NOTIONAL * FEE_RATE
    )

    exit_index = None
    outcome = None
    raw_exit = None

    for index in range(
        entry_index + 1,
        len(df),
    ):

        row = df.iloc[index]

        low = float(row["low"])
        high = float(row["high"])

        if side == "LONG":

            hit_sl = low <= stop
            hit_tp = high >= target

        else:

            hit_sl = high >= stop
            hit_tp = low <= target

        if hit_sl and hit_tp:

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

    if exit_index is None:

        return {
            **candidate,
            "entry_index": entry_index,
            "entry_time": df.index[entry_index],
            "entry": entry,
            "stop": stop,
            "target": target,
            "risk_price": risk_price,
            "structural_target": structural_target,
            "exit_index": np.nan,
            "exit_time": pd.NaT,
            "exit": np.nan,
            "outcome": "UNRESOLVED",
            "gross_pnl": np.nan,
            "fees": np.nan,
            "pnl": np.nan,
            "r_multiple": np.nan,
        }

    exit_price = slipped_exit(
        float(raw_exit),
        side,
    )

    if side == "LONG":

        gross_pnl = (
            exit_price - entry
        ) * quantity

    else:

        gross_pnl = (
            entry - exit_price
        ) * quantity

    exit_fee = (
        abs(exit_price * quantity)
        * FEE_RATE
    )

    fees = (
        entry_fee + exit_fee
    )

    pnl = (
        gross_pnl - fees
    )

    risk_cash = (
        risk_price * quantity
    )

    r_multiple = (
        pnl / risk_cash
    )

    return {
        **candidate,
        "entry_index": entry_index,
        "entry_time": df.index[entry_index],
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk_price": risk_price,
        "structural_target": structural_target,
        "exit_index": exit_index,
        "exit_time": df.index[exit_index],
        "exit": exit_price,
        "outcome": outcome,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "pnl": pnl,
        "r_multiple": r_multiple,
    }


def split_name(
    timestamp: pd.Timestamp,
) -> str:

    if timestamp < RESEARCH_START:
        return "WARMUP"

    if timestamp < DISCOVERY_END:
        return "Discovery"

    if timestamp < DEVELOPMENT_END:
        return "Development"

    if timestamp < OOS_START:
        return "Research_Holdout"

    return "Validation_OOS"


def portfolio_simulation(
    candidates: List[dict],
    data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:

    simulated = []

    for candidate in candidates:

        trade = simulate_candidate(
            data[candidate["symbol"]],
            candidate,
        )

        if trade is None:
            continue

        if (
            trade["entry_time"]
            < RESEARCH_START
        ):
            continue

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

            last_exit[symbol] = (
                pd.Timestamp.max
                .tz_localize("UTC")
            )

    return pd.DataFrame(
        accepted
    )


def metrics(
    trades: pd.DataFrame,
) -> dict:

    closed = trades[
        trades["outcome"].isin(
            ["WIN", "LOSS"]
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
            closed["outcome"] == "WIN"
        ).sum()
    )

    losses = int(
        (
            closed["outcome"] == "LOSS"
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
        .sort_values("entry_time")
    )

    for outcome in (
        ordered["outcome"].tolist()
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
        "trades": len(closed),
        "wins": wins,
        "losses": losses,
        "wr": (
            100.0
            * wins
            / len(closed)
        ),
        "pf": pf,
        "net_r": float(
            closed["r_multiple"].sum()
        ),
        "pnl": float(
            closed["pnl"].sum()
        ),
        "max_streak": max_streak,
    }


def equity_and_dd(
    trades: pd.DataFrame,
) -> Tuple[float, float, float]:

    equity = INITIAL_CAPITAL
    peak = equity
    max_dd = 0.0

    closed = trades[
        trades["outcome"].isin(
            ["WIN", "LOSS"]
        )
        & trades["exit_time"].notna()
    ].copy()

    closed = (
        closed
        .sort_values("exit_time")
    )

    for _, trade in closed.iterrows():

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


def integrity_audit(
    trades: pd.DataFrame,
) -> List[Tuple[str, str]]:

    causal_ok = True
    target_ok = True
    same_symbol_ok = True
    same_candle_ok = True
    rr_ok = True

    for _, trade in trades.iterrows():

        if (
            trade["entry_index"]
            <= trade["ready_index"]
        ):
            causal_ok = False

        if (
            trade["entry_time"]
            <= trade["pattern_confirm_time"]
        ):
            causal_ok = False

        if pd.notna(
            trade["exit_index"]
        ):

            if (
                trade["exit_index"]
                <= trade["entry_index"]
            ):
                same_candle_ok = False

        if trade["side"] == "LONG":

            risk = (
                trade["entry"]
                - trade["stop"]
            )

            if risk <= 0:
                rr_ok = False
            else:

                actual_rr = (
                    trade["target"]
                    - trade["entry"]
                ) / risk

                if abs(
                    actual_rr - RR
                ) > 1e-9:
                    rr_ok = False

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
                - trade["entry"]
            )

            if risk <= 0:
                rr_ok = False
            else:

                actual_rr = (
                    trade["entry"]
                    - trade["target"]
                ) / risk

                if abs(
                    actual_rr - RR
                ) > 1e-9:
                    rr_ok = False

            if (
                trade["target"]
                <
                trade["structural_target"]
                - 1e-12
            ):
                target_ok = False

        if (
            trade["double_2"]
            + PIVOT
            >
            trade["ready_index"]
        ):
            causal_ok = False

        if (
            trade["ob_confirm"]
            >
            trade["ready_index"]
        ):
            causal_ok = False

    ordered = (
        trades
        .sort_values("entry_time")
    )

    for _, group in (
        ordered.groupby("symbol")
    ):

        previous_exit = None

        for _, trade in group.iterrows():

            if (
                previous_exit is not None
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


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {}

    print(
        "Downloading strict Binance Futures 1H data..."
    )

    for symbol in SYMBOLS:

        print(
            f"Loading {symbol}...",
            flush=True,
        )

        data[symbol] = fetch_symbol(
            symbol,
            WARMUP_START,
            OOS_END,
        )

    all_candidates = []

    for symbol, df in data.items():

        candidates = build_candidates(
            df,
            symbol,
        )

        usable = [
            candidate
            for candidate in candidates
            if candidate[
                "pattern_confirm_time"
            ] <= OOS_END
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

    print(
        "Running portfolio simulation..."
    )

    trades = portfolio_simulation(
        all_candidates,
        data,
    )

    if trades.empty:
        raise RuntimeError(
            "No trades produced."
        )

    trades["split"] = (
        trades["entry_time"]
        .map(split_name)
    )

    checks = integrity_audit(
        trades
    )

    audit_passed = all(
        status == "PASSED"
        for _, status in checks
    )

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
            trades["split"] == split
        ]

        lines.append(
            format_metrics(
                split,
                metrics(split_trades),
            )
        )

    lines.append(
        format_metrics(
            "TOTAL",
            metrics(trades),
        )
    )

    (
        final_equity,
        max_dd,
        dd_pct,
    ) = equity_and_dd(trades)

    unresolved = int(
        (
            trades["outcome"]
            == "UNRESOLVED"
        ).sum()
    )

    lines.extend(
        [
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
            "FINAL INTEGRITY AUDIT",
            "=" * 110,
        ]
    )

    for name, status in checks:

        lines.append(
            f"{name:<32}: {status}"
        )

    lines.append(
        f"{'AUDIT STATUS':<32}: "
        f"{'PASSED' if audit_passed else 'FAILED'}"
    )

    lines.extend(
        [
            "",
            "Research protocol:",
            "Real Binance Futures OHLCV only.",
            "No synthetic candles.",
            "No forward fill.",
            "No artificial timeout.",
            "Pivot confirmation is causal.",
            "Entry occurs after structural confirmation.",
            "Fixed RR = 1:2.",
            "One simultaneous trade per symbol.",
            "Different symbols may overlap.",
            "Same-symbol re-entry is blocked until previous exit.",
            f"Fresh OOS: {OOS_START} -> {OOS_END}",
            "",
        ]
    )

    report = "\n".join(lines)

    print(report)

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

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
        f"Saved: {LEDGER_PATH}"
    )

    print(
        f"Saved: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
