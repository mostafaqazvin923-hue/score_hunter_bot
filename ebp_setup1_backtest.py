import io
import time
import zipfile
import calendar
from pathlib import Path

import requests
import numpy as np
import pandas as pd


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
WARMUP_DAYS = 60

RR = 2.0

INITIAL_CAPITAL = 1000.0
MARGIN = 100.0
LEVERAGE = 50.0
NOTIONAL = MARGIN * LEVERAGE

FEE_RATE = 0.0007
SLIPPAGE = 0.0003

PIVOT = 2

EQUAL_ATR = 0.20
EQUAL_PCT = 0.0015

MAX_AB_BARS = 6
MAX_PULLBACK_BARS = 6
MAX_RETRACE = 0.236

MIN_RISK = 0.0005
MAX_RISK = 0.08

ARCHIVE_BASE = (
    "https://data.binance.vision/data/futures/um"
)


# ============================================================
# TIME
# ============================================================

def now_utc():
    return pd.Timestamp.now(tz="UTC")


# ============================================================
# DOWNLOAD BINANCE ARCHIVE
# ============================================================

def download_zip(url):

    for attempt in range(3):

        try:

            response = requests.get(
                url,
                timeout=60
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()

            archive = zipfile.ZipFile(
                io.BytesIO(response.content)
            )

            csv_names = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv")
            ]

            if not csv_names:
                return None

            with archive.open(csv_names[0]) as file:

                return pd.read_csv(
                    file,
                    header=None,
                    low_memory=False
                )

        except Exception as exc:

            if attempt == 2:

                print(
                    f"DOWNLOAD FAILED:\n"
                    f"{url}\n"
                    f"{exc}",
                    flush=True
                )

                return None

            time.sleep(
                1 + attempt
            )

    return None


# ============================================================
# FETCH SYMBOL
# ============================================================

def fetch_symbol(
    symbol,
    start_ms,
    end_ms
):

    start = pd.to_datetime(
        start_ms,
        unit="ms",
        utc=True
    )

    end = pd.to_datetime(
        end_ms,
        unit="ms",
        utc=True
    )

    frames = []

    # --------------------------------------------------------
    # FIX:
    # Do NOT use start.to_period("M") on timezone-aware
    # timestamps. Build naive month anchors instead.
    # --------------------------------------------------------

    start_month = pd.Timestamp(
        year=start.year,
        month=start.month,
        day=1
    )

    end_month = pd.Timestamp(
        year=end.year,
        month=end.month,
        day=1
    )

    months = pd.period_range(
        start=start_month,
        end=end_month,
        freq="M"
    )

    # --------------------------------------------------------
    # Monthly archives first
    # --------------------------------------------------------

    for period in months:

        year = period.year
        month = period.month

        monthly_url = (
            f"{ARCHIVE_BASE}/monthly/klines/"
            f"{symbol}/{INTERVAL}/"
            f"{symbol}-{INTERVAL}-"
            f"{year}-{month:02d}.zip"
        )

        raw = download_zip(
            monthly_url
        )

        if raw is not None:

            frames.append(raw)

            continue

        # ----------------------------------------------------
        # Daily fallback
        # ----------------------------------------------------

        days_in_month = calendar.monthrange(
            year,
            month
        )[1]

        for day in range(
            1,
            days_in_month + 1
        ):

            date_str = (
                f"{year}-"
                f"{month:02d}-"
                f"{day:02d}"
            )

            daily_url = (
                f"{ARCHIVE_BASE}/daily/klines/"
                f"{symbol}/{INTERVAL}/"
                f"{symbol}-{INTERVAL}-"
                f"{date_str}.zip"
            )

            raw = download_zip(
                daily_url
            )

            if raw is not None:
                frames.append(raw)

    if not frames:

        return pd.DataFrame()

    x = pd.concat(
        frames,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Binance archive schema
    # --------------------------------------------------------

    if x.shape[1] < 12:

        raise RuntimeError(
            f"{symbol}: invalid Binance archive format "
            f"columns={x.shape[1]}"
        )

    x = x.iloc[:, :12]

    x.columns = [
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

    # --------------------------------------------------------
    # FIX #1
    #
    # Binance timestamps can be loaded as object/string.
    # Force numeric BEFORE datetime conversion.
    # --------------------------------------------------------

    x["open_time"] = pd.to_numeric(
        x["open_time"],
        errors="coerce"
    )

    x["close_time"] = pd.to_numeric(
        x["close_time"],
        errors="coerce"
    )

    x = x.dropna(
        subset=[
            "open_time",
            "close_time"
        ]
    )

    # --------------------------------------------------------
    # FIX #2
    #
    # Explicitly cast timestamps to int64 milliseconds.
    # This prevents Pandas from trying to parse:
    #
    # "1754006400000"
    #
    # as a date string.
    # --------------------------------------------------------

    x["open_time"] = pd.to_datetime(
        x["open_time"].astype("int64"),
        unit="ms",
        utc=True,
        errors="coerce"
    )

    x["close_time"] = pd.to_datetime(
        x["close_time"].astype("int64"),
        unit="ms",
        utc=True,
        errors="coerce"
    )

    x = x.dropna(
        subset=[
            "open_time",
            "close_time"
        ]
    )

    # --------------------------------------------------------
    # Numeric OHLCV
    # --------------------------------------------------------

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        x[column] = pd.to_numeric(
            x[column],
            errors="coerce"
        )

    x = x.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    x = (
        x.drop_duplicates(
            subset=["open_time"]
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Requested period
    # --------------------------------------------------------

    x = x[
        (x["open_time"] >= start)
        &
        (x["open_time"] < end)
    ]

    # --------------------------------------------------------
    # Never use an incomplete current candle
    # --------------------------------------------------------

    current_time = now_utc()

    x = x[
        x["close_time"] <= current_time
    ]

    x = x.reset_index(
        drop=True
    )

    return x[
        [
            "open_time",
            "close_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]


# ============================================================
# FEATURES
# ============================================================

def add_features(df):

    x = df.copy()

    previous_close = x["close"].shift(1)

    true_range = pd.concat(
        [
            x["high"] - x["low"],
            (
                x["high"]
                -
                previous_close
            ).abs(),
            (
                x["low"]
                -
                previous_close
            ).abs(),
        ],
        axis=1
    ).max(axis=1)

    x["atr"] = (
        true_range
        .rolling(
            14,
            min_periods=14
        )
        .mean()
    )

    x["pivot_high"] = False
    x["pivot_low"] = False

    # --------------------------------------------------------
    # Confirmed pivots.
    #
    # A pivot is only added after PIVOT candles to its right
    # have closed.
    #
    # Therefore no future information is used for the signal.
    # --------------------------------------------------------

    for i in range(
        PIVOT,
        len(x) - PIVOT
    ):

        left_high = x["high"].iloc[
            i - PIVOT:i
        ].max()

        right_high = x["high"].iloc[
            i + 1:i + PIVOT + 1
        ].max()

        left_low = x["low"].iloc[
            i - PIVOT:i
        ].min()

        right_low = x["low"].iloc[
            i + 1:i + PIVOT + 1
        ].min()

        if (
            x["high"].iloc[i]
            >= left_high
            and
            x["high"].iloc[i]
            > right_high
        ):

            x.loc[
                x.index[i],
                "pivot_high"
            ] = True

        if (
            x["low"].iloc[i]
            <= left_low
            and
            x["low"].iloc[i]
            < right_low
        ):

            x.loc[
                x.index[i],
                "pivot_low"
            ] = True

    return x


# ============================================================
# SETUP 1
#
# PDF mechanical translation:
#
# SHORT:
#   downtrend
#   equal highs
#   lowest valley
#   fake upside breakout
#   pullback
#   bearish confirmation
#
# LONG:
#   symmetric reversal
# ============================================================

def find_candidates(
    df,
    symbol
):

    x = add_features(df)

    pivot_highs = []
    pivot_lows = []

    candidates = []

    for i in range(
        len(x)
    ):

        # ----------------------------------------------------
        # Confirm pivot PIVOT bars after the pivot itself.
        # ----------------------------------------------------

        if i >= PIVOT:

            confirmed_index = (
                i - PIVOT
            )

            if x[
                "pivot_high"
            ].iloc[
                confirmed_index
            ]:

                pivot_highs.append(
                    confirmed_index
                )

            if x[
                "pivot_low"
            ].iloc[
                confirmed_index
            ]:

                pivot_lows.append(
                    confirmed_index
                )

            pivot_highs = pivot_highs[-20:]
            pivot_lows = pivot_lows[-20:]

        if (
            len(pivot_highs) < 2
            or
            len(pivot_lows) < 2
        ):

            continue

        h1 = pivot_highs[-2]
        h2 = pivot_highs[-1]

        l1 = pivot_lows[-2]
        l2 = pivot_lows[-1]

        # ====================================================
        # SHORT
        # ====================================================

        downtrend = (
            h2 > h1
            and
            l2 > l1
            and
            x["high"].iloc[h2]
            <
            x["high"].iloc[h1]
            and
            x["low"].iloc[l2]
            <
            x["low"].iloc[l1]
        )

        if downtrend:

            equal_level = (
                x["high"].iloc[h1]
                +
                x["high"].iloc[h2]
            ) / 2.0

            atr_value = x["atr"].iloc[h2]

            if pd.isna(atr_value):

                continue

            tolerance = max(
                float(atr_value)
                * EQUAL_ATR,
                equal_level
                * EQUAL_PCT
            )

            equal_highs = (
                abs(
                    x["high"].iloc[h1]
                    -
                    x["high"].iloc[h2]
                )
                <= tolerance
            )

            if not equal_highs:

                continue

            valleys = [
                p
                for p in pivot_lows
                if p > h2
                and p < i
            ]

            if not valleys:

                continue

            valley = min(
                valleys,
                key=lambda p:
                x["low"].iloc[p]
            )

            valley_price = float(
                x["low"].iloc[valley]
            )

            # ------------------------------------------------
            # Search for fake breakout
            # ------------------------------------------------

            for breakout in range(
                valley + 1,
                min(
                    len(x),
                    valley
                    + MAX_AB_BARS
                    + 1
                )
            ):

                high = float(
                    x["high"].iloc[breakout]
                )

                close = float(
                    x["close"].iloc[breakout]
                )

                # Wick must cross equal-high level.
                if high <= equal_level:
                    continue

                # Close must return below level.
                if close >= equal_level:
                    continue

                move = (
                    equal_level
                    -
                    valley_price
                )

                if move <= 0:

                    continue

                internal = x.iloc[
                    valley + 1:breakout
                ]

                if not internal.empty:

                    retracement = (
                        equal_level
                        -
                        float(
                            internal["low"].min()
                        )
                    ) / move

                    if (
                        retracement
                        >
                        MAX_RETRACE
                    ):

                        continue

                # --------------------------------------------
                # First pullback
                # --------------------------------------------

                for pullback in range(
                    breakout + 1,
                    min(
                        len(x),
                        breakout
                        + MAX_PULLBACK_BARS
                        + 1
                    )
                ):

                    if (
                        x["low"].iloc[pullback]
                        >
                        equal_level
                    ):

                        continue

                    if (
                        x["low"].iloc[pullback]
                        <= valley_price
                    ):

                        break

                    confirmation = (
                        x["close"].iloc[pullback]
                        <
                        x["open"].iloc[pullback]
                        and
                        x["close"].iloc[pullback]
                        <
                        x["low"].iloc[
                            pullback - 1
                        ]
                    )

                    if not confirmation:

                        continue

                    entry_bar = (
                        pullback + 1
                    )

                    if (
                        entry_bar
                        >= len(x)
                    ):

                        break

                    raw_entry = float(
                        x["open"].iloc[
                            entry_bar
                        ]
                    )

                    stop = min(
                        float(
                            x["high"].iloc[
                                breakout
                            ]
                        ),
                        float(
                            x.iloc[
                                breakout:
                                pullback + 1
                            ]["high"].max()
                        )
                    )

                    risk = (
                        stop
                        -
                        raw_entry
                    )

                    if risk <= 0:

                        break

                    risk_pct = (
                        risk
                        /
                        raw_entry
                    )

                    if not (
                        MIN_RISK
                        <= risk_pct
                        <= MAX_RISK
                    ):

                        break

                    target = (
                        raw_entry
                        -
                        RR * risk
                    )

                    # Target must remain at/above
                    # the PDF's valley target.
                    if (
                        target
                        <
                        valley_price
                    ):

                        break

                    candidates.append(
                        {
                            "symbol": symbol,
                            "side": "SHORT",
                            "signal_bar": breakout,
                            "confirm_bar": pullback,
                            "entry_bar": entry_bar,
                            "entry": raw_entry,
                            "stop": stop,
                            "target": target,
                        }
                    )

                    break

                break

        # ====================================================
        # LONG — SYMMETRIC
        # ====================================================

        uptrend = (
            h2 > h1
            and
            l2 > l1
            and
            x["high"].iloc[h2]
            >
            x["high"].iloc[h1]
            and
            x["low"].iloc[l2]
            >
            x["low"].iloc[l1]
        )

        if uptrend:

            equal_level = (
                x["low"].iloc[l1]
                +
                x["low"].iloc[l2]
            ) / 2.0

            atr_value = x["atr"].iloc[l2]

            if pd.isna(atr_value):

                continue

            tolerance = max(
                float(atr_value)
                * EQUAL_ATR,
                equal_level
                * EQUAL_PCT
            )

            equal_lows = (
                abs(
                    x["low"].iloc[l1]
                    -
                    x["low"].iloc[l2]
                )
                <= tolerance
            )

            if not equal_lows:

                continue

            peaks = [
                p
                for p in pivot_highs
                if p > l2
                and p < i
            ]

            if not peaks:

                continue

            peak = max(
                peaks,
                key=lambda p:
                x["high"].iloc[p]
            )

            peak_price = float(
                x["high"].iloc[peak]
            )

            for breakout in range(
                peak + 1,
                min(
                    len(x),
                    peak
                    + MAX_AB_BARS
                    + 1
                )
            ):

                low = float(
                    x["low"].iloc[breakout]
                )

                close = float(
                    x["close"].iloc[breakout]
                )

                if low >= equal_level:

                    continue

                if close <= equal_level:

                    continue

                move = (
                    peak_price
                    -
                    equal_level
                )

                if move <= 0:

                    continue

                internal = x.iloc[
                    peak + 1:breakout
                ]

                if not internal.empty:

                    retracement = (
                        float(
                            internal["high"].max()
                        )
                        -
                        equal_level
                    ) / move

                    if (
                        retracement
                        >
                        MAX_RETRACE
                    ):

                        continue

                for pullback in range(
                    breakout + 1,
                    min(
                        len(x),
                        breakout
                        + MAX_PULLBACK_BARS
                        + 1
                    )
                ):

                    if (
                        x["high"].iloc[pullback]
                        <
                        equal_level
                    ):

                        continue

                    if (
                        x["high"].iloc[pullback]
                        >= peak_price
                    ):

                        break

                    confirmation = (
                        x["close"].iloc[pullback]
                        >
                        x["open"].iloc[pullback]
                        and
                        x["close"].iloc[pullback]
                        >
                        x["high"].iloc[
                            pullback - 1
                        ]
                    )

                    if not confirmation:

                        continue

                    entry_bar = (
                        pullback + 1
                    )

                    if (
                        entry_bar
                        >= len(x)
                    ):

                        break

                    raw_entry = float(
                        x["open"].iloc[
                            entry_bar
                        ]
                    )

                    stop = max(
                        float(
                            x["low"].iloc[
                                breakout
                            ]
                        ),
                        float(
                            x.iloc[
                                breakout:
                                pullback + 1
                            ]["low"].min()
                        )
                    )

                    risk = (
                        raw_entry
                        -
                        stop
                    )

                    if risk <= 0:

                        break

                    risk_pct = (
                        risk
                        /
                        raw_entry
                    )

                    if not (
                        MIN_RISK
                        <= risk_pct
                        <= MAX_RISK
                    ):

                        break

                    target = (
                        raw_entry
                        +
                        RR * risk
                    )

                    if (
                        target
                        >
                        peak_price
                    ):

                        break

                    candidates.append(
                        {
                            "symbol": symbol,
                            "side": "LONG",
                            "signal_bar": breakout,
                            "confirm_bar": pullback,
                            "entry_bar": entry_bar,
                            "entry": raw_entry,
                            "stop": stop,
                            "target": target,
                        }
                    )

                    break

                break

    if not candidates:

        return pd.DataFrame()

    return (
        pd.DataFrame(candidates)
        .drop_duplicates(
            subset=[
                "symbol",
                "entry_bar"
            ]
        )
        .sort_values(
            "entry_bar"
        )
        .reset_index(drop=True)
    )


# ============================================================
# SLIPPAGE
# ============================================================

def apply_entry_slippage(
    price,
    side
):

    if side == "LONG":

        return price * (
            1 + SLIPPAGE
        )

    return price * (
        1 - SLIPPAGE
    )


def apply_exit_slippage(
    price,
    side
):

    if side == "LONG":

        return price * (
            1 - SLIPPAGE
        )

    return price * (
        1 + SLIPPAGE
    )


# ============================================================
# SIMULATION
# ============================================================

def simulate_symbol(
    df,
    candidates
):

    if candidates.empty:

        return pd.DataFrame()

    trades = []

    occupied_until = -1

    for _, setup in candidates.sort_values(
        "entry_bar"
    ).iterrows():

        entry_bar = int(
            setup["entry_bar"]
        )

        # Same symbol cannot overlap.
        if entry_bar <= occupied_until:

            continue

        side = setup["side"]

        entry = apply_entry_slippage(
            float(
                df["open"].iloc[
                    entry_bar
                ]
            ),
            side
        )

        stop = float(
            setup["stop"]
        )

        target = float(
            setup["target"]
        )

        exit_bar = None
        exit_raw = None
        outcome = None

        # ----------------------------------------------------
        # Forward-only trade simulation.
        # ----------------------------------------------------

        for j in range(
            entry_bar,
            len(df)
        ):

            high = float(
                df["high"].iloc[j]
            )

            low = float(
                df["low"].iloc[j]
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

            # Locked rule:
            # same candle SL + TP = LOSS.
            if hit_sl and hit_tp:

                exit_bar = j
                exit_raw = stop
                outcome = "LOSS"

                break

            if hit_sl:

                exit_bar = j
                exit_raw = stop
                outcome = "LOSS"

                break

            if hit_tp:

                exit_bar = j
                exit_raw = target
                outcome = "WIN"

                break

        # ----------------------------------------------------
        # Censored unresolved trade.
        # ----------------------------------------------------

        if exit_bar is None:

            continue

        exit_price = apply_exit_slippage(
            exit_raw,
            side
        )

        risk_price = abs(
            entry - stop
        )

        if risk_price <= 0:

            continue

        if side == "LONG":

            gross_R = (
                exit_price
                -
                entry
            ) / risk_price

        else:

            gross_R = (
                entry
                -
                exit_price
            ) / risk_price

        fee_dollars = (
            NOTIONAL
            * FEE_RATE
            * 2
        )

        one_R_dollars = (
            NOTIONAL
            *
            (
                risk_price
                /
                entry
            )
        )

        fee_R = (
            fee_dollars
            /
            one_R_dollars
        )

        net_R = (
            gross_R
            -
            fee_R
        )

        trades.append(
            {
                "symbol": setup["symbol"],
                "side": side,
                "entry_time":
                    df["open_time"].iloc[
                        entry_bar
                    ],
                "exit_time":
                    df["close_time"].iloc[
                        exit_bar
                    ],
                "entry": entry,
                "exit": exit_price,
                "stop": stop,
                "target": target,
                "outcome": outcome,
                "gross_R": gross_R,
                "fee_R": fee_R,
                "R": net_R,
            }
        )

        # Re-entry only after previous trade closes.
        occupied_until = exit_bar

    if not trades:

        return pd.DataFrame()

    return pd.DataFrame(
        trades
    )


# ============================================================
# STATS
# ============================================================

def calculate_stats(
    trades
):

    if trades.empty:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "wr": np.nan,
            "pf": np.nan,
            "net_R": 0.0,
            "max_streak": 0,
        }

    wins = (
        trades["R"] > 0
    )

    gross_profit = trades.loc[
        wins,
        "R"
    ].sum()

    gross_loss = -trades.loc[
        ~wins,
        "R"
    ].sum()

    if gross_loss > 0:

        pf = (
            gross_profit
            /
            gross_loss
        )

    else:

        pf = np.inf

    streak = 0
    max_streak = 0

    for value in trades["R"]:

        if value <= 0:

            streak += 1

            max_streak = max(
                max_streak,
                streak
            )

        else:

            streak = 0

    return {
        "trades": len(trades),
        "wins": int(
            wins.sum()
        ),
        "losses": int(
            (~wins).sum()
        ),
        "wr": 100 * wins.mean(),
        "pf": pf,
        "net_R": trades["R"].sum(),
        "max_streak": max_streak,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    output = Path(
        "ebp_setup1_outputs"
    )

    output.mkdir(
        exist_ok=True
    )

    end = now_utc().floor("h")

    full_start = (
        end
        -
        pd.Timedelta(
            days=(
                TEST_DAYS
                +
                WARMUP_DAYS
            )
        )
    )

    test_start = (
        end
        -
        pd.Timedelta(
            days=TEST_DAYS
        )
    )

    discovery_end = (
        test_start
        +
        pd.Timedelta(
            days=TEST_DAYS * 0.50
        )
    )

    development_end = (
        discovery_end
        +
        pd.Timedelta(
            days=TEST_DAYS * 0.25
        )
    )

    all_trades = []

    total_candidates = 0

    for symbol in SYMBOLS:

        print(
            f"\nFetching {symbol} ...",
            flush=True
        )

        df = fetch_symbol(
            symbol,
            int(
                full_start.timestamp()
                * 1000
            ),
            int(
                end.timestamp()
                * 1000
            )
        )

        print(
            f"{symbol}: "
            f"{len(df):,} rows",
            flush=True
        )

        if len(df) < 500:

            print(
                f"{symbol}: "
                "INSUFFICIENT DATA",
                flush=True
            )

            continue

        candidates = find_candidates(
            df,
            symbol
        )

        total_candidates += len(
            candidates
        )

        print(
            f"{symbol}: "
            f"{len(candidates):,} candidates",
            flush=True
        )

        trades = simulate_symbol(
            df,
            candidates
        )

        if not trades.empty:

            all_trades.append(
                trades
            )

    # ========================================================
    # COMBINE
    # ========================================================

    if all_trades:

        trades = pd.concat(
            all_trades,
            ignore_index=True
        )

        trades = (
            trades
            .sort_values(
                "entry_time"
            )
            .reset_index(
                drop=True
            )
        )

    else:

        trades = pd.DataFrame()

    trades.to_csv(
        output / "trades.csv",
        index=False
    )

    # ========================================================
    # SPLITS
    # ========================================================

    summary_rows = []

    if not trades.empty:

        splits = [
            (
                "Discovery",
                test_start,
                discovery_end
            ),
            (
                "Development",
                discovery_end,
                development_end
            ),
            (
                "Validation",
                development_end,
                end
            ),
        ]

        for (
            name,
            split_start,
            split_end
        ) in splits:

            subset = trades[
                (
                    trades["entry_time"]
                    >= split_start
                )
                &
                (
                    trades["entry_time"]
                    < split_end
                )
            ]

            stats = calculate_stats(
                subset
            )

            summary_rows.append(
                {
                    "split": name,
                    **stats
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        output / "summary.csv",
        index=False
    )

    # ========================================================
    # PER SYMBOL
    # ========================================================

    per_symbol_rows = []

    if not trades.empty:

        for (
            symbol,
            side
        ), subset in trades.groupby(
            [
                "symbol",
                "side"
            ]
        ):

            stats = calculate_stats(
                subset
            )

            per_symbol_rows.append(
                {
                    "symbol": symbol,
                    "side": side,
                    **stats
                }
            )

    per_symbol = pd.DataFrame(
        per_symbol_rows
    )

    per_symbol.to_csv(
        output / "per_symbol.csv",
        index=False
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    diagnostics = pd.DataFrame(
        [
            {
                "candidate_setups":
                    total_candidates,
                "closed_trades":
                    len(trades)
            }
        ]
    )

    diagnostics.to_csv(
        output / "diagnostics.csv",
        index=False
    )

    # ========================================================
    # REPORT
    # ========================================================

    print("\n")
    print("=" * 72)
    print("EBP SETUP 1 — BACKTEST REPORT")
    print("=" * 72)

    if summary.empty:

        print(
            "NO CLOSED TRADES"
        )

    else:

        print(
            summary.to_string(
                index=False
            )
        )

    print()
    print(
        f"TOTAL CANDIDATES: "
        f"{total_candidates}"
    )

    print(
        f"TOTAL CLOSED TRADES: "
        f"{len(trades)}"
    )

    print("=" * 72)


if __name__ == "__main__":

    main()
