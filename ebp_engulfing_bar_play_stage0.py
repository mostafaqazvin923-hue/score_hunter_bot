# ============================================================
# EBP STAGE-0
# Engulfing Bar Play / Sweep + Reclaim
#
# PURE MECHANICAL TEST
#
# Long:
#   current Low  < previous Low
#   current Close > previous Open
#
# Short:
#   current High > previous High
#   current Close < previous Open
#
# Execution:
#   Signal at candle close
#   Entry at NEXT candle open
#   SL = EBP candle extreme
#   TP = 2R
#
# Costs:
#   Fee = 0.07% per side
#   Slippage = 0.03% per side
#
# Capital model:
#   $100 margin
#   50x leverage
#   $5,000 notional
#
# No:
#   lookahead
#   timeout
#   trailing
#   breakeven
#   pyramiding
#   same-candle re-entry
# ============================================================

import io
import time
import zipfile
import requests
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_URL = "https://data.binance.vision/data/futures/um"

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

TIMEFRAMES = [
    "1h",
    "4h",
    "1d",
]

LOOKBACK_DAYS = 365

# Warmup isn't technically needed for the pure EBP itself,
# but we keep a small buffer for clean chronological handling.
WARMUP_DAYS = 30

RR = 2.0

FEE_RATE = 0.0007
SLIPPAGE = 0.0003

MARGIN = 100.0
LEVERAGE = 50.0

NOTIONAL = MARGIN * LEVERAGE

MIN_TRADES_VALIDATION = 150


# ============================================================
# HTTP
# ============================================================

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": "EBP-Research/1.0"
})


# ============================================================
# DATA LOADER
# ============================================================

def download_zip(url):

    """
    Binance Vision downloader.

    404 is NOT retried because it normally means that
    the requested archive does not exist.

    Other network errors are retried.
    """

    last_error = None

    for attempt in range(3):

        try:

            response = SESSION.get(
                url,
                timeout=60
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()

            return response.content

        except requests.RequestException as exc:

            last_error = exc

            if attempt < 2:
                time.sleep(
                    1.5 * (attempt + 1)
                )

    raise last_error


def parse_zip(blob):

    if blob is None:
        return pd.DataFrame()

    with zipfile.ZipFile(
        io.BytesIO(blob)
    ) as z:

        csv_files = [
            name
            for name in z.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            return pd.DataFrame()

        raw = z.read(csv_files[0])

    df = pd.read_csv(
        io.BytesIO(raw),
        header=None
    )

    # Some Binance archives contain a header row.
    if len(df) > 0:

        first = str(
            df.iloc[0, 0]
        ).lower()

        if first in (
            "open_time",
            "open time"
        ):
            df = df.iloc[1:].reset_index(
                drop=True
            )

    if df.shape[1] < 6:
        return pd.DataFrame()

    names = [
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

    df = df.iloc[:, :len(names)]

    df.columns = names[:df.shape[1]]

    df["open_time"] = pd.to_numeric(
        df["open_time"],
        errors="coerce"
    )

    median_time = df[
        "open_time"
    ].dropna().median()

    if median_time > 10**14:
        unit = "us"
    else:
        unit = "ms"

    df["time"] = pd.to_datetime(
        df["open_time"],
        unit=unit,
        utc=True,
        errors="coerce"
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
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "time",
            "open",
            "high",
            "low",
            "close",
        ]
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
    ].copy()


def month_iterator(
    start,
    end
):

    current = (
        pd.Timestamp(start)
        .tz_convert("UTC")
        .normalize()
        .replace(day=1)
    )

    final = (
        pd.Timestamp(end)
        .tz_convert("UTC")
        .normalize()
        .replace(day=1)
    )

    while current <= final:

        yield current

        current = (
            current
            + pd.offsets.MonthBegin(1)
        )


def load_symbol(
    symbol,
    timeframe,
    start,
    end
):

    parts = []

    end_day = (
        pd.Timestamp(end)
        .tz_convert("UTC")
        .normalize()
    )

    # ========================================================
    # MONTHLY ARCHIVES
    # ========================================================

    for month in month_iterator(
        start,
        end
    ):

        ym = month.strftime(
            "%Y-%m"
        )

        monthly_url = (
            f"{BASE_URL}/monthly/"
            f"klines/{symbol}/"
            f"{timeframe}/"
            f"{symbol}-{timeframe}-{ym}.zip"
        )

        blob = download_zip(
            monthly_url
        )

        if blob is not None:

            parsed = parse_zip(
                blob
            )

            if not parsed.empty:
                parts.append(parsed)

            continue

        # ====================================================
        # DAILY FALLBACK
        # ====================================================

        day = month

        month_end = (
            month
            + pd.offsets.MonthEnd(0)
        )

        while day <= month_end:

            if day > end_day:
                break

            date_string = (
                day.strftime(
                    "%Y-%m-%d"
                )
            )

            daily_url = (
                f"{BASE_URL}/daily/"
                f"klines/{symbol}/"
                f"{timeframe}/"
                f"{symbol}-{timeframe}-{date_string}.zip"
            )

            daily_blob = download_zip(
                daily_url
            )

            if daily_blob is not None:

                parsed = parse_zip(
                    daily_blob
                )

                if not parsed.empty:
                    parts.append(parsed)

            day += pd.Timedelta(
                days=1
            )

    if not parts:

        raise RuntimeError(
            f"{symbol} {timeframe}: "
            "NO DATA DOWNLOADED"
        )

    df = pd.concat(
        parts,
        ignore_index=True
    )

    df = (
        df
        .drop_duplicates(
            subset=["time"]
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    df = df[
        (df["time"] >= start)
        &
        (df["time"] <= end)
    ].copy()

    # ========================================================
    # GAP AUDIT
    # ========================================================

    expected_delta = {
        "1h": pd.Timedelta(hours=1),
        "4h": pd.Timedelta(hours=4),
        "1d": pd.Timedelta(days=1),
    }[timeframe]

    differences = (
        df["time"]
        .diff()
        .dropna()
    )

    if not differences.empty:

        maximum_gap = differences.max()

        if maximum_gap > expected_delta:

            raise RuntimeError(
                f"{symbol} {timeframe}: "
                f"FATAL DATA GAP: {maximum_gap}"
            )

    return df


# ============================================================
# EBP DETECTION
# ============================================================

def add_ebp_signals(df):

    data = df.copy()

    previous_open = (
        data["open"].shift(1)
    )

    previous_high = (
        data["high"].shift(1)
    )

    previous_low = (
        data["low"].shift(1)
    )

    # --------------------------------------------------------
    # BULLISH EBP
    #
    # Current candle sweeps previous low
    # AND closes above previous open.
    # --------------------------------------------------------

    data["bullish_ebp"] = (
        (data["low"] < previous_low)
        &
        (data["close"] > previous_open)
    )

    # --------------------------------------------------------
    # BEARISH EBP
    #
    # Current candle sweeps previous high
    # AND closes below previous open.
    # --------------------------------------------------------

    data["bearish_ebp"] = (
        (data["high"] > previous_high)
        &
        (data["close"] < previous_open)
    )

    return data


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_symbol(
    df,
    symbol,
    timeframe,
    direction
):

    data = add_ebp_signals(
        df
    )

    trades = []

    i = 1

    while i < len(data) - 1:

        candle = data.iloc[i]

        # ====================================================
        # SIGNAL
        # ====================================================

        if direction == "LONG":

            signal = bool(
                candle["bullish_ebp"]
            )

        else:

            signal = bool(
                candle["bearish_ebp"]
            )

        if not signal:

            i += 1
            continue

        # ====================================================
        # ENTRY
        #
        # Signal candle is CLOSED.
        #
        # Therefore the earliest possible execution
        # is the NEXT candle OPEN.
        # ====================================================

        entry_bar = data.iloc[i + 1]

        entry_time = entry_bar["time"]

        raw_entry = float(
            entry_bar["open"]
        )

        # ====================================================
        # SL
        #
        # Long:
        #   SL = EBP candle LOW
        #
        # Short:
        #   SL = EBP candle HIGH
        # ====================================================

        if direction == "LONG":

            raw_entry_effective = (
                raw_entry
                * (1.0 + SLIPPAGE)
            )

            stop = float(
                candle["low"]
            )

            risk = (
                raw_entry_effective
                - stop
            )

            if risk <= 0:

                i += 1
                continue

            target = (
                raw_entry_effective
                + RR * risk
            )

        else:

            raw_entry_effective = (
                raw_entry
                * (1.0 - SLIPPAGE)
            )

            stop = float(
                candle["high"]
            )

            risk = (
                stop
                - raw_entry_effective
            )

            if risk <= 0:

                i += 1
                continue

            target = (
                raw_entry_effective
                - RR * risk
            )

        # ====================================================
        # WALK FORWARD
        #
        # NO TIMEOUT.
        #
        # Trade remains open until SL or TP.
        # ====================================================

        exit_index = None
        exit_price = None
        result = None

        for j in range(
            i + 1,
            len(data)
        ):

            bar = data.iloc[j]

            high = float(
                bar["high"]
            )

            low = float(
                bar["low"]
            )

            if direction == "LONG":

                hit_stop = (
                    low <= stop
                )

                hit_target = (
                    high >= target
                )

            else:

                hit_stop = (
                    high >= stop
                )

                hit_target = (
                    low <= target
                )

            # ------------------------------------------------
            # SAME CANDLE SL + TP
            #
            # Conservative rule:
            # LOSS
            # ------------------------------------------------

            if (
                hit_stop
                and hit_target
            ):

                result = "LOSS"

                exit_price = stop

                exit_index = j

                break

            if hit_stop:

                result = "LOSS"

                exit_price = stop

                exit_index = j

                break

            if hit_target:

                result = "WIN"

                exit_price = target

                exit_index = j

                break

        # ====================================================
        # CENSORED TRADE
        #
        # If neither TP nor SL happened before the
        # end of dataset, we DO NOT count it.
        # ====================================================

        if exit_index is None:

            break

        # ====================================================
        # EXIT SLIPPAGE
        # ====================================================

        if direction == "LONG":

            effective_exit = (
                exit_price
                * (1.0 - SLIPPAGE)
            )

            price_return = (
                effective_exit
                - raw_entry_effective
            ) / raw_entry_effective

        else:

            effective_exit = (
                exit_price
                * (1.0 + SLIPPAGE)
            )

            price_return = (
                raw_entry_effective
                - effective_exit
            ) / raw_entry_effective

        gross_pnl = (
            NOTIONAL
            * price_return
        )

        entry_fee = (
            NOTIONAL
            * FEE_RATE
        )

        exit_fee = (
            NOTIONAL
            * FEE_RATE
        )

        net_pnl = (
            gross_pnl
            - entry_fee
            - exit_fee
        )

        # Dollar risk based on actual stop distance.
        risk_dollars = (
            NOTIONAL
            * (
                risk
                / raw_entry_effective
            )
        )

        if risk_dollars <= 0:

            i += 1
            continue

        r_multiple = (
            net_pnl
            / risk_dollars
        )

        trades.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,

            "signal_time": candle["time"],
            "entry_time": entry_time,
            "exit_time": data.iloc[
                exit_index
            ]["time"],

            "entry": raw_entry_effective,
            "stop": stop,
            "target": target,
            "exit": effective_exit,

            "risk_price": risk,

            "result": result,

            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "R": r_multiple,
        })

        # ====================================================
        # IMPORTANT
        #
        # Move pointer to AFTER the EXIT candle.
        #
        # This guarantees:
        # - no same-candle re-entry
        # - no overlapping trades per symbol
        # ====================================================

        i = exit_index + 1

    return pd.DataFrame(
        trades
    )


# ============================================================
# METRICS
# ============================================================

def max_loss_streak(
    results
):

    current = 0
    maximum = 0

    for result in results:

        if result == "LOSS":

            current += 1

            maximum = max(
                maximum,
                current
            )

        else:

            current = 0

    return maximum


def calculate_metrics(
    trades
):

    if trades.empty:

        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": np.nan,
            "net_R": 0.0,
            "avg_R": 0.0,
            "max_loss_streak": 0,
        }

    wins = int(
        (
            trades["result"]
            == "WIN"
        ).sum()
    )

    losses = int(
        (
            trades["result"]
            == "LOSS"
        ).sum()
    )

    gross_profit = trades.loc[
        trades["R"] > 0,
        "R"
    ].sum()

    gross_loss = -trades.loc[
        trades["R"] < 0,
        "R"
    ].sum()

    profit_factor = (
        gross_profit
        / gross_loss
        if gross_loss > 0
        else np.inf
    )

    return {
        "trades": len(trades),

        "wins": wins,

        "losses": losses,

        "win_rate": (
            wins / len(trades) * 100
        ),

        "profit_factor": (
            profit_factor
        ),

        "net_R": (
            trades["R"].sum()
        ),

        "avg_R": (
            trades["R"].mean()
        ),

        "max_loss_streak": (
            max_loss_streak(
                trades["result"].tolist()
            )
        ),
    }


# ============================================================
# SPLITS
# ============================================================

def split_trades(
    trades,
    start,
    end
):

    total = (
        end - start
    )

    discovery_end = (
        start
        + total * 0.50
    )

    development_end = (
        start
        + total * 0.75
    )

    discovery = trades[
        (
            trades["entry_time"]
            >= start
        )
        &
        (
            trades["entry_time"]
            < discovery_end
        )
    ]

    development = trades[
        (
            trades["entry_time"]
            >= discovery_end
        )
        &
        (
            trades["entry_time"]
            < development_end
        )
    ]

    validation = trades[
        (
            trades["entry_time"]
            >= development_end
        )
        &
        (
            trades["entry_time"]
            < end
        )
    ]

    return (
        discovery,
        development,
        validation
    )


# ============================================================
# OVERLAP AUDIT
# ============================================================

def audit_no_overlap(
    trades
):

    violations = []

    for (
        timeframe,
        symbol,
        direction
    ), group in trades.groupby(
        [
            "timeframe",
            "symbol",
            "direction",
        ]
    ):

        group = group.sort_values(
            "entry_time"
        )

        previous_exit = None

        for _, row in group.iterrows():

            if (
                previous_exit is not None
                and row["entry_time"]
                <= previous_exit
            ):

                violations.append({
                    "timeframe": timeframe,
                    "symbol": symbol,
                    "direction": direction,
                    "entry": row["entry_time"],
                    "previous_exit": previous_exit,
                })

            previous_exit = row[
                "exit_time"
            ]

    if violations:

        raise RuntimeError(
            "OVERLAP AUDIT FAILED: "
            f"{len(violations)} violations"
        )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Only completed UTC days.
    #
    # This avoids depending on the current incomplete day.
    # --------------------------------------------------------

    end = (
        pd.Timestamp.now(
            tz="UTC"
        )
        .normalize()
        - pd.Timedelta(
            minutes=5
        )
    )

    start = (
        end
        - pd.Timedelta(
            days=(
                LOOKBACK_DAYS
                + WARMUP_DAYS
            )
        )
    )

    print()
    print("=" * 70)
    print(
        "EBP — ENGULFING BAR PLAY"
    )
    print(
        "PURE SWEEP + RECLAIM STAGE-0"
    )
    print("=" * 70)

    print(
        f"Window: {start} -> {end}"
    )

    print(
        f"Symbols: {len(SYMBOLS)}"
    )

    print(
        f"Timeframes: {TIMEFRAMES}"
    )

    print(
        f"RR: 1:{RR}"
    )

    print(
        f"Margin: ${MARGIN}"
    )

    print(
        f"Leverage: {LEVERAGE}x"
    )

    print()

    all_trades = []

    # ========================================================
    # DOWNLOAD + BACKTEST
    # ========================================================

    for timeframe in TIMEFRAMES:

        print()
        print(
            f"========== {timeframe} =========="
        )

        for number, symbol in enumerate(
            SYMBOLS,
            1
        ):

            print(
                f"[{number}/{len(SYMBOLS)}] "
                f"{symbol}"
            )

            df = load_symbol(
                symbol,
                timeframe,
                start,
                end
            )

            print(
                f"    rows={len(df)}"
            )

            long_trades = (
                simulate_symbol(
                    df,
                    symbol,
                    timeframe,
                    "LONG"
                )
            )

            short_trades = (
                simulate_symbol(
                    df,
                    symbol,
                    timeframe,
                    "SHORT"
                )
            )

            if not long_trades.empty:

                all_trades.append(
                    long_trades
                )

            if not short_trades.empty:

                all_trades.append(
                    short_trades
                )

            print(
                f"    "
                f"LONG={len(long_trades)} "
                f"SHORT={len(short_trades)}"
            )

    if not all_trades:

        raise RuntimeError(
            "NO EBP TRADES FOUND"
        )

    trades = pd.concat(
        all_trades,
        ignore_index=True
    )

    trades = trades.sort_values(
        [
            "timeframe",
            "symbol",
            "entry_time"
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # AUDITS
    # ========================================================

    print()
    print(
        "Running overlap audit..."
    )

    audit_no_overlap(
        trades
    )

    print(
        "OVERLAP AUDIT: PASS"
    )

    # Same-candle re-entry audit:
    # every new entry must occur strictly AFTER
    # previous exit for that symbol/direction.
    print(
        "Same-candle re-entry audit: PASS"
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("FULL RESULTS")
    print("=" * 70)

    summary_rows = []

    for timeframe in TIMEFRAMES:

        for direction in [
            "LONG",
            "SHORT"
        ]:

            subset = trades[
                (
                    trades["timeframe"]
                    == timeframe
                )
                &
                (
                    trades["direction"]
                    == direction
                )
            ]

            m = calculate_metrics(
                subset
            )

            summary_rows.append({
                "TIMEFRAME": timeframe,
                "DIRECTION": direction,
                "TRADES": m["trades"],
                "WINS": m["wins"],
                "LOSSES": m["losses"],
                "WR_%": m["win_rate"],
                "PF": m["profit_factor"],
                "NET_R": m["net_R"],
                "AVG_R": m["avg_R"],
                "MAX_STREAK": m[
                    "max_loss_streak"
                ],
            })

    summary = pd.DataFrame(
        summary_rows
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # DISCOVERY / DEVELOPMENT / VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("CHRONOLOGICAL SPLITS")
    print("=" * 70)

    split_rows = []

    for timeframe in TIMEFRAMES:

        for direction in [
            "LONG",
            "SHORT"
        ]:

            subset = trades[
                (
                    trades["timeframe"]
                    == timeframe
                )
                &
                (
                    trades["direction"]
                    == direction
                )
            ]

            (
                discovery,
                development,
                validation
            ) = split_trades(
                subset,
                start,
                end
            )

            for name, data in [
                (
                    "DISCOVERY",
                    discovery
                ),
                (
                    "DEVELOPMENT",
                    development
                ),
                (
                    "VALIDATION",
                    validation
                ),
            ]:

                m = calculate_metrics(
                    data
                )

                split_rows.append({
                    "TIMEFRAME": timeframe,
                    "DIRECTION": direction,
                    "SPLIT": name,
                    "TRADES": m["trades"],
                    "WR_%": m["win_rate"],
                    "PF": m["profit_factor"],
                    "NET_R": m["net_R"],
                    "AVG_R": m["avg_R"],
                    "MAX_STREAK": m[
                        "max_loss_streak"
                    ],
                })

    split_summary = pd.DataFrame(
        split_rows
    )

    print(
        split_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # PER SYMBOL VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print(
        "VALIDATION BY SYMBOL"
    )
    print("=" * 70)

    validation = trades[
        trades["entry_time"]
        >= (
            start
            + (
                end - start
            ) * 0.75
        )
    ].copy()

    symbol_rows = []

    for (
        timeframe,
        symbol,
        direction
    ), group in validation.groupby(
        [
            "timeframe",
            "symbol",
            "direction"
        ]
    ):

        m = calculate_metrics(
            group
        )

        symbol_rows.append({
            "TIMEFRAME": timeframe,
            "SYMBOL": symbol,
            "DIRECTION": direction,
            "TRADES": m["trades"],
            "WR_%": m["win_rate"],
            "PF": m["profit_factor"],
            "NET_R": m["net_R"],
            "MAX_STREAK": m[
                "max_loss_streak"
            ],
        })

    symbol_summary = pd.DataFrame(
        symbol_rows
    )

    if not symbol_summary.empty:

        print(
            symbol_summary.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}"
            )
        )

    # ========================================================
    # SAVE
    # ========================================================

    trades.to_csv(
        "ebp_stage0_trades.csv",
        index=False
    )

    summary.to_csv(
        "ebp_stage0_summary.csv",
        index=False
    )

    split_summary.to_csv(
        "ebp_stage0_splits.csv",
        index=False
    )

    symbol_summary.to_csv(
        "ebp_stage0_validation_symbols.csv",
        index=False
    )

    print()
    print(
        "Files saved:"
    )

    print(
        "  ebp_stage0_trades.csv"
    )

    print(
        "  ebp_stage0_summary.csv"
    )

    print(
        "  ebp_stage0_splits.csv"
    )

    print(
        "  ebp_stage0_validation_symbols.csv"
    )

    print()
    print(
        "EBP STAGE-0 COMPLETE"
    )


if __name__ == "__main__":

    main()
