import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "ccxt"]
    )
    import ccxt


# ============================================================
# HUNTER-V129
# V123 EXACT-BEHAVIOR AUDIT
#
# PURPOSE:
# Reproduce V123 before making ANY integrity modification.
#
# IMPORTANT:
# This version intentionally preserves the V123 execution model.
# It is an AUDIT version, NOT the final live-safe version.
#
# ============================================================


# ============================================================
# EXACT V123 EXCHANGE CONNECTION
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True,
    "timeout": 20000
})


# ============================================================
# EXACT V123 SYMBOLS
# ============================================================

SYMBOLS = {
    "CRV": "CRV/USDT",
    "DOGE": "DOGE/USDT",
    "ICP": "ICP/USDT",
    "APT": "APT/USDT",
    "PENDLE": "PENDLE/USDT",
    "WIF": "WIF/USDT",
    "ONDO": "ONDO/USDT",
    "NEAR": "NEAR/USDT",
    "SEI": "SEI/USDT",
    "XLM": "XLM/USDT",
    "ADA": "ADA/USDT",
    "BNB": "BNB/USDT",
    "SOL": "SOL/USDT",
    "ETH": "ETH/USDT"
}


# ============================================================
# V123 PARAMETERS
# ============================================================

TIMEFRAME_BASE = "15m"

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

TRADE_MARGIN = 100.0
LEVERAGE = 50.0

DAYS = 365

# V123 processed quarters separately.
# Keep this behavior for audit.
QUARTER_WARMUP_DAYS = 15


# ============================================================
# DATA FETCH
#
# THIS IS THE ORIGINAL V123 STRUCTURE
# ============================================================

def fetch_chunk_data(
    lbank_symbol,
    start_dt,
    end_dt
):

    since_ts = int(
        (
            start_dt -
            timedelta(days=15)
        ).timestamp() * 1000
    )

    end_ts = int(
        end_dt.timestamp() * 1000
    )

    all_ohlcv = []

    current_since = since_ts

    try:

        while current_since < end_ts:

            batch = exchange.fetch_ohlcv(
                lbank_symbol,
                timeframe="15m",
                since=current_since,
                limit=1000
            )

            if not batch:
                break

            all_ohlcv.extend(batch)

            last_ts = batch[-1][0]

            if last_ts <= current_since:
                break

            current_since = last_ts + 1

            if len(batch) < 1000:
                break

            if last_ts >= end_ts:
                break

            time.sleep(0.2)

    except Exception as e:

        print(
            f"  ERROR fetching "
            f"{lbank_symbol}: {e}"
        )

        return None

    if not all_ohlcv:
        return None

    df = pd.DataFrame(
        all_ohlcv,
        columns=[
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    )

    df["Date"] = pd.to_datetime(
        df["Timestamp"],
        unit="ms"
    )

    df = df[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    ]

    df.dropna(
        inplace=True
    )

    df.drop_duplicates(
        subset=["Date"],
        keep="last",
        inplace=True
    )

    df.sort_values(
        "Date",
        inplace=True
    )

    df.set_index(
        "Date",
        inplace=True
    )

    df = df[
        (df.index >= start_dt) &
        (df.index <= end_dt)
    ]

    return df


# ============================================================
# PREPARE DATA
#
# EXACT V123 CALCULATIONS
# ============================================================

def prepare_data(
    df_15m
):

    # --------------------------------------------------------
    # 4H
    # --------------------------------------------------------

    df_4h = df_15m.resample(
        "4h"
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    df_4h["EMA_50"] = (
        df_4h["Close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    df_4h["EMA_200"] = (
        df_4h["Close"]
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )

    df_4h["Regime_Bullish"] = (
        (df_4h["Close"] > df_4h["EMA_200"]) &
        (df_4h["EMA_50"] > df_4h["EMA_200"])
    )

    df_4h["Regime_Bearish"] = (
        (df_4h["Close"] < df_4h["EMA_200"]) &
        (df_4h["EMA_50"] < df_4h["EMA_200"])
    )


    # --------------------------------------------------------
    # 1H
    #
    # EXACT V123:
    # rolling(5, center=True)
    # --------------------------------------------------------

    df_1h = df_15m.resample(
        "1h"
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    df_1h["Swing_High"] = (
        df_1h["High"]
        .rolling(
            5,
            center=True
        )
        .max()
    )

    df_1h["Swing_Low"] = (
        df_1h["Low"]
        .rolling(
            5,
            center=True
        )
        .min()
    )

    df_1h["ATR"] = (
        (
            df_1h["High"] -
            df_1h["Low"]
        )
        .rolling(14)
        .mean()
    )


    # --------------------------------------------------------
    # 15M
    #
    # EXACT V123
    # --------------------------------------------------------

    df_15m = df_15m.copy()

    df_15m["ATR"] = (
        (
            df_15m["High"] -
            df_15m["Low"]
        )
        .rolling(14)
        .mean()
    )

    df_15m["Body"] = (
        df_15m["Close"] -
        df_15m["Open"]
    ).abs()

    df_15m["Avg_Body"] = (
        df_15m["Body"]
        .rolling(20)
        .mean()
    )

    return (
        df_15m,
        df_1h,
        df_4h
    )


# ============================================================
# V123 TRADE ENGINE
#
# THIS IS INTENTIONALLY THE ORIGINAL EXECUTION MODEL
# ============================================================

def run_v123_engine(
    symbol,
    df_15,
    df_1h,
    df_4h,
    start_dt,
    end_dt
):

    trades = []

    for i in range(
        50,
        len(df_15)
    ):

        t_curr = df_15.index[i]

        if t_curr < start_dt:
            continue

        if t_curr > end_dt:
            break

        # ----------------------------------------------------
        # CURRENT CANDLE
        # ----------------------------------------------------

        c_row = df_15.iloc[i]

        # ----------------------------------------------------
        # PREVIOUS CANDLE
        # ----------------------------------------------------

        p_row = df_15.iloc[i - 1]


        # ----------------------------------------------------
        # V123 HTF SELECTION
        #
        # IMPORTANT:
        # We intentionally preserve <= t_curr.
        #
        # This is one of the things we will audit later.
        # ----------------------------------------------------

        h_sub = df_1h[
            df_1h.index <= t_curr
        ]

        h_4sub = df_4h[
            df_4h.index <= t_curr
        ]

        if len(h_sub) < 10:
            continue

        if len(h_4sub) < 1:
            continue


        regime_bull = bool(
            h_4sub.iloc[-1][
                "Regime_Bullish"
            ]
        )

        regime_bear = bool(
            h_4sub.iloc[-1][
                "Regime_Bearish"
            ]
        )


        # ----------------------------------------------------
        # V123 LIQUIDITY LEVELS
        # ----------------------------------------------------

        recent_lows = (
            h_sub["Low"]
            .iloc[-10:-2]
        )

        recent_highs = (
            h_sub["High"]
            .iloc[-10:-2]
        )

        if len(recent_lows) == 0:
            continue

        if len(recent_highs) == 0:
            continue

        min_support = (
            recent_lows.min()
        )

        max_resistance = (
            recent_highs.max()
        )


        # ----------------------------------------------------
        # V123 SWEEP
        # ----------------------------------------------------

        sweep_low = (
            p_row["Low"] < min_support
        ) and (
            p_row["Close"] > min_support
        )

        sweep_high = (
            p_row["High"] > max_resistance
        ) and (
            p_row["Close"] < max_resistance
        )


        # ----------------------------------------------------
        # V123 DISPLACEMENT
        #
        # IMPORTANT:
        # displacement is CURRENT candle c_row.
        # ----------------------------------------------------

        displacement_up = (
            c_row["Close"] >
            c_row["Open"]
        ) and (
            c_row["Body"] >
            2.0 *
            c_row["Avg_Body"]
        )

        displacement_down = (
            c_row["Close"] <
            c_row["Open"]
        ) and (
            c_row["Body"] >
            2.0 *
            c_row["Avg_Body"]
        )


        # ----------------------------------------------------
        # V123 SIGNAL
        # ----------------------------------------------------

        valid_long = (
            regime_bull
            and
            sweep_low
            and
            displacement_up
        )

        valid_short = (
            regime_bear
            and
            sweep_high
            and
            displacement_down
        )

        if not valid_long and not valid_short:
            continue


        if valid_long:

            side = "LONG"

            entry_price = (
                c_row["Open"] *
                (1.0 + SLIPPAGE)
            )

        else:

            side = "SHORT"

            entry_price = (
                c_row["Open"] *
                (1.0 - SLIPPAGE)
            )


        # ----------------------------------------------------
        # ATR
        # ----------------------------------------------------

        atr = c_row["ATR"]

        if not np.isfinite(atr):
            continue

        if atr <= 0:
            continue


        # ----------------------------------------------------
        # V123 SL / TP
        # ----------------------------------------------------

        if side == "LONG":

            sl = (
                entry_price -
                1.5 * atr
            )

            tp = (
                entry_price +
                3.0 * atr
            )

            be_trigger = (
                entry_price +
                1.5 * atr
            )

        else:

            sl = (
                entry_price +
                1.5 * atr
            )

            tp = (
                entry_price -
                3.0 * atr
            )

            be_trigger = (
                entry_price -
                1.5 * atr
            )


        current_sl = sl

        be_active = False

        outcome = "LOSS"

        exit_price = sl

        exit_index = None


        # ----------------------------------------------------
        # EXACT V123 TRADE SCAN
        #
        # max i + 35
        # ----------------------------------------------------

        end_j = min(
            i + 35,
            len(df_15)
        )

        for j in range(
            i + 1,
            end_j
        ):

            fut = df_15.iloc[j]

            high = fut["High"]
            low = fut["Low"]


            # =================================================
            # LONG
            # =================================================

            if side == "LONG":

                # ------------------------------------------------
                # BE
                # ------------------------------------------------

                if (
                    not be_active
                    and
                    high >= be_trigger
                ):

                    current_sl = (
                        entry_price
                    )

                    be_active = True


                hit_sl = (
                    low <= current_sl
                )

                hit_tp = (
                    high >= tp
                )


                # ------------------------------------------------
                # CONSERVATIVE SL PRIORITY
                # ------------------------------------------------

                if hit_sl:

                    if be_active:

                        outcome = "BE"
                        exit_price = (
                            entry_price
                        )

                    else:

                        outcome = "LOSS"
                        exit_price = (
                            current_sl
                        )

                    exit_index = j

                    break


                if hit_tp:

                    outcome = "WIN"
                    exit_price = tp
                    exit_index = j

                    break


            # =================================================
            # SHORT
            # =================================================

            else:

                if (
                    not be_active
                    and
                    low <= be_trigger
                ):

                    current_sl = (
                        entry_price
                    )

                    be_active = True


                hit_sl = (
                    high >= current_sl
                )

                hit_tp = (
                    low <= tp
                )


                if hit_sl:

                    if be_active:

                        outcome = "BE"
                        exit_price = (
                            entry_price
                        )

                    else:

                        outcome = "LOSS"
                        exit_price = (
                            current_sl
                        )

                    exit_index = j

                    break


                if hit_tp:

                    outcome = "WIN"
                    exit_price = tp
                    exit_index = j

                    break


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # V123 behavior:
        # If no SL/TP hit during scan,
        # outcome remains LOSS and exit_price remains SL.
        #
        # We intentionally preserve this for AUDIT.
        # ----------------------------------------------------

        # ----------------------------------------------------
        # PNL
        # ----------------------------------------------------

        notional = (
            TRADE_MARGIN *
            LEVERAGE
        )

        if side == "LONG":

            price_ret = (
                exit_price -
                entry_price
            ) / entry_price

        else:

            price_ret = (
                entry_price -
                exit_price
            ) / entry_price

        dollar_pnl = (
            notional *
            price_ret
        ) - (
            notional *
            FEE_RATE *
            2
        )


        trades.append({
            "Timestamp": t_curr,
            "ExitTimestamp":
                (
                    df_15.index[
                        exit_index
                    ]
                    if exit_index is not None
                    else None
                ),
            "Symbol": symbol,
            "Side": side,
            "Outcome": outcome,
            "Dollar_PnL":
                dollar_pnl,
            "Entry_Price":
                entry_price,
            "Exit_Price":
                exit_price,
            "Entry_Index": i,
            "Exit_Index": exit_index
        })


    return trades


# ============================================================
# QUARTER GENERATOR
#
# V123 PROCESSED THE YEAR IN QUARTER BLOCKS.
# ============================================================

def make_quarters(
    start_dt,
    end_dt
):

    quarters = []

    current = start_dt

    while current < end_dt:

        quarter_end = min(
            current +
            timedelta(days=90),
            end_dt
        )

        quarters.append(
            (
                current,
                quarter_end
            )
        )

        current = (
            quarter_end +
            timedelta(minutes=15)
        )

    return quarters


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print(
        "HUNTER-V129 — V123 EXACT-BEHAVIOR AUDIT"
    )
    print("=" * 80)

    now = datetime.now()

    start_dt = (
        now -
        timedelta(days=DAYS)
    )

    end_dt = now

    print(
        f"Period: "
        f"{start_dt} -> {end_dt}"
    )

    print()
    print(
        "THIS IS NOT THE FINAL SAFE ENGINE."
    )

    print(
        "PURPOSE: REPRODUCE V123."
    )

    print()
    print(
        "Data:       V123"
    )

    print(
        "Symbols:    V123"
    )

    print(
        "Sweep:      V123"
    )

    print(
        "Displace:   V123"
    )

    print(
        "Entry:      V123"
    )

    print(
        "HTF:        V123"
    )

    print(
        "SL/TP:      V123"
    )

    print(
        "BE:         V123"
    )

    print(
        "Scan:       V123"
    )

    print(
        "Timeout:    V123 behavior"
    )

    print("=" * 80)


    all_trades = []


    # ========================================================
    # EACH SYMBOL
    # ========================================================

    for symbol, lbank_symbol in SYMBOLS.items():

        print()
        print("-" * 80)
        print(
            f"{symbol} -> {lbank_symbol}"
        )
        print("-" * 80)


        # ----------------------------------------------------
        # Fetch EXACT V123-style range
        # ----------------------------------------------------

        df = fetch_chunk_data(
            lbank_symbol,
            start_dt,
            end_dt
        )

        if df is None:

            print(
                "  NO DATA"
            )

            continue


        print(
            f"  Candles: {len(df)}"
        )


        if len(df) < 100:

            print(
                "  INSUFFICIENT DATA"
            )

            continue


        # ----------------------------------------------------
        # Prepare
        # ----------------------------------------------------

        (
            df_15,
            df_1h,
            df_4h
        ) = prepare_data(df)


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Run continuously exactly as V123.
        # ----------------------------------------------------

        symbol_trades = run_v123_engine(
            symbol,
            df_15,
            df_1h,
            df_4h,
            start_dt,
            end_dt
        )


        all_trades.extend(
            symbol_trades
        )


        print(
            f"  Trades: "
            f"{len(symbol_trades)}"
        )


    # ========================================================
    # NO TRADES
    # ========================================================

    if not all_trades:

        print()
        print("=" * 80)
        print(
            "NO TRADES GENERATED"
        )
        print("=" * 80)

        sys.exit(0)


    # ========================================================
    # DATAFRAME
    # ========================================================

    trades_df = pd.DataFrame(
        all_trades
    )

    trades_df.sort_values(
        "Timestamp",
        inplace=True
    )

    trades_df.reset_index(
        drop=True,
        inplace=True
    )


    # ========================================================
    # PERFORMANCE
    # ========================================================

    total_trades = len(
        trades_df
    )

    wins = trades_df[
        trades_df["Outcome"] == "WIN"
    ]

    losses = trades_df[
        trades_df["Outcome"] == "LOSS"
    ]

    bes = trades_df[
        trades_df["Outcome"] == "BE"
    ]


    win_rate = (
        len(wins) /
        total_trades *
        100
    )

    loss_rate = (
        len(losses) /
        total_trades *
        100
    )

    be_rate = (
        len(bes) /
        total_trades *
        100
    )


    net_pnl = float(
        trades_df[
            "Dollar_PnL"
        ].sum()
    )


    gross_profit = float(
        wins[
            "Dollar_PnL"
        ].sum()
    )


    gross_loss = abs(
        float(
            losses[
                "Dollar_PnL"
            ].sum()
        )
    )


    if gross_loss > 0:

        profit_factor = (
            gross_profit /
            gross_loss
        )

    else:

        profit_factor = float(
            "inf"
        )


    avg_win = (
        float(
            wins[
                "Dollar_PnL"
            ].mean()
        )
        if len(wins)
        else 0.0
    )


    avg_loss = (
        float(
            losses[
                "Dollar_PnL"
            ].mean()
        )
        if len(losses)
        else 0.0
    )


    # ========================================================
    # DRAWDOWN
    # ========================================================

    trades_df[
        "Cumulative_PnL"
    ] = (
        trades_df[
            "Dollar_PnL"
        ].cumsum()
    )

    trades_df[
        "Peak"
    ] = (
        trades_df[
            "Cumulative_PnL"
        ].cummax()
    )

    trades_df[
        "Drawdown"
    ] = (
        trades_df[
            "Cumulative_PnL"
        ]
        -
        trades
