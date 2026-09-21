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
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt


# ============================================================
# HUNTER-V128
# V123 STRATEGY + CAUSAL BACKTEST ENGINE
#
# DATA CONNECTION:
# EXACTLY PRESERVED FROM V123
#
# Strategy:
#   4H regime
#   1H liquidity sweep
#   15m displacement
#
# Execution:
#   CLOSED signal candle
#   NEXT 15m OPEN entry
#
# Integrity:
#   NO LOOKAHEAD
#   NO CENTERED ROLLING
#   NO OVERLAPPING POSITION PER SYMBOL
#   NO ARTIFICIAL TIMEOUT LOSS
#   COMPLETED HTF CANDLES ONLY
#
# RR:
#   1 : 2 nominal
# ============================================================


# ============================================================
# EXACT V123 DATA CONNECTION
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True,
    "timeout": 20000
})


# ============================================================
# EXACT V123 SYMBOL FORMAT
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
# PARAMETERS
# ============================================================

TIMEFRAME_BASE = "15m"

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

TRADE_MARGIN = 100.0
LEVERAGE = 50.0

DAYS = 365

# V123 strategy parameters
SWEEP_LOOKBACK = 10
DISPLACEMENT_MULTIPLIER = 2.0

ATR_PERIOD = 14
BODY_PERIOD = 20

SL_ATR = 1.5
TP_ATR = 3.0
BE_ATR = 1.5

# Extra history required for 4H EMA200
WARMUP_DAYS = 60


# ============================================================
# DATA FETCH
#
# THIS IS DELIBERATELY KEPT COMPATIBLE WITH V123
# ============================================================

def fetch_chunk_data(lbank_symbol, start_dt, end_dt):

    since_ts = int(
        (start_dt - timedelta(days=15)).timestamp() * 1000
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
                timeframe=TIMEFRAME_BASE,
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
            f"Error fetching {lbank_symbol}: {e}"
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

    # EXACT V123 timestamp handling
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

    df.reset_index(
        drop=True,
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
# INDICATORS
# ============================================================

def prepare_data(df_15m):

    df_15m = df_15m.copy()

    # --------------------------------------------------------
    # 15M
    # --------------------------------------------------------

    df_15m["Body"] = (
        df_15m["Close"] -
        df_15m["Open"]
    ).abs()

    df_15m["Avg_Body"] = (
        df_15m["Body"]
        .rolling(
            BODY_PERIOD,
            min_periods=BODY_PERIOD
        )
        .mean()
    )

    df_15m["ATR"] = (
        (df_15m["High"] - df_15m["Low"])
        .rolling(
            ATR_PERIOD,
            min_periods=ATR_PERIOD
        )
        .mean()
    )

    # --------------------------------------------------------
    # 1H
    #
    # Same V123 calculation.
    # No center=True.
    # --------------------------------------------------------

    df_1h = df_15m.resample("1h").agg({
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
            min_periods=5
        )
        .max()
    )

    df_1h["Swing_Low"] = (
        df_1h["Low"]
        .rolling(
            5,
            min_periods=5
        )
        .min()
    )

    df_1h["ATR"] = (
        (df_1h["High"] - df_1h["Low"])
        .rolling(
            14,
            min_periods=14
        )
        .mean()
    )

    # --------------------------------------------------------
    # 4H
    # --------------------------------------------------------

    df_4h = df_15m.resample("4h").agg({
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

    return df_15m, df_1h, df_4h


# ============================================================
# GET SIGNAL
#
# IMPORTANT:
# i = NEXT EXECUTION CANDLE
# i-1 = CLOSED SIGNAL CANDLE
# ============================================================

def get_signal(
    df_15m,
    df_1h,
    df_4h,
    i
):

    if i < 50:
        return None

    t_curr = df_15m.index[i]

    # Current candle = execution candle
    c_row = df_15m.iloc[i]

    # Previous candle = fully closed signal candle
    p_row = df_15m.iloc[i - 1]

    # --------------------------------------------------------
    # ONLY COMPLETED HTF CANDLES
    #
    # Important:
    # A 4H candle opened at 08:00 is NOT considered complete
    # at 09:00, 10:00, etc.
    #
    # We therefore use the last HTF candle whose OPEN is
    # strictly earlier than current candle and then verify
    # its closing boundary.
    # --------------------------------------------------------

    h_sub = df_1h[
        df_1h.index < t_curr
    ]

    h_4sub = df_4h[
        df_4h.index < t_curr
    ]

    if len(h_sub) < 10:
        return None

    if len(h_4sub) < 10:
        return None

    # --------------------------------------------------------
    # Make sure selected 1H / 4H candle is completed.
    # --------------------------------------------------------

    completed_1h = h_sub[
        (h_sub.index + pd.Timedelta(hours=1))
        <= t_curr
    ]

    completed_4h = h_4sub[
        (h_4sub.index + pd.Timedelta(hours=4))
        <= t_curr
    ]

    if len(completed_1h) < 10:
        return None

    if len(completed_4h) < 10:
        return None

    latest_4h = completed_4h.iloc[-1]

    regime_bull = bool(
        latest_4h["Regime_Bullish"]
    )

    regime_bear = bool(
        latest_4h["Regime_Bearish"]
    )

    # --------------------------------------------------------
    # V123 LIQUIDITY LEVELS
    # --------------------------------------------------------

    recent_lows = (
        completed_1h["Low"]
        .iloc[-10:-1]
    )

    recent_highs = (
        completed_1h["High"]
        .iloc[-10:-1]
    )

    if len(recent_lows) == 0:
        return None

    if len(recent_highs) == 0:
        return None

    min_support = recent_lows.min()

    max_resistance = recent_highs.max()

    # --------------------------------------------------------
    # EXACT V123 SWEEP LOGIC
    #
    # Sweep is on p_row.
    # NOT pp_row.
    # --------------------------------------------------------

    sweep_low = (
        p_row["Low"] < min_support and
        p_row["Close"] > min_support
    )

    sweep_high = (
        p_row["High"] > max_resistance and
        p_row["Close"] < max_resistance
    )

    # --------------------------------------------------------
    # EXACT V123 DISPLACEMENT LOGIC
    #
    # p_row is fully closed.
    # --------------------------------------------------------

    displacement_up = (
        p_row["Close"] > p_row["Open"]
        and
        p_row["Body"] >
        DISPLACEMENT_MULTIPLIER *
        p_row["Avg_Body"]
    )

    displacement_down = (
        p_row["Close"] < p_row["Open"]
        and
        p_row["Body"] >
        DISPLACEMENT_MULTIPLIER *
        p_row["Avg_Body"]
    )

    valid_long = (
        regime_bull
        and sweep_low
        and displacement_up
    )

    valid_short = (
        regime_bear
        and sweep_high
        and displacement_down
    )

    if valid_long:
        return "LONG"

    if valid_short:
        return "SHORT"

    return None


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    df_15m,
    entry_index,
    side
):

    entry_row = df_15m.iloc[
        entry_index
    ]

    signal_row = df_15m.iloc[
        entry_index - 1
    ]

    # --------------------------------------------------------
    # ENTRY
    # --------------------------------------------------------

    if side == "LONG":

        entry_price = (
            entry_row["Open"] *
            (1.0 + SLIPPAGE)
        )

    else:

        entry_price = (
            entry_row["Open"] *
            (1.0 - SLIPPAGE)
        )

    atr = signal_row["ATR"]

    if not np.isfinite(atr):
        return None

    if atr <= 0:
        return None

    # --------------------------------------------------------
    # FIXED 1:2
    # --------------------------------------------------------

    if side == "LONG":

        sl = (
            entry_price -
            SL_ATR * atr
        )

        tp = (
            entry_price +
            TP_ATR * atr
        )

        be_trigger = (
            entry_price +
            BE_ATR * atr
        )

    else:

        sl = (
            entry_price +
            SL_ATR * atr
        )

        tp = (
            entry_price -
            TP_ATR * atr
        )

        be_trigger = (
            entry_price -
            BE_ATR * atr
        )

    current_sl = sl

    breakeven_activated = False

    # --------------------------------------------------------
    # NO ARTIFICIAL TIMEOUT
    #
    # Continue until SL / TP or dataset end.
    # --------------------------------------------------------

    for j in range(
        entry_index,
        len(df_15m)
    ):

        fut = df_15m.iloc[j]

        high = fut["High"]
        low = fut["Low"]

        # ----------------------------------------------------
        # LONG
        # ----------------------------------------------------

        if side == "LONG":

            # BE activation
            if (
                not breakeven_activated
                and
                high >= be_trigger
            ):

                current_sl = entry_price

                breakeven_activated = True

            hit_sl = (
                low <= current_sl
            )

            hit_tp = (
                high >= tp
            )

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        else:

            if (
                not breakeven_activated
                and
                low <= be_trigger
            ):

                current_sl = entry_price

                breakeven_activated = True

            hit_sl = (
                high >= current_sl
            )

            hit_tp = (
                low <= tp
            )

        # ----------------------------------------------------
        # BOTH HIT
        #
        # Conservative assumption.
        # ----------------------------------------------------

        if hit_sl and hit_tp:

            if breakeven_activated:

                return {
                    "exit_index": j,
                    "exit_timestamp":
                        df_15m.index[j],
                    "Outcome": "BE",
                    "Entry_Price":
                        entry_price,
                    "Exit_Price":
                        entry_price,
                    "BE": True
                }

            return {
                "exit_index": j,
                "exit_timestamp":
                    df_15m.index[j],
                "Outcome": "LOSS",
                "Entry_Price":
                    entry_price,
                "Exit_Price":
                    current_sl,
                "BE": False
            }

        # ----------------------------------------------------
        # SL
        # ----------------------------------------------------

        if hit_sl:

            if breakeven_activated:

                return {
                    "exit_index": j,
                    "exit_timestamp":
                        df_15m.index[j],
                    "Outcome": "BE",
                    "Entry_Price":
                        entry_price,
                    "Exit_Price":
                        entry_price,
                    "BE": True
                }

            return {
                "exit_index": j,
                "exit_timestamp":
                    df_15m.index[j],
                "Outcome": "LOSS",
                "Entry_Price":
                    entry_price,
                "Exit_Price":
                    current_sl,
                "BE": False
            }

        # ----------------------------------------------------
        # TP
        # ----------------------------------------------------

        if hit_tp:

            return {
                "exit_index": j,
                "exit_timestamp":
                    df_15m.index[j],
                "Outcome": "WIN",
                "Entry_Price":
                    entry_price,
                "Exit_Price":
                    tp,
                "BE": breakeven_activated
            }

    # --------------------------------------------------------
    # DATASET ENDED
    #
    # This is NOT a LOSS.
    # This is NOT a WIN.
    # --------------------------------------------------------

    return {
        "exit_index": None,
        "exit_timestamp": None,
        "Outcome": "OPEN_AT_END",
        "Entry_Price": entry_price,
        "Exit_Price": np.nan,
        "BE": breakeven_activated
    }


# ============================================================
# PNL
# ============================================================

def calculate_pnl(
    side,
    entry_price,
    exit_price
):

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

    gross = (
        notional *
        price_ret
    )

    fees = (
        notional *
        FEE_RATE *
        2.0
    )

    return gross - fees


# ============================================================
# ONE SYMBOL
# ============================================================

def run_symbol_backtest(
    symbol,
    df_15m,
    df_1h,
    df_4h,
    start_dt,
    end_dt
):

    trades = []

    i = 50

    while i < len(df_15m):

        t_curr = df_15m.index[i]

        if t_curr < start_dt:

            i += 1

            continue

        if t_curr > end_dt:
            break

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        side = get_signal(
            df_15m,
            df_1h,
            df_4h,
            i
        )

        if side is None:

            i += 1

            continue

        # ----------------------------------------------------
        # SIMULATE
        # ----------------------------------------------------

        trade = simulate_trade(
            df_15m,
            i,
            side
        )

        if trade is None:

            i += 1

            continue

        outcome = trade["Outcome"]

        # ----------------------------------------------------
        # OPEN AT END
        # ----------------------------------------------------

        if outcome == "OPEN_AT_END":

            trades.append({
                "Timestamp": t_curr,
                "ExitTimestamp": None,
                "Symbol": symbol,
                "Side": side,
                "Outcome": "OPEN_AT_END",
                "Dollar_PnL": 0.0,
                "Entry_Price":
                    trade["Entry_Price"],
                "Exit_Price": np.nan,
                "Month":
                    t_curr.strftime("%Y-%m")
            })

            break

        # ----------------------------------------------------
        # CLOSED TRADE
        # ----------------------------------------------------

        pnl = calculate_pnl(
            side,
            trade["Entry_Price"],
            trade["Exit_Price"]
        )

        trades.append({
            "Timestamp": t_curr,
            "ExitTimestamp":
                trade["exit_timestamp"],
            "Symbol": symbol,
            "Side": side,
            "Outcome": outcome,
            "Dollar_PnL": pnl,
            "Entry_Price":
                trade["Entry_Price"],
            "Exit_Price":
                trade["Exit_Price"],
            "Month":
                t_curr.strftime("%Y-%m")
        })

        # ----------------------------------------------------
        # REAL OVERLAP LOCK
        #
        # Next signal can only happen AFTER the exit candle.
        # ----------------------------------------------------

        exit_index = trade[
            "exit_index"
        ]

        if exit_index is None:
            break

        i = exit_index + 1

    return trades


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 78)
    print(
        "HUNTER-V128 — "
        "V123 STRATEGY / CAUSAL BACKTEST"
    )
    print("=" * 78)

    now = datetime.now()

    start_dt = (
        now -
        timedelta(days=DAYS)
    )

    end_dt = now

    # --------------------------------------------------------
    # Warmup data.
    #
    # IMPORTANT:
    # We fetch warmup separately but use the SAME V123
    # fetch function.
    # --------------------------------------------------------

    data_start = (
        start_dt -
        timedelta(days=WARMUP_DAYS)
    )

    print(
        f"Backtest: "
        f"{start_dt.strftime('%Y-%m-%d')} -> "
        f"{end_dt.strftime('%Y-%m-%d')}"
    )

    print(
        f"Warmup: {WARMUP_DAYS} days"
    )

    print(
        "Data engine: V123 COMPATIBLE"
    )

    print(
        "Entry: NEXT 15m OPEN"
    )

    print(
        "RR: 1:2"
    )

    print(
        "Timeout: DISABLED"
    )

    print(
        "Overlap Lock: ENABLED"
    )

    print("=" * 78)

    all_trades = []

    # --------------------------------------------------------
    # EACH SYMBOL
    # --------------------------------------------------------

    for symbol, lbank_symbol in SYMBOLS.items():

        print()
        print("-" * 78)

        print(
            f"{symbol} -> {lbank_symbol}"
        )

        print("-" * 78)

        df_15m = fetch_chunk_data(
            lbank_symbol,
            data_start,
            end_dt
        )

        if df_15m is None:

            print(
                "  No data."
            )

            continue

        if len(df_15m) < 1000:

            print(
                f"  Insufficient data: "
                f"{len(df_15m)}"
            )

            continue

        print(
            f"  Raw candles: "
            f"{len(df_15m)}"
        )

        # ----------------------------------------------------
        # INDICATORS
        # ----------------------------------------------------

        (
            df_15m,
            df_1h,
            df_4h
        ) = prepare_data(
            df_15m
        )

        # ----------------------------------------------------
        # Restrict 15m execution data to actual period
        #
        # HTF datasets remain available with warmup.
        # ----------------------------------------------------

        execution_df = df_15m[
            (df_15m.index >= start_dt) &
            (df_15m.index <= end_dt)
        ].copy()

        if len(execution_df) < 50:

            print(
                "  Not enough execution candles."
            )

            continue

        # ----------------------------------------------------
        # Run
        # ----------------------------------------------------

        symbol_trades = run_symbol_backtest(
            symbol,
            execution_df,
            df_1h,
            df_4h,
            start_dt,
            end_dt
        )

        all_trades.extend(
            symbol_trades
        )

        print(
            f"  Trades found: "
            f"{len(symbol_trades)}"
        )


    # ========================================================
    # NO TRADES
    # ========================================================

    if not all_trades:

        print()
        print("=" * 78)
        print(
            "NO TRADES GENERATED."
        )
        print("=" * 78)

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
    # CLOSED TRADES
    # ========================================================

    closed_df = trades_df[
        trades_df["Outcome"] != "OPEN_AT_END"
    ].copy()

    open_df = trades_df[
        trades_df["Outcome"] == "OPEN_AT_END"
    ].copy()

    n_all = len(trades_df)

    n_closed = len(
        closed_df
    )

    n_open = len(
        open_df
    )


    if n_closed == 0:

        print()
        print(
            "No closed trades."
        )

        print(
            f"Open at end: {n_open}"
        )

        sys.exit(0)


    # ========================================================
    # PERFORMANCE
    # ========================================================

    wins = closed_df[
        closed_df["Outcome"] == "WIN"
    ]

    losses = closed_df[
        closed_df["Outcome"] == "LOSS"
    ]

    bes = closed_df[
        closed_df["Outcome"] == "BE"
    ]

    win_rate = (
        len(wins) /
        n_closed *
        100.0
    )

    loss_rate = (
        len(losses) /
        n_closed *
        100.0
    )

    be_rate = (
        len(bes) /
        n_closed *
        100.0
    )

    net_pnl = float(
        closed_df[
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
        if len(wins) > 0
        else 0.0
    )

    avg_loss = (
        float(
            losses[
                "Dollar_PnL"
            ].mean()
        )
        if len(losses) > 0
        else 0.0
    )


    # ========================================================
    # DRAWDOWN
    # ========================================================

    closed_df[
        "Cumulative_PnL"
    ] = (
        closed_df[
            "Dollar_PnL"
        ].cumsum()
    )

    closed_df[
        "Peak"
    ] = (
        closed_df[
            "Cumulative_PnL"
        ].cummax()
    )

    closed_df[
        "Drawdown"
    ] = (
        closed_df[
            "Cumulative_PnL"
        ]
        -
        closed_df[
            "Peak"
        ]
    )

    max_dd = float(
        closed_df[
            "Drawdown"
        ].min()
    )


    # ========================================================
    # CONSECUTIVE LOSSES
    # ========================================================

    streaks = []

    current_streak = 0

    for outcome in closed_df[
        "Outcome"
    ]:

        if outcome == "LOSS":

            current_streak += 1

        else:

            if current_streak > 0:

                streaks.append(
                    current_streak
                )

            current_streak = 0

    if current_streak > 0:

        streaks.append(
            current_streak
        )

    max_streak = (
        max(streaks)
        if streaks
        else 0
    )


    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 78)
    print(
        "===== HUNTER-V128 "
        "— 1 YEAR RESULT ====="
    )
    print("=" * 78)

    print(
        f"Total Signals/Trades: "
        f"{n_all}"
    )

    print(
        f"Closed Trades:        "
        f"{n_closed}"
    )

    print(
        f"Open At End:          "
        f"{n_open}"
    )

    print(
        f"Trades / Month:       "
        f"{n_closed / 12.0:.1f}"
    )

    print(
        f"Win Rate:             "
        f"{win_rate:.2f}%"
    )

    print(
        f"Loss Rate:            "
        f"{loss_rate:.2f}%"
    )

    print(
        f"Break Even Rate:      "
        f"{be_rate:.2f}%"
    )

    print(
        f"Net PnL:              "
        f"${net_pnl:,.2f}"
    )

    print(
        f"Profit Factor:        "
        f"{profit_factor:.2f}"
    )

    print(
        f"Average Win:          "
        f"${avg_win:,.2f}"
    )

    print(
        f"Average Loss:         "
        f"${avg_loss:,.2f}"
    )

    print(
        f"Max Drawdown:         "
        f"${max_dd:,.2f}"
    )

    print(
        f"Maximum Consecutive "
        f"Losses: {max_streak}"
    )

    print("-" * 78)


    # ========================================================
    # BY SYMBOL
    # ========================================================

    print(
        "BY SYMBOL:"
    )

    for sym in SYMBOLS.keys():

        sub = closed_df[
            closed_df[
                "Symbol"
            ] == sym
        ]

        if len(sub) == 0:
            continue

        wr = (
            sub[
                "Outcome"
            ].eq("WIN").mean()
            * 100.0
        )

        pnl = float(
            sub[
                "Dollar_PnL"
            ].sum()
        )

        print(
            f"  {sym:7} -> "
            f"Trades: {len(sub):4}, "
            f"Win Rate: {wr:6.2f}%, "
            f"PnL: ${pnl:10,.2f}"
        )


    # ========================================================
    # BY MONTH
    # ========================================================

    print("-" * 78)

    print(
        "BY MONTH:"
    )

    for month, sub in (
        closed_df
        .groupby("Month")
    ):

        wr = (
            sub[
                "Outcome"
            ].eq("WIN").mean()
            * 100.0
        )

        pnl = float(
            sub[
                "Dollar_PnL"
            ].sum()
        )

        print(
            f"  {month} -> "
            f"Trades: {len(sub):4}, "
            f"Win Rate: {wr:6.2f}%, "
            f"PnL: ${pnl:10,.2f}"
        )


    # ========================================================
    # INTEGRITY
    # ========================================================

    print("-" * 78)

    print(
        "BACKTEST INTEGRITY:"
    )

    print(
        "  Data Fetch:            V123 COMPATIBLE"
    )

    print(
        "  Signal Candle:         CLOSED"
    )

    print(
        "  Entry:                 NEXT 15m OPEN"
    )

    print(
        "  4H Partial Candle:     EXCLUDED"
    )

    print(
        "  Center=True:           DISABLED"
    )

    print(
        "  Same-Symbol Overlap:   LOCKED"
    )

    print(
        "  Artificial Timeout:    DISABLED"
    )

    print(
        "  RR:                    1:2"
    )

    print("=" * 78)
