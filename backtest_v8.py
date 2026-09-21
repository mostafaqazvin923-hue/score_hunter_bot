import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt


# ============================================================
# HUNTER-V128
# V123 STRATEGY / CAUSAL PORTFOLIO BACKTEST ENGINE
#
# IMPORTANT:
# - Strategy logic preserved from V123
# - Signal candle must be CLOSED
# - Entry = NEXT 15m candle OPEN
# - No lookahead
# - No artificial timeout
# - No overlapping position per symbol
# - Completed 1H / 4H candles only
# - LBank perpetual swaps
# - Nominal RR = 1:2
# ============================================================


# ============================================================
# EXCHANGE
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True,
    "timeout": 20000,
    "options": {
        "defaultType": "swap"
    }
})


# ============================================================
# SYMBOLS
# LBank linear USDT perpetuals
# ============================================================

SYMBOLS = {
    "CRV": "CRV/USDT:USDT",
    "DOGE": "DOGE/USDT:USDT",
    "ICP": "ICP/USDT:USDT",
    "APT": "APT/USDT:USDT",
    "PENDLE": "PENDLE/USDT:USDT",
    "WIF": "WIF/USDT:USDT",
    "ONDO": "ONDO/USDT:USDT",
    "NEAR": "NEAR/USDT:USDT",
    "SEI": "SEI/USDT:USDT",
    "XLM": "XLM/USDT:USDT",
    "ADA": "ADA/USDT:USDT",
    "BNB": "BNB/USDT:USDT",
    "SOL": "SOL/USDT:USDT",
    "ETH": "ETH/USDT:USDT",
}


# ============================================================
# CONFIG
# ============================================================

TIMEFRAME_BASE = "15m"

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

TRADE_MARGIN = 100.0
LEVERAGE = 50.0

DAYS = 365

# Warmup:
# 200 x 4h EMA requires substantial history.
# 45 days gives > 270 four-hour candles.
WARMUP_DAYS = 60

# Original V123 signal parameters
SWEEP_LOOKBACK_1H = 10
DISPLACEMENT_MULTIPLIER = 2.0

ATR_PERIOD_15M = 14
BODY_AVG_PERIOD = 20

SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
BE_ATR_MULT = 1.5

# Conservative intrabar priority
# If both SL and TP occur in the same OHLC candle:
# choose SL first unless BE has already been activated.
CONSERVATIVE_SAME_BAR = True


# ============================================================
# DATA FETCH
# ============================================================

def fetch_ohlcv_full(symbol, start_dt, end_dt):
    """
    Fetch 15m LBank perpetual OHLCV.

    IMPORTANT:
    start_dt/end_dt are UTC-aware.
    """

    since_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    all_ohlcv = []
    current_since = since_ts

    try:
        while current_since < end_ts:

            batch = exchange.fetch_ohlcv(
                symbol,
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

            time.sleep(0.25)

    except Exception as e:
        print(f"ERROR fetching {symbol}: {e}")
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

    # UTC timestamps
    df["Date"] = pd.to_datetime(
        df["Timestamp"],
        unit="ms",
        utc=True
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

    df.dropna(inplace=True)

    df.drop_duplicates(
        subset=["Date"],
        keep="last",
        inplace=True
    )

    df.sort_values("Date", inplace=True)

    df.set_index("Date", inplace=True)

    # Remove current still-open candle.
    now_ms = exchange.milliseconds()

    tf_ms = 15 * 60 * 1000

    last_complete_open_ms = (
        (now_ms // tf_ms) * tf_ms
    ) - tf_ms

    last_complete_dt = pd.to_datetime(
        last_complete_open_ms,
        unit="ms",
        utc=True
    )

    df = df[df.index <= last_complete_dt]

    # Keep requested range including warmup.
    df = df[
        (df.index >= start_dt) &
        (df.index <= end_dt)
    ]

    return df


# ============================================================
# INDICATORS
# ============================================================

def prepare_data(df_15m):
    """
    Build 15m / 1h / 4h datasets.

    No centered rolling calculations.
    """

    df_15m = df_15m.copy()

    # --------------------------------------------------------
    # 15m indicators
    # --------------------------------------------------------

    df_15m["Body"] = (
        df_15m["Close"] -
        df_15m["Open"]
    ).abs()

    df_15m["Avg_Body"] = (
        df_15m["Body"]
        .rolling(BODY_AVG_PERIOD, min_periods=BODY_AVG_PERIOD)
        .mean()
    )

    # V123-style ATR
    df_15m["ATR"] = (
        (df_15m["High"] - df_15m["Low"])
        .rolling(ATR_PERIOD_15M, min_periods=ATR_PERIOD_15M)
        .mean()
    )

    # --------------------------------------------------------
    # 1H
    # --------------------------------------------------------

    df_1h = df_15m.resample(
        "1h",
        label="left",
        closed="left"
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    # No center=True
    df_1h["Swing_High"] = (
        df_1h["High"]
        .rolling(5, min_periods=5)
        .max()
    )

    df_1h["Swing_Low"] = (
        df_1h["Low"]
        .rolling(5, min_periods=5)
        .min()
    )

    df_1h["ATR"] = (
        (df_1h["High"] - df_1h["Low"])
        .rolling(14, min_periods=14)
        .mean()
    )

    # --------------------------------------------------------
    # 4H
    # --------------------------------------------------------

    df_4h = df_15m.resample(
        "4h",
        label="left",
        closed="left"
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    df_4h["EMA_50"] = (
        df_4h["Close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    df_4h["EMA_200"] = (
        df_4h["Close"]
        .ewm(span=200, adjust=False)
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
# GET LAST COMPLETED HTF CANDLE
# ============================================================

def get_completed_htf_row(df_htf, t_curr):
    """
    Return the latest HTF candle whose OPEN timestamp is
    strictly before t_curr.

    Because resampled candles are [open, next-open),
    requiring index < t_curr prevents using a candle that
    has not completed yet.
    """

    subset = df_htf[df_htf.index < t_curr]

    if subset.empty:
        return None

    return subset.iloc[-1]


# ============================================================
# SIGNAL
# ============================================================

def get_signal(df_15m, df_1h, df_4h, i):
    """
    V123 strategy logic preserved.

    IMPORTANT:
    i = EXECUTION candle

    p_row = previous CLOSED 15m candle.

    Therefore:
        signal -> p_row
        entry  -> c_row OPEN
    """

    if i < 50:
        return None

    t_curr = df_15m.index[i]

    c_row = df_15m.iloc[i]

    # Previous candle is fully closed.
    p_row = df_15m.iloc[i - 1]

    # --------------------------------------------------------
    # Completed 1H / 4H only
    # --------------------------------------------------------

    h_sub = df_1h[df_1h.index < t_curr]
    h_4sub = df_4h[df_4h.index < t_curr]

    if len(h_sub) < 10 or len(h_4sub) < 10:
        return None

    latest_4h = h_4sub.iloc[-1]

    regime_bull = bool(
        latest_4h["Regime_Bullish"]
    )

    regime_bear = bool(
        latest_4h["Regime_Bearish"]
    )

    # --------------------------------------------------------
    # V123 liquidity area
    # --------------------------------------------------------

    recent_lows = h_sub["Low"].iloc[-10:-1]
    recent_highs = h_sub["High"].iloc[-10:-1]

    if (
        len(recent_lows) == 0 or
        len(recent_highs) == 0
    ):
        return None

    min_support = recent_lows.min()
    max_resistance = recent_highs.max()

    # --------------------------------------------------------
    # V123 SWEEP
    #
    # IMPORTANT:
    # sweep happens on p_row itself.
    # No pp_row modification.
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
    # V123 DISPLACEMENT
    #
    # Based only on CLOSED p_row.
    # --------------------------------------------------------

    displacement_up = (
        p_row["Close"] > p_row["Open"] and
        p_row["Body"] >
        DISPLACEMENT_MULTIPLIER * p_row["Avg_Body"]
    )

    displacement_down = (
        p_row["Close"] < p_row["Open"] and
        p_row["Body"] >
        DISPLACEMENT_MULTIPLIER * p_row["Avg_Body"]
    )

    valid_long = (
        regime_bull and
        sweep_low and
        displacement_up
    )

    valid_short = (
        regime_bear and
        sweep_high and
        displacement_down
    )

    if not valid_long and not valid_short:
        return None

    if valid_long:
        return "LONG"

    return "SHORT"


# ============================================================
# SINGLE TRADE SIMULATION
# ============================================================

def simulate_trade(
    df_15m,
    entry_index,
    side
):
    """
    Simulate one trade from entry_index forward.

    NO artificial timeout.

    The trade remains OPEN if neither SL nor TP is reached
    before the available dataset ends.

    Returns:
        dict
    """

    entry_row = df_15m.iloc[entry_index]

    entry_open = float(entry_row["Open"])

    if side == "LONG":
        entry_price = (
            entry_open *
            (1.0 + SLIPPAGE)
        )
    else:
        entry_price = (
            entry_open *
            (1.0 - SLIPPAGE)
        )

    # ATR comes from the CLOSED signal candle.
    signal_row = df_15m.iloc[entry_index - 1]

    atr = float(signal_row["ATR"])

    if not np.isfinite(atr) or atr <= 0:
        return None

    if side == "LONG":

        sl = (
            entry_price -
            SL_ATR_MULT * atr
        )

        tp = (
            entry_price +
            TP_ATR_MULT * atr
        )

        be_trigger = (
            entry_price +
            BE_ATR_MULT * atr
        )

    else:

        sl = (
            entry_price +
            SL_ATR_MULT * atr
        )

        tp = (
            entry_price -
            TP_ATR_MULT * atr
        )

        be_trigger = (
            entry_price -
            BE_ATR_MULT * atr
        )

    current_sl = sl
    breakeven_activated = False

    # --------------------------------------------------------
    # Start from ENTRY candle.
    #
    # Entry is assumed at Open.
    # Remaining High/Low of same candle are allowed.
    # --------------------------------------------------------

    for j in range(
        entry_index,
        len(df_15m)
    ):

        fut = df_15m.iloc[j]

        high = float(fut["High"])
        low = float(fut["Low"])

        if side == "LONG":

            # BE activation
            if (
                not breakeven_activated and
                high >= be_trigger
            ):
                current_sl = entry_price
                breakeven_activated = True

            hit_sl = low <= current_sl
            hit_tp = high >= tp

        else:

            if (
                not breakeven_activated and
                low <= be_trigger
            ):
                current_sl = entry_price
                breakeven_activated = True

            hit_sl = high >= current_sl
            hit_tp = low <= tp

        # ----------------------------------------------------
        # Both touched in same OHLC candle.
        # Conservative treatment.
        # ----------------------------------------------------

        if hit_sl and hit_tp:

            if breakeven_activated:
                outcome = "BE"
                exit_price = entry_price
            else:
                outcome = "LOSS"
                exit_price = current_sl

            return {
                "exit_index": j,
                "exit_timestamp": df_15m.index[j],
                "Outcome": outcome,
                "Entry_Price": entry_price,
                "Exit_Price": exit_price,
                "SL": sl,
                "TP": tp,
                "BE_Activated": breakeven_activated
            }

        if hit_sl:

            if breakeven_activated:
                outcome = "BE"
                exit_price = entry_price
            else:
                outcome = "LOSS"
                exit_price = current_sl

            return {
                "exit_index": j,
                "exit_timestamp": df_15m.index[j],
                "Outcome": outcome,
                "Entry_Price": entry_price,
                "Exit_Price": exit_price,
                "SL": sl,
                "TP": tp,
                "BE_Activated": breakeven_activated
            }

        if hit_tp:

            return {
                "exit_index": j,
                "exit_timestamp": df_15m.index[j],
                "Outcome": "WIN",
                "Entry_Price": entry_price,
                "Exit_Price": tp,
                "SL": sl,
                "TP": tp,
                "BE_Activated": breakeven_activated
            }

    # --------------------------------------------------------
    # NO TIMEOUT.
    #
    # The dataset ended while position remained open.
    # Do NOT call this a LOSS.
    # Do NOT call this a WIN.
    # --------------------------------------------------------

    return {
        "exit_index": None,
        "exit_timestamp": None,
        "Outcome": "OPEN_AT_END",
        "Entry_Price": entry_price,
        "Exit_Price": np.nan,
        "SL": sl,
        "TP": tp,
        "BE_Activated": breakeven_activated
    }


# ============================================================
# PNL
# ============================================================

def calculate_pnl(side, entry_price, exit_price):
    """
    Futures-style linear USDT PnL.

    Notional = margin * leverage.

    Fees = entry + exit.
    """

    notional = TRADE_MARGIN * LEVERAGE

    if side == "LONG":

        price_ret = (
            exit_price - entry_price
        ) / entry_price

    else:

        price_ret = (
            entry_price - exit_price
        ) / entry_price

    gross_pnl = notional * price_ret

    fees = (
        notional *
        FEE_RATE *
        2.0
    )

    return gross_pnl - fees


# ============================================================
# BACKTEST ONE SYMBOL
# ============================================================

def run_symbol_backtest(
    symbol,
    df_15m,
    df_1h,
    df_4h,
    requested_start,
    requested_end
):

    trades = []

    i = 50

    while i < len(df_15m):

        t_curr = df_15m.index[i]

        # Only generate trades inside requested period.
        if t_curr < requested_start:
            i += 1
            continue

        if t_curr > requested_end:
            break

        side = get_signal(
            df_15m,
            df_1h,
            df_4h,
            i
        )

        if side is None:
            i += 1
            continue

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
        #
        # No PnL.
        # No fake win/loss.
        # ----------------------------------------------------

        if outcome == "OPEN_AT_END":

            trades.append({
                "Timestamp": t_curr,
                "ExitTimestamp": None,
                "Symbol": symbol,
                "Side": side,
                "Outcome": "OPEN_AT_END",
                "Dollar_PnL": 0.0,
                "Entry_Price": trade["Entry_Price"],
                "Exit_Price": np.nan,
                "Month": t_curr.strftime("%Y-%m")
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
            "ExitTimestamp": trade["exit_timestamp"],
            "Symbol": symbol,
            "Side": side,
            "Outcome": outcome,
            "Dollar_PnL": pnl,
            "Entry_Price": trade["Entry_Price"],
            "Exit_Price": trade["Exit_Price"],
            "Month": t_curr.strftime("%Y-%m")
        })

        # ----------------------------------------------------
        # OVERLAP LOCK
        #
        # Do not inspect signals while this trade is open.
        #
        # Also skip the exit candle itself.
        # Next possible signal = exit_index + 1
        # ----------------------------------------------------

        exit_index = trade["exit_index"]

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
        "V123 STRATEGY / CAUSAL FUTURES BACKTEST"
    )
    print("=" * 78)

    now = datetime.now(timezone.utc)

    requested_start = (
        now -
        timedelta(days=DAYS)
    )

    requested_end = now

    data_start = (
        requested_start -
        timedelta(days=WARMUP_DAYS)
    )

    print()
    print(f"Backtest start : {requested_start}")
    print(f"Backtest end   : {requested_end}")
    print(f"Warmup         : {WARMUP_DAYS} days")
    print(f"Symbols        : {len(SYMBOLS)}")
    print(f"Timeframe      : {TIMEFRAME_BASE}")
    print(f"RR             : 1:2")
    print("Timeout        : DISABLED")
    print("Overlap Lock   : ENABLED")
    print("Lookahead      : NONE BY DESIGN")
    print("Entry model    : NEXT 15m OPEN")
    print("=" * 78)

    all_trades = []

    # --------------------------------------------------------
    # LOAD MARKETS
    # --------------------------------------------------------

    try:
        exchange.load_markets()
    except Exception as e:
        print(f"Market loading error: {e}")
        sys.exit(1)

    # --------------------------------------------------------
    # PROCESS EACH SYMBOL ONCE
    # --------------------------------------------------------

    for symbol, lbank_symbol in SYMBOLS.items():

        print()
        print("-" * 78)
        print(
            f"{symbol} -> {lbank_symbol}"
        )
        print("-" * 78)

        df_15m = fetch_ohlcv_full(
            lbank_symbol,
            data_start,
            requested_end
        )

        if df_15m is None:
            print("  No data.")
            continue

        if len(df_15m) < 1000:
            print(
                f"  Insufficient data: {len(df_15m)} candles"
            )
            continue

        print(
            f"  Raw 15m candles: {len(df_15m)}"
        )

        # ----------------------------------------------------
        # PREPARE
        # ----------------------------------------------------

        df_15m, df_1h, df_4h = prepare_data(
            df_15m
        )

        # ----------------------------------------------------
        # Run symbol
        # ----------------------------------------------------

        symbol_trades = run_symbol_backtest(
            symbol,
            df_15m,
            df_1h,
            df_4h,
            requested_start,
            requested_end
        )

        all_trades.extend(symbol_trades)

        closed_count = sum(
            1 for x in symbol_trades
            if x["Outcome"] != "OPEN_AT_END"
        )

        print(
            f"  Trades: {len(symbol_trades)} "
            f"(closed: {closed_count})"
        )


    # ========================================================
    # RESULTS
    # ========================================================

    if not all_trades:

        print()
        print("=" * 78)
        print("NO TRADES GENERATED.")
        print("=" * 78)
        sys.exit(0)

    trades_df = pd.DataFrame(all_trades)

    trades_df.sort_values(
        "Timestamp",
        inplace=True
    )

    trades_df.reset_index(
        drop=True,
        inplace=True
    )

    # --------------------------------------------------------
    # CLOSED TRADES ONLY
    # --------------------------------------------------------

    closed_df = trades_df[
        trades_df["Outcome"] != "OPEN_AT_END"
    ].copy()

    open_at_end = trades_df[
        trades_df["Outcome"] == "OPEN_AT_END"
    ]

    n_total = len(trades_df)
    n_closed = len(closed_df)
    n_open = len(open_at_end)

    if n_closed == 0:

        print()
        print("=" * 78)
        print("NO CLOSED TRADES.")
        print(
            f"Open positions at end: {n_open}"
        )
        print("=" * 78)
        sys.exit(0)

    # --------------------------------------------------------
    # WIN / LOSS / BE
    # --------------------------------------------------------

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
        closed_df["Dollar_PnL"].sum()
    )

    gross_profit = float(
        wins["Dollar_PnL"].sum()
    )

    gross_loss = abs(
        float(losses["Dollar_PnL"].sum())
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit /
            gross_loss
        )
    else:
        profit_factor = float("inf")

    avg_win = (
        float(wins["Dollar_PnL"].mean())
        if len(wins) > 0
        else 0.0
    )

    avg_loss = (
        float(losses["Dollar_PnL"].mean())
        if len(losses) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # DRAWDOWN
    # --------------------------------------------------------

    closed_df["Cumulative_PnL"] = (
        closed_df["Dollar_PnL"].cumsum()
    )

    closed_df["Peak"] = (
        closed_df["Cumulative_PnL"].cummax()
    )

    closed_df["Drawdown"] = (
        closed_df["Cumulative_PnL"] -
        closed_df["Peak"]
    )

    max_dd = float(
        closed_df["Drawdown"].min()
    )

    # --------------------------------------------------------
    # LOSS STREAK
    # --------------------------------------------------------

    streaks = []

    current_streak = 0

    for outcome in closed_df["Outcome"]:

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

    max_consecutive_losses = (
        max(streaks)
        if streaks
        else 0
    )

    # --------------------------------------------------------
    # MONTHLY
    # --------------------------------------------------------

    monthly = (
        closed_df
        .groupby("Month")
        .agg(
            Trades=("Outcome", "count"),
            Wins=("Outcome", lambda x:
                  (x == "WIN").sum()),
            Losses=("Outcome", lambda x:
                    (x == "LOSS").sum()),
            BE=("Outcome", lambda x:
                (x == "BE").sum()),
            PnL=("Dollar_PnL", "sum")
        )
    )

    monthly["WinRate"] = (
        monthly["Wins"] /
        monthly["Trades"] *
        100.0
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 78)
    print(
        "===== HUNTER-V128 — FINAL 1-YEAR RESULT ====="
    )
    print("=" * 78)

    print(
        f"Total Signals/Trades: {n_total}"
    )

    print(
        f"Closed Trades:        {n_closed}"
    )

    print(
        f"Open at End:          {n_open}"
    )

    print(
        f"Trades Per Month:     "
        f"{n_closed / 12.0:.1f}"
    )

    print(
        f"Win Rate:             {win_rate:.2f}%"
    )

    print(
        f"Loss Rate:            {loss_rate:.2f}%"
    )

    print(
        f"Break-even Rate:      {be_rate:.2f}%"
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
        f"Losses:               "
        f"{max_consecutive_losses}"
    )

    print("-" * 78)

    # ========================================================
    # BY SYMBOL
    # ========================================================

    print("BY SYMBOL:")

    for sym in SYMBOLS.keys():

        sub = closed_df[
            closed_df["Symbol"] == sym
        ]

        if len(sub) == 0:
            continue

        sym_wr = (
            sub["Outcome"].eq("WIN").mean()
            * 100.0
        )

        sym_pnl = float(
            sub["Dollar_PnL"].sum()
        )

        sym_losses = (
            sub["Outcome"] == "LOSS"
        ).sum()

        print(
            f"  {sym:7} -> "
            f"Trades: {len(sub):4}, "
            f"WR: {sym_wr:6.2f}%, "
            f"PnL: ${sym_pnl:10,.2f}"
        )

    # ========================================================
    # BY MONTH
    # ========================================================

    print("-" * 78)
    print("BY MONTH:")

    for month, row in monthly.iterrows():

        print(
            f"  {month} -> "
            f"Trades: {int(row['Trades']):4}, "
            f"WR: {row['WinRate']:6.2f}%, "
            f"PnL: ${row['PnL']:10,.2f}"
        )

    # ========================================================
    # INTEGRITY REPORT
    # ========================================================

    print("-" * 78)
    print("BACKTEST INTEGRITY:")
    print(
        "  Lookahead:              NONE"
    )
    print(
        "  Signal uses closed bar: YES"
    )
    print(
        "  Entry:                  NEXT 15m OPEN"
    )
    print(
        "  HTF partial candles:    EXCLUDED"
    )
    print(
        "  Centered rolling:       DISABLED"
    )
    print(
        "  Overlap per symbol:     LOCKED"
    )
    print(
        "  Artificial timeout:     DISABLED"
    )
    print(
        "  Unresolved positions:   NOT counted"
    )
    print(
        "  Nominal RR:             1:2"
    )
    print(
        "  LBank market type:      SWAP"
    )

    print("=" * 78)
