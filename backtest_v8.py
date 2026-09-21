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
# HUNTER-V130
# HUNTER-V131 — CONTROLLED UNIVERSE TEST
#
# Purpose:
# Reproduce the original V123 behavior before applying
# integrity corrections one at a time.
#
# IMPORTANT:
# This is an AUDIT build, not the final live-safe engine.
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

BASELINE_SYMBOLS = {
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
    "ETH": "ETH/USDT",
    # Controlled test: SOL is removed and LINK is added.
    "LINK": "LINK/USDT",
}

REMOVED_SYMBOL = "SOL"
ADDED_SYMBOL = "LINK"
SYMBOLS = BASELINE_SYMBOLS


# ============================================================
# PARAMETERS
# ============================================================

TIMEFRAME_BASE = "15m"

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

TRADE_MARGIN = 100.0
LEVERAGE = 50.0

DAYS = 365


# ============================================================
# DATA FETCH
#
# Preserved from V123.
# ============================================================

def fetch_chunk_data(lbank_symbol, start_dt, end_dt):
    since_ts = int(
        (start_dt - timedelta(days=15)).timestamp() * 1000
    )

    end_ts = int(end_dt.timestamp() * 1000)

    all_ohlcv = []
    current_since = since_ts

    try:
        while current_since < end_ts:
            batch = exchange.fetch_ohlcv(
                lbank_symbol,
                timeframe="15m",
                since=current_since,
                limit=1000,
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
        print(f"  ERROR fetching {lbank_symbol}: {e}")
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
            "Volume",
        ],
    )

    df["Date"] = pd.to_datetime(
        df["Timestamp"],
        unit="ms",
    )

    df = df[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    ]

    df.dropna(inplace=True)

    df.drop_duplicates(
        subset=["Date"],
        keep="last",
        inplace=True,
    )

    df.sort_values(
        "Date",
        inplace=True,
    )

    df.reset_index(
        drop=True,
        inplace=True,
    )

    df.set_index(
        "Date",
        inplace=True,
    )

    df = df[
        (df.index >= start_dt)
        & (df.index <= end_dt)
    ]

    return df


# ============================================================
# DATA PREPARATION
#
# Original V123 calculations intentionally preserved.
# ============================================================

def prepare_data(df_15m):
    df_15m = df_15m.copy()

    # --------------------------------------------------------
    # 4H
    # --------------------------------------------------------

    df_4h = (
        df_15m.resample("4h")
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
    )

    df_4h["EMA_50"] = df_4h["Close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    df_4h["EMA_200"] = df_4h["Close"].ewm(
        span=200,
        adjust=False,
    ).mean()

    df_4h["Regime_Bullish"] = (
        (df_4h["Close"] > df_4h["EMA_200"])
        & (df_4h["EMA_50"] > df_4h["EMA_200"])
    )

    df_4h["Regime_Bearish"] = (
        (df_4h["Close"] < df_4h["EMA_200"])
        & (df_4h["EMA_50"] < df_4h["EMA_200"])
    )

    # --------------------------------------------------------
    # 1H
    #
    # center=True intentionally preserved for audit.
    # --------------------------------------------------------

    df_1h = (
        df_15m.resample("1h")
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
    )

    df_1h["Swing_High"] = (
        df_1h["High"]
        .rolling(
            5,
            center=True,
        )
        .max()
    )

    df_1h["Swing_Low"] = (
        df_1h["Low"]
        .rolling(
            5,
            center=True,
        )
        .min()
    )

    df_1h["ATR"] = (
        (df_1h["High"] - df_1h["Low"])
        .rolling(14)
        .mean()
    )

    # --------------------------------------------------------
    # 15M
    # --------------------------------------------------------

    df_15m["ATR"] = (
        (df_15m["High"] - df_15m["Low"])
        .rolling(14)
        .mean()
    )

    df_15m["Body"] = (
        df_15m["Close"] - df_15m["Open"]
    ).abs()

    df_15m["Avg_Body"] = (
        df_15m["Body"]
        .rolling(20)
        .mean()
    )

    return df_15m, df_1h, df_4h


# ============================================================
# V123 ENGINE
#
# Intentionally preserves the original execution behavior.
# ============================================================

def run_v123_engine(
    symbol,
    df_15,
    df_1h,
    df_4h,
    start_dt,
    end_dt,
):
    trades = []

    for i in range(50, len(df_15)):
        t_curr = df_15.index[i]

        if t_curr < start_dt:
            continue

        if t_curr > end_dt:
            break

        c_row = df_15.iloc[i]
        p_row = df_15.iloc[i - 1]

        # Original V123 HTF selection.
        h_sub = df_1h[df_1h.index <= t_curr]
        h_4sub = df_4h[df_4h.index <= t_curr]

        if len(h_sub) < 10:
            continue

        if len(h_4sub) < 1:
            continue

        regime_bull = bool(
            h_4sub.iloc[-1]["Regime_Bullish"]
        )

        regime_bear = bool(
            h_4sub.iloc[-1]["Regime_Bearish"]
        )

        recent_lows = h_sub["Low"].iloc[-10:-2]
        recent_highs = h_sub["High"].iloc[-10:-2]

        if len(recent_lows) == 0:
            continue

        if len(recent_highs) == 0:
            continue

        min_support = recent_lows.min()
        max_resistance = recent_highs.max()

        sweep_low = (
            p_row["Low"] < min_support
            and p_row["Close"] > min_support
        )

        sweep_high = (
            p_row["High"] > max_resistance
            and p_row["Close"] < max_resistance
        )

        # Original V123: displacement on CURRENT candle.
        displacement_up = (
            c_row["Close"] > c_row["Open"]
            and c_row["Body"] > 2.0 * c_row["Avg_Body"]
        )

        displacement_down = (
            c_row["Close"] < c_row["Open"]
            and c_row["Body"] > 2.0 * c_row["Avg_Body"]
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

        if not valid_long and not valid_short:
            continue

        if valid_long:
            side = "LONG"
            entry_price = c_row["Open"] * (1.0 + SLIPPAGE)
        else:
            side = "SHORT"
            entry_price = c_row["Open"] * (1.0 - SLIPPAGE)

        atr = c_row["ATR"]

        if not np.isfinite(atr) or atr <= 0:
            continue

        # Original V123 1:2 structure.
        if side == "LONG":
            sl = entry_price - 1.5 * atr
            tp = entry_price + 3.0 * atr
            be_trigger = entry_price + 1.5 * atr
        else:
            sl = entry_price + 1.5 * atr
            tp = entry_price - 3.0 * atr
            be_trigger = entry_price - 1.5 * atr

        current_sl = sl
        be_active = False

        outcome = "LOSS"
        exit_price = sl
        exit_index = None

        # Original V123 scan window.
        for j in range(
            i + 1,
            min(i + 35, len(df_15)),
        ):
            fut = df_15.iloc[j]

            high = fut["High"]
            low = fut["Low"]

            if side == "LONG":
                if not be_active and high >= be_trigger:
                    current_sl = entry_price
                    be_active = True

                hit_sl = low <= current_sl
                hit_tp = high >= tp

                # Conservative SL priority.
                if hit_sl:
                    if be_active:
                        outcome = "BE"
                        exit_price = entry_price
                    else:
                        outcome = "LOSS"
                        exit_price = current_sl

                    exit_index = j
                    break

                if hit_tp:
                    outcome = "WIN"
                    exit_price = tp
                    exit_index = j
                    break

            else:
                if not be_active and low <= be_trigger:
                    current_sl = entry_price
                    be_active = True

                hit_sl = high >= current_sl
                hit_tp = low <= tp

                # Conservative SL priority.
                if hit_sl:
                    if be_active:
                        outcome = "BE"
                        exit_price = entry_price
                    else:
                        outcome = "LOSS"
                        exit_price = current_sl

                    exit_index = j
                    break

                if hit_tp:
                    outcome = "WIN"
                    exit_price = tp
                    exit_index = j
                    break

        # Preserve original V123 unresolved-trade behavior:
        # if neither SL nor TP is hit in the scan window,
        # outcome remains LOSS at SL.

        notional = TRADE_MARGIN * LEVERAGE

        if side == "LONG":
            price_ret = (
                exit_price - entry_price
            ) / entry_price
        else:
            price_ret = (
                entry_price - exit_price
            ) / entry_price

        dollar_pnl = (
            notional * price_ret
            - notional * FEE_RATE * 2.0
        )

        trades.append(
            {
                "Timestamp": t_curr,
                "ExitTimestamp": (
                    df_15.index[exit_index]
                    if exit_index is not None
                    else None
                ),
                "Symbol": symbol,
                "Side": side,
                "Outcome": outcome,
                "Dollar_PnL": dollar_pnl,
                "Entry_Price": entry_price,
                "Exit_Price": exit_price,
                "Entry_Index": i,
                "Exit_Index": exit_index,
            }
        )

    return trades


# ============================================================
# UNIVERSE ANALYSIS
# ============================================================

def summarize_symbol(sub):
    total = len(sub)
    wins = int((sub["Outcome"] == "WIN").sum())
    losses = int((sub["Outcome"] == "LOSS").sum())
    be = int((sub["Outcome"] == "BE").sum())
    pnl = float(sub["Dollar_PnL"].sum())
    gross_profit = float(sub.loc[sub["Outcome"] == "WIN", "Dollar_PnL"].sum())
    gross_loss = abs(float(sub.loc[sub["Outcome"] == "LOSS", "Dollar_PnL"].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    wr = wins / total * 100.0 if total else 0.0

    streak = 0
    max_streak = 0
    for outcome in sub["Outcome"]:
        if outcome == "LOSS":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # Split by time for a simple robustness check. This does not fit
    # parameters; it only checks whether performance persists later.
    if total >= 10:
        split_time = sub["Timestamp"].min() + (sub["Timestamp"].max() - sub["Timestamp"].min()) * 0.70
        val = sub[sub["Timestamp"] >= split_time]
    else:
        val = sub.iloc[0:0]

    val_total = len(val)
    val_wr = ((val["Outcome"] == "WIN").mean() * 100.0) if val_total else 0.0
    val_pnl = float(val["Dollar_PnL"].sum()) if val_total else 0.0

    return {
        "Trades": total,
        "WR": wr,
        "PnL": pnl,
        "PF": pf,
        "MaxLossStreak": max_streak,
        "ValidationTrades": val_total,
        "ValidationWR": val_wr,
        "ValidationPnL": val_pnl,
    }


def print_universe_analysis(trades_df):
    rows = []
    for symbol in SYMBOLS:
        sub = trades_df[trades_df["Symbol"] == symbol].sort_values("Timestamp")
        if len(sub) == 0:
            continue
        r = summarize_symbol(sub)
        r["Symbol"] = symbol
        r["Type"] = "TEST UNIVERSE"
        rows.append(r)

    if not rows:
        return

    u = pd.DataFrame(rows)
    cols = ["Symbol", "Type", "Trades", "WR", "PnL", "PF", "MaxLossStreak", "ValidationTrades", "ValidationWR", "ValidationPnL"]
    u = u[cols]

    print()
    print("=" * 120)
    print("UNIVERSE OPTIMIZATION — FULL PERIOD")
    print("=" * 120)
    print(u.sort_values(["Type", "PnL"], ascending=[True, False]).to_string(index=False, formatters={
        "WR": "{:.2f}".format,
        "PnL": "${:,.2f}".format,
        "PF": "{:.2f}".format,
        "ValidationWR": "{:.2f}".format,
        "ValidationPnL": "${:,.2f}".format,
    }))

    print()
    print("=" * 120)
    print("CONTROLLED REPLACEMENT TEST")
    print("=" * 120)
    candidates = u.copy()
    if len(candidates):
        candidates = candidates.sort_values(
            ["ValidationPnL", "ValidationWR", "PF"],
            ascending=[False, False, False],
        )
        print(candidates.to_string(index=False, formatters={
            "WR": "{:.2f}".format,
            "PnL": "${:,.2f}".format,
            "PF": "{:.2f}".format,
            "ValidationWR": "{:.2f}".format,
            "ValidationPnL": "${:,.2f}".format,
        }))

    print()
    print("NOTE: V131 is a controlled test: SOL removed, LINK added. Accept only if the complete result improves V129 without relying on one metric alone.")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("HUNTER-V131 — V129 BASELINE WITH CONTROLLED LINK REPLACEMENT")
    print("=" * 80)

    now = datetime.now()

    start_dt = now - timedelta(days=DAYS)
    end_dt = now

    print(
        f"Period: {start_dt} -> {end_dt}"
    )

    print()
    print("AUDIT MODE: V123 behavior intentionally preserved.")
    print("Data connection: V123")
    print("Universe: V129 minus SOL + LINK")
    print("HTF selection: V123")
    print("center=True: V123")
    print("Sweep: V123")
    print("Displacement: V123")
    print("Entry: V123")
    print("SL/TP/BE: V123")
    print("35-candle scan: V123")
    print("Unresolved trade behavior: V123")
    print("Overlap behavior: V123")
    print("=" * 80)

    all_trades = []

    for symbol, lbank_symbol in SYMBOLS.items():
        print()
        print("-" * 80)
        print(f"{symbol} -> {lbank_symbol}")
        print("-" * 80)

        df = fetch_chunk_data(
            lbank_symbol,
            start_dt,
            end_dt,
        )

        if df is None:
            print("  NO DATA")
            continue

        print(f"  Candles: {len(df)}")

        if len(df) < 100:
            print("  INSUFFICIENT DATA")
            continue

        df_15, df_1h, df_4h = prepare_data(df)

        symbol_trades = run_v123_engine(
            symbol,
            df_15,
            df_1h,
            df_4h,
            start_dt,
            end_dt,
        )

        all_trades.extend(symbol_trades)

        print(
            f"  Trades: {len(symbol_trades)}"
        )

    if not all_trades:
        print()
        print("=" * 80)
        print("NO TRADES GENERATED")
        print("=" * 80)
        return

    trades_df = pd.DataFrame(all_trades)

    trades_df.sort_values(
        "Timestamp",
        inplace=True,
    )

    trades_df.reset_index(
        drop=True,
        inplace=True,
    )

    total_trades = len(trades_df)

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
        len(wins) / total_trades * 100.0
    )

    loss_rate = (
        len(losses) / total_trades * 100.0
    )

    be_rate = (
        len(bes) / total_trades * 100.0
    )

    net_pnl = float(
        trades_df["Dollar_PnL"].sum()
    )

    gross_profit = float(
        wins["Dollar_PnL"].sum()
    )

    gross_loss = abs(
        float(
            losses["Dollar_PnL"].sum()
        )
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit / gross_loss
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
    # Drawdown
    # --------------------------------------------------------

    trades_df["Cumulative_PnL"] = (
        trades_df["Dollar_PnL"].cumsum()
    )

    trades_df["Peak"] = (
        trades_df["Cumulative_PnL"].cummax()
    )

    trades_df["Drawdown"] = (
        trades_df["Cumulative_PnL"]
        - trades_df["Peak"]
    )

    max_dd = float(
        trades_df["Drawdown"].min()
    )

    # --------------------------------------------------------
    # Consecutive losses
    # --------------------------------------------------------

    loss_streaks = []
    current_streak = 0

    for outcome in trades_df["Outcome"]:
        if outcome == "LOSS":
            current_streak += 1
        else:
            if current_streak > 0:
                loss_streaks.append(
                    current_streak
                )
            current_streak = 0

    if current_streak > 0:
        loss_streaks.append(
            current_streak
        )

    max_streak = (
        max(loss_streaks)
        if loss_streaks
        else 0
    )

    # --------------------------------------------------------
    # Main report
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("===== HUNTER-V129 — V123 AUDIT RESULT =====")
    print("=" * 80)

    print(
        f"Total Trades:          {total_trades}"
    )

    print(
        f"Trades / Month:        {total_trades / 12.0:.1f}"
    )

    print(
        f"Win Rate:              {win_rate:.2f}%"
    )

    print(
        f"Loss Rate:             {loss_rate:.2f}%"
    )

    print(
        f"Break Even Rate:       {be_rate:.2f}%"
    )

    print(
        f"Net PnL:               ${net_pnl:,.2f}"
    )

    print(
        f"Profit Factor:         {profit_factor:.2f}"
    )

    print(
        f"Average Win:           ${avg_win:,.2f}"
    )

    print(
        f"Average Loss:          ${avg_loss:,.2f}"
    )

    print(
        f"Max Drawdown:          ${max_dd:,.2f}"
    )

    print(
        f"Maximum Consecutive Losses: {max_streak}"
    )

    # --------------------------------------------------------
    # By symbol
    # --------------------------------------------------------

    print("-" * 80)
    print("BY SYMBOL:")

    for symbol in SYMBOLS.keys():
        sub = trades_df[
            trades_df["Symbol"] == symbol
        ]

        if len(sub) == 0:
            continue

        wr = (
            sub["Outcome"]
            .eq("WIN")
            .mean()
            * 100.0
        )

        pnl = float(
            sub["Dollar_PnL"].sum()
        )

        print(
            f"  {symbol:7} -> "
            f"Trades: {len(sub):4}, "
            f"Win Rate: {wr:6.2f}%, "
            f"PnL: ${pnl:10,.2f}"
        )

    # --------------------------------------------------------
    # By direction
    # --------------------------------------------------------

    print("-" * 80)
    print("BY DIRECTION:")

    for side in ("LONG", "SHORT"):
        sub = trades_df[
            trades_df["Side"] == side
        ]

        if len(sub) == 0:
            continue

        wr = (
            sub["Outcome"]
            .eq("WIN")
            .mean()
            * 100.0
        )

        pnl = float(
            sub["Dollar_PnL"].sum()
        )

        print(
            f"  {side:7} -> "
            f"Trades: {len(sub):4}, "
            f"Win Rate: {wr:6.2f}%, "
            f"PnL: ${pnl:10,.2f}"
        )

    # --------------------------------------------------------
    # By month
    # --------------------------------------------------------

    print("-" * 80)
    print("BY MONTH:")

    trades_df["Month"] = (
        trades_df["Timestamp"]
        .dt.strftime("%Y-%m")
    )

    for month, sub in trades_df.groupby("Month"):
        wr = (
            sub["Outcome"]
            .eq("WIN")
            .mean()
            * 100.0
        )

        pnl = float(
            sub["Dollar_PnL"].sum()
        )

        print(
            f"  {month} -> "
            f"Trades: {len(sub):4}, "
            f"Win Rate: {wr:6.2f}%, "
            f"PnL: ${pnl:10,.2f}"
        )

    # --------------------------------------------------------
    # Audit target
    # --------------------------------------------------------

    print("-" * 80)
    print("V129 BASELINE REFERENCE:")

    print("  Trades:       1041")
    print("  Win Rate:     53.31%")
    print("  Net PnL:      +$38,076.67")
    print("  Profit Factor: 8.11")
    print("  Max DD:       -$256.08")
    print("  Max Loss Streak: 4")

    print("=" * 80)

    # Universe analysis is deliberately separate from the baseline report.
    print_universe_analysis(trades_df)

    # --------------------------------------------------------
    # Optional CSV artifact for local inspection.
    # --------------------------------------------------------

    try:
        trades_df.to_csv(
            "v129_audit_trades.csv",
            index=False,
        )
        print(
            "Trade log saved: v129_audit_trades.csv"
        )
    except Exception as e:
        print(
            f"Could not save trade log: {e}"
        )


if __name__ == "__main__":
    main()
