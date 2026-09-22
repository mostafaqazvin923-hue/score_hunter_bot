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
# HUNTER-V129 — LIVE-SAFE (CAUSAL & NO LOOKAHEAD)
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True,
    "timeout": 20000,
    "options": {
        "defaultType": "swap"
    }
})

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
    "ETH": "ETH/USDT",
}

TIMEFRAME_BASE = "15m"
SLIPPAGE = 0.0003
FEE_RATE = 0.0007
TRADE_MARGIN = 100.0
LEVERAGE = 50.0
DAYS = 365


def fetch_chunk_data(lbank_symbol, start_dt, end_dt):
    since_ts = int((start_dt - timedelta(days=15)).timestamp() * 1000)
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
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
    )
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.set_index("Date", inplace=True)
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    return df


def prepare_data(df_15m):
    df_15m = df_15m.copy()

    # 4H Regime (فقط گذشته)
    df_4h = (
        df_15m.resample("4h")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )
    df_4h["EMA_50"] = df_4h["Close"].ewm(span=50, adjust=False).mean()
    df_4h["EMA_200"] = df_4h["Close"].ewm(span=200, adjust=False).mean()
    df_4h["Regime_Bullish"] = (df_4h["Close"] > df_4h["EMA_200"]) & (df_4h["EMA_50"] > df_4h["EMA_200"])
    df_4h["Regime_Bearish"] = (df_4h["Close"] < df_4h["EMA_200"]) & (df_4h["EMA_50"] < df_4h["EMA_200"])

    # 1H Structure (بدون center=True و استفاده از shift(1) برای حفظ اصول لایو)
    df_1h = (
        df_15m.resample("1h")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )
    df_1h["Swing_High"] = df_1h["High"].rolling(5).max().shift(1)
    df_1h["Swing_Low"] = df_1h["Low"].rolling(5).min().shift(1)
    df_1h["ATR"] = (df_1h["High"] - df_1h["Low"]).rolling(14).mean()

    # 15M Indicators
    df_15m["ATR"] = (df_15m["High"] - df_15m["Low"]).rolling(14).mean()
    df_15m["Body"] = (df_15m["Close"] - df_15m["Open"]).abs()
    df_15m["Avg_Body"] = df_15m["Body"].rolling(20).mean()

    return df_15m, df_1h, df_4h


def run_live_safe_engine(symbol, df_15, df_1h, df_4h, start_dt, end_dt):
    trades = []

    for i in range(50, len(df_15)):
        t_curr = df_15.index[i]
        if t_curr < start_dt or t_curr > end_dt:
            continue

        c_row = df_15.iloc[i]     # کندل جاری (لحظه ورود در لایو)
        p_row = df_15.iloc[i-1]   # کندل قبلی بسته‌شده (برای بررسی سیگنال)

        h_sub = df_1h[df_1h.index < t_curr]
        h_4sub = df_4h[df_4h.index < t_curr]

        if len(h_sub) < 10 or len(h_4sub) < 1:
            continue

        regime_bull = bool(h_4sub.iloc[-1]["Regime_Bullish"])
        regime_bear = bool(h_4sub.iloc[-1]["Regime_Bearish"])

        recent_lows = h_sub["Low"].iloc[-10:-1]
        recent_highs = h_sub["High"].iloc[-10:-1]

        if len(recent_lows) == 0 or len(recent_highs) == 0:
            continue

        min_support = recent_lows.min()
        max_resistance = recent_highs.max()

        # بررسی سوئیپ و دیسپلیسمنت کاملاً کائوسال روی کندل بسته‌شده قبلی
        sweep_low = (p_row["Low"] < min_support) and (p_row["Close"] > min_support)
        sweep_high = (p_row["High"] > max_resistance) and (p_row["Close"] < max_resistance)

        displacement_up = (p_row["Close"] > p_row["Open"]) and (p_row["Body"] > 2.0 * p_row["Avg_Body"])
        displacement_down = (p_row["Close"] < p_row["Open"]) and (p_row["Body"] > 2.0 * p_row["Avg_Body"])

        valid_long = regime_bull and sweep_low and displacement_up
        valid_short = regime_bear and sweep_high and displacement_down

        if not valid_long and not valid_short:
            continue

        side = "LONG" if valid_long else "SHORT"
        
        # ورود روی Open کندل جاری (دقیقاً معادل اجرای لایو در آغاز کندل جدید)
        entry_price = c_row["Open"] * (1.0 + SLIPPAGE) if side == "LONG" else c_row["Open"] * (1.0 - SLIPPAGE)
        atr = p_row["ATR"]

        if not np.isfinite(atr) or atr <= 0:
            continue

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

        for j in range(i, min(i + 35, len(df_15))):
            fut = df_15.iloc[j]
            high, low = fut["High"], fut["Low"]

            if side == "LONG":
                if not be_active and high >= be_trigger:
                    current_sl = entry_price
                    be_active = True
                if low <= current_sl:
                    outcome = "BE" if be_active else "LOSS"
                    exit_price = entry_price if be_active else current_sl
                    exit_index = j
                    break
                if high >= tp:
                    outcome = "WIN"
                    exit_price = tp
                    exit_index = j
                    break
            else:
                if not be_active and low <= be_trigger:
                    current_sl = entry_price
                    be_active = True
                if high >= current_sl:
                    outcome = "BE" if be_active else "LOSS"
                    exit_price = entry_price if be_active else current_sl
                    exit_index = j
                    break
                if low <= tp:
                    outcome = "WIN"
                    exit_price = tp
                    exit_index = j
                    break

        notional = TRADE_MARGIN * LEVERAGE
        price_ret = (exit_price - entry_price) / entry_price if side == "LONG" else (entry_price - exit_price) / entry_price
        dollar_pnl = (notional * price_ret) - (notional * FEE_RATE * 2.0)

        trades.append({
            "Timestamp": t_curr,
            "ExitTimestamp": df_15.index[exit_index] if exit_index is not None else None,
            "Symbol": symbol,
            "Side": side,
            "Outcome": outcome,
            "Dollar_PnL": dollar_pnl,
            "Entry_Price": entry_price,
            "Exit_Price": exit_price,
        })

    return trades


def main():
    print("=" * 80)
    print("HUNTER-V129 — LIVE-SAFE EXECUTION ENGINE")
    print("=" * 80)

    now = datetime.now()
    start_dt = now - timedelta(days=DAYS)
    end_dt = now

    all_trades = []

    for symbol, lbank_symbol in SYMBOLS.items():
        print(f"Processing {symbol}...")
        df = fetch_chunk_data(lbank_symbol, start_dt, end_dt)
        if df is None or len(df) < 100:
            continue

        df_15, df_1h, df_4h = prepare_data(df)
        symbol_trades = run_live_safe_engine(symbol, df_15, df_1h, df_4h, start_dt, end_dt)
        all_trades.extend(symbol_trades)

    if not all_trades:
        print("No trades generated.")
        return

    trades_df = pd.DataFrame(all_trades)
    total_trades = len(trades_df)
    wins = trades_df[trades_df["Outcome"] == "WIN"]
    losses = trades_df[trades_df["Outcome"] == "LOSS"]
    net_pnl = float(trades_df["Dollar_PnL"].sum())
    win_rate = len(wins) / total_trades * 100.0

    print("=" * 80)
    print(f"LIVE-SAFE RESULTS (Total Trades: {total_trades})")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Net PnL: ${net_pnl:,.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
