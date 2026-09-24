import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd

# ============================================================
# HUNTER-V130-B — STRICT NO-LOOKAHEAD AUDITED ENGINE
# ============================================================

EXCHANGE = ccxt.lbank({
    "enableRateLimit": True,
    "timeout": 20000
})

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "SUI": "SUI/USDT",
    "AVAX": "AVAX/USDT",
    "NEAR": "NEAR/USDT",
    "ADA": "ADA/USDT",
    "BNB": "BNB/USDT",
    "APT": "APT/USDT",
    "CRV": "CRV/USDT",
    "ONDO": "ONDO/USDT",
    "PENDLE": "PENDLE/USDT",
    "ICP": "ICP/USDT",
    "WIF": "WIF/USDT",
}

CORRELATION_CLUSTERS = {
    "BTC": "MAJOR", "ETH": "MAJOR", "SOL": "L1", "SUI": "L1",
    "AVAX": "L1", "NEAR": "L1", "ADA": "L1", "BNB": "L1",
    "APT": "L1", "CRV": "DEFI", "ONDO": "DEFI", "PENDLE": "DEFI",
    "ICP": "OTHER", "WIF": "MEME",
}

TIMEFRAME_BASE = "15m"
SLIPPAGE = 0.0003
FEE_RATE = 0.0007
TRADE_MARGIN = 100.0
LEVERAGE = 50.0
DAYS = 365
MAX_LOSS_STREAK = 4


def fetch_chunk_data(lbank_symbol, start_dt, end_dt):
    since_ts = int((start_dt - timedelta(days=15)).timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    all_ohlcv = []
    current_since = since_ts

    try:
        while current_since < end_ts:
            batch = EXCHANGE.fetch_ohlcv(
                lbank_symbol, timeframe="15m", since=current_since, limit=1000,
            )
            if not batch:
                break
            all_ohlcv.extend(batch)
            last_ts = batch[-1][0]
            if last_ts <= current_since:
                break
            current_since = last_ts + 1
            if len(batch) < 1000 or last_ts >= end_ts:
                break
            time.sleep(0.2)
    except Exception as e:
        print(f"  خطا در دریافت داده {lbank_symbol}: {e}")
        return None

    if not all_ohlcv:
        return None

    df = pd.DataFrame(all_ohlcv, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.set_index("Date", inplace=True)
    return df[(df.index >= start_dt) & (df.index <= end_dt)]


def prepare_data(df_15m):
    df_15m = df_15m.copy()

    # 4H Regime
    df_4h = df_15m.resample("4h").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()
    df_4h["EMA_50"] = df_4h["Close"].ewm(span=50, adjust=False).mean()
    df_4h["EMA_200"] = df_4h["Close"].ewm(span=200, adjust=False).mean()
    df_4h["EMA_Slope"] = df_4h["EMA_200"] - df_4h["EMA_200"].shift(5)

    df_4h["Regime_Bullish"] = (
        (df_4h["Close"] > df_4h["EMA_200"])
        & (df_4h["EMA_50"] > df_4h["EMA_200"])
        & (df_4h["EMA_Slope"] > 0)
    )
    df_4h["Regime_Bearish"] = (
        (df_4h["Close"] < df_4h["EMA_200"])
        & (df_4h["EMA_50"] < df_4h["EMA_200"])
        & (df_4h["EMA_Slope"] < 0)
    )

    # 1H Structure (Safe Shift)
    df_1h = df_15m.resample("1h").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()
    df_1h["Swing_High"] = df_1h["High"].rolling(5).max().shift(1)
    df_1h["Swing_Low"] = df_1h["Low"].rolling(5).min().shift(1)

    # 15M Indicators
    df_15m["ATR"] = (df_15m["High"] - df_15m["Low"]).rolling(14).mean()
    df_15m["Body"] = (df_15m["Close"] - df_15m["Open"]).abs()
    df_15m["Avg_Body"] = df_15m["Body"].rolling(20).mean()
    df_15m["Range"] = df_15m["High"] - df_15m["Low"]

    return df_15m, df_1h, df_4h


def run_v130_engine(symbol, df_15, df_1h, df_4h, start_dt, end_dt):
    trades = []
    cluster = CORRELATION_CLUSTERS.get(symbol, "OTHER")
    consecutive_losses = 0
    pause_counter = 0

    i = 50
    while i < len(df_15) - 1:
        t_curr = df_15.index[i]
        if t_curr < start_dt or t_curr > end_dt:
            i += 1
            continue

        if pause_counter > 0:
            pause_counter -= 1
            if pause_counter == 0:
                consecutive_losses = 0
            i += 1
            continue

        c_row = df_15.iloc[i]
        p_row = df_15.iloc[i - 1]

        h_sub = df_1h[df_1h.index <= t_curr]
        h_4sub = df_4h[df_4h.index <= t_curr]

        if len(h_sub) < 15 or len(h_4sub) < 1:
            i += 1
            continue

        regime_bull = bool(h_4sub.iloc[-1]["Regime_Bullish"])
        regime_bear = bool(h_4sub.iloc[-1]["Regime_Bearish"])

        if not regime_bull and not regime_bear:
            i += 1
            continue

        recent_lows = h_sub["Low"].iloc[-15:-2]
        recent_highs = h_sub["High"].iloc[-15:-2]
        if len(recent_lows) == 0 or len(recent_highs) == 0:
            i += 1
            continue

        min_support = recent_lows.min()
        max_resistance = recent_highs.max()

        atr_15m = c_row["ATR"]
        if not np.isfinite(atr_15m) or atr_15m <= 0:
            i += 1
            continue

        sweep_penetration = 0.15 * atr_15m
        sweep_low = (p_row["Low"] < (min_support - sweep_penetration)) and (p_row["Close"] > min_support)
        sweep_high = (p_row["High"] > (max_resistance + sweep_penetration)) and (p_row["Close"] < max_resistance)

        avg_body = c_row["Avg_Body"]
        if not np.isfinite(avg_body) or avg_body <= 0:
            i += 1
            continue

        displacement_up = (
            regime_bull and sweep_low
            and c_row["Close"] > c_row["Open"]
            and c_row["Body"] >= 0.8 * atr_15m
            and (c_row["Body"] / c_row["Range"] >= 0.60)
        )

        displacement_down = (
            regime_bear and sweep_high
            and c_row["Close"] < c_row["Open"]
            and c_row["Body"] >= 0.8 * atr_15m
            and (c_row["Body"] / c_row["Range"] >= 0.60)
        )

        if not displacement_up and not displacement_down:
            i += 1
            continue

        # اجرای درست: ورود در Open کندل بعدی (i + 1)
        next_row = df_15.iloc[i + 1]
        next_time = df_15.index[i + 1]

        if displacement_up:
            side = "LONG"
            entry_price = next_row["Open"] * (1.0 + SLIPPAGE)
            sl = min_support - (0.15 * atr_15m)
            risk = entry_price - sl
            if risk <= 0: risk = atr_15m
            tp = entry_price + (2.0 * risk)
        else:
            side = "SHORT"
            entry_price = next_row["Open"] * (1.0 - SLIPPAGE)
            sl = max_resistance + (0.15 * atr_15m)
            risk = sl - entry_price
            if risk <= 0: risk = atr_15m
            tp = entry_price - (2.0 * risk)

        # اسکن پوزیشن از کندل ورود به بعد
        outcome = "LOSS"
        exit_price = sl
        exit_time = next_time
        hit_occurred = False

        for j in range(i + 1, len(df_15)):
            fut = df_15.iloc[j]
            f_time = df_15.index[j]
            high, low = fut["High"], fut["Low"]

            if side == "LONG":
                hit_sl = low <= sl
                hit_tp = high >= tp
                if hit_sl and hit_tp:
                    outcome = "LOSS"; exit_price = sl; exit_time = f_time; hit_occurred = True; break
                elif hit_sl:
                    outcome = "LOSS"; exit_price = sl; exit_time = f_time; hit_occurred = True; break
                elif hit_tp:
                    outcome = "WIN"; exit_price = tp; exit_time = f_time; hit_occurred = True; break
            else:
                hit_sl = high >= sl
                hit_tp = low <= tp
                if hit_sl and hit_tp:
                    outcome = "LOSS"; exit_price = sl; exit_time = f_time; hit_occurred = True; break
                elif hit_sl:
                    outcome = "LOSS"; exit_price = sl; exit_time = f_time; hit_occurred = True; break
                elif hit_tp:
                    outcome = "WIN"; exit_price = tp; exit_time = f_time; hit_occurred = True; break

        if not hit_occurred:
            i += 1
            continue

        notional = TRADE_MARGIN * LEVERAGE
        price_ret = (exit_price - entry_price) / entry_price if side == "LONG" else (entry_price - exit_price) / entry_price
        dollar_pnl = (notional * price_ret) - (notional * FEE_RATE * 2.0)

        trades.append({
            "Timestamp": next_time,
            "ExitTimestamp": exit_time,
            "Symbol": symbol,
            "Cluster": cluster,
            "Side": side,
            "Outcome": outcome,
            "Dollar_PnL": dollar_pnl,
            "Entry_Price": entry_price,
            "Exit_Price": exit_price,
        })

        if outcome == "LOSS":
            consecutive_losses += 1
            if consecutive_losses >= MAX_LOSS_STREAK:
                pause_counter = 16
        else:
            consecutive_losses = 0

        i = j + 1  # پرش به بعد از بسته شدن معامله برای جلوگیری از تداخل

    return trades


def main():
    print("=" * 80)
    print("HUNTER-V130-B — AUDITED NO-LOOKAHEAD BACKTEST")
    print("=" * 80)

    now = datetime.now()
    start_dt = now - timedelta(days=DAYS)
    end_dt = now

    all_trades = []
    for symbol, lbank_symbol in SYMBOLS.items():
        print(f"بررسی نماد: {symbol}...")
        df = fetch_chunk_data(lbank_symbol, start_dt, end_dt)
        if df is None or len(df) < 100:
            continue
        df_15, df_1h, df_4h = prepare_data(df)
        symbol_trades = run_v130_engine(symbol, df_15, df_1h, df_4h, start_dt, end_dt)
        all_trades.extend(symbol_trades)
        print(f"  -> تعداد معاملات: {len(symbol_trades)}")

    if not all_trades:
        print("\nهیچ معامله‌ای ثبت نشد.")
        return

    trades_df = pd.DataFrame(all_trades)
    trades_df.sort_values("Timestamp", inplace=True)
    trades_df.reset_index(drop=True, inplace=True)

    total_trades = len(trades_df)
    wins = trades_df[trades_df["Outcome"] == "WIN"]
    losses = trades_df[trades_df["Outcome"] == "LOSS"]
    win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
    net_pnl = float(trades_df["Dollar_PnL"].sum())
    gross_profit = float(wins["Dollar_PnL"].sum()) if len(wins) > 0 else 0.0
    gross_loss = abs(float(losses["Dollar_PnL"].sum())) if len(losses) > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    trades_df["Cumulative_PnL"] = trades_df["Dollar_PnL"].cumsum()
    trades_df["Peak"] = trades_df["Cumulative_PnL"].cummax()
    trades_df["Drawdown"] = trades_df["Cumulative_PnL"] - trades_df["Peak"]
    max_dd = float(trades_df["Drawdown"].min())

    print("\n" + "=" * 80)
    print("📊 گزارش نهایی و کاملاً سالم (HUNTER-V130-B)")
    print("=" * 80)
    print(f"تعداد کل معاملات:        {total_trades}")
    print(f"وین‌ریت (Win Rate):       {win_rate:.2f}%")
    print(f"مجموع سود/زیان خالص:    ${net_pnl:,.2f}")
    print(f"فاکتور سود (Profit Factor): {profit_factor:.2f}")
    print(f"حداکثر افت سرمایه (Max DD): ${max_dd:,.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
