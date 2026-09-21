import os
import subprocess
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

# ============================================================
# HUNTER-V112 — 4H MACRO STRUCTURAL TREND ENGINE
# ============================================================

exchange = ccxt.lbank({"enableRateLimit": True, "timeout": 20000})

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "LINK": "LINK/USDT",
    "UNI": "UNI/USDT",
    "ICP": "ICP/USDT",
    "INJ": "INJ/USDT",
    "ATOM": "ATOM/USDT",
    "RENDER": "RENDER/USDT",
    "XLM": "XLM/USDT",
    "AAVE": "AAVE/USDT",
    "WIF": "WIF/USDT",
    "ONDO": "ONDO/USDT",
    "DOGE": "DOGE/USDT",
    "BNB": "BNB/USDT",
    "ADA": "ADA/USDT",
}

LOOKBACK_DAYS = 180  # داده‌های ۶ ماه گذشته در تایم‌فریم ۴ ساعته
EXEC_TIMEFRAME = "4h"

SLIPPAGE = 0.0005
FEE_RATE = 0.0007
INITIAL_CAPITAL = 2000.0
TRADE_MARGIN = 100.0
LEVERAGE = 10.0  # اهرم امن‌تر برای تایم‌فریم بالا

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)

print("=" * 68)
print("HUNTER-V112 — 4H MACRO TREND ENGINE INITIALIZED")
print("=" * 68)

processed_data = {}

def fetch_ohlcv_data(lbank_symbol, timeframe):
    all_ohlcv = []
    current_since = since_timestamp
    
    try:
        while current_since < exchange.milliseconds():
            batch = exchange.fetch_ohlcv(lbank_symbol, timeframe=timeframe, since=current_since, limit=1000)
            if not batch:
                break
            all_ohlcv.extend(batch)
            last_ts = batch[-1][0]
            if last_ts <= current_since:
                break
            current_since = last_ts + 1
            if len(batch) < 1000:
                break
    except Exception as e:
        print(f"Error fetching {lbank_symbol}: {e}")
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
    return df

for symbol, lbank_symbol in SYMBOLS.items():
    print(f"Downloading 4h data for {symbol}...")
    df_4h = fetch_ohlcv_data(lbank_symbol, EXEC_TIMEFRAME)

    if df_4h is None or len(df_4h) < 200:
        continue

    df_4h.set_index("Date", inplace=True)

    # اندیکاتورهای ساختاری روند کلان
    tr1 = df_4h["High"] - df_4h["Low"]
    tr2 = np.abs(df_4h["High"] - df_4h["Close"].shift(1))
    tr3 = np.abs(df_4h["Low"] - df_4h["Close"].shift(1))
    df_4h["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    
    df_4h["EMA_Fast"] = df_4h["Close"].ewm(span=20, adjust=False).mean()
    df_4h["EMA_Slow"] = df_4h["Close"].ewm(span=50, adjust=False).mean()
    df_4h["EMA_Macro"] = df_4h["Close"].ewm(span=200, adjust=False).mean()

    df_4h.dropna(inplace=True)
    if len(df_4h) > 50:
        processed_data[symbol] = df_4h

print(f"Successfully loaded 4h symbols: {len(processed_data)}")

def run_backtest(processed_data):
    all_trades = []

    for symbol, df in processed_data.items():
        for i in range(200, len(df)):
            ts = df.index[i]
            prev = df.iloc[i - 1]
            curr = df.iloc[i]

            # شرایط روند قدرتمند در تایم فریم ۴ ساعته
            # قیمت بالای میانگین ۲۰۰ و تقاطع ایماهای سریع و کند
            macro_bullish = prev["Close"] > prev["EMA_Macro"]
            macro_bearish = prev["Close"] < prev["EMA_Macro"]

            crossover_long = (prev["EMA_Fast"] > prev["EMA_Slow"]) and (df.iloc[i - 2]["EMA_Fast"] <= df.iloc[i - 2]["EMA_Slow"])
            crossover_short = (prev["EMA_Fast"] < prev["EMA_Slow"]) and (df.iloc[i - 2]["EMA_Fast"] >= df.iloc[i - 2]["EMA_Slow"])

            valid_long = macro_bullish and crossover_long
            valid_short = macro_bearish and crossover_short

            if not (valid_long or valid_short):
                continue

            side = "LONG" if valid_long else "SHORT"
            entry_price = curr["Open"] * (1 + SLIPPAGE) if side == "LONG" else curr["Open"] * (1 - SLIPPAGE)
            atr = curr["ATR"]

            if np.isnan(atr) or atr <= 0:
                continue

            # حد ضرر و حد سود بزرگتر متناسب با تایم فریم ۴ ساعته
            if side == "LONG":
                sl = entry_price - (2.0 * atr)
                tp = entry_price + (5.0 * atr)
            else:
                sl = entry_price + (2.0 * atr)
                tp = entry_price - (5.0 * atr)

            # بررسی نتیجه در کندل‌های بعدی ۴ ساعته (حداکثر ۳۰ کندل معادل ۵ روز)
            outcome = "LOSS"
            exit_price = sl
            for j in range(i + 1, min(i + 30, len(df))):
                future_c = df.iloc[j]
                if side == "LONG":
                    if future_c["Low"] <= sl:
                        exit_price = sl
                        outcome = "LOSS"
                        break
                    if future_c["High"] >= tp:
                        exit_price = tp
                        outcome = "WIN"
                        break
                else:
                    if future_c["High"] >= sl:
                        exit_price = sl
                        outcome = "LOSS"
                        break
                    if future_c["Low"] <= tp:
                        exit_price = tp
                        outcome = "WIN"
                        break

            price_ret = (exit_price - entry_price) / entry_price if side == "LONG" else (entry_price - exit_price) / entry_price
            dollar_pnl = (TRADE_MARGIN * LEVERAGE * price_ret) - (TRADE_MARGIN * LEVERAGE * FEE_RATE * 2)

            all_trades.append({
                "Timestamp": ts,
                "Symbol": symbol,
                "Side": side,
                "Outcome": outcome,
                "Dollar_PnL": dollar_pnl
            })

    return pd.DataFrame(all_trades)

if __name__ == "__main__":
    trades_df = run_backtest(processed_data)
    n = len(trades_df)
    win_rate = (trades_df["Outcome"].eq("WIN").mean() * 100) if n > 0 else 0.0
    net_pnl = float(trades_df["Dollar_PnL"].sum()) if n > 0 else 0.0

    print("=" * 72)
    print("HUNTER-V112 BACKTEST RESULTS (4H MACRO TREND)")
    print("=" * 72)
    print(f"Total Trades: {n}")
    print(f"Overall Win Rate: {win_rate:.2f}%")
    print(f"Net Dollar PnL: ${net_pnl:,.2f}")
    print("=" * 72)
