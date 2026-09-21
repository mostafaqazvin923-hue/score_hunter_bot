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
# HUNTER-V114 — 1-YEAR DAILY TREND PULLBACK ENGINE
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

LOOKBACK_DAYS = 365
EXEC_TIMEFRAME = "1d"

SLIPPAGE = 0.0005
FEE_RATE = 0.0007
INITIAL_CAPITAL = 2000.0
TRADE_MARGIN = 100.0
LEVERAGE = 3.0  # اهرم محافظه‌کارانه برای تایم‌فریم روزانه

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS + 100)
since_timestamp = int(start_date.timestamp() * 1000)

print("=" * 68)
print("HUNTER-V114 — 1-YEAR DAILY PULLBACK ENGINE INITIALIZED")
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
    print(f"Downloading daily data for {symbol}...")
    df_1d = fetch_ohlcv_data(lbank_symbol, EXEC_TIMEFRAME)

    if df_1d is None or len(df_1d) < 200:
        continue

    df_1d.set_index("Date", inplace=True)

    # محاسبه ATR برای حد ضرر داینامیک
    tr1 = df_1d["High"] - df_1d["Low"]
    tr2 = np.abs(df_1d["High"] - df_1d["Close"].shift(1))
    tr3 = np.abs(df_1d["Low"] - df_1d["Close"].shift(1))
    df_1d["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    # میانگین‌های متحرک برای تشخیص روند و پولبک
    df_1d["EMA_20"] = df_1d["Close"].ewm(span=20, adjust=False).mean()
    df_1d["EMA_50"] = df_1d["Close"].ewm(span=50, adjust=False).mean()
    df_1d["EMA_200"] = df_1d["Close"].ewm(span=200, adjust=False).mean()

    cutoff_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
    df_1d = df_1d[df_1d.index >= cutoff_date]

    df_1d.dropna(inplace=True)
    if len(df_1d) > 30:
        processed_data[symbol] = df_1d

print(f"Successfully loaded 1-year pullback symbols: {len(processed_data)}")

def run_backtest(processed_data):
    all_trades = []

    for symbol, df in processed_data.items():
        for i in range(10, len(df)):
            ts = df.index[i]
            prev = df.iloc[i - 1]
            curr = df.iloc[i]

            # فیلتر روند قدرتمند صعودی
            uptrend_strong = (prev["Close"] > prev["EMA_200"]) and (prev["EMA_50"] > prev["EMA_200"])
            
            # شرایط پولبک: قیمت به نزدیک EMA_20 اصلاح کرده بود و الان در حال برگشت (کندل صعودی) است
            near_pullback = prev["Low"] <= prev["EMA_20"] * 1.015
            bullish_bounce = (curr["Close"] > curr["Open"]) and (curr["Close"] > prev["Close"])

            valid_long = uptrend_strong and near_pullback and bullish_bounce

            if not valid_long:
                continue

            side = "LONG"
            entry_price = curr["Open"] * (1 + SLIPPAGE)
            atr = curr["ATR"]

            if np.isnan(atr) or atr <= 0:
                continue

            # حد ضرر پایین‌تر از ATR و حد سود با ریسک به ریوارد ۱ به ۳
            sl = entry_price - (2.0 * atr)
            tp = entry_price + (6.0 * atr)

            outcome = "LOSS"
            exit_price = sl
            for j in range(i + 1, min(i + 30, len(df))):
                future_c = df.iloc[j]
                if future_c["Low"] <= sl:
                    exit_price = sl
                    outcome = "LOSS"
                    break
                if future_c["High"] >= tp:
                    exit_price = tp
                    outcome = "WIN"
                    break

            price_ret = (exit_price - entry_price) / entry_price
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
    print("HUNTER-V114 BACKTEST RESULTS (DAILY TREND PULLBACK)")
    print("=" * 72)
    print(f"Total Trades: {n}")
    print(f"Overall Win Rate: {win_rate:.2f}%")
    print(f"Net Dollar PnL: ${net_pnl:,.2f}")
    print("=" * 72)
