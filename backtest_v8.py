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
# HUNTER-V111 — ULTRA-FAST OPTIMIZED QUANT ENGINE
# ============================================================

exchange = ccxt.lbank({"enableRateLimit": True, "timeout": 15000})

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

LOOKBACK_DAYS = 30  # کاهش به ۳۰ روز برای اجرای فوق‌العاده سریع و بدون تایم‌اوت
EXEC_TIMEFRAME = "15m"
MAX_DAILY_TRADES = 3

SLIPPAGE = 0.0002
FEE_RATE = 0.0007
INITIAL_CAPITAL = 2000.0
TRADE_MARGIN = 100.0
LEVERAGE = 20.0

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)

print("=" * 68)
print("HUNTER-V111 — FAST OPTIMIZED ENGINE INITIALIZED")
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
    print(f"Downloading data for {symbol}...")
    df_15m = fetch_ohlcv_data(lbank_symbol, EXEC_TIMEFRAME)

    if df_15m is None or len(df_15m) < 100:
        continue

    df_15m.set_index("Date", inplace=True)

    # اندیکاتورهای سریع و برداری (Vectorized)
    tr1 = df_15m["High"] - df_15m["Low"]
    tr2 = np.abs(df_15m["High"] - df_15m["Close"].shift(1))
    tr3 = np.abs(df_15m["Low"] - df_15m["Close"].shift(1))
    df_15m["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    df_15m["EMA50"] = df_15m["Close"].ewm(span=50, adjust=False).mean()
    
    delta = df_15m["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df_15m["RSI"] = 100 - (100 / (1 + (gain / loss)))

    df_15m.dropna(inplace=True)
    if len(df_15m) > 50:
        processed_data[symbol] = df_15m

print(f"Successfully loaded symbols: {len(processed_data)}")

def run_backtest(processed_data):
    all_trades = []
    current_day = None
    daily_trade_count = 0

    for symbol, df in processed_data.items():
        for i in range(50, len(df)):
            ts = df.index[i]
            ts_date = ts.date()
            if current_day != ts_date:
                current_day = ts_date
                daily_trade_count = 0

            if daily_trade_count >= MAX_DAILY_TRADES:
                continue

            prev = df.iloc[i - 1]
            curr = df.iloc[i]

            valid_long = (prev["Close"] > prev["EMA50"]) and (45 < prev["RSI"] < 65) and (prev["Close"] > prev["Open"])
            valid_short = (prev["Close"] < prev["EMA50"]) and (35 < prev["RSI"] < 55) and (prev["Close"] < prev["Open"])

            if not (valid_long or valid_short):
                continue

            side = "LONG" if valid_long else "SHORT"
            entry_price = curr["Open"] * (1 + SLIPPAGE) if side == "LONG" else curr["Open"] * (1 - SLIPPAGE)
            atr = curr["ATR"]

            if np.isnan(atr) or atr <= 0:
                continue

            if side == "LONG":
                sl = entry_price - (1.5 * atr)
                tp = entry_price + (3.0 * atr)
            else:
                sl = entry_price + (1.5 * atr)
                tp = entry_price - (3.0 * atr)

            # بررسی نتیجه در کندل‌های بعدی (حداکثر ۲۰ کندل)
            outcome = "LOSS"
            exit_price = sl
            for j in range(i + 1, min(i + 20, len(df))):
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
                "Symbol": symbol,
                "Side": side,
                "Outcome": outcome,
                "Dollar_PnL": dollar_pnl
            })
            daily_trade_count += 1

    return pd.DataFrame(all_trades)

if __name__ == "__main__":
    trades_df = run_backtest(processed_data)
    n = len(trades_df)
    win_rate = (trades_df["Outcome"].eq("WIN").mean() * 100) if n > 0 else 0.0
    net_pnl = float(trades_df["Dollar_PnL"].sum()) if n > 0 else 0.0

    print("=" * 72)
    print("HUNTER-V111 BACKTEST RESULTS (FAST ENGINE)")
    print("=" * 72)
    print(f"Total Trades: {n}")
    print(f"Overall Win Rate: {win_rate:.2f}%")
    print(f"Net Dollar PnL: ${net_pnl:,.2f}")
    print("=" * 72)
