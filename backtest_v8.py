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
# HUNTER-V113 — 1-YEAR DAILY INSTITUTIONAL BREAKOUT & ADX ENGINE
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

LOOKBACK_DAYS = 365  # بک‌تست دقیقاً یک‌ساله در تایم فریم روزانه
EXEC_TIMEFRAME = "1d"

SLIPPAGE = 0.0005
FEE_RATE = 0.0007
INITIAL_CAPITAL = 2000.0
TRADE_MARGIN = 100.0
LEVERAGE = 5.0  # اهرم امن و منطقی برای تایم فریم روزانه

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS + 60)  # بافر برای محاسبه اندیکاتورهای بلندمدت
since_timestamp = int(start_date.timestamp() * 1000)

print("=" * 68)
print("HUNTER-V113 — 1-YEAR DAILY INSTITUTIONAL ENGINE INITIALIZED")
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

    if df_1d is None or len(df_1d) < 100:
        continue

    df_1d.set_index("Date", inplace=True)

    # 1. محاسبه ATR برای مدیریت ریسک
    tr1 = df_1d["High"] - df_1d["Low"]
    tr2 = np.abs(df_1d["High"] - df_1d["Close"].shift(1))
    tr3 = np.abs(df_1d["Low"] - df_1d["Close"].shift(1))
    df_1d["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    # 2. کانال دونچیان (Donchian Channel 20 روزه برای شکست ساختار)
    df_1d["Donchian_High"] = df_1d["High"].shift(1).rolling(20).max()
    df_1d["Donchian_Low"] = df_1d["Low"].shift(1).rolling(20).min()

    # 3. شاخص قدرت روند (ADX) برای جلوگیری از ورود در بازارهای خنثی
    plus_dm = df_1d["High"].diff()
    minus_dm = df_1d["Low"].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    df_1d["Plus_DM"] = pd.Series(plus_dm, index=df_1d.index)
    df_1d["Minus_DM"] = pd.Series(minus_dm, index=df_1d.index)
    
    tr = df_1d["ATR"] * 14
    smoothed_tr = tr.ewm(alpha=1/14, adjust=False).mean()
    smoothed_plus_dm = df_1d["Plus_DM"].ewm(alpha=1/14, adjust=False).mean()
    smoothed_minus_dm = df_1d["Minus_DM"].ewm(alpha=1/14, adjust=False).mean()

    plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
    minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df_1d["ADX"] = dx.rolling(14).mean()

    # فیلتر دقیق بازه یک ساله گذشته
    cutoff_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
    df_1d = df_1d[df_1d.index >= cutoff_date]

    df_1d.dropna(inplace=True)
    if len(df_1d) > 30:
        processed_data[symbol] = df_1d

print(f"Successfully loaded 1-year daily symbols: {len(processed_data)}")

def run_backtest(processed_data):
    all_trades = []

    for symbol, df in processed_data.items():
        for i in range(25, len(df)):
            ts = df.index[i]
            prev = df.iloc[i - 1]
            curr = df.iloc[i]

            # استراتژی سازمانی: شکست کانال سقف/کف + فیلتر قدرت روند ADX بالای ۲۵
            adx_strong = prev["ADX"] > 25
            breakout_long = prev["Close"] >= prev["Donchian_High"]
            breakout_short = prev["Close"] <= prev["Donchian_Low"]

            valid_long = adx_strong and breakout_long
            valid_short = adx_strong and breakout_short

            if not (valid_long or valid_short):
                continue

            side = "LONG" if valid_long else "SHORT"
            entry_price = curr["Open"] * (1 + SLIPPAGE) if side == "LONG" else curr["Open"] * (1 - SLIPPAGE)
            atr = curr["ATR"]

            if np.isnan(atr) or atr <= 0:
                continue

            # حد ضرر و حد سود مبتنی بر ATR روزانه
            if side == "LONG":
                sl = entry_price - (2.5 * atr)
                tp = entry_price + (6.0 * atr)
            else:
                sl = entry_price + (2.5 * atr)
                tp = entry_price - (6.0 * atr)

            # بررسی نتیجه در کندل‌های بعدی روزانه (حداکثر ۲۰ روز معاملاتی)
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
    print("HUNTER-V113 BACKTEST RESULTS (1-YEAR DAILY BREAKOUT)")
    print("=" * 72)
    print(f"Total Trades: {n}")
    print(f"Overall Win Rate: {win_rate:.2f}%")
    print(f"Net Dollar PnL: ${net_pnl:,.2f}")
    print("=" * 72)
