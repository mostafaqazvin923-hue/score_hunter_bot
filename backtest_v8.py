import os
import subprocess
import sys
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import pandas as pd
import numpy as np

exchange = ccxt.lbank({'enableRateLimit': True})
SYMBOLS = {
    "BTC": "BTC/USDT", "ETH": "ETH/USDT", "SOL": "SOL/USDT",
    "XRP": "XRP/USDT", "ADA": "ADA/USDT", "AVAX": "AVAX/USDT",
    "LINK": "LINK/USDT", "NEAR": "NEAR/USDT", "SUI": "SUI/USDT", "DOT": "DOT/USDT"
}

start_date = datetime.now() - timedelta(days=365)
since_timestamp = int(start_date.timestamp() * 1000)

print("============================================================")
print("📥 دانلود داده‌ها برای موتور تک‌تیرانداز (Macro Trend + Liquidity Sweep + R:R 1:2)")
print("============================================================")

data_1h = {}
for symbol, lbank_symbol in SYMBOLS.items():
    all_ohlcv = []
    current_since = since_timestamp
    now_timestamp = exchange.milliseconds()
    
    while current_since < now_timestamp:
        try:
            ohlcv = exchange.fetch_ohlcv(lbank_symbol, timeframe='1h', since=current_since, limit=1000)
            if not ohlcv: break
            current_since = ohlcv[-1][0] + 1
            all_ohlcv.extend(ohlcv)
            if len(ohlcv) < 1000: break
        except:
            break
            
    if all_ohlcv:
        df1h = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df1h['Date'] = pd.to_datetime(df1h['Timestamp'], unit='ms')
        df1h = df1h[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna().drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
        data_1h[symbol] = df1h

def calculate_indicators(df):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    return df

print("\n🚀 اجرای موتور تک‌تیرانداز با ریسک به ریوارد ۱ به ۲...")

all_portfolio_trades = []

for symbol, df1h in data_1h.items():
    if len(df1h) < 400: continue
    df1h = calculate_indicators(df1h)
    
    df4h = df1h.set_index('Date').resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna().reset_index()
    df4h = calculate_indicators(df4h)
    df1h['Date_4H'] = df1h['Date'].dt.floor('4h')
    df4h_indexed = df4h.set_index('Date')
    
    locked_until_index = 0
    for i in range(250, len(df1h) - 70):
        if i < locked_until_index: continue
        
        c = df1h.iloc[i]
        t4h_time = c['Date_4H']
        if t4h_time not in df4h_indexed.index: continue
        r4h = df4h_indexed.loc[t4h_time]
        
        is_bullish_macro = (r4h['Close'] > r4h['EMA_50']) and (r4h['EMA_50'] > r4h['EMA_200'])
        is_bearish_macro = (r4h['Close'] < r4h['EMA_50']) and (r4h['EMA_50'] < r4h['EMA_200'])
        
        if not is_bullish_macro and not is_bearish_macro: continue
        
        lookback = df1h.iloc[i-25:i]
        local_high = lookback['High'].max()
        local_low = lookback['Low'].min()
        
        if is_bullish_macro:
            swept_low = (c['Low'] < local_low) and (c['Close'] > local_low) and (c['Close'] > c['Open'])
            if not swept_low: continue
            
            entry_price = c['Close']
            sl = c['Low'] - (0.4 * c['ATR'])
            risk = entry_price - sl
            
            if risk <= 0 or (risk / entry_price) > 0.035: continue
            
            # تنظیم ریسک به ریوارد روی ۱ به ۲
            tp = entry_price + (2.0 * risk)
            
            outcome = 'OPEN'
            exit_idx = i + 1
            for j in range(i + 1, min(i + 80, len(df1h))):
                f_c = df1h.iloc[j]
                exit_idx = j
                if f_c['Low'] <= sl:
                    outcome = 'LOSS'
                    break
                elif f_c['High'] >= tp:
                    outcome = 'WIN'
                    break
                    
            if outcome in ['WIN', 'LOSS']:
                all_portfolio_trades.append({'Symbol': symbol, 'Outcome': outcome})
                locked_until_index = exit_idx
                
        elif is_bearish_macro:
            swept_high = (c['High'] > local_high) and (c['Close'] < local_high) and (c['Close'] < c['Open'])
            if not swept_high: continue
            
            entry_price = c['Close']
            sl = c['High'] + (0.4 * c['ATR'])
            risk = sl - entry_price
            
            if risk <= 0 or (risk / entry_price) > 0.035: continue
            
            # تنظیم ریسک به ریوارد روی ۱ به ۲
            tp = entry_price - (2.0 * risk)
            
            outcome = 'OPEN'
            exit_idx = i + 1
            for j in range(i + 1, min(i + 80, len(df1h))):
                f_c = df1h.iloc[j]
                exit_idx = j
                if f_c['High'] >= sl:
                    outcome = 'LOSS'
                    break
                elif f_c['Low'] <= tp:
                    outcome = 'WIN'
                    break
                    
            if outcome in ['WIN', 'LOSS']:
                all_portfolio_trades.append({'Symbol': symbol, 'Outcome': outcome})
                locked_until_index = exit_idx

print("\n============================================================")
print("📊 گزارش نهایی موتور تک‌تیرانداز (Sniper Macro Liquidity + R:R 1:2)")
print("============================================================")
if all_portfolio_trades:
    pf_df = pd.DataFrame(all_portfolio_trades)
    wins = len(pf_df[pf_df['Outcome'] == 'WIN'])
    losses = len(pf_df[pf_df['Outcome'] == 'LOSS'])
    total = len(pf_df)
    win_rate = (wins / total) * 100 if total > 0 else 0
    net_score = (wins * 2.0) - losses
    
    print(pf_df['Outcome'].value_counts())
    print(f"🔸 تعداد کل معاملات: {total}")
    print(f"🎯 وین‌ریت جدید: {win_rate:.2f}%")
    print(f"💰 امتیاز سود خالص (Net Score با R:R 1:2): {net_score:.2f}R")
else:
    print("معامله‌ای ثبت نشد.")
