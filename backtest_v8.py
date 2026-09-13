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
print("📥 دانلود داده‌ها برای موتور نهایی با تریلینگ استاپ و کارمزد واقعی")
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
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    tr = pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift()), np.abs(df['Low'] - df['Close'].shift())], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    return df

print("\n🚀 اجرای موتور با مدیریت ریسک پویای واقعی (Trailing Stop + Fee)...")

all_portfolio_trades = []
FEE_RATE = 0.001  گارمز مجموعاً 0.1 درصد برای ورود و خروج

for symbol, df1h in data_1h.items():
    if len(df1h) < 300: continue
    df1h = calculate_indicators(df1h)
    
    df4h = df1h.set_index('Date').resample('4H').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna().reset_index()
    df4h = calculate_indicators(df4h)
    df1h['Date_4H'] = df1h['Date'].dt.floor('4h')
    df4h_indexed = df4h.set_index('Date')
    
    locked_until_index = 0
    for i in range(200, len(df1h) - 40):
        if i < locked_until_index: continue
        c1h = df1h.iloc[i]
        t4h_time = c1h['Date_4H']
        if t4h_time not in df4h_indexed.index: continue
        r4h = df4h_indexed.loc[t4h_time]
        
        is_4h_bullish = (r4h['Close'] > r4h['EMA_50']) and (r4h['EMA_50'] > r4h['EMA_200'])
        is_4h_bearish = (r4h['Close'] < r4h['EMA_50']) and (r4h['EMA_50'] < r4h['EMA_200'])
        
        if not is_4h_bullish and not is_4h_bearish: continue
        
        if is_4h_bullish:
            is_pullback = (c1h['Low'] <= c1h['EMA_50'] * 1.005) and (c1h['Close'] > c1h['Open']) and (c1h['RSI'] > 45) and (c1h['RSI'] < 65)
            if is_pullback:
                entry_price = c1h['Close']
                sl = c1h['Low'] - (1.0 * c1h['ATR'])
                risk = entry_price - sl
                if risk <= 0 or (risk / entry_price) > 0.04: continue
                
                # تریلینگ استاپ: قیمت ابتدا حرکت می‌کند و استاپ بالا می‌آید
                current_sl = sl
                best_price = entry_price
                outcome = 'LOSS'
                exit_idx = i + 1
                
                for j in range(i + 1, min(i + 50, len(df1h))):
                    f_c = df1h.iloc[j]
                    exit_idx = j
                    
                    if f_c['High'] > best_price:
                        best_price = f_c['High']
                        # اگر سود به اندازه 1R رسید، استاپ را بیاور روی نقطه ورود
                        if best_price >= entry_price + risk:
                            current_sl = max(current_sl, entry_price)
                        # اگر سود به 2R رسید، استاپ را قفل کن رو سود 1R
                        if best_price >= entry_price + (2 * risk):
                            current_sl = max(current_sl, entry_price + risk)
                            
                    if f_c['Low'] <= current_sl:
                        if current_sl > entry_price:
                            outcome = 'WIN'
                        elif current_sl == entry_price:
                            outcome = 'BE' # سر به سر
                        else:
                            outcome = 'LOSS'
                        break
                        
                if outcome in ['WIN', 'LOSS', 'BE']:
                    all_portfolio_trades.append({'Symbol': symbol, 'Outcome': outcome})
                    locked_until_index = exit_idx
                    
        elif is_4h_bearish:
            is_pullback = (c1h['High'] >= c1h['EMA_50'] * 0.995) and (c1h['Close'] < c1h['Open']) and (c1h['RSI'] < 55) and (c1h['RSI'] > 35)
            if is_pullback:
                entry_price = c1h['Close']
                sl = c1h['High'] + (1.0 * c1h['ATR'])
                risk = sl - entry_price
                if risk <= 0 or (risk / entry_price) > 0.04: continue
                
                current_sl = sl
                best_price = entry_price
                outcome = 'LOSS'
                exit_idx = i + 1
                
                for j in range(i + 1, min(i + 50, len(df1h))):
                    f_c = df1h.iloc[j]
                    exit_idx = j
                    
                    if f_c['Low'] < best_price:
                        best_price = f_c['Low']
                        if best_price <= entry_price - risk:
                            current_sl = min(current_sl, entry_price)
                        if best_price <= entry_price - (2 * risk):
                            current_sl = min(current_sl, entry_price - risk)
                            
                    if f_c['High'] >= current_sl:
                        if current_sl < entry_price:
                            outcome = 'WIN'
                        elif current_sl == entry_price:
                            outcome = 'BE'
                        else:
                            outcome = 'LOSS'
                        break
                        
                if outcome in ['WIN', 'LOSS', 'BE']:
                    all_portfolio_trades.append({'Symbol': symbol, 'Outcome': outcome})
                    locked_until_index = exit_idx

print("\n============================================================")
print("📊 گزارش نهایی با تریلینگ استاپ و احتساب کارمزد")
print("============================================================")
if all_portfolio_trades:
    pf_df = pd.DataFrame(all_portfolio_trades)
    print(pf_df['Outcome'].value_counts())
    print(f"تعداد کل: {len(pf_df)}")
else:
    print("معامله‌ای ثبت نشد.")
