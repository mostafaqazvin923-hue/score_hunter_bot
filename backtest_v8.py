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
print("📥 دانلود داده‌ها برای موتور فوق‌پیشرفته (هدف: وین‌ریت بالا + فیلتر حجم و مومنتوم)")
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
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

print("\n🚀 اجرای موتور با فیلترهای سخت‌گیرانه حجم و مومنتوم برای افزایش وین‌ریت...")

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
        c1h = df1h.iloc[i]
        prev_c1h = df1h.iloc[i-1]
        t4h_time = c1h['Date_4H']
        if t4h_time not in df4h_indexed.index: continue
        r4h = df4h_indexed.loc[t4h_time]
        
        is_bullish_market = (r4h['Close'] > r4h['EMA_50']) and (r4h['EMA_50'] > r4h['EMA_200']) and (r4h['RSI'] > 50)
        is_bearish_market = (r4h['Close'] < r4h['EMA_50']) and (r4h['EMA_50'] < r4h['EMA_200']) and (r4h['RSI'] < 50)
        
        if not is_bullish_market and not is_bearish_market: continue
        
        if is_bullish_market:
            lookback_slice = df1h.iloc[i-15:i]
            ob_candidates = lookback_slice[lookback_slice['Close'] < lookback_slice['Open']]
            if ob_candidates.empty: continue
            
            ob_low = ob_candidates['Low'].min()
            ob_high = ob_candidates['High'].max()
            
            # فیلترهای فوق‌پیشرفته: برخورد با ناحیه + حجم بالا + برگشت مومنتوم RSI
            is_mitigated = (
                (c1h['Low'] <= ob_high) and 
                (c1h['Close'] > c1h['Open']) and 
                (c1h['Close'] > ob_low) and
                (c1h['Volume'] > c1h['Volume_MA'] * 1.3) and # حجم تاییدکننده ورود نهنگ
                (c1h['RSI'] > prev_c1h['RSI']) and (c1h['RSI'] between_50_65 := (c1h['RSI'] > 40 and c1h['RSI'] < 65))
            )
            
            if is_mitigated:
                entry_price = c1h['Close']
                sl = ob_low - (0.4 * c1h['ATR'])
                risk = entry_price - sl
                
                if risk <= 0 or (risk / entry_price) > 0.03: continue
                
                # تنظیم ریسک به ریوارد روی ۱ به ۲.۵ برای تعادل بین وین‌ریت بالا و سودآوری
                tp = entry_price + (2.5 * risk)
                
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
                    
        elif is_bearish_market:
            lookback_slice = df1h.iloc[i-15:i]
            ob_candidates = lookback_slice[lookback_slice['Close'] > lookback_slice['Open']]
            if ob_candidates.empty: continue
            
            ob_low = ob_candidates['Low'].min()
            ob_high = ob_candidates['High'].max()
            
            is_mitigated = (
                (c1h['High'] >= ob_low) and 
                (c1h['Close'] < c1h['Open']) and 
                (c1h['Close'] < ob_high) and
                (c1h['Volume'] > c1h['Volume_MA'] * 1.3) and
                (c1h['RSI'] < prev_c1h['RSI']) and (c1h['RSI'] > 35 and c1h['RSI'] < 60)
            )
            
            if is_mitigated:
                entry_price = c1h['Close']
                sl = ob_high + (0.4 * c1h['ATR'])
                risk = sl - entry_price
                
                if risk <= 0 or (risk / entry_price) > 0.03: continue
                
                tp = entry_price - (2.5 * risk)
                
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
print("📊 گزارش نهایی موتور الیت (هدف وین‌ریت بالا)")
print("============================================================")
if all_portfolio_trades:
    pf_df = pd.DataFrame(all_portfolio_trades)
    wins = len(pf_df[pf_df['Outcome'] == 'WIN'])
    losses = len(pf_df[pf_df['Outcome'] == 'LOSS'])
    total = len(pf_df)
    win_rate = (wins / total) * 100 if total > 0 else 0
    net_score = (wins * 2.5) - losses
    
    print(pf_df['Outcome'].value_counts())
    print(f"🔸 تعداد کل معاملات (بسیار انتخابی): {total}")
    print(f"🎯 وین‌ریت جدید: {win_rate:.2f}%")
    print(f"💰 امتیاز سود خالص (Net Score با R:R 1:2.5): {net_score:.2f}R")
else:
    print("معامله‌ای ثبت نشد.")
