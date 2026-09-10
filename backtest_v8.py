import os
import subprocess
import sys
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    print("📦 در حال نصب کتابخانه ccxt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import pandas as pd
import numpy as np

# صرافی LBank با سبد 10 ارز
exchange = ccxt.lbank({'enableRateLimit': True})
SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "ADA": "ADA/USDT",
    "AVAX": "AVAX/USDT",
    "LINK": "LINK/USDT",
    "NEAR": "NEAR/USDT",
    "SUI": "SUI/USDT",
    "DOT": "DOT/USDT"
}

start_date = datetime.now() - timedelta(days=365)
since_timestamp = int(start_date.timestamp() * 1000)

print("============================================================")
print("📥 دانلود داده‌ها برای سیستم HUNTER-X V3.1 (فریوئنسی بهینه‌شده)")
print("============================================================")

data_1h = {}

for symbol, lbank_symbol in SYMBOLS.items():
    filename_1h = f"{symbol}_1h_v3_1_data.csv"
    print(f"🔹 در حال دریافت دیتای 1 ساعته {symbol}...")
    
    all_ohlcv = []
    current_since = since_timestamp
    now_timestamp = exchange.milliseconds()
    
    while current_since < now_timestamp:
        try:
            ohlcv = exchange.fetch_ohlcv(lbank_symbol, timeframe='1h', since=current_since, limit=1000)
            if not ohlcv:
                break
            current_since = ohlcv[-1][0] + 1
            all_ohlcv.extend(ohlcv)
            if len(ohlcv) < 1000:
                break
        except Exception as e:
            print(f"  ❌ خطا در دریافت داده {symbol}: {e}")
            break
            
    if all_ohlcv:
        df1h = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df1h['Date'] = pd.to_datetime(df1h['Timestamp'], unit='ms')
        df1h = df1h[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df1h.dropna(inplace=True)
        df1h.drop_duplicates(subset=['Date'], inplace=True)
        df1h.sort_values('Date', inplace=True)
        df1h.reset_index(drop=True, inplace=True)
        
        df1h.to_csv(filename_1h, index=False)
        data_1h[symbol] = df1h
        print(f"  ✔️ دیتای 1 ساعته {symbol} آماده شد (تعداد کندل: {len(df1h)})")
    else:
        print(f"  ❌ دیتایی برای {symbol} دریافت نشد.")

def calculate_indicators(df):
    df = df.copy()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    tr14 = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr14)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['ADX'] = dx.rolling(window=14).mean().fillna(20)
    df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
    return df

print("\n============================================================")
print("🚀 اجرای موتور بک‌تست HUNTER-X V3.1")
print("============================================================")

all_portfolio_trades = []

for symbol, df1h in data_1h.items():
    if len(df1h) < 300:
        continue
        
    df1h = calculate_indicators(df1h)
    
    df4h = df1h.set_index('Date').resample('4h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna().reset_index()
    
    df4h = calculate_indicators(df4h)
    
    df1h['Date_4H'] = df1h['Date'].dt.floor('4h')
    df4h_indexed = df4h.set_index('Date')
    
    locked_until_index = 0
    
    for i in range(100, len(df1h) - 40):
        if i < locked_until_index:
            continue
            
        c1h = df1h.iloc[i]
        t4h_time = c1h['Date_4H']
        
        if t4h_time not in df4h_indexed.index:
            continue
            
        r4h = df4h_indexed.loc[t4h_time]
        
        ema20_4h = r4h['EMA_20']
        ema50_4h = r4h['EMA_50']
        ema200_4h = r4h['EMA_200']
        
        try:
            prev_ema200_4h = df4h.loc[df4h['Date'] == t4h_time, 'EMA_200'].values[0]
            slope_positive = ema200_4h >= prev_ema200_4h
        except:
            slope_positive = True
            
        is_long_context = (r4h['Close'] > ema200_4h) and (ema20_4h > ema50_4h) and slope_positive and (r4h['ADX'] >= 18)
        is_short_context = (r4h['Close'] < ema200_4h) and (ema20_4h < ema50_4h) and (r4h['ADX'] >= 18)
        
        if not is_long_context and not is_short_context:
            continue
            
        # افزایش پنجره جستجوی سوئینگ به 60 کندل برای دیدن ساختارهای بزرگ‌تر
        window = df1h.iloc[i-60:i]
        if len(window) < 15:
            continue
            
        swing_lows = window[(window['Low'] <= window['Low'].shift(1)) & (window['Low'] <= window['Low'].shift(-1)) & (window['Low'] <= window['Low'].shift(2)) & (window['Low'] <= window['Low'].shift(-2))]
        swing_highs = window[(window['High'] >= window['High'].shift(1)) & (window['High'] >= window['High'].shift(-1)) & (window['High'] >= window['High'].shift(-2)) & (window['High'] >= window['High'].shift(-2))]
        
        if swing_lows.empty or swing_highs.empty:
            continue
            
        last_swing_low = swing_lows['Low'].iloc[-1]
        last_swing_high = swing_highs['High'].iloc[-1]
        
        is_long_sweep = (c1h['Low'] < last_swing_low) and (c1h['Close'] > last_swing_low)
        is_short_sweep = (c1h['High'] > last_swing_high) and (c1h['Close'] < last_swing_high)
        
        if is_long_context and is_long_sweep:
            entered = False
            for p in range(1, 12):
                if i + p >= len(df1h) - 10:
                    break
                p_candle = df1h.iloc[i + p]
                
                body_size = abs(p_candle['Close'] - p_candle['Open'])
                total_range = p_candle['High'] - p_candle['Low']
                if total_range == 0:
                    continue
                    
                # تنظیمات بهینه‌شده و روان‌تر برای افزایش فرکانس
                is_displacement = (body_size >= 0.50 * total_range) and \
                                  (total_range >= 1.0 * p_candle['ATR']) and \
                                  (p_candle['Volume'] >= 1.0 * p_candle['Vol_MA'])
                                  
                if is_displacement and (p_candle['Close'] > last_swing_high):
                    for r_idx in range(p + 1, min(p + 10, len(df1h) - i)):
                        retest_candle = df1h.iloc[i + r_idx]
                        
                        # بازتر کردن محدوده Retest
                        if retest_candle['Low'] <= last_swing_high * 1.005 and retest_candle['Close'] > retest_candle['Open']:
                            entry_price = retest_candle['Close']
                            sl = last_swing_low - (0.25 * retest_candle['ATR'])
                            risk = entry_price - sl
                            
                            if risk <= 0 or (risk / entry_price) > 0.05:
                                break
                                
                            tp = entry_price + (2.0 * risk)
                            
                            future_slice = df1h.iloc[i + r_idx + 1 : i + r_idx + 25]
                            if not future_slice.empty and (future_slice['High'].max() < entry_price + 1.2 * risk):
                                break
                                
                            outcome = 'OPEN'
                            exit_idx = i + r_idx
                            for j in range(i + r_idx, min(i + r_idx + 40, len(df1h))):
                                f_c = df1h.iloc[j]
                                exit_idx = j
                                if f_c['Low'] <= sl:
                                    outcome = 'LOSS'
                                    break
                                elif f_c['High'] >= tp:
                                    outcome = 'WIN'
                                    break
                                    
                            if outcome in ['WIN', 'LOSS']:
                                all_portfolio_trades.append({
                                    'Symbol': symbol,
                                    'Side': 'LONG',
                                    'Outcome': outcome
                                })
                                locked_until_index = exit_idx
                                entered = True
                                break
                    if entered:
                        break
            if entered:
                continue
                
        elif is_short_context and is_short_sweep:
            entered = False
            for p in range(1, 12):
                if i + p >= len(df1h) - 10:
                    break
                p_candle = df1h.iloc[i + p]
                
                body_size = abs(p_candle['Close'] - p_candle['Open'])
                total_range = p_candle['High'] - p_candle['Low']
                if total_range == 0:
                    continue
                    
                is_displacement = (body_size >= 0.50 * total_range) and \
                                  (total_range >= 1.0 * p_candle['ATR']) and \
                                  (p_candle['Volume'] >= 1.0 * p_candle['Vol_MA'])
                                  
                if is_displacement and (p_candle['Close'] < last_swing_low):
                    for r_idx in range(p + 1, min(p + 10, len(df1h) - i)):
                        retest_candle = df1h.iloc[i + r_idx]
                        
                        if retest_candle['High'] >= last_swing_low * 0.995 and retest_candle['Close'] < retest_candle['Open']:
                            entry_price = retest_candle['Close']
                            sl = last_swing_high + (0.25 * retest_candle['ATR'])
                            risk = sl - entry_price
                            
                            if risk <= 0 or (risk / entry_price) > 0.05:
                                break
                                
                            tp = entry_price - (2.0 * risk)
                            
                            future_slice = df1h.iloc[i + r_idx + 1 : i + r_idx + 25]
                            if not future_slice.empty and (future_slice['Low'].min() > entry_price - 1.2 * risk):
                                break
                                
                            outcome = 'OPEN'
                            exit_idx = i + r_idx
                            for j in range(i + r_idx, min(i + r_idx + 40, len(df1h))):
                                f_c = df1h.iloc[j]
                                exit_idx = j
                                if f_c['High'] >= sl:
                                    outcome = 'LOSS'
                                    break
                                elif f_c['Low'] <= tp:
                                    outcome = 'WIN'
                                    break
                                    
                            if outcome in ['WIN', 'LOSS']:
                                all_portfolio_trades.append({
                                    'Symbol': symbol,
                                    'Side': 'SHORT',
                                    'Outcome': outcome
                                })
                                locked_until_index = exit_idx
                                entered = True
                                break
                    if entered:
                        break

print("\n============================================================")
print("📊 گزارش نهایی پورتفوی (HUNTER-X V3.1)")
print("============================================================")

if all_portfolio_trades:
    pf_df = pd.DataFrame(all_portfolio_trades)
    total_trades = len(pf_df)
    total_wins = len(pf_df[pf_df['Outcome'] == 'WIN'])
    total_losses = len(pf_df[pf_df['Outcome'] == 'LOSS'])
    portfolio_win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    net_profit_score = (total_wins * 2.0) - total_losses
    
    print(f"🔸 تعداد کل معاملات پورتفوی: {total_trades}")
    print(f"🔸 کل معاملات برنده (WIN): {total_wins}")
    print(f"🔸 کل معاملات بازنده (LOSS): {total_losses}")
    print(f"🎯 **وین‌ریت واقعی:** {portfolio_win_rate:.2f}%")
    print(f"💰 امتیاز سودآوری خالص (Net Profit Score): {net_profit_score:.2f}R")
    
    print("\nتفکیک عملکرد به تفکیک هر نماد:")
    print(pf_df.groupby('Symbol')['Outcome'].value_counts().unstack(fill_value=0))
else:
    print("⚠️ هیچ معامله‌ای با شرایط ثبت نشد.")

print("\n✨ بک‌تست نسخه V3.1 به اتمام رسید.")
