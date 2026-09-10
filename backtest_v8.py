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
print("📥 دانلود داده‌ها و اجرای بک‌تست بدون نگاه به آینده (Fixed 1% SL / 2% TP)")
print("============================================================")

data_1h = {}

for symbol, lbank_symbol in SYMBOLS.items():
    filename_1h = f"{symbol}_1h_clean_data.csv"
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
    df['ADX'] = dx.rolling(window=14).mean()
    return df

print("\n============================================================")
print("🚀 اجرای موتور بک‌تست پاک‌سازی شده (بدون Look-ahead)")
print("============================================================")

all_portfolio_trades = []

for symbol, df1h in data_1h.items():
    if len(df1h) < 300:
        continue
        
    df1h = calculate_indicators(df1h)
    
    # ساخت تایم‌فریم 4 ساعته بدون لوک‌آهد (با برچسب لبه چپ)
    df4h = df1h.set_index('Date').resample('4h', label='left', closed='left').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna().reset_index()
    
    df4h = calculate_indicators(df4h)
    
    locked_until_index = 0
    
    for i in range(200, len(df1h) - 40):
        if i < locked_until_index:
            continue
            
        c1h = df1h.iloc[i]
        current_time = c1h['Date']
        
        # فیلتر 4 ساعته کاملاً بسته‌شده (بدون دسترسی به کندل در حال تشکیل)
        closed_4h_time = current_time - timedelta(hours=4)
        available_4h = df4h[df4h['Date'] <= closed_4h_time]
        if len(available_4h) < 2:
            continue
            
        r4h = available_4h.iloc[-1]
        prev_r4h = available_4h.iloc[-2]
        
        if pd.isna(r4h['ADX']) or pd.isna(r4h['RSI']):
            continue
            
        slope_positive = r4h['EMA_200'] >= prev_r4h['EMA_200']
        
        ema20_4h = r4h['EMA_20']
        ema50_4h = r4h['EMA_50']
        ema200_4h = r4h['EMA_200']
        
        is_long_regime = (r4h['Close'] > ema200_4h) and (ema20_4h > ema50_4h) and (ema50_4h > ema200_4h) and slope_positive and (r4h['ADX'] >= 20) and (r4h['RSI'] > 55)
        is_short_regime = (r4h['Close'] < ema200_4h) and (ema20_4h < ema50_4h) and (ema50_4h < ema200_4h) and (r4h['ADX'] >= 20) and (r4h['RSI'] < 45)
        
        if not is_long_regime and not is_short_regime:
            continue
            
        lookback_slice = df1h.iloc[i-15:i]
        struct_high = lookback_slice['High'].max()
        struct_low = lookback_slice['Low'].min()
        
        avg_vol = lookback_slice['Volume'].mean()
        is_breakout_long = (c1h['Close'] > struct_high) and (c1h['Volume'] >= avg_vol * 1.1)
        is_breakout_short = (c1h['Close'] < struct_low) and (c1h['Volume'] >= avg_vol * 1.1)
        
        if is_long_regime and is_breakout_long:
            entered = False
            for p in range(1, 14):
                if i + p >= len(df1h) - 10:
                    break
                p_candle = df1h.iloc[i + p]
                
                if p_candle['Low'] <= struct_high * 1.003: 
                    if p_candle['Close'] > p_candle['Open'] and p_candle['RSI'] > 50:
                        entry_price = p_candle['Close']
                        
                        # ✨ قانون جدید: استاپ ثابت 1 درصدی و تیک‌پرافیت ثابت 2 درصدی
                        sl = entry_price * 0.99
                        tp = entry_price * 1.02
                        
                        outcome = 'OPEN'
                        exit_idx = i + p + 1
                        for j in range(i + p + 1, min(i + p + 60, len(df1h))):
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
                continue
                
        elif is_short_regime and is_breakout_short:
            entered = False
            for p in range(1, 14):
                if i + p >= len(df1h) - 10:
                    break
                p_candle = df1h.iloc[i + p]
                
                if p_candle['High'] >= struct_low * 0.997:
                    if p_candle['Close'] < p_candle['Open'] and p_candle['RSI'] < 50:
                        entry_price = p_candle['Close']
                        
                        # ✨ قانون جدید: استاپ ثابت 1 درصدی و تیک‌پرافیت ثابت 2 درصدی برای شورت
                        sl = entry_price * 1.01
                        tp = entry_price * 0.98
                        
                        outcome = 'OPEN'
                        exit_idx = i + p + 1
                        for j in range(i + p + 1, min(i + p + 60, len(df1h))):
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

print("\n============================================================")
print("📊 گزارش تجمیعی نهایی پورتفوی (پاک‌سازی شده از تقلب نگاه به آینده)")
print("============================================================")

if all_portfolio_trades:
    pf_df = pd.DataFrame(all_portfolio_trades)
    total_trades = len(pf_df)
    total_wins = len(pf_df[pf_df['Outcome'] == 'WIN'])
    total_losses = len(pf_df[pf_df['Outcome'] == 'LOSS'])
    portfolio_win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    net_profit_score = (total_wins * 2.0) - total_losses
    
    print(f"🔸 تعداد کل معاملات کل سبد (پورتفوی): {total_trades}")
    print(f"🔸 کل معاملات برنده (WIN): {total_wins}")
    print(f"🔸 کل معاملات بازنده (LOSS): {total_losses}")
    print(f"🎯 **وین‌ریت تجمیعی کل پورتفوی (Portfolio Win Rate):** {portfolio_win_rate:.2f}%")
    print(f"💰 امتیاز سودآوری خالص (Net Profit Score): {net_profit_score:.2f}R")
    
    print("\nتفکیک عملکرد به تفکیک هر نماد:")
    print(pf_df.groupby('Symbol')['Outcome'].value_counts().unstack(fill_value=0))
else:
    print("⚠️ هیچ معامله‌ای با شرایط ثبت نشد.")

print("\n✨ بک‌تست پاک‌سازی شده به اتمام رسید.")
