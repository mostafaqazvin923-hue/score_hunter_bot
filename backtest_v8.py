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
print("📥 دانلود داده‌ها برای سیستم HUNTER-X V6 (Stateful Structure + Pullback)")
print("============================================================")

data_1h = {}

for symbol, lbank_symbol in SYMBOLS.items():
    filename_1h = f"{symbol}_1h_v6_data.csv"
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
print("🚀 اجرای موتور بک‌تست HUNTER-X V6")
print("============================================================")

all_detailed_trades = []

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
    
    # شناسایی پیوت‌های Stateful بدون Look-ahead (با تاخیر 2 کندل تایید)
    df1h['Pivot_High'] = np.nan
    df1h['Pivot_Low'] = np.nan
    
    highs = df1h['High'].values
    lows = df1h['Low'].values
    
    for i in range(2, len(df1h) - 2):
        if highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
            df1h.loc[i+2, 'Pivot_High'] = highs[i] # ثبت دقیقاً بعد از 2 کندل تأیید
        if lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
            df1h.loc[i+2, 'Pivot_Low'] = lows[i]
            
    locked_until_index = 0
    
    for i in range(200, len(df1h) - 40):
        if i < locked_until_index:
            continue
            
        c1h = df1h.iloc[i]
        prev_c1h = df1h.iloc[i-1]
        current_time = c1h['Date']
        
        # رژیم 4 ساعته بدون Look-ahead
        closed_4h_time = current_time - timedelta(hours=4)
        available_4h = df4h[df4h['Date'] <= closed_4h_time]
        if available_4h.empty:
            continue
            
        r4h = available_4h.iloc[-1]
        try:
            prev_r4h = available_4h.iloc[-2]
            ema200_slope_up = r4h['EMA_200'] >= prev_r4h['EMA_200']
            ema200_slope_down = r4h['EMA_200'] <= prev_r4h['EMA_200']
        except:
            ema200_slope_up = True
            ema200_slope_down = True
            
        is_long_regime = (r4h['Close'] > r4h['EMA_200']) and ema200_slope_up and (r4h['ADX'] >= 20)
        is_short_regime = (r4h['Close'] < r4h['EMA_200']) and ema200_slope_down and (r4h['ADX'] >= 20)
        
        if not is_long_regime and not is_short_regime:
            continue
            
        # استخراج آخرین پیوت‌های ثبت‌شده تا لحظه جاری (کاملاً Stateful و بدون آینده‌نگری)
        history_slice = df1h.iloc[:i]
        valid_pivots_high = history_slice['Pivot_High'].dropna()
        valid_pivots_low = history_slice['Pivot_Low'].dropna()
        
        if valid_pivots_high.empty or valid_pivots_low.empty:
            continue
            
        last_ph = valid_pivots_high.iloc[-1]
        last_pl = valid_pivots_low.iloc[-1]
        
        # فیلتر حجم ثانویه
        volume_ok = c1h['Volume'] >= (c1h['Vol_MA'] * 1.0)
        
        # شرایط LONG: شکست ساختار صعودی + پولبک
        if is_long_regime and volume_ok:
            # بررسی اینکه آیا شکستِ سقف قبلی رخ داده و الان پولبک به آن زده شده
            body_size = abs(c1h['Close'] - c1h['Open'])
            total_range = c1h['High'] - c1h['Low']
            if total_range == 0:
                continue
                
            is_breakout = (c1h['Close'] > last_ph) and (body_size >= 0.5 * total_range)
            
            # جستجوی پولبک و کندل Continuation در چند کندل اخیر
            for r_idx in range(1, 6):
                if i + r_idx >= len(df1h) - 10:
                    break
                retest_candle = df1h.iloc[i + r_idx]
                
                # پولبک به حوالی سطح شکسته‌شده (last_ph) و تایید با کندل صعودی
                if retest_candle['Low'] <= last_ph * 1.005 and retest_candle['Close'] > retest_candle['Open']:
                    entry_price = retest_candle['Close']
                    sl = last_pl - (0.25 * retest_candle['ATR'])
                    risk = entry_price - sl
                    
                    if (risk >= 0.5 * retest_candle['ATR']) and (risk <= 2.5 * retest_candle['ATR']) and (risk / entry_price <= 0.05):
                        tp = entry_price + (2.0 * risk)
                        
                        outcome = 'OPEN'
                        exit_idx = i + r_idx
                        max_favorable = 0
                        max_adverse = 0
                        
                        for j in range(i + r_idx, min(i + r_idx + 50, len(df1h))):
                            f_c = df1h.iloc[j]
                            exit_idx = j
                            
                            current_mfe = f_c['High'] - entry_price
                            current_mae = entry_price - f_c['Low']
                            if current_mfe > max_favorable: max_favorable = current_mfe
                            if current_mae > max_adverse: max_adverse = current_mae
                            
                            if f_c['Low'] <= sl:
                                outcome = 'LOSS'
                                break
                            elif f_c['High'] >= tp:
                                outcome = 'WIN'
                                break
                                
                        if outcome in ['WIN', 'LOSS']:
                            all_detailed_trades.append({
                                'Symbol': symbol,
                                'Side': 'LONG',
                                'Outcome': outcome,
                                'MFE': max_favorable / risk,
                                'MAE': max_adverse / risk
                            })
                            locked_until_index = exit_idx
                            break
                            
        # شرایط SHORT: شکست ساختار نزولی + پولبک
        elif is_short_regime and volume_ok:
            body_size = abs(c1h['Close'] - c1h['Open'])
            total_range = c1h['High'] - c1h['Low']
            if total_range == 0:
                continue
                
            is_breakout = (c1h['Close'] < last_pl) and (body_size >= 0.5 * total_range)
            
            for r_idx in range(1, 6):
                if i + r_idx >= len(df1h) - 10:
                    break
                retest_candle = df1h.iloc[i + r_idx]
                
                if retest_candle['High'] >= last_pl * 0.995 and retest_candle['Close'] < retest_candle['Open']:
                    entry_price = retest_candle['Close']
                    sl = last_ph + (0.25 * retest_candle['ATR'])
                    risk = sl - entry_price
                    
                    if (risk >= 0.5 * retest_candle['ATR']) and (risk <= 2.5 * retest_candle['ATR']) and (risk / entry_price <= 0.05):
                        tp = entry_price - (2.0 * risk)
                        
                        outcome = 'OPEN'
                        exit_idx = i + r_idx
                        max_favorable = 0
                        max_adverse = 0
                        
                        for j in range(i + r_idx, min(i + r_idx + 50, len(df1h))):
                            f_c = df1h.iloc[j]
                            exit_idx = j
                            
                            current_mfe = entry_price - f_c['Low']
                            current_mae = f_c['High'] - entry_price
                            if current_mfe > max_favorable: max_favorable = current_mfe
                            if current_mae > max_adverse: max_adverse = current_mae
                            
                            if f_c['High'] >= sl:
                                outcome = 'LOSS'
                                break
                            elif f_c['Low'] <= tp:
                                outcome = 'WIN'
                                break
                                
                        if outcome in ['WIN', 'LOSS']:
                            all_detailed_trades.append({
                                'Symbol': symbol,
                                'Side': 'SHORT',
                                'Outcome': outcome,
                                'MFE': max_favorable / risk,
                                'MAE': max_adverse / risk
                            })
                            locked_until_index = exit_idx
                            break

print("\n============================================================")
print("📊 گزارش تفکیکی و جامع نسخه HUNTER-X V6")
print("============================================================")

if all_detailed_trades:
    trades_df = pd.DataFrame(all_detailed_trades)
    trades_df.to_csv("detailed_trades_v6.csv", index=False)
    
    total_t = len(trades_df)
    total_w = len(trades_df[trades_df['Outcome'] == 'WIN'])
    total_l = len(trades_df[trades_df['Outcome'] == 'LOSS'])
    wr = (total_w / total_t) * 100 if total_t > 0 else 0
    net_score = (total_w * 2.0) - total_l
    
    print(f"🔸 کل معاملات پورتفوی: {total_t} | برد: {total_w} | باخت: {total_l}")
    print(f"🎯 **وین‌ریت کل:** {wr:.2f}% | سود خالص: {net_score:.2f}R")
    print(f"📈 میانگین MFE: {trades_df['MFE'].mean():.2f}R | میانگین MAE: {trades_df['MAE'].mean():.2f}R")
    
    print("\n--- عملکرد تفکیکی به تفکیک نماد و سمت (Symbol & Side) ---")
    summary = trades_df.groupby(['Symbol', 'Side'])['Outcome'].value_counts().unstack(fill_value=0)
    print(summary)
else:
    print("⚠️ هیچ معامله‌ای ثبت نشد.")

print("\n✨ بک‌تست V6 به پایان رسید و لاگ‌ها ذخیره شدند.")
