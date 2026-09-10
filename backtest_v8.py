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
print("📥 دانلود داده‌ها برای سیستم HUNTER-X V7 (Compression + Expansion)")
print("============================================================")

data_1h = {}

for symbol, lbank_symbol in SYMBOLS.items():
    filename_1h = f"{symbol}_1h_v7_data.csv"
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
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # اندیکاتور فشردگی (Bollinger Bands Width یا ATR Ratio)
    df['ATR_MA'] = df['ATR'].rolling(window=20).mean()
    df['Compression'] = df['ATR'] < (df['ATR_MA'] * 0.8) # نشانه‌ی فشردگی نوسان
    
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
print("🚀 اجرای موتور بک‌تست پیشرفته HUNTER-X V7")
print("============================================================")

all_trades = []
COMMISSION_RATE = 0.0008 # 0.08% کارمزد رفت و برگشت کل
SLIPPAGE_RATE = 0.0005   # 0.05% اسلیپیج

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
    
    # جداسازی In-Sample (80%) و Out-of-Sample (20%)
    split_idx = int(len(df1h) * 0.8)
    
    locked_until_index = 0
    
    for i in range(200, len(df1h) - 40):
        if i < locked_until_index:
            continue
            
        c1h = df1h.iloc[i]
        prev_c1h = df1h.iloc[i-1]
        current_time = c1h['Date']
        
        # تعیین بخش داده (In-Sample یا OOS)
        data_sample = "OOS" if i >= split_idx else "IS"
        
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
            
        is_long_regime = (r4h['Close'] > r4h['EMA_200']) and ema200_slope_up and (r4h['ADX'] >= 22)
        is_short_regime = (r4h['Close'] < r4h['EMA_200']) and ema200_slope_down and (r4h['ADX'] >= 22)
        
        if not is_long_regime and not is_short_regime:
            continue
            
        # بررسی فشردگی نوسان در کندل‌های قبل (آیا اخیراً فشرده بوده؟)
        recent_compression = df1h.iloc[i-5:i]['Compression'].any()
        if not recent_compression:
            continue
            
        # انفجار بریک‌آوت (Expansion): حجم بالا + ATR بالا + بدنه قوی
        body_size = abs(c1h['Close'] - c1h['Open'])
        total_range = c1h['High'] - c1h['Low']
        if total_range == 0:
            continue
            
        is_expansion = (body_size >= 0.6 * total_range) and \
                       (total_range >= 1.3 * c1h['ATR']) and \
                       (c1h['Volume'] >= 1.3 * c1h['Vol_MA'])
                       
        if not is_expansion:
            continue
            
        # محدوده معاملاتی گذشته برای تعیین استابلاستر ساختاری
        window = df1h.iloc[i-30:i]
        recent_high = window['High'].max()
        recent_low = window['Low'].min()
        
        # شرایط LONG
        if is_long_regime and (c1h['Close'] > recent_high):
            raw_entry = c1h['Close']
            entry_price = raw_entry * (1 + SLIPPAGE_RATE)
            sl = recent_low - (0.5 * c1h['ATR'])
            risk = entry_price - sl
            
            if (risk >= 0.5 * c1h['ATR']) and (risk <= 2.5 * c1h['ATR']) and (risk / entry_price <= 0.05):
                tp = entry_price + (2.0 * risk)
                
                outcome = 'OPEN'
                exit_idx = i
                max_favorable = 0
                max_adverse = 0
                
                for j in range(i, min(i + 50, len(df1h))):
                    f_c = df1h.iloc[j]
                    exit_idx = j
                    
                    cur_mfe = f_c['High'] - entry_price
                    cur_mae = entry_price - f_c['Low']
                    if cur_mfe > max_favorable: max_favorable = cur_mfe
                    if cur_mae > max_adverse: max_adverse = cur_mae
                    
                    if f_c['Low'] <= sl:
                        outcome = 'LOSS'
                        break
                    elif f_c['High'] >= tp:
                        outcome = 'WIN'
                        break
                        
                if outcome in ['WIN', 'LOSS']:
                    net_R = 2.0 if outcome == 'WIN' else -1.0
                    net_R -= (COMMISSION_RATE * 2) # کسر کارمزد
                    
                    all_trades.append({
                        'Symbol': symbol,
                        'Side': 'LONG',
                        'Sample': data_sample,
                        'Outcome': outcome,
                        'NetR': net_R,
                        'Date': c1h['Date'],
                        'MFE': max_favorable / risk,
                        'MAE': max_adverse / risk
                    })
                    locked_until_index = exit_idx
                    
        # شرایط SHORT
        elif is_short_regime and (c1h['Close'] < recent_low):
            raw_entry = c1h['Close']
            entry_price = raw_entry * (1 - SLIPPAGE_RATE)
            sl = recent_high + (0.5 * c1h['ATR'])
            risk = sl - entry_price
            
            if (risk >= 0.5 * c1h['ATR']) and (risk <= 2.5 * c1h['ATR']) and (risk / entry_price <= 0.05):
                tp = entry_price - (2.0 * risk)
                
                outcome = 'OPEN'
                exit_idx = i
                max_favorable = 0
                max_adverse = 0
                
                for j in range(i, min(i + 50, len(df1h))):
                    f_c = df1h.iloc[j]
                    exit_idx = j
                    
                    cur_mfe = entry_price - f_c['Low']
                    cur_mae = f_c['High'] - entry_price
                    if cur_mfe > max_favorable: max_favorable = cur_mfe
                    if cur_mae > max_adverse: max_adverse = cur_mae
                    
                    if f_c['High'] >= sl:
                        outcome = 'LOSS'
                        break
                    elif f_c['Low'] <= tp:
                        outcome = 'WIN'
                        break
                        
                if outcome in ['WIN', 'LOSS']:
                    net_R = 2.0 if outcome == 'WIN' else -1.0
                    net_R -= (COMMISSION_RATE * 2)
                    
                    all_trades.append({
                        'Symbol': symbol,
                        'Side': 'SHORT',
                        'Sample': data_sample,
                        'Outcome': outcome,
                        'NetR': net_R,
                        'Date': c1h['Date'],
                        'MFE': max_favorable / risk,
                        'MAE': max_adverse / risk
                    })
                    locked_until_index = exit_idx

print("\n============================================================")
print("📊 گزارش جامع عملکرد HUNTER-X V7")
print("============================================================")

if all_trades:
    tdf = pd.DataFrame(all_trades)
    tdf.to_csv("detailed_trades_v7.csv", index=False)
    
    for sample_type in ['IS', 'OOS']:
        sample_df = tdf[tdf['Sample'] == sample_type]
        print(f"\n--- نتایج بخش {sample_type} (تعداد معاملات: {len(sample_df)}) ---")
        if sample_df.empty:
            print("معامله‌ای ثبت نشد.")
            continue
            
        t_count = len(sample_df)
        w_count = len(sample_df[sample_df['Outcome'] == 'WIN'])
        l_count = len(sample_df[sample_df['Outcome'] == 'LOSS'])
        win_rate = (w_count / t_count) * 100
        
        gross_profit = sample_df[sample_df['NetR'] > 0]['NetR'].sum()
        gross_loss = abs(sample_df[sample_df['NetR'] < 0]['NetR'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan
        
        expectancy = sample_df['NetR'].mean()
        total_net_r = sample_df['NetR'].sum()
        
        # محاسبه Max Drawdown
        sample_df = sample_df.sort_values('Date')
        sample_df['CumulativeR'] = sample_df['NetR'].cumsum()
        sample_df['Peak'] = sample_df['CumulativeR'].cummax()
        sample_df['Drawdown'] = sample_df['Peak'] - sample_df['CumulativeR']
        max_dd = sample_df['Drawdown'].max()
        
        print(f"🎯 وین‌ریت: {win_rate:.2f}% | سود خالص: {total_net_r:.2f}R")
        print(f"⚖️ Profit Factor: {profit_factor:.2f} | Expectancy: {expectancy:.3f}R")
        print(f"📉 Max Drawdown: {max_dd:.2f}R")
        print(f"📈 میانگین MFE: {sample_df['MFE'].mean():.2f}R | میانگین MAE: {sample_df['MAE'].mean():.2f}R")
        
    print("\n--- عملکرد تفکیکی نمادها کل ---")
    print(tdf.groupby('Symbol')['Outcome'].value_counts().unstack(fill_value=0))
else:
    print("⚠️ هیچ معامله‌ای با شرایط جدید ثبت نشد.")

print("\n✨ بک‌تست V7 به پایان رسید.")
