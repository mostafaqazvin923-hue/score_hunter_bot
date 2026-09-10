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
print("📥 دانلود داده‌ها برای سیستم HUNTER-X V9 (Corrected & Clean)")
print("============================================================")

data_1h = {}

for symbol, lbank_symbol in SYMBOLS.items():
    filename_1h = f"{symbol}_1h_v9_corrected_data.csv"
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
print("🚀 اجرای موتور بک‌تست اصلاح‌شده HUNTER-X V9")
print("============================================================")

all_trades = []
COMMISSION_RATE = 0.0008 # 0.08% کارمزد کل رفت‌وبرگشت (بدون ضرب در 2 اضافی)
SLIPPAGE_RATE = 0.0004   # 0.04% اسلیپیج

for symbol, df1h in data_1h.items():
    if len(df1h) < 300:
        continue
        
    df1h = calculate_indicators(df1h)
    
    # ساخت تایم‌فریم 4 ساعته کاملاً ایزوله
    df4h = df1h.set_index('Date').resample('4h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna().reset_index()
    
    df4h = calculate_indicators(df4h)
    
    # تفکیک 80% In-Sample و 20% Out-of-Sample
    split_idx = int(len(df1h) * 0.8)
    locked_until_index = 0
    
    i = 200
    while i < len(df1h) - 40:
        if i < locked_until_index:
            i += 1
            continue
            
        c1h = df1h.iloc[i]
        current_time = c1h['Date']
        data_sample = "OOS" if i >= split_idx else "IS"
        
        # 1️⃣ فیلتر اصلی 4H (بدون لوک‌آهد)
        closed_4h_time = current_time - timedelta(hours=4)
        available_4h = df4h[df4h['Date'] <= closed_4h_time]
        if available_4h.empty:
            i += 1
            continue
            
        r4h = available_4h.iloc[-1]
        try:
            prev_r4h = available_4h.iloc[-2]
            ema200_slope_up = r4h['EMA_200'] >= prev_r4h['EMA_200']
            ema200_slope_down = r4h['EMA_200'] <= prev_r4h['EMA_200']
        except:
            ema200_slope_up = True
            ema200_slope_down = True
            
        is_long_regime = (r4h['Close'] > r4h['EMA_200']) and (r4h['EMA_50'] > r4h['EMA_200']) and \
                         (r4h['EMA_20'] > r4h['EMA_50']) and (r4h['ADX'] >= 18) and ema200_slope_up
                         
        is_short_regime = (r4h['Close'] < r4h['EMA_200']) and (r4h['EMA_50'] < r4h['EMA_200']) and \
                          (r4h['EMA_20'] < r4h['EMA_50']) and (r4h['ADX'] >= 18) and ema200_slope_down
                          
        if not is_long_regime and not is_short_regime:
            i += 1
            continue
            
        # 2️⃣ پیدا کردن Impulse روی 1H
        body_size = abs(c1h['Close'] - c1h['Open'])
        total_range = c1h['High'] - c1h['Low']
        if total_range == 0:
            i += 1
            continue
            
        is_long_impulse = is_long_regime and (c1h['Close'] > c1h['Open']) and \
                          (body_size >= 0.55 * c1h['ATR']) and (total_range >= 1.0 * c1h['ATR']) and \
                          (c1h['Close'] >= c1h['Low'] + 0.75 * total_range) and (c1h['Volume'] >= c1h['Vol_MA'])
                          
        is_short_impulse = is_short_regime and (c1h['Close'] < c1h['Open']) and \
                           (body_size >= 0.55 * c1h['ATR']) and (total_range >= 1.0 * c1h['ATR']) and \
                           (c1h['Close'] <= c1h['High'] - 0.75 * total_range) and (c1h['Volume'] >= c1h['Vol_MA'])
                           
        if not is_long_impulse and not is_short_impulse:
            i += 1
            continue
            
        impulse_idx = i
        impulse_candle = c1h
        impulse_range = impulse_candle['High'] - impulse_candle['Low']
        
        # محاسبه سطوح فیبوناچی 50% تا 61.8%
        if is_long_impulse:
            fib_50 = impulse_candle['High'] - 0.5 * impulse_range
            fib_618 = impulse_candle['High'] - 0.618 * impulse_range
        else:
            fib_50 = impulse_candle['Low'] + 0.5 * impulse_range
            fib_618 = impulse_candle['Low'] + 0.618 * impulse_range
        
        # 3️⃣ & 4️⃣ بررسی Pullback و Reclaim در 1 تا 6 کندل بعد
        found_setup = False
        pullback_low = float('inf')
        pullback_high = float('-inf')
        reclaim_idx = -1
        
        for p_offset in range(1, 7):
            curr_idx = impulse_idx + p_offset
            if curr_idx >= len(df1h) - 10:
                break
            p_candle = df1h.iloc[curr_idx]
            
            if p_candle['Low'] < pullback_low: pullback_low = p_candle['Low']
            if p_candle['High'] > pullback_high: pullback_high = p_candle['High']
            
            if is_long_impulse:
                if p_candle['Close'] < impulse_candle['Low']:
                    break
                
                # شرط ناحیه فیبو (بین 50 تا 61.8 درصد) یا برخورد به EMA20
                touched_fib_zone = (p_candle['Low'] <= fib_50) and (p_candle['Low'] >= fib_618)
                touched_pullback_zone = touched_fib_zone or (p_candle['Low'] <= p_candle['EMA_20'])
                
                prev_p = df1h.iloc[curr_idx - 1]
                is_reclaim = (p_candle['Close'] > p_candle['Open']) and \
                             (p_candle['Close'] > prev_p['High']) and \
                             (p_candle['Close'] > p_candle['EMA_20']) and \
                             (p_candle['Volume'] >= 0.9 * p_candle['Vol_MA'])
                             
                if touched_pullback_zone and is_reclaim:
                    reclaim_idx = curr_idx
                    found_setup = True
                    break
                    
            elif is_short_impulse:
                if p_candle['Close'] > impulse_candle['High']:
                    break
                    
                touched_fib_zone = (p_candle['High'] >= fib_50) and (p_candle['High'] <= fib_618)
                touched_pullback_zone = touched_fib_zone or (p_candle['High'] >= p_candle['EMA_20'])
                
                prev_p = df1h.iloc[curr_idx - 1]
                is_reclaim = (p_candle['Close'] < p_candle['Open']) and \
                             (p_candle['Close'] < prev_p['Low']) and \
                             (p_candle['Close'] < p_candle['EMA_20']) and \
                             (p_candle['Volume'] >= 0.9 * p_candle['Vol_MA'])
                             
                if touched_pullback_zone and is_reclaim:
                    reclaim_idx = curr_idx
                    found_setup = True
                    break
                    
        if not found_setup or reclaim_idx + 1 >= len(df1h):
            i += 1
            continue
            
        # 5️⃣ ورود در Open کندل 1H بعدی
        entry_candle_idx = reclaim_idx + 1
        entry_candle = df1h.iloc[entry_candle_idx]
        
        if is_long_impulse:
            entry_price = entry_candle['Open'] * (1 + SLIPPAGE_RATE)
            sl = pullback_low - (0.25 * entry_candle['ATR'])
            risk = entry_price - sl
            
            if not (0.5 * entry_candle['ATR'] <= risk <= 2.0 * entry_candle['ATR']):
                i = entry_candle_idx
                continue
                
            tp = entry_price + (2.0 * risk)
            
            outcome = 'OPEN'
            exit_idx = entry_candle_idx
            max_favorable = 0
            max_adverse = 0
            
            for j in range(entry_candle_idx, min(entry_candle_idx + 50, len(df1h))):
                f_c = df1h.iloc[j]
                exit_idx = j
                
                cur_mfe = f_c['High'] - entry_price
                cur_mae = entry_price - f_c['Low']
                if cur_mfe > max_favorable: max_favorable = cur_mfe
                if cur_mae > max_adverse: max_adverse = cur_mae
                
                hit_tp = f_c['High'] >= tp
                hit_sl = f_c['Low'] <= sl
                
                if hit_sl:
                    outcome = 'LOSS'
                    break
                elif hit_tp:
                    outcome = 'WIN'
                    break
                    
            if outcome in ['WIN', 'LOSS']:
                net_R = 2.0 if outcome == 'WIN' else -1.0
                net_R -= COMMISSION_RATE # کسر کارمزد صحیح (بدون ضرب در 2 اضافی)
                
                all_trades.append({
                    'Symbol': symbol,
                    'Side': 'LONG',
                    'Sample': data_sample,
                    'Outcome': outcome,
                    'NetR': net_R,
                    'Date': entry_candle['Date'],
                    'MFE': max_favorable / risk,
                    'MAE': max_adverse / risk
                })
                locked_until_index = max(exit_idx + 6, entry_candle_idx + 6)
                i = locked_until_index
                continue
                
        elif is_short_impulse:
            entry_price = entry_candle['Open'] * (1 - SLIPPAGE_RATE)
            sl = pullback_high + (0.25 * entry_candle['ATR'])
            risk = sl - entry_price
            
            if not (0.5 * entry_candle['ATR'] <= risk <= 2.0 * entry_candle['ATR']):
                i = entry_candle_idx
                continue
                
            tp = entry_price - (2.0 * risk)
            
            outcome = 'OPEN'
            exit_idx = entry_candle_idx
            max_favorable = 0
            max_adverse = 0
            
            for j in range(entry_candle_idx, min(entry_candle_idx + 50, len(df1h))):
                f_c = df1h.iloc[j]
                exit_idx = j
                
                cur_mfe = entry_price - f_c['Low']
                cur_mae = f_c['High'] - entry_price
                if cur_mfe > max_favorable: max_favorable = cur_mfe
                if cur_mae > max_adverse: max_adverse = cur_mae
                
                hit_tp = f_c['Low'] <= tp
                hit_sl = f_c['High'] >= sl
                
                if hit_sl:
                    outcome = 'LOSS'
                    break
                elif hit_tp:
                    outcome = 'WIN'
                    break
                    
            if outcome in ['WIN', 'LOSS']:
                net_R = 2.0 if outcome == 'WIN' else -1.0
                net_R -= COMMISSION_RATE
                
                all_trades.append({
                    'Symbol': symbol,
                    'Side': 'SHORT',
                    'Sample': data_sample,
                    'Outcome': outcome,
                    'NetR': net_R,
                    'Date': entry_candle['Date'],
                    'MFE': max_favorable / risk,
                    'MAE': max_adverse / risk
                })
                locked_until_index = max(exit_idx + 6, entry_candle_idx + 6)
                i = locked_until_index
                continue
                
        i += 1

print("\n============================================================")
print("📊 گزارش جامع و حرفه‌ای سیستم HUNTER-X V9 (Corrected)")
print("============================================================")

if all_trades:
    tdf = pd.DataFrame(all_trades)
    tdf.to_csv("detailed_trades_v9_corrected.csv", index=False)
    
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
        
        sample_df = sample_df.sort_values('Date')
        sample_df['CumulativeR'] = sample_df['NetR'].cumsum()
        sample_df['Peak'] = sample_df['CumulativeR'].cummax()
        sample_df['Drawdown'] = sample_df['Peak'] - sample_df['CumulativeR']
        max_dd = sample_df['Drawdown'].max()
        
        print(f"🎯 تعداد کل: {t_count} | برد: {w_count} | باخت: {l_count}")
        print(f"🎯 وین‌ریت: {win_rate:.2f}% | سود خالص: {total_net_r:.2f}R")
        print(f"⚖️ Profit Factor: {profit_factor:.2f} | Expectancy: {expectancy:.3f}R")
        print(f"📉 Max Drawdown: {max_dd:.2f}R")
        print(f"📈 میانگین MFE: {sample_df['MFE'].mean():.2f}R | میانگین MAE: {sample_df['MAE'].mean():.2f}R")
        
    print("\n--- عملکرد تفکیکی نمادها و سمت‌ها (کل پورتفوی) ---")
    print(tdf.groupby(['Symbol', 'Side'])['Outcome'].value_counts().unstack(fill_value=0))
else:
    print("⚠️ هیچ معامله‌ای ثبت نشد.")

print("\n✨ بک‌تست V9 اصلاح‌شده به پایان رسید.")
