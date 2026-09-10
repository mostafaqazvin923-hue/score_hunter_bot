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
print("📥 دانلود داده‌ها برای سیستم HUNTER-X V5 (Trend + Pullback + Momentum)")
print("============================================================")

data_1h = {}

for symbol, lbank_symbol in SYMBOLS.items():
    filename_1h = f"{symbol}_1h_v5_data.csv"
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
print("🚀 اجرای موتور بک‌تست HUNTER-X V5")
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
    
    locked_until_index = 0
    
    for i in range(100, len(df1h) - 40):
        if i < locked_until_index:
            continue
            
        c1h = df1h.iloc[i]
        prev_c1h = df1h.iloc[i-1]
        current_time = c1h['Date']
        
        # 1. رژیم روند 4 ساعته (بدون Look-ahead با استفاده از کندل بسته شده قبلی 4 ساعته)
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
            
        is_long_context = (r4h['Close'] > r4h['EMA_200']) and (r4h['EMA_20'] > r4h['EMA_50']) and \
                          (r4h['EMA_50'] > r4h['EMA_200']) and ema200_slope_up and (r4h['ADX'] >= 20)
                          
        is_short_context = (r4h['Close'] < r4h['EMA_200']) and (r4h['EMA_20'] < r4h['EMA_50']) and \
                           (r4h['EMA_50'] < r4h['EMA_200']) and ema200_slope_down and (r4h['ADX'] >= 20)
                           
        if not is_long_context and not is_short_context:
            continue
            
        # 2. بررسی ساختار بازار (Pivot با 2 کندل سمت راست برای اجتناب از Look-ahead)
        # برای سادگی و دقت بدون تاخیر زیاد، از سویینگ‌های تاییدشده در پنجره گذشته استفاده می‌کنیم
        window = df1h.iloc[i-50:i-2] # کسر 2 کندل برای رعایت ریجکت سمت راست
        if len(window) < 15:
            continue
            
        swing_lows = window[(window['Low'] <= window['Low'].shift(1)) & (window['Low'] <= window['Low'].shift(-1)) & 
                            (window['Low'] <= window['Low'].shift(2)) & (window['Low'] <= window['Low'].shift(-2))]
        swing_highs = window[(window['High'] >= window['High'].shift(1)) & (window['High'] >= window['High'].shift(-1)) & 
                             (window['High'] >= window['High'].shift(2)) & (window['High'] >= window['High'].shift(-2))]
                             
        if swing_lows.empty or swing_highs.empty:
            continue
            
        last_swing_low = swing_lows['Low'].iloc[-1]
        last_swing_high = swing_highs['High'].iloc[-1]
        
        # 3. بررسی پولبک و مومنتوم ریکلیم در 1H
        if is_long_context:
            # پولبک به حوالی EMA20 یا EMA50 (بدون نفوذ کامل به زیر EMA50)
            in_pullback = (prev_c1h['Low'] <= prev_c1h['EMA_20']) or (prev_c1h['Low'] <= prev_c1h['EMA_50'])
            not_broken_structure = prev_c1h['Close'] >= prev_c1h['EMA_50']
            
            # مومنتوم ریکلیم
            momentum_reclaim = (c1h['Close'] > c1h['Open']) and (c1h['Close'] > prev_c1h['High']) and (c1h['RSI'] > 50)
            
            # حجم تاییدکننده
            volume_confirmed = c1h['Volume'] >= (c1h['Vol_MA'] * 1.10)
            
            # فیلتر بیش‌ازحد کشیده‌نشدن از EMA20
            not_overextended = abs(c1h['Close'] - c1h['EMA_20']) <= (1.2 * c1h['ATR'])
            
            # ساختار بازار صعودی (Higher High / Higher Low نسبی)
            structure_ok = last_swing_low >= window['Low'].iloc[0]
            
            if in_pullback and not_broken_structure and momentum_reclaim and volume_confirmed and not_overextended and structure_ok:
                entry_price = c1h['Close']
                sl = last_swing_low - (0.25 * c1h['ATR'])
                risk = entry_price - sl
                
                # فیلتر اندازه ریسک (بین 0.6 تا 2.0 ATR)
                if (risk >= 0.6 * c1h['ATR']) and (risk <= 2.0 * c1h['ATR']) and (risk / entry_price <= 0.05):
                    tp = entry_price + (2.0 * risk)
                    
                    outcome = 'OPEN'
                    exit_idx = i
                    for j in range(i, min(i + 40, len(df1h))):
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
                        continue
                        
        elif is_short_context:
            in_pullback = (prev_c1h['High'] >= prev_c1h['EMA_20']) or (prev_c1h['High'] >= prev_c1h['EMA_50'])
            not_broken_structure = prev_c1h['Close'] <= prev_c1h['EMA_50']
            
            momentum_reclaim = (c1h['Close'] < c1h['Open']) and (c1h['Close'] < prev_c1h['Low']) and (c1h['RSI'] < 50)
            volume_confirmed = c1h['Volume'] >= (c1h['Vol_MA'] * 1.10)
            not_overextended = abs(c1h['Close'] - c1h['EMA_20']) <= (1.2 * c1h['ATR'])
            structure_ok = last_swing_high <= window['High'].iloc[0]
            
            if in_pullback and not_broken_structure and momentum_reclaim and volume_confirmed and not_overextended and structure_ok:
                entry_price = c1h['Close']
                sl = last_swing_high + (0.25 * c1h['ATR'])
                risk = sl - entry_price
                
                if (risk >= 0.6 * c1h['ATR']) and (risk <= 2.0 * c1h['ATR']) and (risk / entry_price <= 0.05):
                    tp = entry_price - (2.0 * risk)
                    
                    outcome = 'OPEN'
                    exit_idx = i
                    for j in range(i, min(i + 40, len(df1h))):
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
                        continue

print("\n============================================================")
print("📊 گزارش نهایی پورتفوی (HUNTER-X V5)")
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

print("\n✨ بک‌تست نسخه V5 به اتمام رسید.")
