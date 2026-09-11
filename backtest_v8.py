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

# اتصال به صرافی LBank و تعریف سبد 10 ارز
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
print("📥 دانلود داده‌های 1 ساعته واقعی از ال‌بنک برای 10 ارز مشخص")
print("============================================================")

data_1h = {}

for symbol, lbank_symbol in SYMBOLS.items():
    filename_1h = f"{symbol}_1h_lbank_data.csv"
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
        print(f"  ✔️ دیتای {symbol} آماده شد (تعداد کندل: {len(df1h)})")
    else:
        print(f"  ❌ دیتایی برای {symbol} دریافت نشد.")

# تابع محاسبه اندیکاتورهای پیشرفته (شامل ADX برای سنجش قدرت روند)
def calculate_indicators(df):
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2.0)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2.0)
    
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
    
    # محاسبه ADX برای فیلتر کردن بازارهای رِنج و ضعیف
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    tr14 = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / (tr14 + 1e-9))
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / (tr14 + 1e-9))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['ADX'] = dx.rolling(window=14).mean().fillna(20)
    
    return df

print("\n============================================================")
print("🚀 اجرای موتور بک‌تست بهینه‌شده با فیلترهای سخت‌گیرانه وین‌ریت بالا")
print("============================================================")

all_portfolio_trades = []

for symbol, df1h in data_1h.items():
    if len(df1h) < 300:
        continue
        
    df1h = calculate_indicators(df1h)
    
    # استفاده از '4h' به جای '4H' طبق استانداردهای جدید پانداس
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
    
    for i in range(200, len(df1h) - 40):
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
        
        # فیلتر روند سخت‌گیرانه 4 ساعته همراه با ADX بالا (روند قدرتمند)
        is_long_regime = (r4h['Close'] > ema200_4h) and (ema20_4h > ema50_4h) and (r4h['ADX'] > 25)
        is_short_regime = (r4h['Close'] < ema200_4h) and (ema20_4h < ema50_4h) and (r4h['ADX'] > 25)
        
        if not is_long_regime and not is_short_regime:
            continue
            
        # فیلتر حجم و تاییدیه پولبک دقیق در 1 ساعته
        avg_vol = df1h.iloc[i-20:i]['Volume'].mean()
        is_volume_confirmed = c1h['Volume'] > (avg_vol * 1.2)
        
        is_long_trigger = (c1h['Low'] <= c1h['BB_Lower']) and (c1h['RSI'] < 30) and is_volume_confirmed
        is_short_trigger = (c1h['High'] >= c1h['BB_Upper']) and (c1h['RSI'] > 70) and is_volume_confirmed
        
        if is_long_regime and is_long_trigger:
            entry_price = c1h['Close']
            sl = c1h['Low'] - (1.5 * c1h['ATR'])
            risk = entry_price - sl
            
            if risk <= 0:
                continue
                
            tp = entry_price + (2.0 * risk)
            
            outcome = 'OPEN'
            exit_idx = i
            for j in range(i + 1, min(i + 50, len(df1h))):
                future_c = df1h.iloc[j]
                exit_idx = j
                if future_c['Low'] <= sl:
                    outcome = 'LOSS'
                    break
                elif future_c['High'] >= tp:
                    outcome = 'WIN'
                    break
                    
            if outcome in ['WIN', 'LOSS']:
                all_portfolio_trades.append({
                    'Symbol': symbol,
                    'Side': 'LONG',
                    'Outcome': outcome
                })
                locked_until_index = exit_idx
                
        elif is_short_regime and is_short_trigger:
            entry_price = c1h['Close']
            sl = c1h['High'] + (1.5 * c1h['ATR'])
            risk = sl - entry_price
            
            if risk <= 0:
                continue
                
            tp = entry_price - (2.0 * risk)
            
            outcome = 'OPEN'
            exit_idx = i
            for j in range(i + 1, min(i + 50, len(df1h))):
                future_c = df1h.iloc[j]
                exit_idx = j
                if future_c['High'] >= sl:
                    outcome = 'LOSS'
                    break
                elif future_c['Low'] <= tp:
                    outcome = 'WIN'
                    break
                    
            if outcome in ['WIN', 'LOSS']:
                all_portfolio_trades.append({
                    'Symbol': symbol,
                    'Side': 'SHORT',
                    'Outcome': outcome
                })
                locked_until_index = exit_idx

print("\n============================================================")
print("📊 گزارش نهایی بک‌تست بهینه‌شده پورتفوی ارز دیجیتال")
print("============================================================")

if all_portfolio_trades:
    pf_df = pd.DataFrame(all_portfolio_trades)
    total_trades = len(pf_df)
    total_wins = len(pf_df[pf_df['Outcome'] == 'WIN'])
    total_losses = len(pf_df[pf_df['Outcome'] == 'LOSS'])
    portfolio_win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    net_profit_score = (total_wins * 2.0) - total_losses
    
    print(f"🔸 تعداد کل معاملات کل سبد: {total_trades}")
    print(f"🔸 معاملات برنده (WIN): {total_wins}")
    print(f"🔸 معاملات بازنده (LOSS): {total_losses}")
    print(f"🎯 **وین‌ریت تجمیعی پورتفوی:** {portfolio_win_rate:.2f}%")
    print(f"💰 **امتیاز سودآوری خالص (رتبه R):** {net_profit_score:.2f}R")
    
    print("\nتفکیک عملکرد به تفکیک هر ارز:")
    print(pf_df.groupby('Symbol')['Outcome'].value_counts().unstack(fill_value=0))
else:
    print("⚠️ هیچ معامله‌ای با شرایط تعیین شده ثبت نشد.")

print("\n✨ بک‌تست بهینه‌شده با موفقیت به پایان رسید.")
