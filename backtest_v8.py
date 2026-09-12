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
print("📥 دانلود داده‌های 1 ساعته و ساخت دیتای 4 ساعته و 1 ساعته همگام")
print("============================================================")

data_1h = {}
data_4h = {}

for symbol, lbank_symbol in SYMBOLS.items():
    filename_1h = f"{symbol}_1h_multitf_data.csv"
    print(f"🔹 در حال دریافت و پردازش دیتای {symbol}...")
    
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
        df1h.set_index('Date', inplace=True)
        
        # ساخت کندل‌های 4 ساعته استاندارد برای تعیین روند کلان
        df4h = df1h.resample('4H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna().reset_index()
        
        df1h.reset_index(inplace=True)
        
        data_1h[symbol] = df1h
        data_4h[symbol] = df4h
        print(f"  ✔️ دیتای {symbol} آماده شد (1 ساعته: {len(df1h)} کندل | 4 ساعته: {len(df4h)} کندل)")
    else:
        print(f"  ❌ دیتایی برای {symbol} دریافت نشد.")

def calculate_indicators(df):
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
    return df

print("\n============================================================")
print("🚀 اجرای موتور بک‌تست حرفه‌ای دوطرفه (روند 4H + پولبک/حجم 1H + TP 2% / SL 1%)")
print("============================================================")

all_portfolio_trades = []

for symbol in SYMBOLS.keys():
    if symbol not in data_1h or symbol not in data_4h:
        continue
        
    df1h = data_1h[symbol].copy()
    df4h = data_4h[symbol].copy()
    
    if len(df1h) < 300 or len(df4h) < 100:
        continue
        
    df1h = calculate_indicators(df1h)
    df4h = calculate_indicators(df4h)
    
    # مپ کردن تاریخ 4 ساعته روی دیتافریم 1 ساعته برای دسترسی سریع به روند کلان
    df1h['Date_4H'] = df1h['Date'].dt.floor('4h')
    df4h_indexed = df4h.set_index('Date')
    
    locked_until_index = 0
    
    # پیمایش کندل به کندل در تایم‌فریم 1 ساعته
    for i in range(250, len(df1h) - 20):
        
        # رعایت کامل قفل پوزیشن (عدم همپوشانی و جلوگیری از سیگنال روی کندل تسویه)
        if i < locked_until_index:
            continue
            
        c1h = df1h.iloc[i]
        t4h_time = c1h['Date_4H']
        
        if t4h_time not in df4h_indexed.index:
            continue
            
        r4h = df4h_indexed.loc[t4h_time]
        
        # 1. تشخیص جهت روند کلان در تایم‌فریم 4 ساعته
        is_long_macro = (r4h['Close'] > r4h['EMA_200']) and (r4h['EMA_20'] > r4h['EMA_50']) and (r4h['EMA_50'] > r4h['EMA_200'])
        is_short_macro = (r4h['Close'] < r4h['EMA_200']) and (r4h['EMA_20'] < r4h['EMA_50']) and (r4h['EMA_50'] < r4h['EMA_200'])
        
        if not is_long_macro and not is_short_macro:
            continue
            
        # بررسی حجم و فیلتر انرژی در تایم‌فریم 1 ساعته (کندل بسته شده)
        # حجم کندل فعلی باید بالاتر از میانگین حجم باشد تا حرکت فیک نباشد
        if c1h['Volume'] < c1h['Vol_MA'] * 1.1:
            continue
            
        # بررسی پولبک و تاییدیه مومنتوم
        lookback_slice = df1h.iloc[i-5:i]
        
        # شرایط لانگ: روند 4 ساعته صعودی + کندل 1 ساعته صعودی قدرتمند بعد از پولبک
        if is_long_macro:
            is_bullish_candle = (c1h['Close'] > c1h['Open']) and ((c1h['High'] - c1h['Close']) < (c1h['Close'] - c1h['Open']))
            if not is_bullish_candle:
                continue
                
            # ورود در بازگشایی کندل بعدی (جلوگیری از ورود وسط کندل)
            entry_idx = i + 1
            if entry_idx >= len(df1h):
                break
            
            entry_candle = df1h.iloc[entry_idx]
            entry_price = entry_candle['Open']
            
            tp = entry_price * 1.02  # حد سود ثابت 2 درصد
            sl = entry_price * 0.99  # حد ضرر ثابت 1 درصد
            
            outcome = None
            exit_idx = entry_idx
            
            # بررسی تحقق هدایت قیمت بدون نگاه به آینده
            for j in range(entry_idx, len(df1h)):
                f_c = df1h.iloc[j]
                exit_idx = j
                
                # اولویت بررسی حد ضرر برای محافظه‌کاری در بک‌تست
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
                # قفل کردن ربات تا حداقل یک کندل بعد از تسویه کامل معامله
                locked_until_index = exit_idx + 1
                
        # شرایط شورت: روند 4 ساعته نزولی + کندل 1 ساعته نزولی قدرتمند
        elif is_short_macro:
            is_bearish_candle = (c1h['Close'] < c1h['Open']) and ((c1h['Close'] - c1h['Low']) < (c1h['Open'] - c1h['Close']))
            if not is_bearish_candle:
                continue
                
            entry_idx = i + 1
            if entry_idx >= len(df1h):
                break
                
            entry_candle = df1h.iloc[entry_idx]
            entry_price = entry_candle['Open']
            
            tp = entry_price * 0.98  # حد سود ثابت 2 درصد پایین‌تر
            sl = entry_price * 1.01  # حد ضرر ثابت 1 درصد بالاتر
            
            outcome = None
            exit_idx = entry_idx
            
            for j in range(entry_idx, len(df1h)):
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
                locked_until_index = exit_idx + 1

print("\n============================================================")
print("📊 گزارش نهایی پورتفوی دوطرفه (روند 4H + تریگر 1H + TP 2% / SL 1%)")
print("============================================================")

if all_portfolio_trades:
    pf_df = pd.DataFrame(all_portfolio_trades)
    total_trades = len(pf_df)
    total_wins = len(pf_df[pf_df['Outcome'] == 'WIN'])
    total_losses = len(pf_df[pf_df['Outcome'] == 'LOSS'])
    portfolio_win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    
    # ریسک به ریوارد 1 به 2 (سود 2% در برابر ضرر 1%)
    net_profit_score = (total_wins * 2.0) - total_losses
    
    print(f"🔸 تعداد کل معاملات پورتفوی: {total_trades}")
    print(f"🔸 معاملات برنده (WIN): {total_wins}")
    print(f"🔸 معاملات بازنده (LOSS): {total_losses}")
    print(f"🎯 **وین‌ریت تجمیعی پورتفوی:** {portfolio_win_rate:.2f}%")
    print(f"💰 **امتیاز سودآوری خالص (Net Profit Score):** {net_profit_score:.2f}R")
    
    print("\nتفکیک عملکرد به تفکیک هر نماد:")
    print(pf_df.groupby('Symbol')['Outcome'].value_counts().unstack(fill_value=0))
else:
    print("⚠️ هیچ معامله‌ای با این شرایط ثبت نشد.")

print("\n✨ پایان بک‌تست یک‌ساله.")
