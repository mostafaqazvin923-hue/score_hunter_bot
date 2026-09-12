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

# اتصال به صرافی LBank با سبد 10 ارز مشخص‌شده
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
print("📥 دانلود داده‌های 1 ساعته و ساخت کندل‌های 4 ساعته برای LBank")
print("============================================================")

data_4h = {}

for symbol, lbank_symbol in SYMBOLS.items():
    filename_4h = f"{symbol}_4h_data.csv"
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
        
        # تبدیل دیتای 1 ساعته به 4 ساعته استاندارد
        df4h = df1h.resample('4H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna().reset_index()
        
        df4h.to_csv(filename_4h, index=False)
        data_4h[symbol] = df4h
        print(f"  ✔️ دیتای 4 ساعته {symbol} آماده شد (تعداد کندل: {len(df4h)})")
    else:
        print(f"  ❌ دیتایی برای {symbol} دریافت نشد.")

def calculate_indicators(df):
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # محاسبه ATR برای سنجش نوسان و قابلیت تاچ شدن اهداف
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    df['ATR_Pct'] = df['ATR'] / df['Close'] # درصد نوسان واقعی قیمت
    
    # حجم میانگین
    df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
    return df

print("\n============================================================")
print("🚀 اجرای موتور بک‌تست اختصاصی 4 ساعته (TP: 1% | SL: 0.5%)")
print("============================================================")

all_portfolio_trades = []

for symbol, df4h in data_4h.items():
    if len(df4h) < 250:
        continue
        
    df4h = calculate_indicators(df4h)
    locked_until_index = 0
    
    for i in range(200, len(df4h) - 15):
        if i < locked_until_index:
            continue
            
        row = df4h.iloc[i]
        
        # 1. فیلتر شرایط بازار و قابلیت تاچ (بررسی نوسان ATR و حجم معاملات)
        # بررسی می‌کنیم که آیا نوسان بازار نه آنقدر مرده است که تاچ نشود و نه آنقدر هیجانی که اسلیپیج بالا دهد
        atr_pct = row['ATR_Pct']
        is_market_touchable = (0.003 <= atr_pct <= 0.03) and (row['Volume'] >= row['Vol_MA'] * 0.9)
        
        if not is_market_touchable:
            continue
            
        # 2. بررسی روند در تایم فریم 4 ساعته
        close = row['Close']
        ema20 = row['EMA_20']
        ema50 = row['EMA_50']
        ema200 = row['EMA_200']
        
        is_long_trend = (close > ema200) and (ema20 > ema50) and (ema50 > ema200)
        is_short_trend = (close < ema200) and (ema20 < ema50) and (ema50 < ema200)
        
        if not is_long_trend and not is_short_trend:
            continue
            
        # تنظیمات دقیق ریسک به ریوارد (حد سود 1% و حد ضرر 0.5%)
        if is_long_trend:
            entry_price = close
            tp = entry_price * 1.01   # حد سود ثابت 1 درصد
            sl = entry_price * 0.995  # حد ضرر ثابت 0.5 درصد
            
            outcome = 'OPEN'
            exit_idx = i
            # بررسی کندل‌های بعدی 4 ساعته برای چک کردن تاچ شدن TP یا SL
            for j in range(i + 1, min(i + 15, len(df4h))):
                future_candle = df4h.iloc[j]
                exit_idx = j
                
                # اولویت چک کردن حد ضرر برای احتیاط بیشتر در بک‌تست
                if future_candle['Low'] <= sl:
                    outcome = 'LOSS'
                    break
                elif future_candle['High'] >= tp:
                    outcome = 'WIN'
                    break
                    
            if outcome in ['WIN', 'LOSS']:
                all_portfolio_trades.append({
                    'Symbol': symbol,
                    'Side': 'LONG',
                    'Outcome': outcome
                })
                locked_until_index = exit_idx
                
        elif is_short_trend:
            entry_price = close
            tp = entry_price * 0.99   # حد سود ثابت 1 درصد
            sl = entry_price * 1.005  # حد ضرر ثابت 0.5 درصد
            
            outcome = 'OPEN'
            exit_idx = i
            for j in range(i + 1, min(i + 15, len(df4h))):
                future_candle = df4h.iloc[j]
                exit_idx = j
                
                if future_candle['High'] >= sl:
                    outcome = 'LOSS'
                    break
                elif future_candle['Low'] <= tp:
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
print("📊 گزارش نهایی بک‌تست پورتفوی 4 ساعته (TP: 1% و SL: 0.5%)")
print("============================================================")

if all_portfolio_trades:
    pf_df = pd.DataFrame(all_portfolio_trades)
    total_trades = len(pf_df)
    total_wins = len(pf_df[pf_df['Outcome'] == 'WIN'])
    total_losses = len(pf_df[pf_df['Outcome'] == 'LOSS'])
    portfolio_win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    
    # چون ریوارد 1 به 2 است (سود 1% در برابر ضرر 0.5%): ضریب امتیاز برد 2 است
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

print("\n✨ پایان بک‌تست اختصاصی 4 ساعته.")
