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

# استفاده از صرافی MEXC برای تایم‌فریم ۱ ساعته
exchange = ccxt.mexc({'enableRateLimit': True})
SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT"
}

# محاسبه بازه زمانی دقیق یک سال گذشته
start_date = datetime.now() - timedelta(days=365)
since_timestamp = int(start_date.timestamp() * 1000)

print("============================================================")
print("📥 دانلود داده‌های یک‌ساله واقعی ۱ ساعته (1h) بدون محدودیت")
print("============================================================")

for symbol, mexc_symbol in SYMBOLS.items():
    filename = f"{symbol}_1h.csv"
    print(f"🔹 در حال دریافت تاریخچه کامل یک‌ساله {symbol} در تایم‌فریم ۱ ساعته...")
    
    all_ohlcv = []
    current_since = since_timestamp
    now_timestamp = exchange.milliseconds()
    
    while current_since < now_timestamp:
        try:
            # دریافت کندل‌های ۱ ساعته با لیمیت ۱۰۰۰ عددی (هر بار حدود ۴۱ روز)
            ohlcv = exchange.fetch_ohlcv(mexc_symbol, timeframe='1h', since=current_since, limit=1000)
            if not ohlcv:
                break
            
            current_since = ohlcv[-1][0] + 1
            all_ohlcv.extend(ohlcv)
            
            if len(ohlcv) < 1000:
                break
        except Exception as e:
            print(f"  ❌ خطا در دریافت داده‌ها برای {symbol}: {e}")
            break
            
    if all_ohlcv:
        df = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)
        df.drop_duplicates(subset=['Date'], inplace=True)
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        df.to_csv(filename, index=False)
        print(f"  ✔️ فایل {filename} با موفقیت ساخته شد (تعداد کل کندل‌ها: {len(df)})")
    else:
        print(f"  ❌ هیچ داده‌ای برای {symbol} دریافت نشد.")

print("\n============================================================")
print("🚀 اجرای موتور بک‌تست روی دیتای واقعی یک‌ساله ۱ ساعته")
print("============================================================")

for symbol in SYMBOLS.keys():
    filename = f"{symbol}_1h.csv"
    if not os.path.exists(filename):
        continue
        
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # فیلتر روند (SMA 200) و سقف/کف‌های ۲۰ دوره‌ای برای تایم‌فریم ۱ ساعته
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Prev_High'] = df['High'].shift(1).rolling(window=20).max()
    df['Prev_Low'] = df['Low'].shift(1).rolling(window=20).min()
    
    trades = []
    sl_pct = 0.020  # استاپ لاس مناسب برای تایم‌فریم ۱ ساعته (۲ درصد)
    tp_pct = 0.045  # حد سود با ریسک به ریوارد حدود 1:2.2 (۴.۵ درصد)
    
    for i in range(200, len(df) - 30):
        current = df.iloc[i]
        close = current['Close']
        sma = current['SMA_200']
        
        if pd.isna(sma) or pd.isna(current['Prev_High']) or pd.isna(current['Prev_Low']):
            continue
            
        is_long_trap = (current['Low'] < current['Prev_Low']) and (close > current['Prev_Low']) and (close > sma)
        is_short_trap = (current['High'] > current['Prev_High']) and (close < current['Prev_High']) and (close < sma)
        
        if is_long_trap or is_short_trap:
            entry_price = close
            entry_time = current['Date']
            side = 'LONG' if is_long_trap else 'SHORT'
            
            sl = entry_price * (1 - sl_pct) if side == 'LONG' else entry_price * (1 + sl_pct)
            tp = entry_price * (1 + tp_pct) if side == 'LONG' else entry_price * (1 - tp_pct)
                
            outcome = 'OPEN'
            for j in range(i + 1, min(i + 30, len(df))):
                future_candle = df.iloc[j]
                if side == 'LONG':
                    if future_candle['Low'] <= sl:
                        outcome = 'LOSS'
                        break
                    elif future_candle['High'] >= tp:
                        outcome = 'WIN'
                        break
                else:
                    if future_candle['High'] >= sl:
                        outcome = 'LOSS'
                        break
                    elif future_candle['Low'] <= tp:
                        outcome = 'WIN'
                        break
            
            if outcome in ['WIN', 'LOSS']:
                trades.append({
                    'Time': entry_time,
                    'Side': side,
                    'Entry': entry_price,
                    'Outcome': outcome
                })
                
    if trades:
        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['Outcome'] == 'WIN'])
        losses = len(trades_df[trades_df['Outcome'] == 'LOSS'])
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        net_profit = (wins * 2.25) - losses
        
        print(f"📊 نتایج یک‌ساله ۱ ساعته برای {symbol}:")
        print(f"   - تعداد کل معاملات: {total_trades} | برنده: {wins} | بازنده: {losses}")
        print(f"   - **وین‌ریت واقعی (Win Rate):** {win_rate:.2f}%")
        print(f"   - امتیاز عملکرد (Profit Score): {net_profit:.2f}")
    else:
        print(f"⚠️ معامله‌ای ثبت نشد.")

print("\n✨ بک‌تست یک‌ساله ۱ ساعته به اتمام رسید.")
