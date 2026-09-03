import os
import subprocess
import sys

# 1. بررسی و نصب خودکار ccxt در صورت نیاز
try:
    import ccxt
except ImportError:
    print("📦 در حال نصب کتابخانه ccxt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import pandas as pd
import numpy as np

# 2. دانلود خودکار داده‌ها از صرافی کراکن
exchange = ccxt.kraken({'enableRateLimit': True})
SYMBOLS = {
    "BTC": "BTC/USD",
    "ETH": "ETH/USD",
    "SOL": "SOL/USD",
    "XRP": "XRP/USD"
}

print("============================================================")
print("📥 دریافت داده‌های ۱۵ دقیقه‌ای واقعی از صرافی کراکن")
print("============================================================")

for symbol, kraken_symbol in SYMBOLS.items():
    filename = f"{symbol}_15m.csv"
    print(f"🔹 در حال دریافت داده‌های {symbol}...")
    try:
        ohlcv = exchange.fetch_ohlcv(kraken_symbol, timeframe='15m', limit=2000)
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df.dropna(inplace=True)
            df.sort_values('Date', inplace=True)
            df.reset_index(drop=True, inplace=True)
            df.to_csv(filename, index=False)
            print(f"  ✔️ فایل {filename} ساخته شد ({len(df)} کندل).")
    except Exception as e:
        print(f"  ❌ خطا در دریافت {symbol}: {e}")

print("\n============================================================")
print("🚀 اجرای موتور بک‌تست استراتژی تله‌ی نهنگ + فیلتر روند")
print("============================================================")

for symbol in SYMBOLS.keys():
    filename = f"{symbol}_15m.csv"
    if not os.path.exists(filename):
        print(f"\n⚠️ فایل {filename} یافت نشد.")
        continue
        
    print(f"\n🔹 تحلیل و بک‌تست روی {symbol}...")
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # فیلتر روند (SMA 200) و سطوح نقدینگی
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Prev_High'] = df['High'].shift(1).rolling(window=20).max()
    df['Prev_Low'] = df['Low'].shift(1).rolling(window=20).min()
    
    trades = []
    sl_pct = 0.015
    tp_pct = 0.030
    
    for i in range(200, len(df) - 50):
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
            
            if side == 'LONG':
                sl = entry_price * (1 - sl_pct)
                tp = entry_price * (1 + tp_pct)
            else:
                sl = entry_price * (1 + sl_pct)
                tp = entry_price * (1 - tp_pct)
                
            outcome = 'OPEN'
            for j in range(i + 1, min(i + 50, len(df))):
                future_candle = df.iloc[j]
                high = future_candle['High']
                low = future_candle['Low']
                
                if side == 'LONG':
                    if low <= sl:
                        outcome = 'LOSS'
                        break
                    elif high >= tp:
                        outcome = 'WIN'
                        break
                else:
                    if high >= sl:
                        outcome = 'LOSS'
                        break
                    elif low <= tp:
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
        net_profit = (wins * 2) - losses
        
        print(f"📊 نتایج نهایی برای {symbol}:")
        print(f"   - تعداد معاملات: {total_trades} | برنده: {wins} | بازنده: {losses}")
        print(f"   - **وین‌ریت (Win Rate):** {win_rate:.2f}%")
        print(f"   - امتیاز عملکرد (Profit Score): {net_profit}")
    else:
        print(f"⚠️ معامله‌ای با شرایط فیلتر روند روی {symbol} ثبت نشد.")

print("\n✨ بک‌تست به اتمام رسید.")
