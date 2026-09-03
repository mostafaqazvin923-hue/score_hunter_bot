import os
import requests
import zipfile
import io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT"
}

# بازه زمانی یک‌ساله دقیق از آرشیو ماهانه بایننس (بدون محدودیت API)
months = pd.date_range(start="2025-09-01", end="2026-08-01", freq='MS')
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

print("============================================================")
print("📥 دانلود مستقیم داده‌های یک‌ساله کامل ۱۵ دقیقه‌ای از آرشیو عمومی")
print("============================================================")

for symbol, binance_symbol in SYMBOLS.items():
    filename = f"{symbol}_15m.csv"
    print(f"\n🔹 در حال دریافت تاریخچه کامل یک‌ساله {symbol}...")
    all_dfs = []
    
    for dt in months:
        year = dt.year
        month = f"{dt.month:02d}"
        url = f"https://data.binance.vision/data/spot/monthly/klines/{binance_symbol}/15m/{binance_symbol}-15m-{year}-{month}.zip"
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    csv_filename = z.namelist()[0]
                    csv_bytes = z.read(csv_filename)
                    df_month = pd.read_csv(io.BytesIO(csv_bytes), header=None, usecols=[0, 1, 2, 3, 4, 5],
                                          names=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    
                    df_month['Timestamp'] = pd.to_numeric(df_month['Timestamp'], errors='coerce')
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        df_month[col] = pd.to_numeric(df_month[col], errors='coerce')
                        
                    df_month.dropna(subset=['Timestamp', 'Close'], inplace=True)
                    df_month['Date'] = pd.to_datetime(df_month['Timestamp'], unit='ms', errors='coerce')
                    df_month.dropna(subset=['Date'], inplace=True)
                    
                    if not df_month.empty:
                        all_dfs.append(df_month[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']])
        except Exception:
            pass
            
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.sort_values('Date', inplace=True)
        final_df.drop_duplicates(subset=['Date'], inplace=True)
        final_df.reset_index(drop=True, inplace=True)
        
        final_df.to_csv(filename, index=False)
        print(f"  ✔️ فایل {filename} با موفقیت ساخته شد (تعداد کل کندل‌ها: {len(final_df)})")
    else:
        print(f"  ❌ خطا در دریافت داده‌های {symbol}")

print("\n============================================================")
print("🚀 اجرای موتور بک‌تست روی داده‌های واقعی ۳۵ هزار کندلی یک‌ساله")
print("============================================================")

for symbol in SYMBOLS.keys():
    filename = f"{symbol}_15m.csv"
    if not os.path.exists(filename):
        continue
        
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'])
    
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
            
            sl = entry_price * (1 - sl_pct) if side == 'LONG' else entry_price * (1 + sl_pct)
            tp = entry_price * (1 + tp_pct) if side == 'LONG' else entry_price * (1 - tp_pct)
                
            outcome = 'OPEN'
            for j in range(i + 1, min(i + 50, len(df))):
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
        net_profit = (wins * 2) - losses
        
        print(f"📊 نتایج واقعی یک‌ساله برای {symbol}:")
        print(f"   - تعداد کل معاملات: {total_trades} | برنده: {wins} | بازنده: {losses}")
        print(f"   - **وین‌ریت نهایی (Win Rate):** {win_rate:.2f}%")
        print(f"   - امتیاز عملکرد (Profit Score): {net_profit}")
    else:
        print(f"⚠️ معامله‌ای ثبت نشد.")

print("\n✨ بک‌تست یک‌ساله کامل به پایان رسید.")
