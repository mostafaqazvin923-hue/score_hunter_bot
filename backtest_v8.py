import os
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime

SYMBOLS = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "XRP-USD": "XRPUSDT"
}

# بازه زمانی امن: از اکتبر 2025 تا پایان جولای 2026
months = pd.date_range(start="2025-10-01", end="2026-07-01", freq='MS')

# هدر مرورگر برای جلوگیری از بلاک شدن توسط سرور بایننس
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

print("============================================================")
print("📥 دانلود داده‌های ۱۵ دقیقه‌ای از بایننس با هدر امن مرورگر")
print("============================================================")

for symbol_key, symbol_binance in SYMBOLS.items():
    filename = f"{symbol_key.split('-')[0]}_15m.csv"
    print(f"\n🔹 در حال آماده‌سازی داده‌های {symbol_key}...")
    
    all_dfs = []
    
    for dt in months:
        year = dt.year
        month = f"{dt.month:02d}"
        
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol_binance}/15m/{symbol_binance}-15m-{year}-{month}.zip"
        
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
                        print(f"  ✔️ ماه {year}-{month} با موفقیت دانلود شد ({len(df_month)} کندل).")
            else:
                print(f"  ⚠️ ماه {year}-{month} در دسترس نبود (کد خطا: {response.status_code})")
        except Exception as e:
            print(f"  ❌ خطا در دانلود ماه {year}-{month}: {e}")
            
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.sort_values('Date', inplace=True)
        final_df.reset_index(drop=True, inplace=True)
        
        final_df.to_csv(filename, index=False)
        print(f"✅ فایل نهایی {filename} ساخته شد! (تعداد کل کندل‌ها: {len(final_df)})")
    else:
        print(f"❌ هیچ داده‌ای برای {symbol_key} دریافت نشد.")

print("\n✨ آماده‌سازی فایل‌ها کامل شد.")
