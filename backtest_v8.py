import os
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime, timedelta

SYMBOLS = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "XRP-USD": "XRPUSDT"
}

end_date = datetime.now()
start_date = end_date - timedelta(days=365)
months = pd.date_range(start=start_date, end=end_date, freq='MS')

print("============================================================")
print("📥 دانلود داده‌های واقعی یک‌ساله ۱۵ دقیقه‌ای از آرشیو بایننس")
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
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    csv_filename = z.namelist()[0]
                    csv_bytes = z.read(csv_filename)
                    df_month = pd.read_csv(io.BytesIO(csv_bytes), header=None, usecols=[0, 1, 2, 3, 4, 5],
                                          names=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    
                    # پاکسازی و اعتبارسنجیِ ماهانه (جلوگیری از انتقال خطا به کل دیتا)
                    df_month['Timestamp'] = pd.to_numeric(df_month['Timestamp'], errors='coerce')
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        df_month[col] = pd.to_numeric(df_month[col], errors='coerce')
                        
                    df_month.dropna(subset=['Timestamp', 'Close'], inplace=True)
                    df_month['Date'] = pd.to_datetime(df_month['Timestamp'], unit='ms', errors='coerce')
                    df_month.dropna(subset=['Date'], inplace=True)
                    
                    if not df_month.empty:
                        all_dfs.append(df_month[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']])
                        print(f"  ✔️ ماه {year}-{month} پاکسازی و اضافه شد ({len(df_month)} کندل سالم).")
            else:
                print(f"  ⚠️ ماه {year}-{month} موجود نبود.")
        except Exception as e:
            print(f"  ❌ خطا در دانلود ماه {year}-{month}: {e}")
            
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.sort_values('Date', inplace=True)
        final_df.reset_index(drop=True, inplace=True)
        
        final_df.to_csv(filename, index=False)
        print(f"✅ فایل نهایی {filename} با موفقیت ساخته شد! (تعداد کل کندل‌ها: {len(final_df)})")
    else:
        print(f"❌ هیچ داده‌ای برای {symbol_key} دریافت نشد.")

print("\n✨ دانلود و آماده‌سازی تمام فایل‌ها با موفقیت به پایان رسید.")
