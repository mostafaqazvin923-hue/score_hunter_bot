import os
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime, timedelta

# نمادها به فرمت بایننس
SYMBOLS = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "XRP-USD": "XRPUSDT"
}

# بازه زمانی: یک سال گذشته
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# تولید لیست ماه‌ها برای دانلود
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
                    with z.open(csv_filename) as f:
                        df_month = pd.read_csv(f, header=None, usecols=[0, 1, 2, 3, 4, 5],
                                              names=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                        all_dfs.append(df_month)
                print(f"  ✔️ ماه {year}-{month} دانلود شد.")
            else:
                print(f"  ⚠️ ماه {year}-{month} موجود نبود (یا هنوز در آرشیو آپلود نشده).")
        except Exception as e:
            print(f"  ❌ خطا در دانلود ماه {year}-{month}: {e}")
            
    if all_dfs:
        # ترکیب تمام ماه‌ها به یک فایل یک‌ساله
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # تبدیل امن ستون‌ها به عدد
        for col in ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        
        # حذف ردیف‌هایی که واقعاً مقدارشان خالی است (نه کل دیتا)
        final_df.dropna(subset=['Timestamp', 'Close'], inplace=True)
        
        # تبدیل استاندارد تایم‌استمپ میلی‌ثانیه به تاریخ
        final_df['Date'] = pd.to_datetime(final_df['Timestamp'], unit='ms')
        
        # مرتب‌سازی ستون‌ها و حذف ستون اضافی Timestamp
        final_df = final_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        final_df.sort_values('Date', inplace=True)
        final_df.reset_index(drop=True, inplace=True)
        
        # ذخیره نهایی به صورت CSV محلی
        final_df.to_csv(filename, index=False)
        print(f"✅ فایل نهایی {filename} با موفقیت ساخته شد! (تعداد کل کندل‌ها: {len(final_df)})")
    else:
        print(f"❌ هیچ داده‌ای برای {symbol_key} دریافت نشد.")

print("\n✨ دانلود و آماده‌سازی تمام فایل‌ها با موفقیت به پایان رسید.")
