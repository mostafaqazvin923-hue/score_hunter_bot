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

# بازه زمانی: یک سال گذشته (مثلا از سپتامبر ۲۰۲۵ تا سپتامبر ۲۰۲۶)
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
        
        # لینک فایل ماهانه‌ی ۱۵ دقیقه در بایننس ویزن
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol_binance}/15m/{symbol_binance}-15m-{year}-{month}.zip"
        
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    csv_filename = z.namelist()[0]
                    with z.open(csv_filename) as f:
                        # بایننس هدر ندارد، نام ستون‌ها را دستی تعریف می‌کنیم
                        df_month = pd.read_csv(f, header=None, usecols=[0, 1, 2, 3, 4, 5],
                                              names=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                        all_dfs.append(df_month)
                print(f"  ✔️ ماه {year}-{month} دانلود شد.")
            else:
                print(f"  ⚠️ ماه {year}-{month} موجود نبود (یا لینک تغییر کرده).")
        except Exception as e:
            print(f"  ❌ خطا در دانلود ماه {year}-{month}: {e}")
            
    if all_dfs:
        # ترکیب تمام ماه‌ها به یک فایل یک‌ساله
        final_df = pd.concat(all_dfs, ignore_index=True)
        # تبدیل Timestamp به تاریخ خوانا
        final_df['Timestamp'] = pd.to_datetime(final_df['Timestamp'], unit='ms')
        final_df.rename(columns={'Timestamp': 'Date'}, inplace=True)
        
        # ذخیره نهایی به صورت CSV محلی
        final_df.to_csv(filename, index=False)
        print(f"✅ فایل نهایی {filename} با موفقیت ساخته شد! (تعداد کل کندل‌ها: {len(final_df)})")
    else:
        print(f"❌ هیچ داده‌ای برای {symbol_key} دریافت نشد.")

print("\n✨ دانلود تمام فایل‌ها به پایان رسید. حالا می‌توانید اسکریپت بک‌تست را اجرا کنید.")
