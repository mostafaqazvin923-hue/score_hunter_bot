import os
import subprocess
import sys

# نصب خودکار ccxt اگر روی محیط موجود نباشد
try:
    import ccxt
except ImportError:
    print("📦 در حال نصب کتابخانه ccxt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import pandas as pd

# استفاده از صرافی کراکن (بدون تحریم روی سرورهای ابری و با داده‌های کاملاً واقعی)
exchange = ccxt.kraken({
    'enableRateLimit': True,
})

SYMBOLS = {
    "BTC-USD": "BTC/USD",
    "ETH-USD": "ETH/USD",
    "SOL-USD": "SOL/USD",
    "XRP-USD": "XRP/USD"
}

print("============================================================")
print("📥 دانلود داده‌های ۱۵ دقیقه‌ای واقعی و بدون فیلتر از صرافی کراکن")
print("============================================================")

for symbol_key, kraken_symbol in SYMBOLS.items():
    filename = f"{symbol_key.split('-')[0]}_15m.csv"
    print(f"\n🔹 در حال دریافت داده‌های {symbol_key}...")
    
    try:
        # دریافت کندل‌های ۱۵ دقیقه‌ای واقعی از کراکن
        ohlcv = exchange.fetch_ohlcv(kraken_symbol, timeframe='15m', limit=2000)
        
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # حذف مقادیر خالی احتمالی
            df.dropna(inplace=True)
            df.sort_values('Date', inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            df.to_csv(filename, index=False)
            print(f"✅ فایل نهایی {filename} با موفقیت ساخته شد! (تعداد کل کندل‌ها: {len(df)})")
        else:
            print(f"❌ داده‌ای برای {symbol_key} دریافت نشد.")
            
    except Exception as e:
        print(f"❌ خطا در دریافت اطلاعات {symbol_key}: {e}")

print("\n✨ دریافت داده‌های واقعی به پایان رسید. حالا موتور بک‌تست آماده‌ی اجراست.")
