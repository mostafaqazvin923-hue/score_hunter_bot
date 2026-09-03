import os
import ccxt
import pandas as pd

# استفاده از صرافی کوکوین که تحریم و محدودیت آی‌پی بایننس را روی سرورهای ابری ندارد
exchange = ccxt.kucoin({
    'enableRateLimit': True,
})

SYMBOLS = {
    "BTC-USD": "BTC/USDT",
    "ETH-USD": "ETH/USDT",
    "SOL-USD": "SOL/USDT",
    "XRP-USD": "XRP/USDT"
}

print("============================================================")
print("📥 دانلود داده‌های ۱۵ دقیقه‌ای از صرافی کوکوین (بدون تحریم و محدودیت)")
print("============================================================")

for symbol_key, kucoin_symbol in SYMBOLS.items():
    filename = f"{symbol_key.split('-')[0]}_15m.csv"
    print(f"\n🔹 در حال دانلود داده‌های {symbol_key} از کوکوین...")
    
    try:
        # دریافت داده‌های کندل ۱۵ دقیقه (Fetch OHLCV)
        # صرافی‌ها معمولاً در هر درخواست تعداد محدودی کندل می‌دهند، اما کوکوین تاریخچه خوبی دارد
        ohlcv = exchange.fetch_ohlcv(kucoin_symbol, timeframe='15m', limit=1500)
        
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            df.to_csv(filename, index=False)
            print(f"✅ فایل نهایی {filename} با موفقیت ساخته شد! (تعداد کل کندل‌ها: {len(df)})")
        else:
            print(f"❌ داده‌ای برای {symbol_key} دریافت نشد.")
            
    except Exception as e:
        print(f"❌ خطا در دریافت اطلاعات {symbol_key}: {e}")

print("\n✨ فرآیند دریافت داده‌ها به پایان رسید.")
