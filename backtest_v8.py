import os
import subprocess
import sys

# نصب خودکار پکیج‌های مورد نیاز در صورت عدم حضور
try:
    from backtesting import Backtest, Strategy
    import ccxt
except ImportError:
    print("📦 در حال نصب کتابخانه‌های backtesting و ccxt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "backtesting", "ccxt"])
    from backtesting import Backtest, Strategy
    import ccxt

import pandas as pd
import numpy as np

# تعریف استراتژی حرفه‌ای با Backtesting.py
class HunterXStrategy(Strategy):
    fast_period = 50
    slow_period = 200
    risk_reward = 2.5
    atr_multiplier = 1.5

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # محاسبه اندیکاتورها به صورت ایمن (بدون نگاه به آینده)
        self.ema50 = self.I(lambda x: pd.Series(x).ewm(span=self.fast_period, adjust=False).mean(), close)
        self.ema200 = self.I(lambda x: pd.Series(x).ewm(span=self.slow_period, adjust=False).mean(), close)
        
        # محاسبه ATR برای تعیین حد ضرر پویا
        tr = pd.concat([high - low, np.abs(high - close.shift()), np.abs(low - close.shift())], axis=1).max(axis=1)
        self.atr = self.I(lambda x: pd.Series(x).rolling(window=14).mean(), tr)

    def next(self):
        if self.position:
            return  # اگر معامله بازی داریم، کاری انجام نده

        c = self.data.Close[-1]
        o = self.data.Open[-1]
        h = self.data.High[-1]
        l = self.data.Low[-1]
        
        # شرایط روند صعودی در ساختار کلان
        uptrend = (self.ema50[-1] > self.ema200[-1]) and (c > self.ema50[-1])
        
        # شرط پولبک و بازگشت (کندل صعودی قدرتمند پس از اصلاح)
        pullback_signal = (c > o) and (self.data.Close[-2] < self.ema50[-2])
        
        if uptrend and pullback_signal:
            atr_val = self.atr[-1]
            if pd.isna(atr_val) or atr_val <= 0:
                return
                
            # تعیین حد ضرر و حد سود منطقی بدون تقلب
            sl = l - (self.atr_multiplier * atr_val)
            risk = c - sl
            if risk <= 0:
                return
                
            tp = c + (self.risk_reward * risk)
            
            # ورود به معامله خرید (لانگ)
            self.buy(sl=sl, tp=tp)


print("============================================================")
print("📥 دانلود داده‌های واقعی از صرافی LBank برای تست ساختار جدید")
print("============================================================")

exchange = ccxt.lbank({'enableRateLimit': True})
symbol = 'BTC/USDT'  # تست اولیه روی بیت‌کوین برای ارزیابی ساختار

ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=1000)
df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
df.set_index('Date', inplace=True)
df.drop(columns=['Timestamp'], inplace=True)

# پاکسازی مقادیر خالی احتمالی
df.dropna(inplace=True)

print(f"✔️ {len(df)} کندل ۱ ساعته دریافت شد.")
print("🚀 در حال اجرای موتور محاسباتی Backtesting.py...")

# تنظیمات بک‌تست (سرمایه اولیه 10,000 دلار، کمیسیون استاندارد 0.1 درصد)
bt = Backtest(df, HunterXStrategy, cash=10000, commission=.001, exclusive_orders=True)
stats = bt.run()

print("\n============================================================")
print("📊 گزارش نهایی عملکرد ربات با چارچوب استاندارد Backtesting.py")
print("============================================================")
print(stats)

# ذخیره گزارش خروجی به صورت HTML برای بررسی دقیق چارت‌ها در گیت‌هاب
output_file = 'backtest_report.html'
bt.plot(filename=output_file, open_browser=False)
print(f"✨ فایل گزارش گرافیکی در {output_file} ذخیره شد.")
