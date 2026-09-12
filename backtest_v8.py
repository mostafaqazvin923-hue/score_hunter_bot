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

# صرافی LBank با سبد 10 ارز معتبر
exchange = ccxt.lbank({'enableRateLimit': True})
SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "ADA": "ADA/USDT",
    "AVAX": "AVAX/USDT",
    "LINK": "LINK/USDT",
    "NEAR": "NEAR/USDT",
    "SUI": "SUI/USDT",
    "DOT": "DOT/USDT"
}

start_date = datetime.now() - timedelta(days=365)
since_timestamp = int(start_date.timestamp() * 1000)

print("============================================================")
print("📥 دانلود داده‌های 1 ساعته برای ستاپ شکار نقدینگی (Liquidity Sweep)")
print("============================================================")

data_1h = {}

for symbol, lbank_symbol in SYMBOLS.items():
    print(f"🔹 در حال دریافت دیتای {symbol}...")
    all_ohlcv = []
    current_since = since_timestamp
    now_timestamp = exchange.milliseconds()
    
    while current_since < now_timestamp:
        try:
            ohlcv = exchange.fetch_ohlcv(lbank_symbol, timeframe='1h', since=current_since, limit=1000)
            if not ohlcv:
                break
            current_since = ohlcv[-1][0] + 1
            all_ohlcv.extend(ohlcv)
            if len(ohlcv) < 1000:
                break
        except Exception as e:
            break
            
    if all_ohlcv:
        df1h = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df1h['Date'] = pd.to_datetime(df1h['Timestamp'], unit='ms')
        df1h.set_index('Date', inplace=True)
        data_1h[symbol] = df1h
        print(f"  ✔️ دیتای {symbol} آماده شد (تعداد کندل: {len(df1h)})")

def calculate_smart_money_indicators(df):
    # کانال Donchian برای سقف و کف 24 ساعته (24 کندل گذشته)
    df['Highest_24'] = df['High'].shift(1).rolling(window=24).max()
    df['Lowest_24'] = df['Low'].shift(1).rolling(window=24).min()
    
    # محاسبه ATR برای حد سود و ضرر پویا
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # میانگین حجم برای تشخیص حجم نهادی
    df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
    return df

print("\n============================================================")
print("🚀 اجرای موتور بک‌تست هوشمند (شکار نقدینگی + ATR پویا + ریسک به ریوارد 1:2)")
print("============================================================")

all_portfolio_trades = []

for symbol, df1h in data_1h.items():
    if len(df1h) < 100:
        continue
        
    df1h = calculate_smart_money_indicators(df1h)
    locked_until_index = 0
    
    for i in range(50, len(df1h) - 20):
        # بررسی قفل پوزیشن و بافر پس از تسویه
        if i < locked_until_index:
            continue
            
        c1h = df1h.iloc[i]
        atr = c1h['ATR']
        vol_ma = c1h['Vol_MA']
        
        # فیلتر پیش‌نیاز سلامت بازار (بررسی دسترس‌پذیری نوسان کافی)
        if pd.isna(atr) or atr <= 0 or c1h['Volume'] < vol_ma * 1.4:
            continue
            
        highest_24 = c1h['Highest_24']
        lowest_24 = c1h['Lowest_24']
        
        if pd.isna(highest_24) or pd.isna(lowest_24):
            continue
            
        # 1. ستاپ لانگ: شکار نقدینگی کف (قیمت پایین‌تر از کف 24 ساعته نفوذ کرده اما به داخل برگشته و بسته شده)
        is_sweep_low = (c1h['Low'] < lowest_24) and (c1h['Close'] > lowest_24) and (c1h['Close'] > c1h['Open'])
        
        # 2. ستاپ شورت: شکار نقدینگی سقف (قیمت بالاتر از سقف 24 ساعته نفوذ کرده اما به داخل برگشته و بسته شده)
        is_sweep_high = (c1h['High'] > highest_24) and (c1h['Close'] < highest_24) and (c1h['Close'] < c1h['Open'])
        
        if not is_sweep_low and not is_sweep_high:
            continue
            
        if is_sweep_low:
            # ورود در بازگشایی کندل بعدی (بدون نگاه به آینده)
            entry_idx = i + 1
            if entry_idx >= len(df1h):
                break
                
            entry_price = df1h.iloc[entry_idx]['Open']
            
            # تعیین حد ضرر و حد سود پویا بر اساس ATR (ریسک به ریوارد دقیق 1 به 2)
            sl = entry_price - (1.5 * atr)
            tp = entry_price + (3.0 * atr)  # دقیقاً دو برابر فاصله ریسک
            
            outcome = None
            exit_idx = entry_idx
            
            for j in range(entry_idx, len(df1h)):
                f_c = df1h.iloc[j]
                exit_idx = j
                
                # محافظه‌کاری: بررسی اولویت حد ضرر در صورت هم‌پوشانی در یک کندل
                if f_c['Low'] <= sl:
                    outcome = 'LOSS'
                    break
                elif f_c['High'] >= tp:
                    outcome = 'WIN'
                    break
                    
            if outcome in ['WIN', 'LOSS']:
                all_portfolio_trades.append({
                    'Symbol': symbol,
                    'Side': 'LONG',
                    'Outcome': outcome
                })
                # قفل کردن ربات تا حداقل یک کندل بعد از تسویه کامل
                locked_until_index = exit_idx + 1
                
        elif is_sweep_high:
            entry_idx = i + 1
            if entry_idx >= len(df1h):
                break
                
            entry_price = df1h.iloc[entry_idx]['Open']
            
            sl = entry_price + (1.5 * atr)
            tp = entry_price - (3.0 * atr)  # ریسک به ریوارد 1 به 2
            
            outcome = None
            exit_idx = entry_idx
            
            for j in range(entry_idx, len(df1h)):
                f_c = df1h.iloc[j]
                exit_idx = j
                
                if f_c['High'] >= sl:
                    outcome = 'LOSS'
                    break
                elif f_c['Low'] <= tp:
                    outcome = 'WIN'
                    break
                    
            if outcome in ['WIN', 'LOSS']:
                all_portfolio_trades.append({
                    'Symbol': symbol,
                    'Side': 'SHORT',
                    'Outcome': outcome
                })
                locked_until_index = exit_idx + 1

print("\n============================================================")
print("📊 گزارش نهایی پورتفوی هوشمند (شکار نقدینگی + ATR پویا)")
print("============================================================")

if all_portfolio_trades:
    pf_df = pd.DataFrame(all_portfolio_trades)
    total_trades = len(pf_df)
    total_wins = len(pf_df[pf_df['Outcome'] == 'WIN'])
    total_losses = len(pf_df[pf_df['Outcome'] == 'LOSS'])
    portfolio_win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    net_profit_score = (total_wins * 2.0) - total_losses
    
    print(f"🔸 تعداد کل معاملات پورتفوی: {total_trades}")
    print(f"🔸 معاملات برنده (WIN): {total_wins}")
    print(f"🔸 معاملات بازنده (LOSS): {total_losses}")
    print(f"🎯 **وین‌ریت تجمیعی پورتفوی:** {portfolio_win_rate:.2f}%")
    print(f"💰 **امتیاز سودآوری خالص (Net Profit Score):** {net_profit_score:.2f}R")
    
    print("\nتفکیک عملکرد به تفکیک هر نماد:")
    print(pf_df.groupby('Symbol')['Outcome'].value_counts().unstack(fill_value=0))
else:
    print("⚠️ هیچ معامله‌ای با این شرایط ثبت نشد.")

print("\n✨ پایان بک‌تست هوشمند.")
