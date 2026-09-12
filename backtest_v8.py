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

# اتصال به صرافی LBank با سبد 10 ارز معتبر
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
print("📥 دانلود داده‌های 1 ساعته و ساخت کندل‌های 4 ساعته استاندارد")
print("============================================================")

data_4h = {}

for symbol, lbank_symbol in SYMBOLS.items():
    filename_4h = f"{symbol}_4h_pro_data.csv"
    print(f"🔹 در حال دریافت و پردازش دیتای {symbol}...")
    
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
            print(f"  ❌ خطا در دریافت داده {symbol}: {e}")
            break
            
    if all_ohlcv:
        df1h = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df1h['Date'] = pd.to_datetime(df1h['Timestamp'], unit='ms')
        df1h.set_index('Date', inplace=True)
        
        # تبدیل دقیق 1 ساعته به 4 ساعته بدون نشت داده
        df4h = df1h.resample('4H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna().reset_index()
        
        df4h.to_csv(filename_4h, index=False)
        data_4h[symbol] = df4h
        print(f"  ✔️ دیتای 4 ساعته {symbol} آماده شد (تعداد کندل: {len(df4h)})")
    else:
        print(f"  ❌ دیتایی برای {symbol} دریافت نشد.")

def calculate_advanced_indicators(df):
    # میانگین‌های متحرک جهت تشخیص روند
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # محاسبه ATR برای حد ضرر پویا
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # محاسبه ADX برای سنجش قدرت روند (فیلتر بازار سایدوی)
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    tr14 = df['ATR']
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / (tr14 + 1e-9))
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / (tr14 + 1e-9))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['ADX'] = dx.rolling(window=14).mean().fillna(20)
    
    # میانگین حجم
    df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
    return df

print("\n============================================================")
print("🚀 اجرای موتور بک‌تست حرفه‌ای (ATR Stop-Loss & ADX Filter)")
print("============================================================")

all_portfolio_trades = []

for symbol, df4h in data_4h.items():
    if len(df4h) < 250:
        continue
        
    df4h = calculate_advanced_indicators(df4h)
    
    # متغیر قفل پوزیشن برای جلوگیری از همپوشانی
    locked_until_index = 0
    
    for i in range(200, len(df4h) - 15):
        
        # اگر پوزیشنی باز است، حق ورود نداریم
        if i < locked_until_index:
            continue
            
        row = df4h.iloc[i]
        
        # فیلتر کیفیت بازار: ADX باید بالای 20 باشد تا روند قدرت داشته باشد + حجم مناسب
        if row['ADX'] < 20 or row['Volume'] < row['Vol_MA'] * 0.8:
            continue
            
        close = row['Close']
        ema20 = row['EMA_20']
        ema50 = row['EMA_50']
        ema200 = row['EMA_200']
        atr = row['ATR']
        
        if pd.isna(atr) or atr <= 0:
            continue
            
        is_long_trend = (close > ema200) and (ema20 > ema50) and (ema50 > ema200)
        is_short_trend = (close < ema200) and (ema20 < ema50) and (ema50 < ema200)
        
        if not is_long_trend and not is_short_trend:
            continue
            
        # استراتژی مدیریت ریسک پویا بر اساس ATR (ریسک به ریوارد 1 به 2)
        if is_long_trend:
            entry_price = close
            sl = entry_price - (1.0 * atr)  # حد ضرر به اندازه 1 برابر ATR
            tp = entry_price + (2.0 * atr)  # حد سود به اندازه 2 برابر ATR (ریسک به ریوارد 1:2)
            
            outcome = None
            exit_idx = i
            
            # بررسی بدون نگاه به آینده در کندل‌های بعدی
            for j in range(i + 1, len(df4h)):
                future_candle = df4h.iloc[j]
                exit_idx = j
                
                # اولویت با حد ضرر در صورت برخورد در یک کندل
                if future_candle['Low'] <= sl:
                    outcome = 'LOSS'
                    break
                elif future_candle['High'] >= tp:
                    outcome = 'WIN'
                    break
            
            if outcome in ['WIN', 'LOSS']:
                all_portfolio_trades.append({
                    'Symbol': symbol,
                    'Side': 'LONG',
                    'Outcome': outcome
                })
                # قفل کردن کامل پوزیشن تا حداقل یک کندل بعد از تسویه
                locked_until_index = exit_idx + 1
                
        elif is_short_trend:
            entry_price = close
            sl = entry_price + (1.0 * atr)
            tp = entry_price - (2.0 * atr)
            
            outcome = None
            exit_idx = i
            
            for j in range(i + 1, len(df4h)):
                future_candle = df4h.iloc[j]
                exit_idx = j
                
                if future_candle['High'] >= sl:
                    outcome = 'LOSS'
                    break
                elif future_candle['Low'] <= tp:
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
print("📊 گزارش نهایی پورتفوی حرفه‌ای (بدون نگاه به آینده و بدون همپوشانی)")
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
    print("⚠️ هیچ معامله‌ای با این فیلترها ثبت نشد.")

print("\n✨ پایان بک‌تست.")
