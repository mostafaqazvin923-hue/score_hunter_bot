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
print("📥 دانلود داده‌های 1 ساعته برای اجرای استراتژی وین‌ریت بالا")
print("============================================================")

data_1h = {}

for symbol, lbank_symbol in SYMBOLS.items():
    filename_1h = f"{symbol}_1h_smart_money.csv"
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
        df1h = df1h[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df1h.dropna(inplace=True)
        df1h.drop_duplicates(subset=['Date'], inplace=True)
        df1h.sort_values('Date', inplace=True)
        df1h.reset_index(drop=True, inplace=True)
        data_1h[symbol] = df1h

def calculate_smart_money_indicators(df):
    # تعیین روند ساختاری با میانگین‌های بلندمدت
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # محاسبه ATR برای مدیریت ریسک و حد ضرر دقیق
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # شناسایی سطوح نقدینگی (سقف‌ها و کف‌های ۲۰ کندل گذشته)
    df['Swing_High'] = df['High'].rolling(window=20).max()
    df['Swing_Low'] = df['Low'].rolling(window=20).min()
    
    return df

print("\n============================================================")
print("🚀 اجرای موتور بک‌تست اختصاصی وین‌ریت بالا (ریسک به ریوارد 1 به 2)")
print("============================================================")

all_portfolio_trades = []

for symbol, df1h in data_1h.items():
    if len(df1h) < 300:
        continue
        
    df1h = calculate_smart_money_indicators(df1h)
    
    locked_until_index = 0
    
    # حلقه بررسی کندل به کندل بدون نگاه به آینده در لحظه تصمیم
    for i in range(200, len(df1h) - 40):
        if i < locked_until_index:
            continue
            
        c = df1h.iloc[i]
        prev_c = df1h.iloc[i-1]
        
        # فیلتر سخت‌گیرانه روند کلان
        is_bullish_trend = c['Close'] > c['EMA_200'] and c['EMA_50'] > c['EMA_200']
        is_bearish_trend = c['Close'] < c['EMA_200'] and c['EMA_50'] < c['EMA_200']
        
        if not is_bullish_trend and not is_bearish_trend:
            continue
            
        # فیلتر حجم نوسانی برای جلوگیری از ورود در بازارهای کم‌حجم
        avg_vol = df1h.iloc[i-20:i]['Volume'].mean()
        high_volume = c['Volume'] > (avg_vol * 1.5)
        
        if not high_volume:
            continue
            
        # منطق لانگ: قیمت کف ۲۰ کندل گذشته را هپ کرده (تصفیه نقدینگی) و با قدرت به داخل برگشته
        is_long_sweep = (prev_c['Low'] <= df1h.iloc[i-1]['Swing_Low']) and (c['Close'] > c['Open']) and (c['Close'] > prev_c['High'])
        # منطق شورت: قیمت سقف ۲۰ کندل گذشته را هپ کرده و با قدرت ریجکت شده
        is_short_sweep = (prev_c['High'] >= df1h.iloc[i-1]['Swing_High']) and (c['Close'] < c['Open']) and (c['Close'] < prev_c['Low'])
        
        if is_bullish_trend and is_long_sweep:
            entry_price = c['Close']
            sl = prev_c['Low'] - (0.5 * c['ATR'])
            risk = entry_price - sl
            
            if risk <= 0 or (risk / entry_price) > 0.03: # محدودیت ریسک معقول
                continue
                
            tp = entry_price + (2.0 * risk) # ریوارد دقیق 1 به 2
            
            outcome = 'OPEN'
            exit_idx = i
            for j in range(i + 1, min(i + 50, len(df1h))):
                future_c = df1h.iloc[j]
                exit_idx = j
                if future_c['Low'] <= sl:
                    outcome = 'LOSS'
                    break
                elif future_c['High'] >= tp:
                    outcome = 'WIN'
                    break
                    
            if outcome in ['WIN', 'LOSS']:
                all_portfolio_trades.append({
                    'Symbol': symbol,
                    'Side': 'LONG',
                    'Outcome': outcome
                })
                locked_until_index = exit_idx
                
        elif is_bearish_trend and is_short_sweep:
            entry_price = c['Close']
            sl = prev_c['High'] + (0.5 * c['ATR'])
            risk = sl - entry_price
            
            if risk <= 0 or (risk / entry_price) > 0.03:
                continue
                
            tp = entry_price - (2.0 * risk) # ریوارد دقیق 1 به 2
            
            outcome = 'OPEN'
            exit_idx = i
            for j in range(i + 1, min(i + 50, len(df1h))):
                future_c = df1h.iloc[j]
                exit_idx = j
                if future_c['High'] >= sl:
                    outcome = 'LOSS'
                    break
                elif future_c['Low'] <= tp:
                    outcome = 'WIN'
                    break
                    
            if outcome in ['WIN', 'LOSS']:
                all_portfolio_trades.append({
                    'Symbol': symbol,
                    'Side': 'SHORT',
                    'Outcome': outcome
                })
                locked_until_index = exit_idx

print("\n============================================================")
print("📊 گزارش نهایی پورتفوی (استراتژی نقدینگی و R=2)")
print("============================================================")

if all_portfolio_trades:
    pf_df = pd.DataFrame(all_portfolio_trades)
    total_trades = len(pf_df)
    total_wins = len(pf_df[pf_df['Outcome'] == 'WIN'])
    total_losses = len(pf_df[pf_df['Outcome'] == 'LOSS'])
    portfolio_win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    net_profit_score = (total_wins * 2.0) - total_losses
    
    print(f"🔸 تعداد کل معاملات کل سبد: {total_trades}")
    print(f"🔸 معاملات برنده (WIN): {total_wins}")
    print(f"🔸 معاملات بازنده (LOSS): {total_losses}")
    print(f"🎯 **وین‌ریت تجمیعی پورتفوی:** {portfolio_win_rate:.2f}%")
    print(f"💰 **امتیاز سودآوری خالص:** {net_profit_score:.2f}R")
    
    print("\nتفکیک عملکرد به تفکیک هر ارز:")
    print(pf_df.groupby('Symbol')['Outcome'].value_counts().unstack(fill_value=0))
else:
    print("⚠️ هیچ معامله‌ای با شرایط ثبت نشد.")

print("\n✨ تست به پایان رسید.")
