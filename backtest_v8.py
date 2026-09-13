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

# صرافی LBank با سبد 10 ارز
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
print("📥 دانلود داده‌های 1 ساعته برای سیستم برگشت از اسوینگ‌های کلیدی")
print("============================================================")

data_1h = {}

for symbol, lbank_symbol in SYMBOLS.items():
    filename_1h = f"{symbol}_1h_swing_rejection_data.csv"
    print(f"🔹 در حال دریافت دیتای 1 ساعته {symbol}...")
    
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
        df1h = df1h[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df1h.dropna(inplace=True)
        df1h.drop_duplicates(subset=['Date'], inplace=True)
        df1h.sort_values('Date', inplace=True)
        df1h.reset_index(drop=True, inplace=True)
        
        df1h.to_csv(filename_1h, index=False)
        data_1h[symbol] = df1h
        print(f"  ✔️ دیتای 1 ساعته {symbol} آماده شد (تعداد کندل: {len(df1h)})")
    else:
        print(f"  ❌ دیتایی برای {symbol} دریافت نشد.")

def calculate_atr(df):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    return df

print("\n============================================================")
print("🚀 اجرای موتور برگشت از اسوینگ خالص (R:R 1:1)")
print("============================================================")

all_portfolio_trades = []

for symbol, df1h in data_1h.items():
    if len(df1h) < 200:
        continue
        
    df1h = calculate_atr(df1h)
    
    # ساختار روند بزرگتر با میانگین متحرک ساده 50 کندلی برای جهت‌گیری کلی
    df1h['SMA_50'] = df1h['Close'].rolling(window=50).mean()
    df1h['Volume_MA'] = df1h['Volume'].rolling(window=20).mean()
    
    locked_until_index = 0
    
    for i in range(100, len(df1h) - 30):
        if i < locked_until_index:
            continue
            
        c = df1h.iloc[i]
        prev_c = df1h.iloc[i-1]
        
        # تعریف اسوینگ‌های مهم در 30 کندل گذشته
        lookback = df1h.iloc[i-30:i]
        swing_high = lookback['High'].max()
        swing_low = lookback['Low'].min()
        
        candle_range = c['High'] - c['Low']
        if candle_range == 0:
            isnan = np.isnan(c['ATR'])
            if isnan: continue
            continue
            
        body_size = abs(c['Close'] - c['Open'])
        lower_wick = c['Close'] - c['Low'] if c['Close'] > c['Open'] else c['Open'] - c['Low']
        upper_wick = c['High'] - c['Open'] if c['Close'] > c['Open'] else c['High'] - c['Close']
        
        # شروط برگشت قیمت (ریجکشن تمیز از کف یا سقف اسوینگ قبلی به همراه حجم بالا)
        is_near_support = c['Low'] <= swing_low * 1.002
        is_near_resistance = c['High'] >= swing_high * 0.998
        
        bullish_rejection = is_near_support and (lower_wick >= candle_range * 0.6) and (c['Close'] > c['Open']) and (c['Volume'] > c['Volume_MA'] * 1.5)
        bearish_rejection = is_near_resistance and (upper_wick >= candle_range * 0.6) and (c['Close'] < c['Open']) and (c['Volume'] > c['Volume_MA'] * 1.5)
        
        if bullish_rejection and (c['Close'] > c['SMA_50']):
            entry_price = c['Close']
            sl = c['Low'] - (0.5 * c['ATR'])
            risk = entry_price - sl
            
            if risk > 0 and (risk / entry_price) <= 0.04:
                tp = entry_price + (1.0 * risk) # ریسک به ریوارد دقیقاً ۱ به ۱
                
                outcome = 'OPEN'
                exit_idx = i + 1
                for j in range(i + 1, min(i + 35, len(df1h))):
                    f_c = df1h.iloc[j]
                    exit_idx = j
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
                    locked_until_index = exit_idx
                    
        elif bearish_rejection and (c['Close'] < c['SMA_50']):
            entry_price = c['Close']
            sl = c['High'] + (0.5 * c['ATR'])
            risk = sl - entry_price
            
            if risk > 0 and (risk / entry_price) <= 0.04:
                tp = entry_price - (1.0 * risk) # ریسک به ریوارد دقیقاً ۱ به ۱
                
                outcome = 'OPEN'
                exit_idx = i + 1
                for j in range(i + 1, min(i + 35, len(df1h))):
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
                    locked_until_index = exit_idx

print("\n============================================================")
print("📊 گزارش نهایی پورتفوی سیستم اسوینگ ریجکشن (R:R 1:1)")
print("============================================================")

if all_portfolio_trades:
    pf_df = pd.DataFrame(all_portfolio_trades)
    total_trades = len(pf_df)
    total_wins = len(pf_df[pf_df['Outcome'] == 'WIN'])
    total_losses = len(pf_df[pf_df['Outcome'] == 'LOSS'])
    portfolio_win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    net_profit_score = (total_wins * 1.0) - total_losses
    
    print(f"🔸 تعداد کل معاملات کل سبد (پورتفوی): {total_trades}")
    print(f"🔸 کل معاملات برنده (WIN): {total_wins}")
    print(f"🔸 کل معاملات بازنده (LOSS): {total_losses}")
    print(f"🎯 **وین‌ریت تجمیعی کل پورتفوی (Portfolio Win Rate):** {portfolio_win_rate:.2f}%")
    print(f"💰 امتیاز سودآوری خالص (Net Profit Score): {net_profit_score:.2f}R")
    
    print("\nتفکیک عملکرد به تفکیک هر نماد:")
    print(pf_df.groupby('Symbol')['Outcome'].value_counts().unstack(fill_value=0))
else:
    print("⚠️ هیچ معامله‌ای با شرایط ثبت نشد.")

print("\n✨ بک‌تست به اتمام رسید.")
