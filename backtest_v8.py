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
print("📥 دانلود داده‌ها برای موتور اسمارت‌مانی و اسویپ نقدینگی (Liquidity Sweep)")
print("============================================================")

data_1h = {}

for symbol, lbank_symbol in SYMBOLS.items():
    filename_1h = f"{symbol}_1h_smart_money_data.csv"
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
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    return df

print("\n============================================================")
print("🚀 اجرای موتور اسمارت‌مانی (Liquidity Sweep & Structure Shift)")
print("============================================================")

all_portfolio_trades = []

for symbol, df1h in data_1h.items():
    if len(df1h) < 100:
        continue
        
    df1h = calculate_atr(df1h)
    locked_until_index = 0
    
    # بررسی ساختار در پنجره‌های متوالی
    for i in range(30, len(df1h) - 40):
        if i < locked_until_index:
            continue
            
        c = df1h.iloc[i]
        prev_c = df1h.iloc[i-1]
        
        # تعیین محدوده استخر نقدینگی در 24 کندل گذشته
        lookback = df1h.iloc[i-24:i]
        liq_high = lookback['High'].max()
        liq_low = lookback['Low'].min()
        
        # 1. سناریوی لیکویید شدن کف (Bearish Sweep / Fake Breakdown -> Bullish Reversal)
        # قیمت به زیر کف قبلی نفوذ کرده (Low < liq_low) اما سریع برگشته و Close بالای کف قبلی بسته شده
        is_low_swept = (df1h.iloc[i-1]['Low'] < liq_low) or (c['Low'] < liq_low)
        bullish_sweep = is_low_swept and (c['Close'] > liq_low) and (c['Close'] > c['Open']) and (c['Volume'] > c['Volume_MA'] * 1.2)
        
        # 2. سناریوی لیکویید شدن سقف (Bullish Sweep / Fake Breakout -> Bearish Reversal)
        is_high_swept = (df1h.iloc[i-1]['High'] > liq_high) or (c['High'] > liq_high)
        bearish_sweep = is_high_swept and (c['Close'] < liq_high) and (c['Close'] < c['Open']) and (c['Volume'] > c['Volume_MA'] * 1.2)
        
        if bullish_sweep:
            entry_price = c['Close']
            # استاپ لاس پشت کندل اسویپ شده قرار می‌گیرد
            sl = min(c['Low'], df1h.iloc[i-1]['Low']) - (0.3 * c['ATR'])
            risk = entry_price - sl
            
            if risk > 0 and (risk / entry_price) <= 0.04:
                # ریسک به ریوارد پویا 1 به 2
                tp = entry_price + (2.0 * risk)
                
                outcome = 'OPEN'
                exit_idx = i + 1
                for j in range(i + 1, min(i + 40, len(df1h))):
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
                    
        elif bearish_sweep:
            entry_price = c['Close']
            sl = max(c['High'], df1h.iloc[i-1]['High']) + (0.3 * c['ATR'])
            risk = sl - entry_price
            
            if risk > 0 and (risk / entry_price) <= 0.04:
                tp = entry_price - (2.0 * risk)
                
                outcome = 'OPEN'
                exit_idx = i + 1
                for j in range(i + 1, min(i + 40, len(df1h))):
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
print("📊 گزارش نهایی پورتفوی اسمارت‌مانی (Liquidity Sweep R:R 1:2)")
print("============================================================")

if all_portfolio_trades:
    pf_df = pd.DataFrame(all_portfolio_trades)
    total_trades = len(pf_df)
    total_wins = len(pf_df[pf_df['Outcome'] == 'WIN'])
    total_losses = len(pf_df[pf_df['Outcome'] == 'LOSS'])
    portfolio_win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    net_profit_score = (total_wins * 2.0) - total_losses
    
    print(f"🔸 تعداد کل معاملات کل سبد (پورتفوی): {total_trades}")
    print(f"🔸 کل معاملات برنده (WIN): {total_wins}")
    print(f"🔸 کل معاملات بازنده (LOSS): {total_losses}")
    print(f"🎯 **وین‌ریت تجمیعی کل پورتفوی (Portfolio Win Rate):** {portfolio_win_rate:.2f}%")
    print(f"💰 امتیاز سودآوری خالص (Net Profit Score با R:R 1:2): {net_profit_score:.2f}R")
    
    print("\nتفکیک عملکرد به تفکیک هر نماد:")
    print(pf_df.groupby('Symbol')['Outcome'].value_counts().unstack(fill_value=0))
else:
    print("⚠️ هیچ معامله‌ای با شرایط ثبت نشد.")

print("\n✨ بک‌تست به اتمام رسید.")
