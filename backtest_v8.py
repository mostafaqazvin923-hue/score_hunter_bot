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

# صرافی LBank با سبد 10 ارز برتر
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
print("📥 دانلود داده‌ها برای سیستم فوق‌العاده سخت‌گیر (Hyper-Selective)")
print("============================================================")

data_1h = {}
data_4h = {}

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
        
        df4h = df1h.resample('4H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna().reset_index()
        
        df1h.reset_index(inplace=True)
        
        data_1h[symbol] = df1h
        data_4h[symbol] = df4h
        print(f"  ✔️ دیتای {symbol} آماده شد.")

def calculate_hyper_indicators(df):
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # ATR برای تعیین حدود دقیق
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
    return df

print("\n============================================================")
print("🚀 اجرای موتور بک‌تست فوق‌العاده گزینشی (هدف: وین‌ریت بالا)")
print("============================================================")

all_portfolio_trades = []

for symbol in SYMBOLS.keys():
    if symbol not in data_1h or symbol not in data_4h:
        continue
        
    df1h = data_1h[symbol].copy()
    df4h = data_4h[symbol].copy()
    
    if len(df1h) < 300 or len(df4h) < 100:
        continue
        
    df1h = calculate_hyper_indicators(df1h)
    df4h = calculate_hyper_indicators(df4h)
    
    df1h['Date_4H'] = df1h['Date'].dt.floor('4h')
    df4h_indexed = df4h.set_index('Date')
    
    locked_until_index = 0
    
    for i in range(250, len(df1h) - 20):
        if i < locked_until_index:
            continue
            
        c1h = df1h.iloc[i]
        t4h_time = c1h['Date_4H']
        
        if t4h_time not in df4h_indexed.index:
            continue
            
        r4h = df4h_indexed.loc[t4h_time]
        
        # فیلتر کلان فوق‌العاده سخت‌گیرانه (تایید قطعی روند در 4H)
        is_strong_bull = (r4h['Close'] > r4h['EMA_200']) and (r4h['EMA_50'] > r4h['EMA_200'])
        is_strong_bear = (r4h['Close'] < r4h['EMA_200']) and (r4h['EMA_50'] < r4h['EMA_200'])
        
        atr = c1h['ATR']
        vol_ma = c1h['Vol_MA']
        
        # فیلتر حجم انفجاری (حداقل 1.8 برابر میانگین برای تایید ورود پول نهادی واقعی)
        if pd.isna(atr) or atr <= 0 or c1h['Volume'] < vol_ma * 1.8:
            continue
            
        # بررسی ساختار قیمت در 1 ساعته (باید بدنه کندل خیلی قدرتمند باشد)
        body_size = abs(c1h['Close'] - c1h['Open'])
        candle_range = c1h['High'] - c1h['Low']
        if candle_range == 0:
            continue
            
        is_clean_body = (body_size / candle_range) > 0.65  # حداقل 65 درصد کندل بدنه خالص باشد
        
        if is_strong_bull and (c1h['Close'] > c1h['Open']) and is_clean_body:
            # تایید ورود لانگ با بالاترین کیفیت
            entry_idx = i + 1
            if entry_idx >= len(df1h):
                break
                
            entry_price = df1h.iloc[entry_idx]['Open']
            sl = entry_price - (1.2 * atr)  # استاپ نزدیک و امن پشت ساختار
            tp = entry_price + (2.4 * atr)  # ریسک به ریوارد دقیق 1 به 2
            
            outcome = None
            exit_idx = entry_idx
            
            for j in range(entry_idx, len(df1h)):
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
                locked_until_index = exit_idx + 3  # استراحت طولانی‌تر برای فیلتر نویزها
                
        elif is_strong_bear and (c1h['Close'] < c1h['Open']) and is_clean_body:
            # تایید ورود شورت با بالاترین کیفیت
            entry_idx = i + 1
            if entry_idx >= len(df1h):
                break
                
            entry_price = df1h.iloc[entry_idx]['Open']
            sl = entry_price + (1.2 * atr)
            tp = entry_price - (2.4 * atr)  # ریسک به ریوارد 1 به 2
            
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
                locked_until_index = exit_idx + 3

print("\n============================================================")
print("📊 گزارش نهایی پورتفوی فوق‌العاده گزینشی (Hyper-Selective)")
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

print("\n✨ پایان بک‌تست.")
