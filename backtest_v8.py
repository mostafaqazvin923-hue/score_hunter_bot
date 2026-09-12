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
print("📥 دانلود داده‌ها برای نسخه ارتقایافته HUNTER-X PRO V11")
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

def calculate_indicators(df):
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['EMA_50_Slope'] = df['EMA_50'].diff()
    
    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR 14
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
    
    # Donchian Channel 20 (بدون نگاه به آینده با shift)
    df['Donchian_High'] = df['High'].rolling(window=20).max().shift(1)
    df['Donchian_Low'] = df['Low'].rolling(window=20).min().shift(1)
    
    # ADX 14
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    tr14 = tr.rolling(window=14).sum()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14).sum() / tr14.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14).sum() / tr14.replace(0, np.nan)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    df['ADX'] = dx.rolling(window=14).mean()
    
    return df

print("\n============================================================")
print("🚀 اجرای موتور بک‌تست HUNTER-X PRO V11 (بهینه‌شده برای افزایش وین‌ریت)")
print("============================================================")

all_portfolio_trades = []

for symbol in SYMBOLS.keys():
    if symbol not in data_1h or symbol not in data_4h:
        continue
        
    df1h = data_1h[symbol].copy()
    df4h = data_4h[symbol].copy()
    
    if len(df1h) < 300 or len(df4h) < 100:
        continue
        
    df1h = calculate_indicators(df1h)
    df4h = calculate_indicators(df4h)
    
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
        
        # فیلترهای روند 4H با ADX سخت‌گیرانه‌تر (بالای 25 برای قدرت بیشتر روند)
        long_4h = (r4h['EMA_50'] > r4h['EMA_200']) and (r4h['Close'] > r4h['EMA_50']) and (r4h['EMA_50_Slope'] > 0) and (r4h['ADX'] > 25) and (r4h['RSI'] > 55)
        short_4h = (r4h['EMA_50'] < r4h['EMA_200']) and (r4h['Close'] < r4h['EMA_50']) and (r4h['EMA_50_Slope'] < 0) and (r4h['ADX'] > 25) and (r4h['RSI'] < 45)
        
        atr = c1h['ATR']
        vol_ma = c1h['Vol_MA']
        
        if pd.isna(atr) or atr <= 0:
            continue
            
        # بررسی حجم و کندل Displacement با فیلتر دقیق‌تر
        vol_expansion = c1h['Volume'] >= (1.3 * vol_ma) # سخت‌گیری بیشتر روی حجم
        body_size = abs(c1h['Close'] - c1h['Open'])
        candle_range = c1h['High'] - c1h['Low']
        displacement = (candle_range > 0) and ((body_size / candle_range) > 0.60) and (candle_range > (atr * 0.9))
        
        if not (vol_expansion and displacement):
            continue
            
        # بررسی شکست Donchian روی 1H
        is_donchian_long_breakout = c1h['Close'] > c1h['Donchian_High']
        is_donchian_short_breakout = c1h['Close'] < c1h['Donchian_Low']
        
        # اجرای لانگ
        if long_4h and is_donchian_long_breakout:
            entry_idx = i + 1
            if entry_idx >= len(df1h):
                break
                
            entry_price = df1h.iloc[entry_idx]['Open']
            recent_low = df1h['Low'].iloc[max(0, i-5):i+1].min()
            sl = recent_low - (0.2 * atr)
            risk = entry_price - sl
            if risk <= 0:
                risk = 1.0 * atr
                sl = entry_price - risk
                
            tp = entry_price + (2.0 * risk)
            
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
                locked_until_index = exit_idx + 2
                
        # اجرای شورت
        elif short_4h and is_donchian_short_breakout:
            entry_idx = i + 1
            if entry_idx >= len(df1h):
                break
                
            entry_price = df1h.iloc[entry_idx]['Open']
            recent_high = df1h['High'].iloc[max(0, i-5):i+1].max()
            sl = recent_high + (0.2 * atr)
            risk = sl - entry_price
            if risk <= 0:
                risk = 1.0 * atr
                sl = entry_price + risk
                
            tp = entry_price - (2.0 * risk)
            
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
                locked_until_index = exit_idx + 2

print("\n============================================================")
print("📊 گزارش نهایی پورتفوی ستاپ HUNTER-X PRO V11 (ارتقایافته)")
print("============================================================")

if all_portfolio_trades:
    pf_df = pd.DataFrame(all_portfolio_trades)
    total_trades = len(pf_df)
    total_wins = len(pf_df[pf_df['Outcome'] == 'WIN'])
    total_losses = len(pf_df[pf_df['Outcome'] == 'LOSS'])
    
    win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    
    print(f"🔸 تعداد کل معاملات پورتفوی: {total_trades}")
    print(f"🔸 معاملات برنده (WIN): {total_wins}")
    print(f"🔸 معاملات بازنده (LOSS): {total_losses}")
    print(f"🎯 **وین‌ریت کلی:** {win_rate:.2f}%")
    
    print("\nتفکیک عملکرد به تفکیک هر نماد:")
    print(pf_df.groupby('Symbol')['Outcome'].value_counts().unstack(fill_value=0))
else:
    print("⚠️ هیچ معامله‌ای با این شرایط ثبت نشد.")

print("\n✨ پایان بک‌تست V11.")
