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

# صرافی LBank با سبد گسترده‌تر (20 ارز برتر برای جبران تعداد معاملات با تنوع بازار)
exchange = ccxt.lbank({'enableRateLimit': True})
SYMBOLS = {
    "BTC": "BTC/USDT", "ETH": "ETH/USDT", "SOL": "SOL/USDT", "XRP": "XRP/USDT",
    "ADA": "ADA/USDT", "AVAX": "AVAX/USDT", "LINK": "LINK/USDT", "NEAR": "NEAR/USDT",
    "SUI": "SUI/USDT", "DOT": "DOT/USDT", "RENDER": "RENDER/USDT", "ARB": "ARB/USDT",
    "OP": "OP/USDT", "INJ": "INJ/USDT", "TIA": "TIA/USDT", "APT": "APT/USDT",
    "POL": "POL/USDT", "ICP": "ICP/USDT", "XLM": "XLM/USDT", "ATOM": "ATOM/USDT"
}

start_date = datetime.now() - timedelta(days=365)
since_timestamp = int(start_date.timestamp() * 1000)

print("============================================================")
print("📥 دانلود داده‌ها برای نسخه HUNTER-X PRO V14 (تمرکز روی حداکثر وین‌ریت)")
print("============================================================")

data_1h = {}
data_4h = {}
data_1d = {}

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
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna().reset_index()
        
        df1d = df1h.resample('1D').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna().reset_index()
        
        df1h.reset_index(inplace=True)
        
        data_1h[symbol] = df1h
        data_4h[symbol] = df4h
        data_1d[symbol] = df1d
        print(f"  ✔️ دیتای {symbol} آماده شد.")

def calculate_indicators(df):
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    df['Vol_MA'] = df['Volume'].rolling(window=20).mean()
    
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
print("🚀 اجرای موتور بک‌تست HUNTER-X PRO V14 (استراتژی فوق‌العاده باکیفیت)")
print("============================================================")

all_portfolio_trades = []

for symbol in SYMBOLS.keys():
    if symbol not in data_1h or symbol not in data_4h or symbol not in data_1d:
        continue
        
    df1h = data_1h[symbol].copy()
    df4h = data_4h[symbol].copy()
    df1d = data_1d[symbol].copy()
    
    if len(df1h) < 300 or len(df4h) < 100 or len(df1d) < 50:
        continue
        
    df1h = calculate_indicators(df1h)
    df4h = calculate_indicators(df4h)
    df1d = calculate_indicators(df1d)
    
    df1h['Date_4H'] = df1h['Date'].dt.floor('4h')
    df1h['Date_1D'] = df1h['Date'].dt.floor('1d')
    
    df4h_indexed = df4h.set_index('Date')
    df1d_indexed = df1d.set_index('Date')
    
    locked_until_index = 0
    
    for i in range(250, len(df1h) - 20):
        if i < locked_until_index:
            continue
            
        c1h = df1h.iloc[i]
        t4h_time = c1h['Date_4H']
        t1d_time = c1h['Date_1D']
        
        if t4h_time not in df4h_indexed.index or t1d_time not in df1d_indexed.index:
            continue
            
        r4h = df4h_indexed.loc[t4h_time]
        r1d = df1d_indexed.loc[t1d_time]
        
        # فیلتر بسیار سنگین روند در روزانه و ۴ ساعته
        trend_daily = r1d['Close'] > r1d['EMA_50']
        trend_4h = (r4h['EMA_50'] > r4h['EMA_200']) and (r4h['Close'] > r4h['EMA_50']) and (r4h['ADX'] > 28)
        
        atr = c1h['ATR']
        vol_ma = c1h['Vol_MA']
        
        if pd.isna(atr) or atr <= 0:
            continue
            
        # شرایط تاییدیه فوق‌العاده قوی در ۱ ساعته (پولبک عمیق + حجم بالا + کندل بازگشتی قدرتمند)
        prev_rsi = df1h.iloc[i-1]['RSI'] if i > 0 else 50
        is_deep_pullback = (prev_rsi < 42) and (c1h['RSI'] > prev_rsi)
        strong_volume = c1h['Volume'] >= (1.5 * vol_ma)
        bullish_candle = (c1h['Close'] > c1h['Open']) and ((c1h['Close'] - c1h['Open']) / (c1h['High'] - c1h['Low'] + 1e-9) > 0.6)
        
        valid_setup = trend_daily and trend_4h and is_deep_pullback and strong_volume and bullish_candle
        
        if valid_setup:
            entry_idx = i + 1
            if entry_idx >= len(df1h):
                break
                
            entry_price = df1h.iloc[entry_idx]['Open']
            recent_low = df1h['Low'].iloc[max(0, i-5):i+1].min()
            sl = recent_low - (0.4 * atr)
            risk = entry_price - sl
            if risk <= 0:
                risk = 1.0 * atr
                sl = entry_price - risk
                
            tp = entry_price + (2.5 * risk) # ریسک به ریوارد 1 به 2.5 برای سودآوری بالا
            
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
                locked_until_index = exit_idx + 3

print("\n============================================================")
print("📊 گزارش نهایی پورتفوی ستاپ HUNTER-X PRO V14 (کیفیت حداکثری)")
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

print("\n✨ پایان بک‌تست V14.")
