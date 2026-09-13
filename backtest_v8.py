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
print("📥 دانلود داده‌های 1 ساعته برای هدف وین‌ریت 70% (فیلترهای فوق‌سخت‌گیرانه)")
print("============================================================")

data_1h = {}

for symbol, lbank_symbol in SYMBOLS.items():
    filename_1h = f"{symbol}_1h_elite_data.csv"
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

def calculate_indicators(df):
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
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
    
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    tr14 = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr14)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['ADX'] = dx.rolling(window=14).mean().fillna(20)
    
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    return df

print("\n============================================================")
print("🚀 اجرای موتور بک‌تست با فیلترهای الیت (هدف: وین‌ریت بالا | R:R 1:1)")
print("============================================================")

all_portfolio_trades = {}

for symbol, df1h in data_1h.items():
    if len(df1h) < 300:
        continue
        
    df1h = calculate_indicators(df1h)
    
    df4h = df1h.set_index('Date').resample('4h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna().reset_index()
    
    df4h = calculate_indicators(df4h)
    df1h['Date_4H'] = df1h['Date'].dt.floor('4h')
    df4h_indexed = df4h.set_index('Date')
    
    symbol_trades = []
    locked_until_index = 0
    
    for i in range(200, len(df1h) - 10):
        if i < locked_until_index:
            continue
            
        c1h = df1h.iloc[i]
        t4h_time = c1h['Date_4H']
        
        if t4h_time not in df4h_indexed.index:
            continue
            
        r4h = df4h_indexed.loc[t4h_time]
        
        # روند فوق‌العاده قدرتمند با ADX بالای 35 (حذف کامل بازارهای نویزدار)
        is_elite_uptrend = (r4h['Close'] > r4h['EMA_200']) and (r4h['EMA_20'] > r4h['EMA_50']) and (r4h['ADX'] >= 35)
        is_elite_downtrend = (r4h['Close'] < r4h['EMA_200']) and (r4h['EMA_20'] < r4h['EMA_50']) and (r4h['ADX'] >= 35)
        
        if not is_elite_uptrend and not is_elite_downtrend:
            continue
            
        # شرط پولبک به ناحیه ارزش (نزدیک بودن قیمت به EMA 20 یا EMA 50)
        is_near_ema_long = (c1h['Low'] <= c1h['EMA_20']) and (c1h['Close'] > c1h['EMA_50'])
        is_near_ema_short = (c1h['High'] >= c1h['EMA_20']) and (c1h['Close'] < c1h['EMA_50'])
        
        candle_range = c1h['High'] - c1h['Low']
        if candle_range == 0:
            continue
            
        lower_wick = c1h['Close'] - c1h['Low'] if c1h['Close'] > c1h['Open'] else c1h['Open'] - c1h['Low']
        upper_wick = c1h['High'] - c1h['Open'] if c1h['Close'] > c1h['Open'] else c1h['High'] - c1h['Close']
        
        # تاییدیه پین‌بار / ریجکشن ساختاری تمیز به همراه حجم انفجاری (2 برابر میانگین)
        is_clean_bullish_pin = (lower_wick >= candle_range * 0.6) and (c1h['Close'] > c1h['Open']) and (c1h['Volume'] > c1h['Volume_MA20'] * 2.0)
        is_clean_bearish_pin = (upper_wick >= candle_range * 0.6) and (c1h['Close'] < c1h['Open']) and (c1h['Volume'] > c1h['Volume_MA20'] * 2.0)
        
        is_long_signal = is_elite_uptrend and is_near_ema_long and is_clean_bullish_pin and (45 < c1h['RSI'] < 65)
        is_short_signal = is_elite_downtrend and is_near_ema_short and is_clean_bearish_pin and (35 < c1h['RSI'] < 55)
        
        if is_long_signal:
            entry_price = c1h['Close']
            sl = c1h['Low'] - (0.5 * c1h['ATR'])
            risk = entry_price - sl
            
            if risk > 0 and (risk / entry_price) <= 0.04:
                tp = entry_price + (1.0 * risk)
                
                outcome = None
                exit_idx = i + 1
                for j in range(i + 1, len(df1h)):
                    f_c = df1h.iloc[j]
                    exit_idx = j
                    
                    hit_sl = f_c['Low'] <= sl
                    hit_tp = f_c['High'] >= tp
                    
                    if hit_sl and hit_tp:
                        outcome = 'LOSS'
                        break
                    elif hit_sl:
                        outcome = 'LOSS'
                        break
                    elif hit_tp:
                        outcome = 'WIN'
                        break
                        
                if outcome in ['WIN', 'LOSS']:
                    symbol_trades.append({
                        'Symbol': symbol,
                        'Side': 'LONG',
                        'Outcome': outcome
                    })
                    locked_until_index = exit_idx
                    
        elif is_short_signal:
            entry_price = c1h['Close']
            sl = c1h['High'] + (0.5 * c1h['ATR'])
            risk = sl - entry_price
            
            if risk > 0 and (risk / entry_price) <= 0.04:
                tp = entry_price - (1.0 * risk)
                
                outcome = None
                exit_idx = i + 1
                for j in range(i + 1, len(df1h)):
                    f_c = df1h.iloc[j]
                    exit_idx = j
                    
                    hit_sl = f_c['High'] >= sl
                    hit_tp = f_c['Low'] <= tp
                    
                    if hit_sl and hit_tp:
                        outcome = 'LOSS'
                        break
                    elif hit_sl:
                        outcome = 'LOSS'
                        break
                    elif hit_tp:
                        outcome = 'WIN'
                        break
                        
                if outcome in ['WIN', 'LOSS']:
                    symbol_trades.append({
                        'Symbol': symbol,
                        'Side': 'SHORT',
                        'Outcome': outcome
                    })
                    locked_until_index = exit_idx

    if symbol_trades:
        all_portfolio_trades[symbol] = symbol_trades

print("\n============================================================")
print("📊 گزارش نهایی پورتفوی با فیلترهای الیت (R:R 1:1)")
print("============================================================")

flat_trades = []
for sym, trades in all_portfolio_trades.items():
    flat_trades.extend(trades)

if flat_trades:
    pf_df = pd.DataFrame(flat_trades)
    total_trades = len(pf_df)
    total_wins = len(pf_df[pf_df['Outcome'] == 'WIN'])
    total_losses = len(pf_df[pf_df['Outcome'] == 'LOSS'])
    win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    net_profit = (total_wins * 1.0) - total_losses
    
    print(f"🔸 تعداد کل معاملات پورتفو: {total_trades}")
    print(f"🔸 کل برنده (WIN): {total_wins}")
    print(f"🔸 کل بازنده (LOSS): {total_losses}")
    print(f"🎯 **وین‌ریت واقعی پورتفوی:** {win_rate:.2f}%")
    print(f"💰 **سود خالص (بر حسب واحد R):** {net_profit:.2f}R")
    
    print("\nتفکیک عملکرد نمادها:")
    print(pf_df.groupby('Symbol')['Outcome'].value_counts().unstack(fill_value=0))
else:
    print("⚠️ هیچ معامله‌ای ثبت نشد.")

print("\n✨ اجرای اسکریپت با موفقیت به پایان رسید.")
