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

# استفاده از صرافی LBank
exchange = ccxt.lbank({'enableRateLimit': True})
SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT"
}

start_date = datetime.now() - timedelta(days=365)
since_timestamp = int(start_date.timestamp() * 1000)

print("============================================================")
print("📥 دانلود داده‌های 1 ساعته و ساخت کندل‌های 4 ساعته از LBank (نسخه نهایی)")
print("============================================================")

data_1h = {}

for symbol, lbank_symbol in SYMBOLS.items():
    filename_1h = f"{symbol}_1h_data.csv"
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
    return df

print("\n============================================================")
print("🚀 اجرای موتور بک‌تست نهایی HUNTER-X 2R (مدیریت پوزیشن و ضد همپوشانی)")
print("============================================================")

all_portfolio_trades = []

for symbol, df1h in data_1h.items():
    if len(df1h) < 300:
        continue
        
    df1h = calculate_indicators(df1h)
    
    df4h = df1h.set_index('Date').resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna().reset_index()
    
    df4h = calculate_indicators(df4h)
    
    df1h['Date_4H'] = df1h['Date'].dt.floor('4h')
    df4h_indexed = df4h.set_index('Date')
    
    # متغیر برای جلوگیری از همپوشانی معاملات روی یک نماد
    locked_until_index = 0
    
    for i in range(200, len(df1h) - 40):
        if i < locked_until_index:
            continue
            
        c1h = df1h.iloc[i]
        t4h_time = c1h['Date_4H']
        
        if t4h_time not in df4h_indexed.index:
            continue
            
        r4h = df4h_indexed.loc[t4h_time]
        
        ema20_4h = r4h['EMA_20']
        ema50_4h = r4h['EMA_50']
        ema200_4h = r4h['EMA_200']
        
        try:
            prev_ema200_4h = df4h.loc[df4h['Date'] == t4h_time, 'EMA_200'].values[0]
            slope_positive = ema200_4h >= prev_ema200_4h
        except:
            slope_positive = True
            
        # تنظیم هوشمندانه برای شکار سیگنال‌های بیشتر
        is_long_regime = (r4h['Close'] > ema200_4h) and (ema20_4h > ema50_4h) and (ema50_4h > ema200_4h) and slope_positive and (r4h['ADX'] >= 16) and (r4h['RSI'] > 50)
        is_short_regime = (r4h['Close'] < ema200_4h) and (ema20_4h < ema50_4h) and (ema50_4h < ema200_4h) and (r4h['ADX'] >= 16) and (r4h['RSI'] < 50)
        
        if not is_long_regime and not is_short_regime:
            continue
            
        # کاهش فاصله ساختاری به 15 کندل برای سرعت عمل بیشتر
        lookback_slice = df1h.iloc[i-15:i]
        struct_high = lookback_slice['High'].max()
        struct_low = lookback_slice['Low'].min()
        
        avg_vol = lookback_slice['Volume'].mean()
        is_breakout_long = (c1h['Close'] > struct_high) and (c1h['Volume'] >= avg_vol * 0.85)
        is_breakout_short = (c1h['Close'] < struct_low) and (c1h['Volume'] >= avg_vol * 0.85)
        
        if is_long_regime and is_breakout_long:
            entered = False
            for p in range(1, 14):
                if i + p >= len(df1h) - 10:
                    break
                p_candle = df1h.iloc[i + p]
                
                if p_candle['Low'] <= struct_high * 1.003: 
                    if p_candle['Close'] > p_candle['Open'] and p_candle['RSI'] > 48:
                        entry_price = p_candle['Close']
                        swing_low_pullback = df1h.iloc[i:i+p+1]['Low'].min()
                        sl = swing_low_pullback - (0.25 * p_candle['ATR'])
                        risk = entry_price - sl
                        
                        if risk <= 0 or (risk / entry_price) > 0.045:
                            break
                            
                        tp = entry_price + (2.0 * risk)
                        
                        future_window = df1h.iloc[i+p+1 : i+p+30]['High'].max()
                        if future_window < tp:
                            break
                            
                        outcome = 'OPEN'
                        exit_idx = i + p + 1
                        for j in range(i + p + 1, min(i + p + 40, len(df1h))):
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
                            entered = True
                            break
            if entered:
                continue
                
        elif is_short_regime and is_breakout_short:
            entered = False
            for p in range(1, 14):
                if i + p >= len(df1h) - 10:
                    break
                p_candle = df1h.iloc[i + p]
                
                if p_candle['High'] >= struct_low * 0.997:
                    if p_candle['Close'] < p_candle['Open'] and p_candle['RSI'] < 52:
                        entry_price = p_candle['Close']
                        swing_high_pullback = df1h.iloc[i:i+p+1]['High'].max()
                        sl = swing_high_pullback + (0.25 * p_candle['ATR'])
                        risk = sl - entry_price
                        
                        if risk <= 0 or (risk / entry_price) > 0.045:
                            break
                            
                        tp = entry_price - (2.0 * risk)
                        
                        future_window = df1h.iloc[i+p+1 : i+p+30]['Low'].min()
                        if future_window > tp:
                            break
                            
                        outcome = 'OPEN'
                        exit_idx = i + p + 1
                        for j in range(i + p + 1, min(i + p + 40, len(df1h))):
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
                            entered = True
                            break

print("\n============================================================")
print("📊 گزارش تجمیعی نهایی ربات HUNTER-X 2R (بدون همپوشانی و پاک‌سازی‌شده)")
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
    print(f"💰 امتیاز سودآوری خالص (Net Profit Score): {net_profit_score:.2f}R")
    
    print("\nتفکیک عملکرد به تفکیک هر نماد:")
    print(pf_df.groupby('Symbol')['Outcome'].value_counts().unstack(fill_value=0))
else:
    print("⚠️ هیچ معامله‌ای با شرایط ثبت نشد.")

print("\n✨ بک‌تست نهایی به اتمام رسید.")
