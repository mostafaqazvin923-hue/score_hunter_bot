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
print("📥 دانلود داده‌ها برای ستاپ جدید Pullback & Continuation")
print("============================================================")

data_1h = {}

for symbol, lbank_symbol in SYMBOLS.items():
    filename_1h = f"{symbol}_1h_pullback_strategy.csv"
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
    df = df.copy()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # ATR برای فیلتر نوسان بازار
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Low'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    df['ATR_MA'] = df['ATR'].rolling(window=50).mean()
    return df

print("\n============================================================")
print("🚀 اجرای موتور بک‌تست ستاپ پولبک و ادامه روند (بدون Look-ahead)")
print("============================================================")

all_trades = []
COMMISSION_RATE = 0.0008 # 0.08% کل کارمزد رفت‌وبرگشت
COMMISSION_PER_SIDE = COMMISSION_RATE / 2
SLIPPAGE_RATE = 0.04 / 100 # 0.04% اسلیپیج

for symbol, df1h in data_1h.items():
    if len(df1h) < 300:
        continue
        
    df1h = calculate_indicators(df1h)
    
    # ساخت تایم‌فریم 4 ساعته بدون لوک‌آهد (با لبه چپ)
    df4h = df1h.set_index('Date').resample('4h', label='left', closed='left').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna().reset_index()
    
    df4h = calculate_indicators(df4h)
    
    locked_until_index = 0
    
    for i in range(250, len(df1h) - 40):
        if i < locked_until_index:
            continue
            
        c1h = df1h.iloc[i]
        current_time = c1h['Date']
        
        # دسترسی به کندل 4 ساعته کاملاً بسته‌شده
        closed_4h_time = current_time - timedelta(hours=4)
        available_4h = df4h[df4h['Date'] <= closed_4h_time]
        if len(available_4h) < 5:
            continue
            
        r4h = available_4h.iloc[-1]
        prev_r4h = available_4h.iloc[-2]
        
        if pd.isna(r4h['EMA_200']) or pd.isna(prev_r4h['EMA_200']):
            continue
            
        # بررسی شیب EMA200 در 4H
        ema200_slope_positive = r4h['EMA_200'] >= prev_r4h['EMA_200']
        ema200_slope_negative = r4h['EMA_200'] <= prev_r4h['EMA_200']
        
        is_long_regime = (r4h['Close'] > r4h['EMA_200']) and (r4h['EMA_50'] > r4h['EMA_200']) and ema200_slope_positive
        is_short_regime = (r4h['Close'] < r4h['EMA_200']) and (r4h['EMA_50'] < r4h['EMA_200']) and ema200_slope_negative
        
        if not is_long_regime and not is_short_regime:
            continue
            
        # فیلتر نوسان 1H (اطمینان از اینکه بازار مرده نیست و نوسان کافی دارد)
        if pd.isna(c1h['ATR']) or pd.isna(c1h['ATR_MA']) or c1h['ATR_MA'] == 0:
            continue
        if c1h['ATR'] < c1h['ATR_MA'] * 0.8: # بازار خیلی کم‌نوسان است
            continue
            
        # بررسی الگو در 1H: حرکت قوی + پولبک + کندل تأیید
        # نگاه به 5 کندل اخیر 1H برای تشخیص پولبک و حرکت قوی
        recent_1h = df1h.iloc[i-5:i+1]
        if len(recent_1h) < 6:
            continue
            
        # تعریف حرکت قوی: یکی از کندل‌های قبلی بدنه بزرگ رو به جلو داشته
        # تعریف پولبک: چند کندل اصلاحی کوچک‌تر خلاف جهت روند
        if is_long_regime:
            # بررسی اینکه آیا کندل جاری یک کندل تأیید صعودی است (Close > Open با بدنه مناسب)
            body_size = c1h['Close'] - c1h['Open']
            avg_recent_body = (recent_1h['Close'] - recent_1h['Open']).abs().mean()
            
            is_bullish_confirmation = (c1h['Close'] > c1h['Open']) and (body_size >= 0.6 * avg_recent_body)
            
            # پولبک کنترل‌شده: حداقل یکی از کندل‌های قبل نزولی بوده یا Low پایین‌تر زده اما ساختار روند 1H نشسته
            has_recent_pullback = recent_1h.iloc[-3]['Close'] < recent_1h.iloc[-4]['Close'] or \
                                  recent_1h.iloc[-2]['Close'] < recent_1h.iloc[-3]['Close']
                                  
            if is_bullish_confirmation and has_recent_pullback:
                entry_candle_idx = i + 1
                if entry_candle_idx >= len(df1h):
                    break
                
                entry_candle = df1h.iloc[entry_candle_idx]
                entry_price = entry_candle['Open'] * (1 + SLIPPAGE_RATE)
                
                sl = entry_price * 0.99
                tp = entry_price * 1.02
                risk = entry_price - sl
                
                outcome = 'OPEN'
                exit_idx = entry_candle_idx
                executed_exit_price = 0
                
                for j in range(entry_candle_idx, min(entry_candle_idx + 50, len(df1h))):
                    f_c = df1h.iloc[j]
                    exit_idx = j
                    
                    if f_c['Low'] <= sl:
                        outcome = 'LOSS'
                        executed_exit_price = sl * (1 - SLIPPAGE_RATE)
                        break
                    elif f_c['High'] >= tp:
                        outcome = 'WIN'
                        executed_exit_price = tp * (1 - SLIPPAGE_RATE)
                        break
                        
                if outcome in ['WIN', 'LOSS']:
                    gross_pnl = executed_exit_price - entry_price
                    entry_fee = entry_price * COMMISSION_PER_SIDE
                    exit_fee = executed_exit_price * COMMISSION_PER_SIDE
                    net_pnl = gross_pnl - entry_fee - exit_fee
                    net_R = net_pnl / risk
                    
                    all_trades.append({
                        'Symbol': symbol,
                        'Side': 'LONG',
                        'Outcome': outcome,
                        'NetR': net_R,
                        'Date': entry_candle['Date']
                    })
                    locked_until_index = exit_idx + 1
                    continue
                    
        elif is_short_regime:
            body_size = c1h['Open'] - c1h['Close']
            avg_recent_body = (recent_1h['Open'] - recent_1h['Close']).abs().mean()
            
            is_bearish_confirmation = (c1h['Close'] < c1h['Open']) and (body_size >= 0.6 * avg_recent_body)
            has_recent_pullback = recent_1h.iloc[-3]['Close'] > recent_1h.iloc[-4]['Close'] or \
                                  recent_1h.iloc[-2]['Close'] > recent_1h.iloc[-3]['Close']
                                  
            if is_bearish_confirmation and has_recent_pullback:
                entry_candle_idx = i + 1
                if entry_candle_idx >= len(df1h):
                    break
                
                entry_candle = df1h.iloc[entry_candle_idx]
                entry_price = entry_candle['Open'] * (1 - SLIPPAGE_RATE)
                
                sl = entry_price * 1.01
                tp = entry_price * 0.98
                risk = sl - entry_price
                
                outcome = 'OPEN'
                exit_idx = entry_candle_idx
                executed_exit_price = 0
                
                for j in range(entry_candle_idx, min(entry_candle_idx + 50, len(df1h))):
                    f_c = df1h.iloc[j]
                    exit_idx = j
                    
                    if f_c['High'] >= sl:
                        outcome = 'LOSS'
                        executed_exit_price = sl * (1 + SLIPPAGE_RATE)
                        break
                    elif f_c['Low'] <= tp:
                        outcome = 'WIN'
                        executed_exit_price = tp * (1 + SLIPPAGE_RATE)
                        break
                        
                if outcome in ['WIN', 'LOSS']:
                    gross_pnl = entry_price - executed_exit_price
                    entry_fee = entry_price * COMMISSION_PER_SIDE
                    exit_fee = executed_exit_price * COMMISSION_PER_SIDE
                    net_pnl = gross_pnl - entry_fee - exit_fee
                    net_R = net_pnl / risk
                    
                    all_trades.append({
                        'Symbol': symbol,
                        'Side': 'SHORT',
                        'Outcome': outcome,
                        'NetR': net_R,
                        'Date': entry_candle['Date']
                    })
                    locked_until_index = exit_idx + 1
                    continue

print("\n============================================================")
print("📊 گزارش عملکرد ستاپ پولبک و ادامه روند")
print("============================================================")

if all_trades:
    tdf = pd.DataFrame(all_trades)
    total_trades = len(tdf)
    wins = len(tdf[tdf['Outcome'] == 'WIN'])
    losses = len(tdf[tdf['Outcome'] == 'LOSS'])
    win_rate = (wins / total_trades) * 100
    total_net_r = tdf['NetR'].sum()
    
    print(f"🎯 تعداد کل معاملات: {total_trades}")
    print(f"🏆 برد: {wins} | ❌ باخت: {losses}")
    print(f"📈 وین‌ریت: {win_rate:.2f}%")
    print(f"💰 مجموع سود خالص (Net R با کسر کارمزد و اسلیپیج): {total_net_r:.2f}R")
    
    print("\n--- تفکیک نمادها ---")
    symbol_grouped = tdf.groupby('Symbol').agg(
        Trades=('Outcome', 'count'),
        Wins=('Outcome', lambda x: (x == 'WIN').sum()),
        Losses=('Outcome', lambda x: (x == 'LOSS').sum()),
        NetR=('NetR', 'sum')
    )
    symbol_grouped['WinRate'] = (symbol_grouped['Wins'] / symbol_grouped['Trades'] * 100).round(2)
    print(symbol_grouped)
else:
    print("⚠️ هیچ معامله‌ای ثبت نشد.")
