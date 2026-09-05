import os
import json
import requests
import subprocess
import sys
from datetime import datetime

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import pandas as pd
import numpy as np

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MANUAL_RUN_ENV = os.getenv("MANUAL_RUN", "false")
MANUAL_RUN = str(MANUAL_RUN_ENV).lower() == "true"
STATE_FILE = "active_trades_state.json"

def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ توکن یا چت‌آیدی تلگرام تنظیم نشده است.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"❌ خطا در ارسال پیام تلگرام: {e}")

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except:
            return {"active": {}, "cooldown": {}}
    return {"active": {}, "cooldown": {}}

def save_and_commit_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
        print("💾 حافظه ربات با موفقیت در فایل محلی ذخیره شد.")
    except Exception as e:
        print(f"❌ خطا در ذخیره فایل وضعیت: {e}")

if MANUAL_RUN:
    send_telegram_message("✅ ربات Score Hunter Pro با سیستم مانیتورینگ روی LBank استارت شد.")

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

state_data = load_state()
active_trades = state_data.get("active", {})
cooldowns = state_data.get("cooldown", {})

print("============================================================")
print("🔍 مانیتورینگ هوشمند پوزیشن‌ها و اسکن بازار...")
print("============================================================")

# ۱. مانیتورینگ دقیق پوزیشن‌های باز
symbols_to_remove = []
for symbol, trade in active_trades.items():
    lbank_symbol = SYMBOLS.get(symbol)
    if not lbank_symbol:
        continue
    try:
        ohlcv = exchange.fetch_ohlcv(lbank_symbol, timeframe='1h', limit=24)
        ticker = exchange.fetch_ticker(lbank_symbol)
        current_price = ticker['last']
        
        trade_time = pd.to_datetime(trade['time'])
        direction = trade['direction']
        tp = trade['tp']
        sl = trade['sl']
        entry = trade['entry_price']
        
        hit_tp = False
        hit_sl = False
        
        # بررسی قیمت لحظه‌ای بازار
        if direction == "LONG":
            if current_price >= tp:
                hit_tp = True
            elif current_price <= sl:
                hit_sl = True
        elif direction == "SHORT":
            if current_price <= tp:
                hit_tp = True
            elif current_price >= sl:
                hit_sl = True
                
        # بررسی کندل‌های تاریخی در صورت نیاز
        if not hit_tp and not hit_sl and ohlcv:
            df_candles = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df_candles['Date'] = pd.to_datetime(df_candles['Timestamp'], unit='ms')
            df_active_period = df_candles[df_candles['Date'] >= trade_time.floor('h')]
            
            for _, row in df_active_period.iterrows():
                if direction == "LONG":
                    if row['High'] >= tp:
                        hit_tp = True
                        break
                    elif row['Low'] <= sl:
                        hit_sl = True
                        break
                elif direction == "SHORT":
                    if row['Low'] <= tp:
                        hit_tp = True
                        break
                    elif row['High'] >= sl:
                        hit_sl = True
                        break
                        
        if hit_tp:
            msg = (
                f"🎯 **حد سود لمس شد (TP Hit)!** 🎉\n"
                f"💎 جفت ارز: `{symbol}USDT`\n"
                f"📍 قیمت ورود: `{entry:.4f}`\n"
                f"🎯 هدف سود: `{tp:.4f}`\n"
                f"✨ پوزیشن با موفقیت بسته شد."
            )
            send_telegram_message(msg)
            symbols_to_remove.append(symbol)
        elif hit_sl:
            msg = (
                f"🛑 **حد ضرر لمس شد (SL Hit)!**\n"
                f"💎 جفت ارز: `{symbol}USDT`\n"
                f"📍 قیمت ورود: `{entry:.4f}`\n"
                f"🛑 حد ضرر: `{sl:.4f}`\n"
                f"⚠️ پوزیشن متوقف شد."
            )
            send_telegram_message(msg)
            symbols_to_remove.append(symbol)
            
    except Exception as e:
        print(f"❌ خطا در مانیتورینگ نماد {symbol}: {e}")

for sym in symbols_to_remove:
    if sym in active_trades:
        cooldowns[sym] = active_trades[sym]["time"]
        del active_trades[sym]

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

# ۲. اسکن سیگنال‌های جدید
for symbol, lbank_symbol in SYMBOLS.items():
    if symbol in active_trades:
        print(f"🔒 نماد {symbol} دارای پوزیشن فعال است؛ اسکن رد شد.")
        continue
        
    try:
        ohlcv = exchange.fetch_ohlcv(lbank_symbol, timeframe='1h', limit=300)
        if not ohlcv or len(ohlcv) < 250:
            continue
            
        df1h = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df1h['Date'] = pd.to_datetime(df1h['Timestamp'], unit='ms')
        df1h = df1h[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df1h.dropna(inplace=True)
        df1h.reset_index(drop=True, inplace=True)
        
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
        
        i = len(df1h) - 2
        c1h = df1h.iloc[i]
        t4h_time = c1h['Date_4H']
        
        candle_time_str = str(c1h['Date'])
        if symbol in cooldowns and cooldowns[symbol] == candle_time_str:
            print(f"⏳ نماد {symbol} روی این کندل قبلاً معامله شده است.")
            continue
        elif symbol in cooldowns and cooldowns[symbol] != candle_time_str:
            del cooldowns[symbol]

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
            
        is_long_regime = (r4h['Close'] > ema200_4h) and (ema20_4h > ema50_4h) and (ema50_4h > ema200_4h) and slope_positive and (r4h['ADX'] >= 16) and (r4h['RSI'] > 50)
        is_short_regime = (r4h['Close'] < ema200_4h) and (ema20_4h < ema50_4h) and (ema50_4h < ema200_4h) and (r4h['ADX'] >= 16) and (r4h['RSI'] < 50)
        
        if not is_long_regime and not is_short_regime:
            continue
            
        lookback_slice = df1h.iloc[i-15:i]
        struct_high = lookback_slice['High'].max()
        struct_low = lookback_slice['Low'].min()
        avg_vol = lookback_slice['Volume'].mean()
        
        is_breakout_long = (c1h['Close'] > struct_high) and (c1h['Volume'] >= avg_vol * 0.85)
        is_breakout_short = (c1h['Close'] < struct_low) and (c1h['Volume'] >= avg_vol * 0.85)
        
        if is_long_regime and is_breakout_long:
            entry_price = c1h['Close']
            swing_low_pullback = lookback_slice['Low'].min()
            sl = swing_low_pullback - (0.25 * c1h['ATR'])
            risk = entry_price - sl
            if risk > 0 and (risk / entry_price) <= 0.045:
                tp = entry_price + (2.0 * risk)
                sl_pct = (risk / entry_price) * 100
                tp_pct = ((tp - entry_price) / entry_price) * 100
                
                active_trades[symbol] = {
                    "symbol": lbank_symbol,
                    "direction": "LONG",
                    "entry_price": entry_price,
                    "tp": tp,
                    "sl": sl,
                    "time": candle_time_str
                }
                
                signal_text = (
                    f"🚀 **سیگنال جدید (LONG)**\n"
                    f"💎 جفت ارز: `{symbol}USDT` (صرافی LBank)\n"
                    f"📍 قیمت ورود: `{entry_price:.4f}`\n"
                    f"🎯 حد سود (TP): `{tp:.4f}` (+{tp_pct:.2f}%)\n"
                    f"🛑 حد ضرر (SL): `{sl:.4f}` (-{sl_pct:.2f}%)\n"
                    f"⚖️ ریسک به ریوارد: `1:2`"
                )
                send_telegram_message(signal_text)
                
        elif is_short_regime and is_breakout_short:
            entry_price = c1h['Close']
            swing_high_pullback = lookback_slice['High'].max()
            sl = swing_high_pullback + (0.25 * c1h['ATR'])
            risk = sl - entry_price
            if risk > 0 and (risk / entry_price) <= 0.045:
                tp = entry_price - (2.0 * risk)
                sl_pct = (risk / entry_price) * 100
                tp_pct = ((entry_price - tp) / entry_price) * 100
                
                active_trades[symbol] = {
                    "symbol": lbank_symbol,
                    "direction": "SHORT",
                    "entry_price": entry_price,
                    "tp": tp,
                    "sl": sl,
                    "time": candle_time_str
                }
                
                signal_text = (
                    f"📉 **سیگنال جدید (SHORT)**\n"
                    f"💎 جفت ارز: `{symbol}USDT` (صرافی LBank)\n"
                    f"📍 قیمت ورود: `{entry_price:.4f}`\n"
                    f"🎯 حد سود (TP): `{tp:.4f}` (+{tp_pct:.2f}%)\n"
                    f"🛑 حد ضرر (SL): `{sl:.4f}` (-{sl_pct:.2f}%)\n"
                    f"⚖️ ریسک به ریوارد: `1:2`"
                )
                send_telegram_message(signal_text)
                
    except Exception as e:
        print(f"❌ خطا در پردازش نماد {symbol}: {e}")

save_and_commit_state({"active": active_trades, "cooldown": cooldowns})
print("✨ پایان اسکن و مانیتورینگ بازار.")
