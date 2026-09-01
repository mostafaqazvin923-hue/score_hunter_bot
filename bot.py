import os
import json
import requests
import pandas as pd
import numpy as np

# تنظیمات تلگرام از محیط گیت‌هاب (Environment Variables)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

STATE_FILE = "active_trades_state.json"

def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials not found!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        return response.json()
    except Exception as e:
        print(f"Error sending telegram message: {e}")

def get_binance_data(symbol="BTCUSDT", interval="1h", limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def run_bot():
    print("[*] Running Score Hunter Pro live check...")
    
    # تشخیص اینکه آیا اجرا به صورت دستی (Workflow Dispatch) بوده یا خودکار
    # اگر فایلگیت‌هاب اکشن متغیر RUN_TYPE را ست کرده باشد یا برای تست دستی اولیه
    is_manual_run = os.environ.get("MANUAL_RUN", "false").lower() == "true"
    if is_manual_run:
        send_telegram_message("🤖 ربات اسکور هانتر با موفقیت اجرا شد و بازار را پایش می‌کند.")

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    state = load_state()

    for symbol in symbols:
        df = get_binance_data(symbol, interval="1h", limit=100)
        if df is None or len(df) < 50:
            continue

        df['rsi'] = calculate_rsi(df['close'], period=14)
        
        # منطق استراتژی (بک‌تست)
        current_close = df['close'].iloc[-2]
        current_rsi = df['rsi'].iloc[-2]
        prev_high = df['high'].iloc[-5:-1].max()

        # بررسی پوزیشن باز برای این ارز
        if symbol in state and state[symbol]:
            trade = state[symbol]
            entry_price = trade['entry_price']
            tp = trade['tp']
            sl = trade['sl']
            
            high_price = df['high'].iloc[-1]
            low_price = df['low'].iloc[-1]

            if high_price >= tp:
                tp_pct = round(((tp - entry_price) / entry_price) * 100, 2)
                send_telegram_message(f"🟢 *TP Hit (سود)*\nرمز ارز: `{symbol}`\nحد سود لمس شد! 🎯\nسود معامله: `+{tp_pct}%`")
                del state[symbol]
            elif low_price <= sl:
                sl_pct = round(((entry_price - sl) / entry_price) * 100, 2)
                send_telegram_message(f"🔴 *SL Hit (ضرر)*\nرمز ارز: `{symbol}`\nحد ضرر لمس شد! 🛑\nضرر معامله: `-{sl_pct}%`")
                del state[symbol]
        else:
            # شرایط ورود به معامله (مطابق بک‌تست)
            # مثال: عبور قیمت از مقاومت قبلی (BOS) همراه با RSI بین 40 تا 70
            if current_close > prev_high and 40 <= current_rsi <= 70:
                entry_price = current_close
                tp = entry_price * 1.025  # فرض مثال 2.5 درصد تی پی
                sl = entry_price * 0.99   # فرض مثال 1 درصد اس ال
                
                tp_pct = round(((tp - entry_price) / entry_price) * 100, 2)
                sl_pct = round(((entry_price - sl) / entry_price) * 100, 2)

                state[symbol] = {
                    "entry_price": entry_price,
                    "tp": tp,
                    "sl": sl
                }
                
                send_telegram_message(
                    f"🚀 *سیگنال جدید (Score Hunter Pro)*\n\n"
                    f"ارز: `{symbol}`\n"
                    f"قیمت ورود: `{entry_price}`\n"
                    f"🎯 حد سود (TP): `{tp}` (`+{tp_pct}%`)\n"
                    f"🛑 حد ضرر (SL): `{sl}` (`-{sl_pct}%`)\n"
                    f"📊 RSI: `{round(current_rsi, 2)}`"
                )

    save_state(state)
    print("[*] State successfully saved.")

if __name__ == "__main__":
    run_bot()
