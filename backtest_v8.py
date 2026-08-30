import requests
import json
import os
from datetime import datetime

# تنظیمات تلگرام (از متغیرهای محیطی گیت‌هاب یا مقادیر پیش‌فرض)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "TOKEN_HERE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "CHAT_ID_HERE")

SYMBOLS = ["BTCUSDT", "SOLUSDT"]
STATE_FILE = "state.json"

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

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"خطا در ارسال پیام به تلگرام: {e}")

def get_market_data(symbol):
    # دریافت قیمت از API صرافی کوینکس (Coinex)
    url = f"https://api.coinex.com/v1/market/ticker?market={symbol}"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        
        # ساختار پاسخ کوینکس: {"code": 0, "data": {"ticker": {"last": "..."}}}
        if "data" in data and "ticker" in data["data"] and "last" in data["data"]["ticker"]:
            return float(data["data"]["ticker"]["last"])
        else:
            print(f"پاسخ کوینکس برای {symbol} نامعتبر بود: {data}")
            return None
    except Exception as e:
        print(f"خطا در دریافت قیمت از کوینکس برای {symbol}: {e}")
        return None

def run_bot():
    print(f"[{datetime.utcnow()}] در حال بررسی بازار...")
    state = load_state()
    
    for symbol in SYMBOLS:
        current_price = get_market_data(symbol)
        if not current_price:
            continue
            
        # بررسی پوزیشن باز قبلی برای این ارز
        if symbol in state:
            pos = state[symbol]
            tp = pos["tp"]
            sl = pos["sl"]
            
            # بررسی اینکه آیا قیمت به حد سود (TP) یا حد ضرر (SL) رسیده است یا خیر
            if current_price >= tp or current_price <= sl:
                print(f"پوزیشن {symbol} بسته شد. قیمت فعلی: {current_price}")
                # پوزیشن بسته شد، پس آن را پاک می‌کنیم تا اجازه صدور سیگنال جدید داده شود
                del state[symbol]
            else:
                print(f"ارز {symbol} پوزیشن باز دارد (ورود: {pos['entry']}, TP: {tp}, SL: {sl}). سیگنال جدید رد شد.")
                continue 
        
        # شرایط صدور سیگنال جدید
        entry_price = current_price
        sl_price = entry_price * (1 - 0.012)  # 1.2 درصد حد ضرر
        tp_price = entry_price * (1 + 0.015)  # 1.5 درصد حد سود
        rsi_val = 55.5  # مقدار نمونه RSI
        
        # ذخیره وضعیت پوزیشن جدید
        state[symbol] = {
            "entry": entry_price,
            "tp": tp_price,
            "sl": sl_price
        }
        
        # ارسال سیگنال به تلگرام
        message = (
            f"🚨 *سیگنال خرید جدید (Score Hunter Pro)* 🚨\n\n"
            f"🔹 ارز: `{symbol}`\n"
            f"⏱ تایم‌فریم: `15m`\n\n"
            f"📥 قیمت ورود: `{entry_price}`\n"
            f"🛑 حد ضرر (SL): `{sl_price:.4f} (-1.2%)`\n"
            f"🎯 حد سود (TP): `{tp_price:.4f} (+1.5%)`\n\n"
            f"📊 شاخص RSI: `{rsi_val}`\n"
            f"⏰ زمان: `{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC`"
        )
        send_telegram_message(message)
        print(f"سیگنال جدید برای {symbol} ارسال شد.")

    save_state(state)

if __name__ == "__main__":
    run_bot()
