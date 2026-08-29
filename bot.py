import json
import urllib.request
import time
import requests

# =========================================================
# 1. تنظیمات تلگرام (این دو مورد را با اطلاعات خودت پر کن)
# =========================================================
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"  # توکن رباتت
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"      # آیدی کانال یا چت شخصی‌ات

# لیست ۱۰ ارز مورد نظر با فرمت OKX
SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "DOGE-USDT", "DYDX-USDT",
    "LINK-USDT", "ADA-USDT", "XRP-USDT", "NEAR-USDT", "AVAX-USDT"
]

# حافظه برای جلوگیری از ارسال سیگنال تکراری
last_signals = {s: None for s in SYMBOLS}

# =========================================================
# 2. توابع ارسال پیام و دریافت دیتا
# =========================================================
def send_telegram_message(message):
    """ارسال پیام به تلگرام"""
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

def get_crypto_klines_okx(symbol, bar="15m", limit=150):
    """دریافت کندل‌ها از OKX (بدون تحریم و بدون کلید)"""
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={bar}&limit={limit}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode())
            if res_data.get('code') != '0' or 'data' not in res_data:
                return None
            
            raw_list = res_data['data']
            raw_list.reverse()
            
            klines = []
            for item in raw_list:
                klines.append({
                    'timestamp': int(item[0]),
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4]),
                    'volume': float(item[5])
                })
            return klines
    except Exception as e:
        print(f"خطا در دریافت دیتای {symbol}: {e}")
        return None

def calculate_ema(prices, span):
    alpha = 2 / (span + 1)
    ema = [prices[0]]
    for p in prices[1:]:
        ema.append(p * alpha + ema[-1] * (1 - alpha))
    return ema

# =========================================================
# 3. منطق تحلیلی استراتژی (نسخه v8.16)
# =========================================================
def analyze_and_signal(symbol):
    klines_15m = get_crypto_klines_okx(symbol=symbol, bar="15m", limit=150)
    if not klines_15m or len(klines_15m) < 100:
        return

    closes = [k['close'] for k in klines_15m]
    highs = [k['high'] for k in klines_15m]
    lows = [k['low'] for k in klines_15m]

    # محاسبه ATR (14)
    atr = []
    for i in range(len(klines_15m)):
        if i == 0:
            atr.append(highs[0] - lows[0])
        else:
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            atr.append(tr)
    
    atr_smooth = [sum(atr[:i+1]) / (i+1) if i < 14 else sum(atr[i-13:i+1]) / 14 for i in range(len(atr))]

    # محاسبه RSI (14)
    rsi = [50.0] * len(closes)
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i-1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
        if i >= 14:
            avg_gain = sum(gains[-14:]) / 14
            avg_loss = sum(losses[-14:]) / 14
            rs = avg_gain / avg_loss if avg_loss != 0 else 100
            rsi[i] = 100.0 - (100.0 / (1.0 + rs)) if avg_loss != 0 else 100.0

    # EMA 100
    ema_htf = calculate_ema(closes, 100)

    # بررسی کندل قبلی (آخرین کندلِ تمام‌شده)
    idx = -2
    c_close = closes[idx]
    c_atr = atr_smooth[idx]
    htf_trend = ema_htf[idx]
    c_rsi = rsi[idx]

    upper_break = max(highs[idx-10:idx])
    lower_break = min(lows[idx-10:idx])

    rr_ratio = 1.5
    atr_sl_mult = 1.5

    signal_type = None
    entry_price = c_close

    # شرط ورود به پوزیشن LONG
    if (c_close > htf_trend) and (c_close > upper_break) and (c_rsi > 50):
        signal_type = "LONG 🟩"
        sl = entry_price - (c_atr * atr_sl_mult)
        tp = entry_price + ((entry_price - sl) * rr_ratio)

    # شرط ورود به پوزیشن SHORT
    elif (c_close < htf_trend) and (c_close < lower_break) and (c_rsi < 50):
        signal_type = "SHORT 🟥"
        sl = entry_price + (c_atr * atr_sl_mult)
        tp = entry_price - ((sl - entry_price) * rr_ratio)

    # اگر سیگنال جدید بود و قبلاً فرستاده نشده بود
    if signal_type and last_signals[symbol] != (signal_type, entry_price):
        last_signals[symbol] = (signal_type, entry_price)
        
        msg = (
            f"🚨 **سیگنال جدید Score Hunter** 🚨\n\n"
            f"📌 **ارز:** `{symbol}`\n"
            f"📊 **موقعیت:** {signal_type}\n"
            f"💵 **قیمت ورود:** `{entry_price:.4f}`\n"
            f"🛑 **حد ضرر (SL):** `{sl:.4f}`\n"
            f"🎯 **حد سود (TP):** `{tp:.4f}`\n\n"
            f"⏰ **تایم‌فریم:** 15m\n"
            f"⚖️ **ریسک به ریوارد:** 1:1.5"
        )
        print(f"[{symbol}] سیگنال جدید پیدا شد -> ارسال به تلگرام")
        send_telegram_message(msg)

# =========================================================
# 4. اجرای مداوم ربات
# =========================================================
def main():
    print("=== ربات Score Hunter Pro با موفقیت روشن شد ===")
    print("در حال اسکن بازار...")
    
    while True:
        for symbol in SYMBOLS:
            analyze_and_signal(symbol)
            time.sleep(1) # وقفه ۱ ثانیه‌ای بین ارزها
        
        # ۶۰ ثانیه صبر می‌کنه و دوباره ۱۰ ارز رو بررسی می‌کنه
        time.sleep(60)

if __name__ == "__main__":
    main()
