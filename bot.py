import json
import os
import urllib.request
import requests

# ================= تنظیمات اختصاصی شما =================
TELEGRAM_BOT_TOKEN = "8937303392:AAGXDckoHV61vY6G0B4VFcHMi90YbhY-jiY"
TELEGRAM_CHAT_ID = "2090120004"

SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "DOGE-USDT", "DYDX-USDT",
    "LINK-USDT", "ADA-USDT", "XRP-USDT", "NEAR-USDT", "AVAX-USDT"
]

DATA_FILE = "active_signals.json"

def load_active_signals():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_active_signals(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

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
        print(f"خطا در ارسال پیام: {e}")

def get_crypto_klines_okx(symbol, bar="15m", limit=150):
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
        print(f"خطا در دریافت دیتا {symbol}: {e}")
        return None

def calculate_ema(prices, span):
    alpha = 2 / (span + 1)
    ema = [prices[0]]
    for p in prices[1:]:
        ema.append(p * alpha + ema[-1] * (1 - alpha))
    return ema

def check_active_positions(symbol, klines, active_signals):
    """بررسی رسیدن پوزیشن‌های فعال به حد سود (TP) یا حد ضرر (SL)"""
    if symbol not in active_signals:
        return

    sig = active_signals[symbol]
    last_candle = klines[-1]
    high = last_candle['high']
    low = last_candle['low']

    result_type = None

    if sig['type'] == 'LONG':
        if low <= sig['sl']:
            result_type = "❌ **حد ضرر (SL) خورد!**"
        elif high >= sig['tp']:
            result_type = "🎯 **حد سود (TP) لمس شد! (+1.5R)**"
    elif sig['type'] == 'SHORT':
        if high >= sig['sl']:
            result_type = "❌ **حد ضرر (SL) خورد!**"
        elif low <= sig['tp']:
            result_type = "🎯 **حد سود (TP) لمس شد! (+1.5R)**"

    if result_type:
        msg = (
            f"🔔 **بسته‌شدن موقعیت**\n\n"
            f"📌 **ارز:** `{symbol}`\n"
            f"📊 **نتیجه:** {result_type}\n"
            f"💵 **قیمت ورود:** `{sig['entry']:.4f}`\n"
            f"🛑 **SL:** `{sig['sl']:.4f}` | 🎯 **TP:** `{sig['tp']:.4f}`"
        )
        send_telegram_message(msg)
        del active_signals[symbol]

def analyze_and_signal(symbol, active_signals):
    klines_15m = get_crypto_klines_okx(symbol=symbol, bar="15m", limit=150)
    if not klines_15m or len(klines_15m) < 100:
        return

    # ابتدا بررسی وضعیت پوزیشن‌های فعال قبلی
    check_active_positions(symbol, klines_15m, active_signals)

    # اگر از قبل پوزیشن فعال داشته باشد، سیگنال جدید صادر نمی‌کند
    if symbol in active_signals:
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
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
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

    # محاسبه EMA 100
    ema_htf = calculate_ema(closes, 100)

    # کندل قبلی (آخرین کندل تمام‌شده)
    idx = -2
    c_close = closes[idx]
    c_atr = atr_smooth[idx]
    htf_trend = ema_htf[idx]
    c_rsi = rsi[idx]

    upper_break = max(highs[idx-10:idx])
    lower_break = min(lows[idx-10:idx])

    rr_ratio = 1.5
    atr_sl_mult = 1.5

    pos_type = None
    entry_price = c_close

    if (c_close > htf_trend) and (c_close > upper_break) and (c_rsi > 50):
        pos_type = "LONG"
        sl = entry_price - (c_atr * atr_sl_mult)
        tp = entry_price + ((entry_price - sl) * rr_ratio)
    elif (c_close < htf_trend) and (c_close < lower_break) and (c_rsi < 50):
        pos_type = "SHORT"
        sl = entry_price + (c_atr * atr_sl_mult)
        tp = entry_price - ((sl - entry_price) * rr_ratio)

    if pos_type:
        active_signals[symbol] = {
            'type': pos_type,
            'entry': entry_price,
            'sl': sl,
            'tp': tp
        }
        symbol_icon = "LONG 🟩" if pos_type == "LONG" else "SHORT 🟥"
        msg = (
            f"🚨 **سیگنال جدید Score Hunter** 🚨\n\n"
            f"📌 **ارز:** `{symbol}`\n"
            f"📊 **موقعیت:** {symbol_icon}\n"
            f"💵 **قیمت ورود:** `{entry_price:.4f}`\n"
            f"🛑 **حد ضرر (SL):** `{sl:.4f}`\n"
            f"🎯 **حد سود (TP):** `{tp:.4f}`\n\n"
            f"⏰ **تایم‌فریم:** 15m\n"
            f"⚖️ **ریسک به ریوارد:** 1:1.5"
        )
        send_telegram_message(msg)

def main():
    active_signals = load_active_signals()
    for symbol in SYMBOLS:
        analyze_and_signal(symbol, active_signals)
    save_active_signals(active_signals)

if __name__ == "__main__":
    main()
