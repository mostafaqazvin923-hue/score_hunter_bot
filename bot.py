import json
import time
import urllib.request
import urllib.parse
import math

# ============================================================
# SCORE HUNTER PRO - LIVE BOT WITH TRADE RESULT TRACKER
# ============================================================

TOKEN = "8937303392:AAGXDckoHV61vY6G0B4VFcHMi90YbhY-jiY"
CHAT_ID = "2090120004"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
TIMEFRAME = "1hour"
TARGET_RR = 2.0

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    data = urllib.parse.urlencode(payload).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read()
    except Exception as e:
        print(f"Telegram Error: {e}")

def fetch_klines(symbol):
    url = f"https://api.coinex.com/v2/spot/kline?market={symbol}&period={TIMEFRAME}&limit=50"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 SCORE-HUNTER-BOT"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            raw = response.read().decode("utf-8")
            payload = json.loads(raw)
            if isinstance(payload, dict) and payload.get("code") == 0:
                rows = payload.get("data", [])
                candles = []
                for row in rows:
                    if isinstance(row, dict):
                        ts = int(float(row.get("created_at", row.get("time", 0))))
                        op = float(row.get("open", 0))
                        hi = float(row.get("high", 0))
                        lo = float(row.get("low", 0))
                        cl = float(row.get("close", 0))
                    elif isinstance(row, list) and len(row) >= 6:
                        ts = int(float(row[0]))
                        op = float(row[1])
                        cl = float(row[2])
                        hi = float(row[3])
                        lo = float(row[4])
                    else:
                        continue
                    candles.append({"timestamp": ts, "open": op, "high": hi, "low": lo, "close": cl})
                candles.sort(key=lambda x: x["timestamp"])
                return candles
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
    return []

def calculate_rsi(candles, period=14):
    if len(candles) < period + 1:
        return 50.0
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        diff = candles[-i]["close"] - candles[-i-1]["close"]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - (100.0 / (1.0 + rs))

def run_bot():
    print("==================================================")
    print("   SCORE HUNTER PRO - LIVE BOT WITH TRACKER      ")
    print("==================================================")
    send_telegram("🚀 **Score Hunter Pro** ربات روشن شد و سیستم پیگیری نتایج پوزیشن‌ها فعال است!")

    last_timestamps = {s: 0 for s in SYMBOLS}
    active_trades = {}  # ذخیره پوزیشن‌های فعال هر ارز برای بررسی نتیجه

    while True:
        for symbol in SYMBOLS:
            candles = fetch_klines(symbol)
            if len(candles) < 20:
                continue

            current_candle = candles[-1]
            c = candles[-2]       # کندل بسته شده‌ی قبلی
            prev_c = candles[-3]
            prev2_c = candles[-4]

            # ۱. بررسی پوزیشن فعال روی این ارز جهت اعلام نتیجه (TP یا SL)
            if symbol in active_trades:
                trade = active_trades[symbol]
                if current_candle["low"] <= trade["stop_loss"]:
                    msg = (
                        f"❌ **حد ضرر معامله لمس شد (SL Hit)**\n\n"
                        f"🪙 ارز: `{symbol}`\n"
                        f"📍 نقطه ورود: `{trade['entry_price']:.4f}`\n"
                        f"🛑 قیمت حد ضرر: `{trade['stop_loss']:.4f}`\n"
                        f"📉 زیان: `-{trade['sl_pct']:.2f}%`"
                    )
                    send_telegram(msg)
                    del active_trades[symbol]
                elif current_candle["high"] >= trade["take_profit"]:
                    msg = (
                        f"✅ **حد سود معامله لمس شد (TP Hit - WON!)**\n\n"
                        f"🪙 ارز: `{symbol}`\n"
                        f"📍 نقطه ورود: `{trade['entry_price']:.4f}`\n"
                        f"🎯 قیمت حد سود: `{trade['take_profit']:.4f}`\n"
                        f"📈 سود خالص: `+{trade['tp_pct']:.2f}%`"
                    )
                    send_telegram(msg)
                    del active_trades[symbol]

            # ۲. جستجوی سیگنال جدید (فقط اگر پوزیشن فعالی روی این ارز باز نباشد)
            if symbol not in active_trades and c["timestamp"] != last_timestamps[symbol]:
                recent_highs = max(x["high"] for x in candles[-15:-2])
                is_bos = c["close"] > recent_highs and (c["close"] - c["open"]) > (c["high"] - c["low"]) * 0.4
                has_bullish_fvg = prev2_c["high"] < c["low"]
                
                current_rsi = calculate_rsi(candles)
                rsi_filter = 40 < current_rsi < 70

                if is_bos and has_bullish_fvg and rsi_filter:
                    entry_price = c["close"]
                    stop_loss = min(prev_c["low"], prev2_c["low"]) - (entry_price * 0.002)
                    risk_dist = entry_price - stop_loss

                    if risk_dist <= 0 or (risk_dist / entry_price) > 0.03:
                        continue

                    take_profit = entry_price + (risk_dist * TARGET_RR)

                    tp_percentage = ((take_profit - entry_price) / entry_price) * 100
                    sl_percentage = ((entry_price - stop_loss) / entry_price) * 100

                    # ثبت پوزیشن فعال
                    active_trades[symbol] = {
                        "entry_price": entry_price,
                        "take_profit": take_profit,
                        "stop_loss": stop_loss,
                        "tp_pct": tp_percentage,
                        "sl_pct": sl_percentage
                    }

                    msg = (
                        f"🟢 **سیگنال جدید لانگ (Long)**\n\n"
                        f"🪙 ارز: `{symbol}`\n"
                        f"📍 نقطه ورود: `{entry_price:.4f}`\n\n"
                        f"🎯 **TP:** `{take_profit:.4f}` (سود: `+`{tp_percentage:.2f}%)\n"
                        f"🛑 **SL:** `{stop_loss:.4f}` (ضرر: `-`{sl_percentage:.2f}%)\n"
                        f"⚖️ ریسک به ریوارد: `1:2`\n"
                        f"📊 RSI: `{current_rsi:.1f}`"
                    )
                    send_telegram(msg)
                    last_timestamps[symbol] = c["timestamp"]

        time.sleep(60)

if __name__ == "__main__":
    run_bot()
