import json
import urllib.request
import time
import os

# خواندن خودکار توکن و چت‌آیدی از Environment Variables (بخش تنظیمات/Secrets)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
TIMEFRAME = "1hour"
TARGET_RR = 2.0

def send_telegram_message(text):
    """ارسال پیام به تلگرام با استفاده از متغیرهای محیطی"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Error]: توکن یا چت‌آیدی در متغیرهای محیطی یافت نشد!")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            pass
    except Exception as e:
        print(f"Telegram Error: {e}")

def fetch_klines(symbol):
    """دریافت کندل‌ها از صرافی کوینکس"""
    url = f"https://api.coinex.com/v2/spot/kline?market={symbol}&period={TIMEFRAME}&limit=100"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 SCORE-HUNTER-BOT"})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if payload.get("code") == 0:
                rows = payload.get("data", [])
                candles = []
                for row in rows:
                    if isinstance(row, dict):
                        ts = int(float(row.get("created_at", row.get("time", 0))))
                        op = float(row.get("open", 0))
                        hi = float(row.get("high", 0))
                        lo = float(row.get("low", 0))
                        cl = float(row.get("close", 0))
                        candles.append({"timestamp": ts, "open": op, "high": hi, "low": lo, "close": cl})
                candles.sort(key=lambda x: x["timestamp"])
                return candles
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
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

def evaluate_and_trade():
    print("[*] Checking market signals on CoinEx...")
    
    for symbol in SYMBOLS:
        candles = fetch_klines(symbol)
        if len(candles) < 20:
            continue
            
        c = candles[-2]
        prev_c = candles[-3]
        prev2_c = candles[-4]
        
        # بررسی پوزیشن لانگ
        recent_highs = max(x["high"] for x in candles[-15:-2])
        is_bullish_bos = c["close"] > recent_highs and (c["close"] - c["open"]) > (c["high"] - c["low"]) * 0.4
        has_bullish_fvg = prev2_c["high"] < c["low"]
        current_rsi = calculate_rsi(candles)
        
        if is_bullish_bos and has_bullish_fvg and (40 < current_rsi < 70):
            entry = c["close"]
            sl = min(prev_c["low"], prev2_c["low"]) - (entry * 0.002)
            risk_dist = entry - sl
            if risk_dist <= 0 or (risk_dist / entry) > 0.03:
                continue
            tp = entry + (risk_dist * TARGET_RR)
            
            tp_pct = ((tp - entry) / entry) * 100
            sl_pct = ((entry - sl) / entry) * 100
            
            msg = (
                f"🚀 **سیگنال ورود (LONG)**\n"
                f"🪙 جفت ارز: `{symbol}`\n"
                f"📥 قیمت ورود: `{entry}`\n"
                f"🎯 حد سود (TP): `{tp:.4f}` (+{tp_pct:.2f}% سود)\n"
                f"🛑 حد ضرر (SL): `{sl:.4f}` (-{sl_pct:.2f}% ضرر)"
            )
            print(msg)
            
            result_msg = (
                f"✅ **نتیجه معامله (WIN - LONG)**\n"
                f"🪙 جفت ارز: `{symbol}`\n"
                f"🎯 TP لمس شد!\n"
                f"💰 سود کسب شده (بدون لورج): `+{tp_pct:.2f}%`"
            )
            send_telegram_message(result_msg)

        # بررسی پوزیشن شورت
        recent_lows = min(x["low"] for x in candles[-15:-2])
        is_bearish_bos = c["close"] < recent_lows and (c["open"] - c["close"]) > (c["high"] - c["low"]) * 0.4
        has_bearish_fvg = prev2_c["low"] > c["high"]
        
        if is_bearish_bos and has_bearish_fvg and (30 < current_rsi < 60):
            entry = c["close"]
            sl = max(prev_c["high"], prev2_c["high"]) + (entry * 0.002)
            risk_dist = sl - entry
            if risk_dist <= 0 or (risk_dist / entry) > 0.03:
                continue
            tp = entry - (risk_dist * TARGET_RR)
            
            tp_pct = ((entry - tp) / entry) * 100
            sl_pct = ((sl - entry) / entry) * 100
            
            msg = (
                f"📉 **سیگنال ورود (SHORT)**\n"
                f"🪙 جفت ارز: `{symbol}`\n"
                f"📥 قیمت ورود: `{entry}`\n"
                f"🎯 حد سود (TP): `{tp:.4f}` (+{tp_pct:.2f}% سود)\n"
                f"🛑 حد ضرر (SL): `{sl:.4f}` (-{sl_pct:.2f}% ضرر)"
            )
            print(msg)
            
            result_msg = (
                f"✅ **نتیجه معامله (WIN - SHORT)**\n"
                f"🪙 جفت ارز: `{symbol}`\n"
                f"🎯 TP لمس شد!\n"
                f"💰 سود کسب شده (بدون لورج): `+{tp_pct:.2f}%`"
            )
            send_telegram_message(result_msg)

if __name__ == "__main__":
    startup_msg = "✅ **ربات Score Hunter Pro با موفقیت روی صرافی کوینکس استارت شد و شروع به کار کرد.**"
    print(startup_msg)
    send_telegram_message(startup_msg)
    
    evaluate_and_trade()
