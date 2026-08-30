import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

# ============================================================
# TELEGRAM & STRATEGY SETTINGS (BTC & SOL - SL: 1.2% / TP: 1.5%)
# ============================================================

TELEGRAM_TOKEN = "8937303392:AAGXDckoHV61vY6G0B4VFcHMi90YbhY-jiY"
CHAT_ID = "2090120004"

BASE_URL = "https://api.coinex.com/v2"
SYMBOLS = ["BTCUSDT", "SOLUSDT"]
TIMEFRAME = "15min"

last_processed_timestamps = {
    "BTCUSDT": 0,
    "SOLUSDT": 0
}

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    data = urllib.parse.urlencode(payload).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status == 200
    except Exception as e:
        print(f"خطا در ارسال پیام به تلگرام: {e}")
        return False

def download_latest_klines(symbol, limit=100):
    url = f"{BASE_URL}/spot/kline?market={symbol}&period={TIMEFRAME}&limit={limit}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 SCORE-HUNTER-LIVE"}
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8")
            payload = json.loads(raw)
            
        if not isinstance(payload, dict) or payload.get("code") != 0:
            return []

        rows = payload.get("data", [])
        candles = []
        for row in rows:
            try:
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

                if ts > 100000000000:
                    ts = ts // 1000

                candles.append({
                    "timestamp": ts,
                    "open": op,
                    "high": hi,
                    "low": lo,
                    "close": cl
                })
            except Exception:
                continue

        candles.sort(key=lambda x: x["timestamp"])
        return candles
    except Exception:
        return []

def calculate_ema(closes, period):
    if len(closes) < period:
        return closes
    ema = [closes[0]]
    multiplier = 2 / (period + 1)
    for price in closes[1:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])
    return ema

def calculate_rsi(candles, period=14):
    if len(candles) < period + 1:
        return [50.0] * len(candles)
    
    rsi_values = [50.0] * len(candles)
    gains = 0.0
    losses = 0.0
    
    for i in range(1, period + 1):
        change = candles[i]["close"] - candles[i - 1]["close"]
        if change > 0:
            gains += change
        else:
            losses -= change
            
    avg_gain = gains / period
    avg_loss = losses / period
    
    if avg_loss == 0:
        rsi_values[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_values[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, len(candles)):
        change = candles[i]["close"] - candles[i - 1]["close"]
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
        if avg_loss == 0:
            rsi_values[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))
            
    return rsi_values

def check_market_signals():
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] در حال بررسی بازار برای BTC و SOL...")
    
    for symbol in SYMBOLS:
        candles = download_latest_klines(symbol, limit=100)
        if len(candles) < 60:
            continue
            
        i = len(candles) - 2
        c = candles[i]
        prev_c = candles[i-1]
        
        if c["timestamp"] <= last_processed_timestamps[symbol]:
            continue

        closes = [item["close"] for item in candles]
        ema_20 = calculate_ema(closes, 20)
        ema_50 = calculate_ema(closes, 50)
        rsi_list = calculate_rsi(candles, 14)

        close_p = c["close"]
        open_p = c["open"]
        
        is_uptrend = ema_20[i] > ema_50[i]
        is_pullback_recovery = (prev_c["low"] <= ema_20[i-1]) and (close_p > ema_20[i])
        rsi = rsi_list[i]
        is_rsi_good = 42 <= rsi <= 62
        is_green = close_p > open_p

        if is_uptrend and is_pullback_recovery and is_rsi_good and is_green:
            has_obstacle = False
            for j in range(max(0, i-3), i):
                upper_wick = candles[j]["high"] - max(candles[j]["open"], candles[j]["close"])
                body = abs(candles[j]["close"] - candles[j]["open"])
                if body > 0 and upper_wick > (body * 2.0):
                    has_obstacle = True
                    break

            if not has_obstacle:
                last_processed_timestamps[symbol] = c["timestamp"]
                
                entry_price = close_p
                stop_loss = entry_price * 0.988     # ۱.۲٪ ضرر
                take_profit = entry_price * 1.015   # ۱.۵٪ سود
                
                msg = (
                    f"🚨 **سیگنال خرید جدید (Score Hunter Pro)** 🚨\n\n"
                    f"🔹 **ارز:** `{symbol}`\n"
                    f"⏱ **تایم‌فریم:** `15m`\n\n"
                    f"📥 **قیمت ورود:** `{entry_price}`\n"
                    f"🛑 **حد ضرر (SL):** `{stop_loss:.4f}` (-1.2%)\n"
                    f"🎯 **حد سود (TP):** `{take_profit:.4f}` (+1.5%)\n\n"
                    f"📊 **شاخص RSI:** `{rsi:.1f}`\n"
                    f"⏰ **زمان:** `{datetime.fromtimestamp(c['timestamp'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC`"
                )
                
                send_telegram_message(msg)
                print(f"سیگنال جدید برای {symbol} ارسال شد!")

if __name__ == "__main__":
    check_market_signals()
