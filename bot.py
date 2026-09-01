import json
import urllib.request
import urllib.parse
import os
import math

# تنظیمات کاملاً منطبق با کد بک‌تست
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8937303392:AAGXDckoHV61vY6G0B4VFcHMi90YbhY-jiY")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "2090120004")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
TIMEFRAME = "1hour"
TARGET_RR = 2.0
STATE_FILE = "active_trades_state.json"

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

def fetch_klines(symbol, limit=100):
    url = f"https://api.coinex.com/v2/spot/kline?market={symbol}&period={TIMEFRAME}&limit={limit}"
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
                if len(candles) > 0:
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

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {"active_trades": {}, "last_timestamps": {}}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def main():
    state = load_state()
    active_trades = state.get("active_trades", {})
    last_timestamps = state.get("last_timestamps", {})

    print("[*] Running Score Hunter Pro live check...")

    for symbol in SYMBOLS:
        candles = fetch_klines(symbol, limit=100)
        if len(candles) < 25:
            continue

        current_candle = candles[-1]
        sub_candles = candles
        c = sub_candles[-2]
        prev_c = sub_candles[-3]
        prev2_c = sub_candles[-4]

        # ۱. بررسی پوزیشن‌های فعال برای اعلام نتیجه (تاچ شدن TP یا SL)
        if symbol in active_trades:
            trade = active_trades[symbol]
            if current_candle["low"] <= trade["stop_loss"]:
                msg = (
                    f"❌ **نتیجه معامله: حد ضرر لمس شد (SL Hit)**\n\n"
                    f"🪙 ارز: `{symbol}`\n"
                    f"📍 نقطه ورود: `{trade['entry_price']:.4f}`\n"
                    f"🛑 قیمت حد ضرر: `{trade['stop_loss']:.4f}`\n"
                    f"📉 درصد ضرر: `-{trade['sl_pct']:.2f}%`"
                )
                send_telegram(msg)
                del active_trades[symbol]
            elif current_candle["high"] >= trade["take_profit"]:
                msg = (
                    f"✅ **نتیجه معامله: حد سود لمس شد (TP Hit - WON!)**\n\n"
                    f"🪙 ارز: `{symbol}`\n"
                    f"📍 نقطه ورود: `{trade['entry_price']:.4f}`\n"
                    f"🎯 قیمت حد سود: `{trade['take_profit']:.4f}`\n"
                    f"📈 درصد سود: `+{trade['tp_pct']:.2f}%`"
                )
                send_telegram(msg)
                del active_trades[symbol]

        # ۲. جستجوی سیگنال جدید و درج درصد سود و ضرر در پیام
        if symbol not in active_trades and c["timestamp"] != last_timestamps.get(symbol, 0):
            recent_highs = max(x["high"] for x in sub_candles[-15:-2])
            is_bos = c["close"] > recent_highs and (c["close"] - c["open"]) > (c["high"] - c["low"]) * 0.4
            has_bullish_fvg = prev2_c["high"] < c["low"]
            
            current_rsi = calculate_rsi(sub_candles)
            rsi_filter = 40 < current_rsi < 70

            if is_bos and has_bullish_fvg and rsi_filter:
                entry_price = c["close"]
                stop_loss = min(prev_c["low"], prev2_c["low"]) - (entry_price * 0.002)
                risk_dist = entry_price - stop_loss

                if risk_dist <= 0 or (risk_dist / entry_price) > 0.03:
                    continue

                take_profit = entry_price + (risk_dist * TARGET_RR)
                
                # محاسبه دقیق درصدها برای نمایش در سیگنال
                tp_percentage = ((take_profit - entry_price) / entry_price) * 100
                sl_percentage = ((entry_price - stop_loss) / entry_price) * 100

                active_trades[symbol] = {
                    "entry_price": entry_price,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "tp_pct": tp_percentage,
                    "sl_pct": sl_percentage
                }

                msg = (
                    f"🟢 **سیگنال جدید لانگ (Score Hunter Pro)**\n\n"
                    f"🪙 ارز: `{symbol}`\n"
                    f"📍 نقطه ورود: `{entry_price:.4f}`\n\n"
                    f"🎯 **TP:** `{take_profit:.4f}` (سود: `+{tp_percentage:.2f}%`)\n"
                    f"🛑 **SL:** `{stop_loss:.4f}` (ضرر: `-{sl_percentage:.2f}%`)\n"
                    f"⚖️ ریسک به ریوارد: `1:2`\n"
                    f"📊 RSI: `{current_rsi:.1f}`"
                )
                send_telegram(msg)
                last_timestamps[symbol] = c["timestamp"]

    state["active_trades"] = active_trades
    state["last_timestamps"] = last_timestamps
    save_state(state)
    print("[*] State successfully saved.")

if __name__ == "__main__":
    main()
