import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ============================================================
# SETTINGS & CONFIGURATION
# ============================================================

BASE_URL = "https://api.coinex.com/v2"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
TIMEFRAME = "1hour"
TARGET_RR = 2.5                     # ریسک به ریوارد ۱ به ۲.۵
FIXED_RISK_AMOUNT = 50.0            # ریسک ثابت معامله (دلار)

# اطلاعات اختصاصی تلگرام شما
TELEGRAM_BOT_TOKEN = "8937303392:AAGXDckoHV61vY6G0B4VFcHMi90YbhY-jiY"
TELEGRAM_CHAT_ID = "2090120004"

STATE_FILE = "active_trades_state.json"

# ============================================================
# TELEGRAM NOTIFIER
# ============================================================

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN:
        print("[Telegram Alert Simulation]:", message)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 SCORE-HUNTER-BOT"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            pass
    except Exception as e:
        print(f"[Telegram Error]: {e}")

# ============================================================
# DATA DOWNLOADER
# ============================================================

def fetch_klines(symbol, limit=200):
    url = f"{BASE_URL}/spot/kline?market={symbol}&period={TIMEFRAME}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 SCORE-HUNTER-LIVE"})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
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
                        vol = float(row.get("volume", 0))
                    elif isinstance(row, list) and len(row) >= 6:
                        ts = int(float(row[0]))
                        op = float(row[1])
                        cl = float(row[2])
                        hi = float(row[3])
                        lo = float(row[4])
                        vol = float(row[5])
                    else:
                        continue
                    
                    if ts > 100000000000:
                        ts = ts // 1000
                    
                    if hi > 0 and lo > 0 and op > 0 and cl > 0:
                        candles.append({
                            "timestamp": ts, "open": op, "high": hi,
                            "low": lo, "close": cl, "volume": vol
                        })
                candles.sort(key=lambda x: x["timestamp"])
                return candles
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
    return []

# ============================================================
# STATE MANAGEMENT (Persistence)
# ============================================================

def load_state():
    if Path(STATE_FILE).exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"Error saving state: {e}")

# ============================================================
# CORE TRADING & MONITORING LOGIC
# ============================================================

def run_bot_cycle():
    active_trades = load_state()
    print(f"\n--- Bot Cycle Started at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} ---")

    for symbol in SYMBOLS:
        candles = fetch_klines(symbol, limit=150)
        if len(candles) < 20:
            continue

        latest_candle = candles[-1]
        current_price = latest_candle["close"]
        current_high = latest_candle["high"]
        current_low = latest_candle["low"]

        # 1. مدیریت و بررسی پوزیشن‌های باز فعال برای این ارز
        if symbol in active_trades:
            trade = active_trades[symbol]
            tp = trade["tp"]
            sl = trade["sl"]
            entry = trade["entry"]
            tp_pct = trade["tp_pct"]
            sl_pct = trade["sl_pct"]

            result = None
            if current_low <= sl:
                result = "LOSS"
            elif current_high >= tp:
                result = "WIN"

            if result:
                profit = (FIXED_RISK_AMOUNT * TARGET_RR) if result == "WIN" else (-FIXED_RISK_AMOUNT)
                emoji = "✅ WIN (موفق)" if result == "WIN" else "❌ LOSS (حد ضرر)"
                
                msg = (
                    f"🎯 **نتیجه معامله ({symbol})**\n\n"
                    f"وضعیت: {emoji}\n"
                    f"نقطه ورود: `{entry:.4f}`\n"
                    f"حد سود (TP): `{tp:.4f}` (`+{tp_pct:.2f}%`)\n"
                    f"حد ضرر (SL): `{sl:.4f}` (`-{sl_pct:.2f}%`)\n"
                    f"سود/زیان خالص: `{profit:+.2f}$`\n"
                    f"زمان بسته شدن: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}`"
                )
                send_telegram_message(msg)
                print(f"Trade closed for {symbol}: {result}")
                del active_trades[symbol]
                save_state(active_trades)
            continue

        # 2. بررسی سیگنال جدید بر اساس استراتژی SMC (BOS + FVG)
        if len(candles) < 15:
            continue

        c = candles[-2]      # آخرین کندل بسته شده
        prev_c = candles[-3]
        prev2_c = candles[-4]

        # شرط BOS (شکست ساختار سقف اخیر)
        recent_highs = max(x["high"] for x in candles[-15:-2])
        is_bos = c["close"] > recent_highs and (c["close"] - c["open"]) > (c["high"] - c["low"]) * 0.5

        # شرط FVG (گپ ارزش منصفانه صعودی)
        has_bullish_fvg = prev2_c["high"] < c["low"]

        if is_bos and has_bullish_fvg:
            entry_price = current_price
            stop_loss = min(prev_c["low"], prev2_c["low"]) - (entry_price * 0.003)
            risk_distance = entry_price - stop_loss

            if risk_distance <= 0 or risk_distance / entry_price > 0.03:
                continue

            take_profit = entry_price + (risk_distance * TARGET_RR)

            # محاسبه درصدهای بدون لورج (اسپات)
            sl_pct = ((entry_price - stop_loss) / entry_price) * 100
            tp_pct = ((take_profit - entry_price) / entry_price) * 100

            # ثبت پوزیشن جدید در فایل استیت
            active_trades[symbol] = {
                "entry": entry_price,
                "sl": stop_loss,
                "tp": take_profit,
                "sl_pct": sl_pct,
                "tp_pct": tp_pct,
                "time": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
            }
            save_state(active_trades)

            # ارسال سیگنال ورود به تلگرام همراه با درصدهای بدون لورج
            signal_msg = (
                f"🚀 **سیگنال ورود جدید (SMC)**\n\n"
                f"ارز: `{symbol}`\n"
                f"تایم‌فریم: `1h`\n"
                f"نقطه ورود (Entry): `{entry_price:.4f}`\n"
                f"حد سود (TP): `{take_profit:.4f}` (`+{tp_pct:.2f}%`)\n"
                f"حد ضرر (SL): `{stop_loss:.4f}` (`-{sl_pct:.2f}%`)\n"
                f"وضعیت: پوزیشن باز شد 🟢"
            )
            send_telegram_message(signal_msg)
            print(f"New signal sent for {symbol}")

# ============================================================
# MAIN ENTRY (GitHub Actions Single Run)
# ============================================================

if __name__ == "__main__":
    try:
        run_bot_cycle()
    except Exception as e:
        print(f"Execution error: {e}")
