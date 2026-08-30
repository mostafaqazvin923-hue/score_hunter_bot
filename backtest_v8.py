import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone

# ============================================================
# SETTINGS (CoinEx API)
# ============================================================

BASE_URL = "https://api.coinex.com/v2"

SYMBOL = "SOLUSDT"
TIMEFRAME = "15m"   # تایم‌فریم ۱۵ دقیقه در کوینکس
TARGET_CANDLES = 1000

# ============================================================
# HTTP & DATA DOWNLOADER (CoinEx)
# ============================================================

def download_klines(symbol, timeframe, limit=1000):
    # کوینکس به راحتی در هر درخواست تا ۱۰۰۰ کندل تاریخچه میده
    url = f"{BASE_URL}/spot/market-kline?market={symbol}&type={timeframe}&limit={limit}"
    
    print(f"در حال دریافت تاریخچه کندل‌ها از CoinEx برای {symbol}...")

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 SCORE-HUNTER-BACKTEST"
        }
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8")
            payload = json.loads(raw)
    except Exception as exc:
        print(f"خطا در ارتباط با سرور کوینکس: {exc}")
        return []

    # ساختار پاسخ کوینکس: {"code": 0, "data": [...]}
    if not isinstance(payload, dict) or payload.get("code") != 0:
        print("پاسخ نامعتبر از صرافی کوینکس")
        return []

    rows = payload.get("data", [])
    candles = []

    # ساختار هر کندل در کوینکس: [time, open, close, high, low, volume, amount, ...]
    for row in rows:
        try:
            if not isinstance(row, list) or len(row) < 6:
                continue

            ts = int(float(row[0]))
            op = float(row[1])
            cl = float(row[2])
            hi = float(row[3])
            lo = float(row[4])
            vol = float(row[5])

            if hi <= 0 or lo <= 0 or op <= 0 or cl <= 0:
                continue

            candles.append({
                "timestamp": ts,
                "open": op,
                "high": hi,
                "low": lo,
                "close": cl,
                "volume": vol,
            })
        except Exception:
            continue

    # مرتب‌سازی بر اساس زمان (قدیمی به جدید)
    candles.sort(key=lambda x: x["timestamp"])
    return candles

# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest():
    candles = download_klines(SYMBOL, TIMEFRAME, TARGET_CANDLES)
    
    print(f"تعداد کل کندل‌های دریافت شده: {len(candles)}")
    
    if len(candles) < 50:
        print("تعداد کندل‌های دریافتی برای بک‌تست کافی نیست.")
        return

    first_time = datetime.fromtimestamp(candles[0]["timestamp"], tz=timezone.utc)
    last_time = datetime.fromtimestamp(candles[-1]["timestamp"], tz=timezone.utc)
    total_days = (candles[-1]["timestamp"] - candles[0]["timestamp"]) / 86400
    
    print(f"📅 بازه زمانی داده‌ها: از {first_time.strftime('%Y-%m-%d %H:%M')} تا {last_time.strftime('%Y-%m-%d %H:%M')} (حدود {total_days:.1f} روز)")

    total_trades = 0
    wins = 0
    losses = 0
    
    print("-" * 50)
    print(f"شروع بک‌تست روی {len(candles)} کندلِ {SYMBOL.upper()}...")
    print("-" * 50)

    for i in range(10, len(candles) - 1):
        candle = candles[i]
        
        close_price = candle["close"]
        open_price = candle["open"]
        
        is_green = close_price > open_price

        if is_green:
            entry_price = close_price
            stop_loss = entry_price * 0.99
            take_profit = entry_price * 1.02
            
            total_trades += 1
            
            trade_result = None
            for future_candle in candles[i+1:]:
                if future_candle["low"] <= stop_loss:
                    trade_result = "LOSS"
                    break
                elif future_candle["high"] >= take_profit:
                    trade_result = "WIN"
                    break
            
            if trade_result == "WIN":
                wins += 1
            elif trade_result == "LOSS":
                losses += 1
            else:
                total_trades -= 1

    print("-" * 50)
    print("📊 نتایج نهایی بک‌تست:")
    print(f"🔹 کل معاملات انجام شده: {total_trades}")
    print(f"✅ معاملات موفق (Win): {wins}")
    print(f"❌ معاملات ناموفق (Loss): {losses}")
    
    if total_trades > 0:
        win_rate = (wins / total_trades) * 100
        print(f"🎯 وین‌ریت استراتژی: {win_rate:.2f}%")
        print(f"📈 میانگین تعداد معامله در روز: {total_trades / max(total_days, 0.1):.1f}")
    else:
        print("هیچ معامله‌ای با این شرایط در این بازه فعال نشد.")

if __name__ == "__main__":
    run_backtest()
