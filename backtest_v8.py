import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone

# ============================================================
# SETTINGS (CoinEx API v2)
# ============================================================

BASE_URL = "https://api.coinex.com/v2"

SYMBOL = "SOLUSDT"
TIMEFRAME = "15min"
TARGET_CANDLES = 1000

INITIAL_CAPITAL = 1000.0  # سرمایه اولیه (دلار)
RISK_PERCENT = 0.05       # ریسک ۵ درصد از کل سرمایه در هر معامله

# ============================================================
# HTTP & DATA DOWNLOADER (CoinEx)
# ============================================================

def download_klines(symbol, timeframe, limit=1000):
    url = f"{BASE_URL}/spot/kline?market={symbol}&period={timeframe}&limit={limit}"
    
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

    if not isinstance(payload, dict) or payload.get("code") != 0:
        print("پاسخ نامعتبر از صرافی کوینکس:", payload.get("message"))
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
                vol = float(row.get("volume", 0))
            elif isinstance(row, list) and len(row) >= 6:
                ts = int(float(row[0]))
                op = float(row[1])
                cl = float(row[2])
                hi = float(row.get("high", 3))
                lo = float(row.get("low", 4))
                vol = float(row.get("volume", 5))
            else:
                ts = int(float(row[0]))
                op = float(row[1])
                cl = float(row[2])
                hi = float(row[3])
                lo = float(row.get("low", 4))
                vol = float(row.get("volume", 5))

            if ts > 100000000000:
                ts = ts // 1000

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

    candles.sort(key=lambda x: x["timestamp"])
    return candles

# ============================================================
# BACKTEST ENGINE (High Win-Rate Strategy with Trend Filter)
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

    capital = INITIAL_CAPITAL
    total_trades = 0
    wins = 0
    losses = 0
    
    print("-" * 50)
    print(f"شروع بک‌تست روی {len(candles)} کندلِ {SYMBOL.upper()} با سرمایه اولیه {INITIAL_CAPITAL}$...")
    print("-" * 50)

    for i in range(25, len(candles) - 1):
        current_candle = candles[i]
        prev_candle = candles[i - 1]
        
        close_price = current_candle["close"]
        open_price = current_candle["open"]
        volume = current_candle["volume"]
        
        # محاسبه میانگین متحرک ساده (SMA 20) برای تشخیص روند صعودی
        sma_20 = sum(c["close"] for c in candles[i-20:i]) / 20
        avg_volume = sum(c["volume"] for c in candles[i-20:i]) / 20

        # فیلترهای سخت‌گیرانه برای بالا بردن وین‌ریت:
        # ۱. قیمت بالای میانگین متحرک ۲۰ دوره‌ای باشد (روند صعودی تایید شود)
        is_uptrend = close_price > sma_20
        # ۲. کندل صعودی قوی
        is_green = close_price > open_price
        # ۳. شکستن سقف کندل قبل با حجم بالاتر از میانگین
        is_breakout = close_price > prev_candle["high"]
        is_volume_ok = volume > (avg_volume * 1.1)

        if is_uptrend and is_green and is_breakout and is_volume_ok:
            entry_price = close_price
            stop_loss = entry_price * 0.99      # حد ضرر ۱ درصد (تطبیق‌پذیرتر)
            take_profit = entry_price * 1.02    # حد سود ۲ درصد (وین‌ریت بالاتر)
            
            trade_result = None
            for future_candle in candles[i+1:]:
                if future_candle["low"] <= stop_loss:
                    trade_result = "LOSS"
                    break
                elif future_candle["high"] >= take_profit:
                    trade_result = "WIN"
                    break
            
            total_trades += 1
            risk_amount = capital * RISK_PERCENT

            if trade_result == "WIN":
                wins += 1
                profit_amount = risk_amount * 2.0
                capital += profit_amount
            elif trade_result == "LOSS":
                losses += 1
                capital -= risk_amount
            else:
                total_trades -= 1

    print("-" * 50)
    print("📊 نتایج نهایی بک‌تست:")
    print(f"🔹 کل معاملات انجام شده: {total_trades}")
    print(f"✅ معاملات موفق (Win): {wins}")
    print(f"❌ معاملات ناموفق (Loss): {losses}")
    
    if total_trades > 0:
        win_rate = (wins / total_trades) * 100
        net_profit = capital - INITIAL_CAPITAL
        profit_percentage = (net_profit / INITIAL_CAPITAL) * 100
        
        print(f"🎯 وین‌ریت استراتژی: {win_rate:.2f}%")
        print(f"💰 سرمایه نهایی: {capital:.2f}$")
        print(f"📈 سود/زیان خالص: {net_profit:+.2f}$ ({profit_percentage:+.2f}%)")
        print(f"📈 میانگین تعداد معامله در روز: {total_trades / max(total_days, 0.1):.1f}")
    else:
        print("هیچ معامله‌ای با این شرایط در این بازه فعال نشد.")

if __name__ == "__main__":
    run_backtest()
