import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

# ============================================================
# SETTINGS
# ============================================================

BASE_URLS = [
    "https://api.lbank.info",
    "https://api.lbkex.com",
    "https://www.lbkex.net",
]

SYMBOL = "sol_usdt"
TIMEFRAME = "minute15"
TARGET_CANDLES = 500  # تعداد کندل برای بک‌تست
PAGE_SIZE = 200        # اندازه هر صفحه درخواست

# ============================================================
# HTTP & DATA DOWNLOADER
# ============================================================

def http_get(url, params, timeout=20):
    query = urllib.parse.urlencode(params)
    full_url = url + "/v2/kline.do?" + query

    request = urllib.request.Request(
        full_url,
        headers={
            "User-Agent": "Mozilla/5.0 SCORE-HUNTER-BACKTEST"
        }
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")

    return json.loads(raw)

def extract_data(payload):
    if isinstance(payload, dict):
        data = payload.get("data")
        if data is None:
            return []
        return data
    if isinstance(payload, list):
        return payload
    return []

def download_klines(symbol, timeframe, target_count):
    all_rows = {}
    end_time = int(time.time())
    pages = 0
    last_oldest = None

    print(f"در حال دریافت تاریخچه کندل ها از LBank برای {symbol}...")

    while len(all_rows) < target_count:
        pages += 1
        if pages > 50:
            break

        params = {
            "symbol": symbol,
            "size": PAGE_SIZE,
            "type": timeframe,
            "time": end_time,
        }

        payload = None
        last_error = None

        for base in BASE_URLS:
            try:
                payload = http_get(base, params)
                break
            except Exception as exc:
                last_error = exc
                continue

        if payload is None:
            print(f"خطا در ارتباط با سرورهای ال بنک: {last_error}")
            break

        rows = extract_data(payload)
        if not rows:
            break

        added = 0
        for row in rows:
            if not isinstance(row, list) or len(row) < 6:
                continue

            try:
                ts = int(float(row[0]))
                op = float(row[1])
                hi = float(row[2])
                lo = float(row[3])
                cl = float(row[4])
                vol = float(row[5])

                if hi <= 0 or lo <= 0 or op <= 0 or cl <= 0:
                    continue

                all_rows[ts] = {
                    "timestamp": ts,
                    "open": op,
                    "high": hi,
                    "low": lo,
                    "close": cl,
                    "volume": vol,
                }
                added += 1
            except Exception:
                continue

        if not all_rows:
            break

        oldest = min(all_rows.keys())
        if last_oldest == oldest:
            break
        last_oldest = oldest
        end_time = oldest - 1

        if added == 0:
            break

        time.sleep(0.05)

    candles = list(all_rows.values())
    candles.sort(key=lambda x: x["timestamp"])

    if len(candles) > target_count:
        candles = candles[-target_count:]

    return candles

# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest():
    candles = download_klines(SYMBOL, TIMEFRAME, TARGET_CANDLES)
    
    if len(candles) < 50:
        print("تعداد کندل های دریافتی برای بک تست کافی نیست.")
        return

    total_trades = 0
    wins = 0
    losses = 0
    
    print("-" * 50)
    print(f"شروع بک تست روی {len(candles)} کندلِ {SYMBOL.upper()}...")
    print("-" * 50)

    for i in range(20, len(candles) - 1):
        prev_candle = candles[i - 1]
        current_candle = candles[i]
        
        close_price = current_candle["close"]
        volume = current_candle["volume"]
        
        avg_volume = sum(c["volume"] for c in candles[i-20:i]) / 20

        is_bullish_momentum = close_price > prev_candle["high"]
        is_volume_confirmed = volume > (avg_volume * 1.5)

        if is_bullish_momentum and is_volume_confirmed:
            entry_price = close_price
            stop_loss = entry_price * 0.985
            take_profit = entry_price * 1.03
            
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
    print(f"نتایج نهایی بک تست:")
    print(f"کل معاملات انجام شده: {total_trades}")
    print(f"معاملات موفق (Win): {wins}")
    print(f"معاملات ناموفق (Loss): {losses}")
    
    if total_trades > 0:
        win_rate = (wins / total_trades) * 100
        print(f"وین ریت استراتژی: {win_rate:.2f}%")
    else:
        print("هیچ معامله ای با این شرایط در این بازه فعال نشد.")

if __name__ == "__main__":
    run_backtest()
