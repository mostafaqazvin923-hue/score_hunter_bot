import json
import urllib.request

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
TIMEFRAME = "1hour"
TARGET_RR = 2.0

def fetch_all_klines(symbol):
    """دریافت دیتای تاریخی کامل از کوینکس برای بک‌تست"""
    all_candles = []
    # دریافت کندل‌ها با لیمیت بالا برای بک‌تست
    url = f"https://api.coinex.com/v2/spot/kline?market={symbol}&period={TIMEFRAME}&limit=1000"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 SCORE-HUNTER-BACKTEST"})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if payload.get("code") == 0:
                rows = payload.get("data", [])
                for row in rows:
                    if isinstance(row, dict):
                        ts = int(float(row.get("created_at", row.get("time", 0))))
                        op = float(row.get("open", 0))
                        hi = float(row.get("high", 0))
                        lo = float(row.get("low", 0))
                        cl = float(row.get("close", 0))
                        all_candles.append({"timestamp": ts, "open": op, "high": hi, "low": lo, "close": cl})
                all_candles.sort(key=lambda x: x["timestamp"])
    except Exception as e:
        print(f"Error fetching backtest data for {symbol}: {e}")
    return all_candles

def calculate_rsi(candles, index, period=14):
    if index < period:
        return 50.0
    gains, losses = 0.0, 0.0
    for i in range(index - period + 1, index + 1):
        diff = candles[i]["close"] - candles[i-1]["close"]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - (100.0 / (1.0 + rs))

def run_backtest():
    total_trades = 0
    total_wins = 0
    long_count = 0
    short_count = 0
    net_profit_pct = 0.0

    print("=== در حال اجرای بک‌تست دوطرفه (Long & Short) با آمار تفکیکی ===")
    
    for symbol in SYMBOLS:
        candles = fetch_all_klines(symbol)
        if len(candles) < 50:
            continue
            
        for i in range(20, len(candles) - 1):
            c = candles[i]
            prev_c = candles[i-1]
            prev2_c = candles[i-2]
            
            # --- بررسی لانگ ---
            recent_highs = max(x["high"] for x in candles[i-15:i])
            is_bullish_bos = c["close"] > recent_highs and (c["close"] - c["open"]) > (c["high"] - c["low"]) * 0.4
            has_bullish_fvg = prev2_c["high"] < c["low"]
            current_rsi = calculate_rsi(candles, i)
            
            if is_bullish_bos and has_bullish_fvg and (40 < current_rsi < 70):
                entry = c["close"]
                sl = min(prev_c["low"], prev2_c["low"]) - (entry * 0.002)
                risk_dist = entry - sl
                if risk_dist <= 0 or (risk_dist / entry) > 0.03:
                    continue
                tp = entry + (risk_dist * TARGET_RR)
                
                # شبیه‌سازی نتیجه در آینده کندل‌ها
                trade_result = simulate_trade(candles, i + 1, entry, tp, sl, is_long=True)
                if trade_result is not None:
                    total_trades += 1
                    long_count += 1
                    if trade_result == "WIN":
                        total_wins += 1
                        net_profit_pct += ((tp - entry) / entry) * 100
                    else:
                        net_profit_pct -= ((entry - sl) / entry) * 100
                continue

            # --- بررسی شورت ---
            recent_lows = min(x["low"] for x in candles[i-15:i])
            is_bearish_bos = c["close"] < recent_lows and (c["open"] - c["close"]) > (c["high"] - c["low"]) * 0.4
            has_bearish_fvg = prev2_c["low"] > c["high"]
            
            if is_bearish_bos and has_bearish_fvg and (30 < current_rsi < 60):
                entry = c["close"]
                sl = max(prev_c["high"], prev2_c["high"]) + (entry * 0.002)
                risk_dist = sl - entry
                if risk_dist <= 0 or (risk_dist / entry) > 0.03:
                    continue
                tp = entry - (risk_dist * TARGET_RR)
                
                trade_result = simulate_trade(candles, i + 1, entry, tp, sl, is_long=False)
                if trade_result is not None:
                    total_trades += 1
                    short_count += 1
                    if trade_result == "WIN":
                        total_wins += 1
                        net_profit_pct += ((entry - tp) / entry) * 100
                    else:
                        net_profit_pct -= ((sl - entry) / entry) * 100

    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    print("\n========================================")
    print("           گزارش نهایی بک‌تست           ")
    print("========================================")
    print(f"📊 کل معاملات انجام شده: {total_trades}")
    print(f"🟢 تعداد معاملات لانگ (LONG): {long_count}")
    print(f"🔴 تعداد معاملات شورت (SHORT): {short_count}")
    print(f"🏆 وین‌ریت کل (Win Rate): {win_rate:.2f}%")
    print(f"💰 سود خالص بدون لورج: {net_profit_pct:.2f}%")
    print("========================================")

def simulate_trade(candles, start_idx, entry, tp, sl, is_long):
    """بررسی برخورد قیمت به TP یا SL در کندل‌های بعدی"""
    for idx in range(start_idx, len(candles)):
        candle = candles[idx]
        if is_long:
            # ابتدا بررسی حد ضرر یا حد سود در کندل
            hit_sl = candle["low"] <= sl
            hit_tp = candle["high"] >= tp
            if hit_sl and hit_tp:
                return "LOSS"  # محافظه‌کارانه در صورت برخورد همزمان
            elif hit_sl:
                return "LOSS"
            elif hit_tp:
                return "WIN"
        else:
            hit_sl = candle["high"] >= sl
            hit_tp = candle["low"] <= tp
            if hit_sl and hit_tp:
                return "LOSS"
            elif hit_sl:
                return "LOSS"
            elif hit_tp:
                return "WIN"
    return None

if __name__ == "__main__":
    run_backtest()
