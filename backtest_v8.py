import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

# ============================================================
# SETTINGS (4-Asset Portfolio / Fixed $50 Margin per Trade)
# ============================================================

BASE_URL = "https://api.coinex.com/v2"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
TIMEFRAME = "15min"
TARGET_CANDLES = 17500
PAGE_LIMIT = 1000

INITIAL_CAPITAL = 1000.0
FIXED_RISK_AMOUNT = 50.0  # مارجین و ریسک ثابت ۵۰ دلار برای هر معامله

# ============================================================
# HTTP & PAGINATION DATA DOWNLOADER
# ============================================================

def download_klines(symbol, timeframe, target_count):
    all_rows = {}
    end_time = None
    pages = 0
    max_pages = 25

    while len(all_rows) < target_count and pages < max_pages:
        pages += 1
        url = f"{BASE_URL}/spot/kline?market={symbol}&period={timeframe}&limit={PAGE_LIMIT}"
        if end_time:
            url += f"&end_time={end_time}"

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 SCORE-HUNTER-BACKTEST"}
        )

        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read().decode("utf-8")
                payload = json.loads(raw)
        except Exception:
            break

        if not isinstance(payload, dict) or payload.get("code") != 0:
            break

        rows = payload.get("data", [])
        if not rows:
            break

        added = 0
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
                    cl = float(row[2]) if len(row) > 2 else float(row[2])
                    hi = float(row[3]) if len(row) > 3 else float(row[3])
                    lo = float(row[4]) if len(row) > 4 else float(row[4])
                    vol = float(row.get("5", row[5])) if len(row) > 5 else float(row[5])
                else:
                    continue

                if ts > 100000000000:
                    ts = ts // 1000

                if hi <= 0 or lo <= 0 or op <= 0 or cl <= 0:
                    continue

                if ts not in all_rows:
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

        if added == 0:
            break

        oldest_ts = min(all_rows.keys())
        end_time = oldest_ts * 1000
        time.sleep(0.1)

    candles = list(all_rows.values())
    candles.sort(key=lambda x: x["timestamp"])

    if len(candles) > target_count:
        candles = candles[-target_count:]

    return candles

# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_ema(closes, period):
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

# ============================================================
# PORTFOLIO BACKTEST ENGINE (Fixed $50 Margin)
# ============================================================

def run_portfolio_backtest():
    all_trades = []
    global_min_ts = float('inf')
    global_max_ts = 0

    print("در حال اجرای استراتژی با مارجین ثابت ۵۰ دلار روی ۴ ارز...")

    for symbol in SYMBOLS:
        candles = download_klines(symbol, TIMEFRAME, TARGET_CANDLES)
        if len(candles) < 100:
            continue

        if candles[0]["timestamp"] < global_min_ts:
            global_min_ts = candles[0]["timestamp"]
        if candles[-1]["timestamp"] > global_max_ts:
            global_max_ts = candles[-1]["timestamp"]

        closes = [c["close"] for c in candles]
        ema_20 = calculate_ema(closes, 20)
        ema_50 = calculate_ema(closes, 50)
        rsi_list = calculate_rsi(candles, 14)

        for i in range(50, len(candles) - 1):
            c = candles[i]
            prev_c = candles[i-1]
            
            close_p = c["close"]
            open_p = c["open"]
            
            is_uptrend = ema_20[i] > ema_50[i]
            is_pullback_recovery = (prev_c["low"] <= ema_20[i-1]) and (close_p > ema_20[i])
            rsi = rsi_list[i]
            is_rsi_good = 42 <= rsi <= 62
            is_green = close_p > open_p

            if is_uptrend and is_pullback_recovery and is_rsi_good and is_green:
                entry_price = close_p
                stop_loss = entry_price * 0.988
                take_profit = entry_price * 1.015
                
                trade_result = None
                exit_ts = c["timestamp"]
                for future_c in candles[i+1:]:
                    if future_c["low"] <= stop_loss:
                        trade_result = "LOSS"
                        exit_ts = future_c["timestamp"]
                        break
                    elif future_c["high"] >= take_profit:
                        trade_result = "WIN"
                        exit_ts = future_c["timestamp"]
                        break
                
                if trade_result:
                    all_trades.append({
                        "entry_time": c["timestamp"],
                        "exit_time": exit_ts,
                        "symbol": symbol,
                        "result": trade_result
                    })

    all_trades.sort(key=lambda x: x["entry_time"])
    total_days = (global_max_ts - global_min_ts) / 86400

    capital = INITIAL_CAPITAL
    wins = 0
    losses = 0

    for trade in all_trades:
        if capital < FIXED_RISK_AMOUNT:
            break

        if trade["result"] == "WIN":
            wins += 1
            capital += FIXED_RISK_AMOUNT * 1.25
        elif trade["result"] == "LOSS":
            losses += 1
            capital -= FIXED_RISK_AMOUNT

    print("\n" + "=" * 50)
    print("نتایج نهایی سبد ۴ ارز با مارجین ثابت ۵۰ دلاری:")
    print("=" * 50)
    total_trades = wins + losses
    print("کل معاملات کل سبد: " + str(total_trades))
    print("معاملات موفق (Win): " + str(wins))
    print("معاملات ناموفق (Loss): " + str(losses))
    
    if total_trades > 0:
        win_rate = (wins / total_trades) * 100
        net_profit = capital - INITIAL_CAPITAL
        profit_percentage = (net_profit / INITIAL_CAPITAL) * 100
        
        print(f"وین‌ریت کل سبد: {win_rate:.2f}%")
        print(f"سرمایه نهایی سبد: {capital:.2f}$")
        print(f"سود/زیان خالص کل: {net_profit:+.2f}$ ({profit_percentage:+.2f}%)")
        print(f"میانگین تعداد معامله در روز: {total_trades / max(total_days, 0.1):.2f}")
    else:
        print("هیچ معامله‌ای انجام نشد.")

if __name__ == "__main__":
    run_portfolio_backtest()
