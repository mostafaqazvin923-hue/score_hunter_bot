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
    
    print("در حال دریافت تاریخچه کندل‌ها از CoinEx برای " + symbol + "...")

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
        print("خطا در ارتباط با سرور کوینکس: " + str(exc))
        return []

    if not isinstance(payload, dict) or payload.get("code") != 0:
        print("پاسخ نامعتبر از صرافی کوینکس")
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
                cl = float(row.get("2", row[2])) if len(row) > 2 else float(row[2])
                hi = float(row.get("3", row[3])) if len(row) > 3 else float(row[3])
                lo = float(row.get("4", row[4])) if len(row) > 4 else float(row[4])
                vol = float(row.get("5", row[5])) if len(row) > 5 else float(row[5])
            else:
                continue

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
# TECHNICAL INDICATORS (Pure Python RSI)
# ============================================================

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
# BACKTEST ENGINE
# ============================================================

def run_backtest():
    candles = download_klines(SYMBOL, TIMEFRAME, TARGET_CANDLES)
    
    print("تعداد کل کندل‌های دریافت شده: " + str(len(candles)))
    
    if len(candles) < 50:
        print("تعداد کندل‌های دریافتی برای بک‌تست کافی نیست.")
        return

    first_time = datetime.fromtimestamp(candles[0]["timestamp"], tz=timezone.utc)
    last_time = datetime.fromtimestamp(candles[-1]["timestamp"], tz=timezone.utc)
    total_days = (candles[-1]["timestamp"] - candles[0]["timestamp"]) / 86400
    
    print("بازه زمانی داده‌ها: از " + first_time.strftime('%Y-%m-%d %H:%M') + " تا " + last_time.strftime('%Y-%m-%d %H:%M'))

    rsi_list = calculate_rsi(candles, period=14)

    capital = INITIAL_CAPITAL
    total_trades = 0
    wins = 0
    losses = 0
    
    print("-" * 50)
    print("شروع بک‌تست روی " + str(len(candles)) + " کندل...")
    print("-" * 50)

    for i in range(30, len(candles) - 1):
        current_candle = candles[i]
        close_price = current_candle["close"]
        open_price = current_candle["open"]
        volume = current_candle["volume"]
        
        sma_50 = sum(c["close"] for c in candles[i-50:i]) / 50
        avg_volume = sum(c["volume"] for c in candles[i-20:i]) / 20
        current_rsi = rsi_list[i]
        prev_rsi = rsi_list[i - 1]

        is_uptrend = close_price > sma_50
        is_green = close_price > open_price
        is_rsi_bullish = (prev_rsi <= 52 and current_rsi > 52) or (45 <= current_rsi <= 65 and current_rsi > prev_rsi)
        is_volume_ok = volume > (avg_volume * 1.2)

        if is_uptrend and is_green and is_rsi_bullish and is_volume_ok:
            entry_price = close_price
            stop_loss = entry_price * 0.988
            take_profit = entry_price * 1.025
            
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
    print("نتایج نهایی بک‌تست:")
    print("کل معاملات انجام شده: " + str(total_trades))
    print("معاملات موفق (Win): " + str(wins))
    print("معاملات ناموفق (Loss): " + str(losses))
    
    if total_trades > 0:
        win_rate = (wins / total_trades) * 100
        net_profit = capital - INITIAL_CAPITAL
        profit_percentage = (net_profit / INITIAL_CAPITAL) * 100
        
        print(f"وین‌ریت استراتژی: {win_rate:.2f}%")
        print(f"سرمایه نهایی: {capital:.2f}$")
        print(f"سود/زیان خالص: {net_profit:+.2f}$ ({profit_percentage:+.2f}%)")
    else:
        print("هیچ معامله‌ای با این شرایط در این بازه فعال نشد.")

if __name__ == "__main__":
    run_backtest()
