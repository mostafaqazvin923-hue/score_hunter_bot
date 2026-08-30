import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

# ============================================================
# SETTINGS (Optimized Portfolio: BTC, ETH, SOL, LINK, DOGE, AVAX)
# ============================================================

BASE_URL = "https://api.coinex.com/v2"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "DOGEUSDT", "AVAXUSDT"]
TIMEFRAME = "15min"
TARGET_CANDLES = 17500
PAGE_LIMIT = 1000

INITIAL_CAPITAL = 1000.0
RISK_PERCENT = 0.02

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
                    vol = float(row[5]) if len(row) > 5 else float(row[5])
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

def calculate_atr(candles, period=14):
    atr_values = [0.0] * len(candles)
    tr_list = [candles[0]["high"] - candles[0]["low"]]
    
    for i in range(1, len(candles)):
        h_l = candles[i]["high"] - candles[i]["low"]
        h_pc = abs(candles[i]["high"] - candles[i-1]["close"])
        l_pc = abs(candles[i]["low"] - candles[i-1]["close"])
        tr = max(h_l, h_pc, l_pc)
        tr_list.append(tr)
        
    if len(tr_list) >= period:
        atr_values[period-1] = sum(tr_list[:period]) / period
        for i in range(period, len(candles)):
            atr_values[i] = (atr_values[i-1] * (period - 1) + tr_list[i]) / period
    return atr_values

# ============================================================
# MAIN PORTFOLIO BACKTEST (Optimized New Basket)
# ============================================================

def run_portfolio_backtest():
    all_trades = []
    global_min_ts = float('inf')
    global_max_ts = 0

    print("در حال استخراج سیگنال‌ها روی سبد جدید (BTC, ETH, SOL, LINK, DOGE, AVAX)...")

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
        atr_list = calculate_atr(candles, 14)

        for i in range(50, len(candles) - 1):
            c = candles[i]
            prev_c = candles[i-1]
            
            close_p = c["close"]
            open_p = c["open"]
            high_p = c["high"]
            low_p = c["low"]
            atr = atr_list[i]
            
            if atr <= 0:
                continue

            # ۱. روند صعودی قدرتمند
            is_uptrend = ema_20[i] > ema_50[i]
            
            # ۲. پولبک استاندارد به EMA 20
            is_pullback_recovery = (prev_c["low"] <= ema_20[i-1]) and (close_p > ema_20[i])
            
            # ۳. فیلتر RSI دقیق در محدوده کاملاً صعودی و ایمن
            rsi = rsi_list[i]
            is_rsi_good = 50 <= rsi <= 62
            
            # ۴. تاییدیه بدنه کندل صعودی قوی
            candle_body = abs(close_p - open_p)
            candle_range = high_p - low_p
            is_strong_body = candle_range > 0 and (candle_body / candle_range) >= 0.45
            is_green = close_p > open_p

            if is_uptrend and is_pullback_recovery and is_rsi_good and is_strong_body and is_green:
                entry_price = close_p
                # مدیریت ریسک پویا بر اساس ATR
                stop_loss = entry_price - (1.1 * atr)
                take_profit = entry_price + (1.6 * atr)
                
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
    total_trades = len(all_trades)
    wins = 0
    losses = 0

    for trade in all_trades:
        risk_amount = capital * RISK_PERCENT
        if trade["result"] == "WIN":
            wins += 1
            capital += risk_amount * 1.45
        else:
            losses += 1
            capital -= risk_amount

    print("\n" + "=" * 50)
    print("نتایج نهایی سبد جدید (BTC, ETH, SOL, LINK, DOGE, AVAX):")
    print("=" * 50)
    print(f"کل معاملات سبد: {total_trades}")
    print(f"معاملات موفق (Win): {wins}")
    print(f"معاملات ناموفق (Loss): {losses}")
    
    if total_trades > 0:
        win_rate = (wins / total_trades) * 100
        net_profit = capital - INITIAL_CAPITAL
        profit_percentage = (net_profit / INITIAL_CAPITAL) * 100
        
        print(f"وین‌ریت کل سبد: {win_rate:.2f}%")
        print(f"سرمایه نهایی سبد: {capital:.2f}$")
        print(f"سود/زیان خالص کل: {net_profit:+.2f}$ ({profit_percentage:+.2f}%)")
        print(f"میانگین کل معاملات در روز (کل سبد): {total_trades / max(total_days, 0.1):.2f}")
    else:
        print("هیچ معامله‌ای انجام نشد.")

if __name__ == "__main__":
    run_portfolio_backtest()
