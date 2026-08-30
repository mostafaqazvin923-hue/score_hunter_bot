import json
import time
import urllib.request
from datetime import datetime, timezone

# ============================================================
# SETTINGS (Professional High-Winrate Setup)
# ============================================================

BASE_URL = "https://api.coinex.com/v2"

SYMBOLS = ["BTCUSDT", "SOLUSDT"]
TIMEFRAME = "15min"
TARGET_CANDLES = 17500
PAGE_LIMIT = 1000

INITIAL_CAPITAL_PER_SYMBOL = 500.0  
FIXED_RISK_AMOUNT = 50.0            # ریسک ثابت ۵۰ دلار برای هر معامله

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
            headers={"User-Agent": "Mozilla/5.0 SCORE-HUNTER-PROFESSIONAL"}
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
                    cl = float(row[2])
                    hi = float(row[3])
                    lo = float(row[4])
                    vol = float(row[5])
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

def calculate_sma(values, period):
    sma = []
    for i in range(len(values)):
        if i < period - 1:
            sma.append(sum(values[:i+1]) / (i + 1))
        else:
            sma.append(sum(values[i-period+1:i+1]) / period)
    return sma

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
# PROFESSIONAL BACKTEST ENGINE (High Win-Rate Rules)
# ============================================================

def run_backtest():
    total_portfolio_profit = 0.0
    total_trades_all = 0
    total_wins_all = 0
    total_losses_all = 0
    total_days = 0

    print("=" * 60)
    print("گزارش تست حرفه‌ای با فیلترهای سخت‌گیرانه روند و حجم روی BTC و SOL:")
    print("=" * 60)

    for symbol in SYMBOLS:
        candles = download_klines(symbol, TIMEFRAME, TARGET_CANDLES)
        if len(candles) < 150:
            continue

        min_ts = candles[0]["timestamp"]
        max_ts = candles[-1]["timestamp"]
        total_days = (max_ts - min_ts) / 86400

        closes = [c["close"] for c in candles]
        volumes = [c["volume"] for c in candles]
        
        ema_20 = calculate_ema(closes, 20)
        ema_50 = calculate_ema(closes, 50)
        ema_100 = calculate_ema(closes, 100)
        sma_vol_20 = calculate_sma(volumes, 20)
        rsi_list = calculate_rsi(candles, 14)

        trades = []
        active_position = None  # قفل پوزیشن برای جلوگیری از تداخل

        for i in range(100, len(candles) - 1):
            c = candles[i]
            prev_c = candles[i-1]
            
            close_p = c["close"]
            open_p = c["open"]
            
            # مدیریت پوزیشن باز فعلی
            if active_position is not None:
                if c["low"] <= active_position["sl"]:
                    trades.append("LOSS")
                    active_position = None
                elif c["high"] >= active_position["tp"]:
                    trades.append("WIN")
                    active_position = None
                else:
                    continue

            # فیلترهای حرفه‌ای و سخت‌گیرانه برای بالا بردن دقت و وین‌ریت
            is_strong_uptrend = (ema_20[i] > ema_50[i]) and (ema_50[i] > ema_100[i])
            is_pullback = (prev_c["low"] <= ema_20[i-1]) and (close_p > ema_20[i])
            
            rsi = rsi_list[i]
            is_rsi_optimal = 50 <= rsi <= 60  # محدوده‌ی دقیق مومنتوم صعودی سالم
            
            is_green_candle = close_p > open_p
            is_volume_confirmed = c["volume"] > (sma_vol_20[i] * 1.2)  # حجم معاملات بالاتر از میانگین

            if is_strong_uptrend and is_pullback and is_rsi_optimal and is_green_candle and is_volume_confirmed:
                entry_price = close_p
                
                # تنظیمات ریسک به ریوارد حرفه‌ای (SL دقیق و TP هدفمند)
                stop_loss = entry_price * 0.992     # 0.8% حد ضرر فشرده‌تر و امن‌تر
                take_profit = entry_price * 1.018   # 1.8% حد سود منطقی برای R:R بالا
                
                # بررسی موانع قیمتی در کندل‌های قبلی
                has_obstacle = False
                for j in range(max(0, i-3), i):
                    upper_wick = candles[j]["high"] - max(candles[j]["open"], candles[j]["close"])
                    body = abs(candles[j]["close"] - candles[j]["open"])
                    if body > 0 and upper_wick > (body * 1.8):
                        has_obstacle = True
                        break

                if has_obstacle:
                    continue

                trade_result = None
                for future_c in candles[i+1:]:
                    if future_c["low"] <= stop_loss:
                        trade_result = "LOSS"
                        break
                    elif future_c["high"] >= take_profit:
                        trade_result = "WIN"
                        break
                
                if trade_result:
                    trades.append(trade_result)
                else:
                    active_position = {"tp": take_profit, "sl": stop_loss}

        wins = trades.count("WIN")
        losses = trades.count("LOSS")
        symbol_total_trades = wins + losses
        symbol_win_rate = (wins / symbol_total_trades * 100) if symbol_total_trades > 0 else 0.0
        
        # محاسبه سود خالص با ضریب ریسک به ریوارد جدید (تخمین نسبت ~2.25)
        symbol_profit = (wins * FIXED_RISK_AMOUNT * 2.25) - (losses * FIXED_RISK_AMOUNT)
        
        total_trades_all += symbol_total_trades
        total_wins_all += wins
        total_losses_all += losses
        total_portfolio_profit += symbol_profit

        print(f"\nارز: {symbol}")
        print(f"  - تعداد کل معاملات حرفه‌ای: {symbol_total_trades}")
        print(f"  - موفق (Win): {wins} | ناموفق (Loss): {losses}")
        print(f"  - وین‌ریت: {symbol_win_rate:.2f}%")
        print(f"  - سود/زیان خالص: {symbol_profit:+.2f}$")
        print("-" * 40)

    print("\n" + "=" * 60)
    print("برآیند نهایی ستاپ حرفه‌ای:")
    print("=" * 60)
    print(f"کل معاملات کل سبد: {total_trades_all}")
    print(f"معاملات موفق (Win): {total_wins_all}")
    print(f"معاملات ناموفق (Loss): {total_losses_all}")
    
    portfolio_wr = (total_wins_all / total_trades_all * 100) if total_trades_all > 0 else 0.0
    final_capital = 1000.0 + total_portfolio_profit
    portfolio_roi = (total_portfolio_profit / 1000.0) * 100

    print(f"وین‌ریت کل سبد: {portfolio_wr:.2f}%")
    print(f"سرمایه نهایی سبد: {final_capital:.2f}$")
    print(f"سود خالص کل سبد: {total_portfolio_profit:+.2f}$ ({portfolio_roi:+.2f}%)")

if __name__ == "__main__":
    run_backtest()
