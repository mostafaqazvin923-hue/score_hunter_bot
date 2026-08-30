import json
import time
import urllib.request
from datetime import datetime, timezone

# ============================================================
# SETTINGS (Multi-Coin 70% Win-Rate Setup & R:R 1:2)
# ============================================================

BASE_URL = "https://api.coinex.com/v2"

# گسترش سبد ارزها (چندارزی)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
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
            headers={"User-Agent": "Mozilla/5.0 SCORE-HUNTER-ELITE-70"}
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

def calculate_atr(candles, period=14):
    atr = [candles[0]["high"] - candles[0]["low"]]
    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i-1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        if i < period:
            atr.append((atr[-1] * i + tr) / (i + 1))
        else:
            atr.append((atr[-1] * (period - 1) + tr) / period)
    return atr

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
# ELITE BACKTEST ENGINE (Targeting ~70% Win-Rate & R:R 1:2)
# ============================================================

def run_backtest():
    total_portfolio_profit = 0.0
    total_trades_all = 0
    total_wins_all = 0
    total_losses_all = 0
    total_days = 0

    print("=" * 60)
    print("گزارش تست چندارزی (سودآور و با وین‌ریت بالا / R:R 1:2):")
    print("=" * 60)

    for symbol in SYMBOLS:
        candles = download_klines(symbol, TIMEFRAME, TARGET_CANDLES)
        if len(candles) < 200:
            continue

        min_ts = candles[0]["timestamp"]
        max_ts = candles[-1]["timestamp"]
        total_days = (max_ts - min_ts) / 86400

        closes = [c["close"] for c in candles]
        volumes = [c["volume"] for c in candles]
        
        ema_50 = calculate_ema(closes, 50)
        ema_200 = calculate_ema(closes, 200)
        atr_list = calculate_atr(candles, 14)
        sma_vol_50 = calculate_sma(volumes, 50)
        rsi_list = calculate_rsi(candles, 14)

        trades = []
        active_position = None  # قفل پوزیشن برای جلوگیری از تداخل

        for i in range(200, len(candles) - 1):
            c = candles[i]
            close_p = c["close"]
            
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

            # استراتژی الیت جهت رسیدن به وین‌ریت بالا
            # 1. روند بلندمدت صعودی (قیمت بالای EMA 200 و EMA 50 بالای EMA 200)
            is_macro_uptrend = (close_p > ema_200[i]) and (ema_50[i] > ema_200[i])
            
            # 2. اصلاح قیمتی تمیز و تایید بازگشت با قدرت ساختار بازار
            is_structure_bullish = (c["close"] > c["open"]) and (c["close"] > candles[i-1]["high"])
            
            # 3. RSI در ناحیه سلامت روند (بین ۵۲ تا ۶۸ بدون اشباع خطرناک)
            rsi = rsi_list[i]
            is_rsi_safe = 52 <= rsi <= 68
            
            # 4. تایید حجم معاملات بالاتر از میانگین ۵۰ کندل اخیر
            is_volume_strong = c["volume"] > (sma_vol_50[i] * 1.3)

            if is_macro_uptrend and is_structure_bullish and is_rsi_safe and is_volume_strong:
                entry_price = close_p
                current_atr = atr_list[i]
                
                # تنظیم دقیق حد ضرر بر اساس ATR و حد سود دقیقاً ۲ برابر (R:R 1:2)
                stop_loss = entry_price - (current_atr * 1.2)
                take_profit = entry_price + (current_atr * 2.4)
                
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
        
        # محاسبه سود خالص با ریسک به ریوارد دقیق ۱ به ۲ (هر برد = ۲ برابر ریسک)
        symbol_profit = (wins * FIXED_RISK_AMOUNT * 2.0) - (losses * FIXED_RISK_AMOUNT)
        
        total_trades_all += symbol_total_trades
        total_wins_all += wins
        total_losses_all += losses
        total_portfolio_profit += symbol_profit

        print(f"\nارز: {symbol}")
        print(f"  - تعداد کل معاملات: {symbol_total_trades}")
        print(f"  - موفق (Win): {wins} | ناموفق (Loss): {losses}")
        print(f"  - وین‌ریت: {symbol_win_rate:.2f}%")
        print(f"  - سود/زیان خالص: {symbol_profit:+.2f}$")
        print("-" * 40)

    print("\n" + "=" * 60)
    print("برآیند نهایی سبد چندارزی (هدف Win-Rate بالا و R:R 1:2):")
    print("=" * 60)
    print(f"کل معاملات کل سبد: {total_trades_all}")
    print(f"معاملات موفق (Win): {total_wins_all}")
    print(f"معاملات ناموفق (Loss): {total_losses_all}")
    
    portfolio_wr = (total_wins_all / total_trades_all * 100) if total_trades_all > 0 else 0.0
    initial_total_capital = len(SYMBOLS) * INITIAL_CAPITAL_PER_SYMBOL
    final_capital = initial_total_capital + total_portfolio_profit
    portfolio_roi = (total_portfolio_profit / initial_total_capital) * 100

    print(f"وین‌ریت کل سبد: {portfolio_wr:.2f}%")
    print(f"سرمایه نهایی کل سبد: {final_capital:.2f}$")
    print(f"سود خالص کل سبد: {total_portfolio_profit:+.2f}$ ({portfolio_roi:+.2f}%)")

if __name__ == "__main__":
    run_backtest()
