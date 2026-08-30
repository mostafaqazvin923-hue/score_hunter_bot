import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

# ============================================================
# SETTINGS (Price Action Momentum Breakout - 6 Months)
# ============================================================

BASE_URL = "https://api.coinex.com/v2"

SYMBOL = "SOLUSDT"
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

    print(f"در حال دریافت تاریخچه ۶ ماهه کندل‌ها از CoinEx برای {symbol}...")

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
        except Exception as exc:
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
        time.sleep(0.2)

    candles = list(all_rows.values())
    candles.sort(key=lambda x: x["timestamp"])

    if len(candles) > target_count:
        candles = candles[-target_count:]

    return candles

# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_sma(values, period):
    sma = []
    for i in range(len(values)):
        if i < period - 1:
            sma.append(sum(values[:i+1]) / (i + 1))
        else:
            sma.append(sum(values[i-period+1:i+1]) / period)
    return sma

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
# BACKTEST ENGINE (Price Action Breakout Strategy)
# ============================================================

def run_backtest():
    candles = download_klines(SYMBOL, TIMEFRAME, TARGET_CANDLES)
    
    print("تعداد کل کندل‌های دریافت شده: " + str(len(candles)))
    
    if len(candles) < 100:
        print("تعداد کندل‌های دریافتی برای بک‌تست کافی نیست.")
        return

    first_time = datetime.fromtimestamp(candles[0]["timestamp"], tz=timezone.utc)
    last_time = datetime.fromtimestamp(candles[-1]["timestamp"], tz=timezone.utc)
    total_days = (candles[-1]["timestamp"] - candles[0]["timestamp"]) / 86400
    
    print(f"بازه زمانی دقیق داده‌ها: از {first_time.strftime('%Y-%m-%d %H:%M')} تا {last_time.strftime('%Y-%m-%d %H:%M')} (حدود {total_days:.1f} روز)")

    closes = [c["close"] for c in candles]
    sma_50 = calculate_sma(closes, 50)
    atr_list = calculate_atr(candles, 14)

    capital = INITIAL_CAPITAL
    total_trades = 0
    wins = 0
    losses = 0
    
    print("-" * 50)
    print("شروع بک‌تست استراتژی پرایس اکشن بریک‌اوت روی " + str(len(candles)) + " کندل...")
    print("-" * 50)

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

        # بررسی روند میان‌مدت (قیمت بالای میانگین متحرک ۵۰)
        is_uptrend = close_p > sma_50[i]
        
        # شکست سقف ۵ کندل گذشته (Breakout)
        recent_highest = max(x["high"] for x in candles[i-5:i])
        is_breakout = close_p > recent_highest
        
        # کندل صعودی پرقدرت (بدنه کندل حداقل ۶۰ درصد کل طول کندل باشد تا از پین‌بارهای جعلی جلوگیری شود)
        candle_body = abs(close_p - open_p)
        candle_range = high_p - low_p
        is_strong_body = candle_range > 0 and (candle_body / candle_range) >= 0.55
        
        # حجم معاملات بالاتر از میانگین ۱۰ کندل گذشته
        avg_vol = sum(x["volume"] for x in candles[i-10:i]) / 10
        is_volume_ok = c["volume"] > (avg_vol * 1.15)

        if is_uptrend and is_breakout and is_strong_body and is_volume_ok:
            entry_price = close_p
            stop_loss = entry_price - (1.5 * atr)    # حد ضرر استاندارد مبتنی بر ATR
            take_profit = entry_price + (2.5 * atr)  # ریسک به ریوارد ۱ به ۱.۶۷ برای افزایش وین‌ریت
            
            trade_result = None
            for future_c in candles[i+1:]:
                if future_c["low"] <= stop_loss:
                    trade_result = "LOSS"
                    break
                elif future_c["high"] >= take_profit:
                    trade_result = "WIN"
                    break
            
            if trade_result:
                total_trades += 1
                risk_amount = capital * RISK_PERCENT

                if trade_result == "WIN":
                    wins += 1
                    capital += risk_amount * 1.67
                elif trade_result == "LOSS":
                    losses += 1
                    capital -= risk_amount

    print("-" * 50)
    print("نتایج نهایی بک‌تست پرایس اکشن:")
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
        print(f"میانگین تعداد معامله در روز: {total_trades / max(total_days, 0.1):.2f}")
    else:
        print("هیچ معامله‌ای با این شرایط در این بازه فعال نشد.")

if __name__ == "__main__":
    run_backtest()
