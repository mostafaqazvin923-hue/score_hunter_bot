import json
import time
import urllib.request
from datetime import datetime, timezone

# ============================================================
# SETTINGS (Multi-Timeframe Ichimoku: 4h Trend + 1h Entry / R:R 1:2)
# ============================================================

BASE_URL = "https://api.coinex.com/v2"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
TIMEFRAME_1H = "1hour"
TARGET_CANDLES_1H = 8000  # معادل حدود ۱۱ ماه داده ۱ ساعته
PAGE_LIMIT = 1000

INITIAL_CAPITAL_PER_SYMBOL = 500.0  
FIXED_RISK_AMOUNT = 50.0            # ریسک ثابت ۵۰ دلار برای هر معامله

# ============================================================
# HTTP & PAGINATION DATA DOWNLOADER (1-Hour Candles)
# ============================================================

def download_klines(symbol, timeframe, target_count):
    all_rows = {}
    end_time = None
    pages = 0
    max_pages = 15

    while len(all_rows) < target_count and pages < max_pages:
        pages += 1
        url = f"{BASE_URL}/spot/kline?market={symbol}&period={timeframe}&limit={PAGE_LIMIT}"
        if end_time:
            url += f"&end_time={end_time}"

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 SCORE-HUNTER-MTF-ICHIMOKU"}
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

# تجمیع کندل‌های ۱ ساعته به ۴ ساعته برای فیلتر روند
def build_4h_candles(candles_1h):
    candles_4h = []
    chunk = []
    for c in candles_1h:
        chunk.append(c)
        if len(chunk) == 4:
            ts = chunk[0]["timestamp"]
            op = chunk[0]["open"]
            hi = max(x["high"] for x in chunk)
            lo = min(x["low"] for x in chunk)
            cl = chunk[-1]["close"]
            vol = sum(x["volume"] for x in chunk)
            candles_4h.append({
                "timestamp": ts,
                "open": op,
                "high": hi,
                "low": lo,
                "close": cl,
                "volume": vol
            })
            chunk = []
    return candles_4h

# ============================================================
# ICHIMOKU CALCULATOR
# ============================================================

def calculate_donchian(candles, period, start_idx):
    if start_idx < period - 1:
        slice_candles = candles[:start_idx+1]
    else:
        slice_candles = candles[start_idx-period+1:start_idx+1]
    highest = max(c["high"] for c in slice_candles)
    lowest = min(c["low"] for c in slice_candles)
    return (highest + lowest) / 2.0

def calculate_ichimoku_series(candles):
    tenkan_sen = []
    kijun_sen = []
    senkou_a = []
    senkou_b = []

    for i in range(len(candles)):
        t = calculate_donchian(candles, 9, i)
        tenkan_sen.append(t)
        k = calculate_donchian(candles, 26, i)
        kijun_sen.append(k)
        sa = (t + k) / 2.0
        senkou_a.append(sa)
        sb = calculate_donchian(candles, 52, i)
        senkou_b.append(sb)

    return tenkan_sen, kijun_sen, senkou_a, senkou_b

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
# MULTI-TIMEFRAME BACKTEST ENGINE
# ============================================================

def run_backtest():
    total_portfolio_profit = 0.0
    total_trades_all = 0
    total_wins_all = 0
    total_losses_all = 0

    print("=" * 60)
    print("گزارش تست ایچیموکو چندتایم‌فریمی (روند 4h + ورود 1h / R:R 1:2):")
    print("=" * 60)

    for symbol in SYMBOLS:
        candles_1h = download_klines(symbol, TIMEFRAME_1H, TARGET_CANDLES_1H)
        if len(candles_1h) < 300:
            continue

        candles_4h = build_4h_candles(candles_1h)
        if len(candles_4h) < 60:
            continue

        # محاسبه ایچیموکو برای هر دو تایم‌فریم
        t_1h, k_1h, sa_1h, sb_1h = calculate_ichimoku_series(candles_1h)
        rsi_1h = calculate_rsi(candles_1h, 14)

        t_4h, k_4h, sa_4h, sb_4h = calculate_ichimoku_series(candles_4h)

        trades = []
        active_position = None

        # مپ کردن زمان 1h به 4h
        for i in range(100, len(candles_1h) - 1):
            c_1h = candles_1h[i]
            prev_1h = candles_1h[i-1]
            close_1h = c_1h["close"]
            open_1h = c_1h["open"]
            ts_1h = c_1h["timestamp"]

            # مدیریت پوزیشن باز
            if active_position is not None:
                if c_1h["low"] <= active_position["sl"]:
                    trades.append("LOSS")
                    active_position = None
                elif c_1h["high"] >= active_position["tp"]:
                    trades.append("WIN")
                    active_position = None
                else:
                    continue

            # پیدا کردن کندل معادل 4h برای تشخیص روند
            valid_4h_indices = [idx for idx, c4 in enumerate(candles_4h) if c4["timestamp"] <= ts_1h]
            if not valid_4h_indices:
                continue
            idx_4h = valid_4h_indices[-1]

            # 1. شرط روند ۴ ساعته (صعودی قدرتمند)
            cloud_top_4h = max(sa_4h[idx_4h], sb_4h[idx_4h])
            is_4h_uptrend = (candles_4h[idx_4h]["close"] > cloud_top_4h) and (t_4h[idx_4h] > k_4h[idx_4h])

            if not is_4h_uptrend:
                continue

            # 2. شرط نقطه ورود ۱ ساعته (پولبک به کیجون‌سن یا کراس تنکان و کیجون)
            is_1h_kijun_bounce = (prev_1h["low"] <= k_1h[i]) and (close_1h > k_1h[i]) and (close_1h > open_1h)
            is_1h_rsi_safe = 50 <= rsi_1h[i] <= 68

            if is_1h_kijun_bounce and is_1h_rsi_safe:
                entry_price = close_1h
                
                # تعیین حد ضرر پشت کیجون‌سن ۱ ساعته و حد سود دقیقاً ۲ برابر (R:R 1:2)
                risk_distance = entry_price - k_1h[i]
                if risk_distance <= 0 or risk_distance / entry_price > 0.025:
                    risk_distance = entry_price * 0.01  # حد ضرر محافظتی ۱٪

                stop_loss = entry_price - risk_distance
                take_profit = entry_price + (risk_distance * 2.0)

                trade_result = None
                for future_c in candles_1h[i+1:]:
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
    print("برآیند نهایی سبد چندتایم‌فریمی (4h Trend + 1h Entry):")
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
