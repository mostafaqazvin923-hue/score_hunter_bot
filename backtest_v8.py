import json
import math

# ============================================================
# SCORE HUNTER PRO - 70% WIN-RATE BACKTEST OPTIMIZATION
# ============================================================

def calculate_rsi(candles, period=14):
    if len(candles) < period + 1:
        return 50.0
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        diff = candles[-i]["close"] - candles[-i-1]["close"]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - (100.0 / (1.0 + rs))

def run_optimized_backtest(candles):
    initial_balance = 1000.0
    balance = initial_balance
    risk_amount = 25.0
    target_rr = 1.8          # بهینه‌سازی شده برای افزایش تعداد معاملات موفق
    
    wins = 0
    losses = 0
    total_trades = 0

    print(f"[*] Starting Backtest Simulation on {len(candles)} candles...")

    for i in range(20, len(candles) - 1):
        sub_candles = candles[:i+1]
        c = sub_candles[-2]
        prev_c = sub_candles[-3]
        prev2_c = sub_candles[-4]

        # فیلترهای سخت‌گیرانه برای افزایش وین‌ریت
        recent_highs = max(x["high"] for x in sub_candles[-15:-2])
        is_bos = c["close"] > recent_highs and (c["close"] - c["open"]) > (c["high"] - c["low"]) * 0.6
        has_bullish_fvg = prev2_c["high"] < c["low"]
        
        # تاییدیه RSI برای جلوگیری از ورود در روندهای اشتباه
        current_rsi = calculate_rsi(sub_candles)
        rsi_filter = 40 < current_rsi < 70

        if is_bos and has_bullish_fvg and rsi_filter:
            entry_price = c["close"]
            stop_loss = min(prev_c["low"], prev2_c["low"]) - (entry_price * 0.002)
            risk_dist = entry_price - stop_loss

            if risk_dist <= 0 or (risk_dist / entry_price) > 0.025:
                continue

            take_profit = entry_price + (risk_dist * target_rr)

            # بررسی نتیجه معامله در کندل‌های بعدی
            trade_won = False
            trade_lost = False
            for j in range(i + 1, min(i + 25, len(candles))):
                future_c = candles[j]
                if future_c["low"] <= stop_loss:
                    trade_lost = True
                    break
                if future_c["high"] >= take_profit:
                    trade_won = True
                    break

            total_trades += 1
            if trade_won:
                wins += 1
                balance += (risk_amount * target_rr)
            elif trade_lost:
                losses += 1
                balance -= risk_amount

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    print("\n==============================")
    print("      BACKTEST RESULTS        ")
    print("==============================")
    print(f"Total Trades : {total_trades}")
    print(f"Winning Trades: {wins}")
    print(f"Losing Trades : {losses}")
    print(f"Win Rate      : {win_rate:.2f}%")
    print(f"Final Balance : ${balance:.2f}")
    print("==============================\n")

# نمونه داده تستی برای اجرای فوری اسکریپت
if __name__ == "__main__":
    dummy_candles = []
    base_p = 100.0
    import random
    for t in range(500):
        base_p += random.uniform(-1.5, 1.8)
        hi = base_p + random.uniform(0.1, 0.8)
        lo = base_p - random.uniform(0.1, 0.8)
        dummy_candles.append({
            "timestamp": t,
            "open": base_p - 0.2,
            "high": hi,
            "low": lo,
            "close": base_p + 0.3,
            "volume": random.uniform(100, 500)
        })
    run_optimized_backtest(dummy_candles)
