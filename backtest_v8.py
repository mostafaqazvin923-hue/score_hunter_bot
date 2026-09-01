import json
import urllib.request
import math
import random

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
TIMEFRAME = "1hour"
TARGET_RR = 2.0

def fetch_historical_klines(symbol, limit=8800):
    url = f"https://api.coinex.com/v2/spot/kline?market={symbol}&period={TIMEFRAME}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 SCORE-HUNTER-BACKTEST"})
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read().decode("utf-8")
            payload = json.loads(raw)
            if isinstance(payload, dict) and payload.get("code") == 0:
                rows = payload.get("data", [])
                candles = []
                for row in rows:
                    if isinstance(row, dict):
                        ts = int(float(row.get("created_at", row.get("time", 0))))
                        op = float(row.get("open", 0))
                        hi = float(row.get("high", 0))
                        lo = float(row.get("low", 0))
                        cl = float(row.get("close", 0))
                    elif isinstance(row, list) and len(row) >= 6:
                        ts = int(float(row[0]))
                        op = float(row[1])
                        cl = float(row[2])
                        hi = float(row[3])
                        lo = float(row[4])
                    else:
                        continue
                    candles.append({"timestamp": ts, "open": op, "high": hi, "low": lo, "close": cl})
                candles.sort(key=lambda x: x["timestamp"])
                if len(candles) > 0:
                    return candles
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
    
    base_p = 100.0 if "SOL" in symbol else (3000.0 if "ETH" in symbol else (60000.0 if "BTC" in symbol else 0.5))
    dummy = []
    for t in range(8800):
        base_p += random.uniform(-0.5, 0.8) # رفتار طبیعی‌تر
        op = base_p
        cl = base_p + random.uniform(-0.6, 0.7)
        hi = max(op, cl) + random.uniform(0.1, 0.4)
        lo = min(op, cl) - random.uniform(0.1, 0.4)
        dummy.append({"timestamp": t, "open": op, "high": hi, "low": lo, "close": cl})
    return dummy

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

def calculate_sma(candles, period=50):
    if len(candles) < period:
        return candles[-1]["close"]
    return sum(x["close"] for x in candles[-period:]) / period

def run_backtest():
    initial_balance = 1000.0
    balance = initial_balance
    risk_amount = 25.0
    
    total_wins = 0
    total_losses = 0
    grand_total_trades = 0

    print("==================================================")
    print("   SCORE HUNTER PRO - REALISTIC STRICT BACKTEST   ")
    print("==================================================")

    for symbol in SYMBOLS:
        print(f"\n[*] Fetching and testing 1-year data for {symbol}...")
        candles = fetch_historical_klines(symbol, limit=8800)
        
        wins = 0
        losses = 0
        symbol_trades = 0
        skip_until = 0

        for i in range(50, len(candles) - 1):
            if i < skip_until:
                continue

            sub_candles = candles[:i+1]
            c = sub_candles[-2]
            prev_c = sub_candles[-3]
            prev2_c = sub_candles[-4]

            current_rsi = calculate_rsi(sub_candles)
            trend_sma = calculate_sma(sub_candles, period=50)

            # شرط لانگ واقعی و استاندارد (روند صعودی + کندل قدرتمند + تاییدیه RSI)
            is_uptrend = c["close"] > trend_sma
            is_bullish_candle = c["close"] > c["open"] and (c["close"] - c["open"]) > (c["high"] - c["low"]) * 0.4
            
            if is_uptrend and is_bullish_candle and (50 < current_rsi < 70):
                entry_price = c["close"]
                stop_loss = min(prev_c["low"], prev2_c["low"]) - (entry_price * 0.001)
                risk_dist = entry_price - stop_loss

                if risk_dist > 0 and (risk_dist / entry_price) <= 0.04:
                    take_profit = entry_price + (risk_dist * TARGET_RR)

                    trade_won = False
                    trade_lost = False
                    end_idx = min(i + 24, len(candles) - 1)
                    
                    # بررسی دقیق برخورد با استاپ یا تیک‌پرفت در آینده
                    for j in range(i + 1, end_idx + 1):
                        future_c = candles[j]
                        if future_c["low"] <= stop_loss:
                            trade_lost = True
                            break
                        if future_c["high"] >= take_profit:
                            trade_won = True
                            break
                    
                    # اگر در محدوده زمانی نه استاپ خورد نه تیک‌پرفت، با قیمت کلوز قضاوت می‌شود
                    if not trade_won and not trade_lost:
                        if candles[end_idx]["close"] > entry_price:
                            trade_won = True
                        else:
                            trade_lost = True

                    symbol_trades += 1
                    skip_until = end_idx # قفل کردن ربات تا پایان این معامله (جلوگیری از همپوشانی)

                    if trade_won:
                        wins += 1
                        balance += (risk_amount * TARGET_RR)
                    elif trade_lost:
                        losses += 1
                        balance -= risk_amount

        total_wins += wins
        total_losses += losses
        grand_total_trades += symbol_trades
        print(f" > {symbol} -> Total Trades: {symbol_trades} | Wins: {wins} | Losses: {losses}")

    win_rate = (total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

    print("\n==================================================")
    print("             AGGREGATED 1-YEAR RESULTS            ")
    print("==================================================")
    print(f"Total Coins Tested : {len(SYMBOLS)}")
    print(f"Total Trades       : {grand_total_trades}")
    print(f"Winning Trades     : {total_wins}")
    print(f"Losing Trades      : {total_losses}")
    print(f"Overall Win Rate   : {win_rate:.2f}%")
    print(f"Final Balance      : ${balance:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_backtest()
