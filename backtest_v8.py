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
                        cl = float(row.get("close", 0))
                        hi = float(row.get("high", 0))
                        lo = float(row.get("low", 0))
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
        base_p += random.uniform(-1.5, 1.5)
        op = base_p
        cl = base_p + random.uniform(-1.2, 1.2)
        hi = max(op, cl) + random.uniform(0.1, 0.6)
        lo = min(op, cl) - random.uniform(0.1, 0.6)
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
    grand_total_longs = 0
    grand_total_shorts = 0

    print("==================================================")
    print("   SCORE HUNTER PRO - ORGANIC SMART BACKTEST     ")
    print("==================================================")

    for symbol in SYMBOLS:
        print(f"\n[*] Fetching and testing 1-year data for {symbol}...")
        candles = fetch_historical_klines(symbol, limit=8800)
        
        wins = 0
        losses = 0
        symbol_trades = 0
        symbol_longs = 0
        symbol_shorts = 0
        
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
            trade_taken = False

            # --- تشخیص ارگانیک شورت توسط ربات ---
            past_lows = min(x["low"] for x in sub_candles[-10:-2])
            is_bearish_structure = (c["close"] < past_lows) and (c["close"] < trend_sma)
            strong_bearish_momentum = (c["open"] - c["close"]) > (c["high"] - c["low"]) * 0.5

            if is_bearish_structure and strong_bearish_momentum and (current_rsi < 42):
                entry_price = c["close"]
                stop_loss = max(prev_c["high"], prev2_c["high"]) + (entry_price * 0.001)
                risk_dist = stop_loss - entry_price

                if risk_dist > 0 and (risk_dist / entry_price) <= 0.04:
                    take_profit = entry_price - (risk_dist * TARGET_RR)

                    trade_won = False
                    trade_lost = False
                    end_idx = min(i + 24, len(candles) - 1)
                    
                    for j in range(i + 1, end_idx + 1):
                        future_c = candles[j]
                        if future_c["high"] >= stop_loss:
                            trade_lost = True
                            break
                        if future_c["low"] <= take_profit:
                            trade_won = True
                            break
                    
                    if not trade_won and not trade_lost:
                        if candles[end_idx]["close"] < entry_price:
                            trade_won = True
                        else:
                            trade_lost = True

                    symbol_trades += 1
                    symbol_shorts += 1
                    grand_total_shorts += 1
                    skip_until = end_idx
                    trade_taken = True

                    if trade_won:
                        wins += 1
                        balance += (risk_amount * TARGET_RR)
                    elif trade_lost:
                        losses += 1
                        balance -= risk_amount

            # --- تشخیص ارگانیک لانگ توسط ربات (اگر شورت تریگر نشد) ---
            if not trade_taken:
                past_highs = max(x["high"] for x in sub_candles[-10:-2])
                is_bullish_structure = (c["close"] > past_highs) and (c["close"] > trend_sma)
                strong_bullish_momentum = (c["close"] - c["open"]) > (c["high"] - c["low"]) * 0.5

                if is_bullish_structure and strong_bullish_momentum and (current_rsi > 58):
                    entry_price = c["close"]
                    stop_loss = min(prev_c["low"], prev2_c["low"]) - (entry_price * 0.001)
                    risk_dist = entry_price - stop_loss

                    if risk_dist > 0 and (risk_dist / entry_price) <= 0.04:
                        take_profit = entry_price + (risk_dist * TARGET_RR)

                        trade_won = False
                        trade_lost = False
                        end_idx = min(i + 24, len(candles) - 1)
                        
                        for j in range(i + 1, end_idx + 1):
                            future_c = candles[j]
                            if future_c["low"] <= stop_loss:
                                trade_lost = True
                                break
                            if future_c["high"] >= take_profit:
                                trade_won = True
                                break
                        
                        if not trade_won and not trade_lost:
                            if candles[end_idx]["close"] > entry_price:
                                trade_won = True
                            else:
                                trade_lost = True

                        symbol_trades += 1
                        symbol_longs += 1
                        grand_total_longs += 1
                        skip_until = end_idx

                        if trade_won:
                            wins += 1
                            balance += (risk_amount * TARGET_RR)
                        elif trade_lost:
                            losses += 1
                            balance -= risk_amount

        total_wins += wins
        total_losses += losses
        grand_total_trades += symbol_trades
        print(f" > {symbol} -> Total: {symbol_trades} (Longs: {symbol_symbol_longs if 'symbol_symbol_longs' in locals() else symbol_longs}, Shorts: {symbol_shorts}) | Wins: {wins} | Losses: {losses}")

    win_rate = (total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

    print("\n==================================================")
    print("             AGGREGATED 1-YEAR RESULTS            ")
    print("==================================================")
    print(f"Total Coins Tested : {len(SYMBOLS)}")
    print(f"Total Trades       : {grand_total_trades}")
    print(f"  - Total Longs    : {grand_total_longs}")
    print(f"  - Total Shorts     : {grand_total_shorts}")
    print(f"Winning Trades     : {total_wins}")
    print(f"Losing Trades      : {total_losses}")
    print(f"Overall Win Rate   : {win_rate:.2f}%")
    print(f"Final Balance      : ${balance:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_backtest()
