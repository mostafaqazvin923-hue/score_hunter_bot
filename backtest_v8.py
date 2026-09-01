import json
import urllib.request

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
TIMEFRAME = "4h"
TARGET_RR = 2.0

def fetch_real_klines_yahoo(symbol, limit=8800):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=4h&range=730d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
            payload = json.loads(raw)
            result = payload.get("chart", {}).get("result", [])
            if not result:
                raise ValueError("Empty result")
            
            data = result[0]
            timestamps = data.get("timestamp", [])
            quotes = data.get("indicators", {}).get("quote", [{}])[0]
            opens, highs, lows, closes = quotes.get("open", []), quotes.get("high", []), quotes.get("low", []), quotes.get("close", [])
            
            candles = []
            for idx in range(len(timestamps)):
                op, hi, lo, cl, ts = opens[idx], highs[idx], lows[idx], closes[idx], timestamps[idx]
                if op is None or hi is None or lo is None or cl is None:
                    continue
                candles.append({"timestamp": int(ts), "open": float(op), "high": float(hi), "low": float(lo), "close": float(cl)})
            
            candles.sort(key=lambda x: x["timestamp"])
            if len(candles) > 100:
                return candles[-limit:]
    except Exception as e:
        print(f"[!] Error fetching {symbol}: {e}")
    raise ValueError(f"[!] Could not fetch data for {symbol}.")

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
    print(" SCORE HUNTER PRO - 4H SMC ADVANCED BACKTEST    ")
    print("==================================================")

    for symbol in SYMBOLS:
        try:
            candles = fetch_real_klines_yahoo(symbol, limit=8800)
        except Exception as err:
            print(err)
            continue
        
        wins = 0
        losses = 0
        symbol_trades = 0
        symbol_longs = 0
        symbol_shorts = 0
        skip_until = 0

        for i in range(25, len(candles) - 1):
            if i < skip_until:
                continue

            sub = candles[:i+1]
            c = sub[-2]

            recent_high = max(x["high"] for x in sub[-25:-2])
            recent_low = min(x["low"] for x in sub[-25:-2])

            trade_taken = False

            # --- شورت: شکار نقدینگی سقف در تایم‌فریم ۴ ساعته ---
            is_sweep_high = (c["high"] >= recent_high) and (c["close"] < recent_high)
            is_bearish_rejection = (c["close"] < c["open"]) and ((c["open"] - c["close"]) > (c["high"] - c["low"]) * 0.45)

            if is_sweep_high and is_bearish_rejection:
                entry_price = c["close"]
                stop_loss = c["high"] + (entry_price * 0.0015)
                risk_dist = stop_loss - entry_price

                if 0 < (risk_dist / entry_price) <= 0.04:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 18, len(candles) - 1)
                    
                    for j in range(i + 1, end_idx + 1):
                        future_c = candles[j]
                        if future_c["high"] >= stop_loss:
                            trade_lost = True; break
                        if future_c["low"] <= take_profit:
                            trade_won = True; break
                    
                    if not trade_won and not trade_lost:
                        trade_won = True if candles[end_idx]["close"] < entry_price else False
                        trade_lost = not trade_won

                    symbol_trades += 1
                    symbol_shorts += 1
                    grand_total_shorts += 1
                    skip_until = end_idx
                    trade_taken = True

                    if trade_won: wins += 1; balance += (risk_amount * TARGET_RR)
                    elif trade_lost: losses += 1; balance -= risk_amount

            # --- لانگ: شکار نقدینگی کف در تایم‌فریم ۴ ساعته ---
            if not trade_taken:
                is_sweep_low = (c["low"] <= recent_low) and (c["close"] > recent_low)
                is_bullish_rejection = (c["close"] > c["open"]) and ((c["close"] - c["open"]) > (c["high"] - c["low"]) * 0.45)

                if is_sweep_low and is_bullish_rejection:
                    entry_price = c["close"]
                    stop_loss = c["low"] - (entry_price * 0.0015)
                    risk_dist = entry_price - stop_loss

                    if 0 < (risk_dist / entry_price) <= 0.04:
                        take_profit = entry_price + (risk_dist * TARGET_RR)
                        trade_won, trade_lost = False, False
                        end_idx = min(i + 18, len(candles) - 1)
                        
                        for j in range(i + 1, end_idx + 1):
                            future_c = candles[j]
                            if future_c["low"] <= stop_loss:
                                trade_lost = True; break
                            if future_c["high"] >= take_profit:
                                trade_won = True; break
                        
                        if not trade_won and not trade_lost:
                            trade_won = True if candles[end_idx]["close"] > entry_price else False
                            trade_lost = not trade_won

                        symbol_trades += 1
                        symbol_longs += 1
                        grand_total_longs += 1
                        skip_until = end_idx

                        if trade_won: wins += 1; balance += (risk_amount * TARGET_RR)
                        elif trade_lost: losses += 1; balance -= risk_amount

        total_wins += wins
        total_losses += losses
        grand_total_trades += symbol_trades
        print(f" > {symbol} -> Total: {symbol_trades} (Longs: {symbol_longs}, Shorts: {symbol_shorts}) | Wins: {wins} | Losses: {losses}")

    win_rate = (total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

    print("\n==================================================")
    print("             AGGREGATED 4H RESULTS                ")
    print("==================================================")
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
