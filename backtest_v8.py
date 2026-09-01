import json
import urllib.request

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
TIMEFRAME = "1h"

def fetch_real_klines_yahoo(symbol, limit=8800):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1h&range=730d"
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

def calculate_ema(candles, period):
    if len(candles) < period:
        return candles[-1]["close"]
    multiplier = 2 / (period + 1)
    ema = sum(x["close"] for x in candles[:period]) / period
    for c in candles[period:]:
        ema = (c["close"] - ema) * multiplier + ema
    return ema

def calculate_rsi(candles, period=14):
    if len(candles) < period + 1:
        return 50.0
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        change = candles[-i]["close"] - candles[-i-1]["close"]
        if change > 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

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
    print(" SCORE HUNTER PRO - STRICT DAILY 2-3 TRADES     ")
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

        for i in range(50, len(candles) - 1):
            if i < skip_until:
                continue

            sub = candles[:i+1]
            c = sub[-2]
            prev_c = sub[-3]

            ema20 = calculate_ema(sub, 20)
            ema50 = calculate_ema(sub, 50)
            ema200 = calculate_ema(sub, 200) # فیلتر روند کلان
            rsi = calculate_rsi(sub, 14)
            trade_taken = False

            # قدرت روند بسیار بالا برای جلوگیری از اورتررید
            trend_strength = abs(ema20 - ema50) / c["close"]
            if trend_strength < 0.007:
                continue

            # --- شورت با فیلتر سنگین روند و مومنتوم ---
            is_down_trend = (ema20 < ema50) and (c["close"] < ema200)
            is_strong_short = (c["close"] < c["open"]) and ((c["open"] - c["close"]) > (c["high"] - c["low"]) * 0.65) and (rsi < 42)

            if is_down_trend and is_strong_short:
                entry_price = c["close"]
                stop_loss = prev_c["high"] + (entry_price * 0.0015)
                risk_dist = stop_loss - entry_price

                if 0 < (risk_dist / entry_price) <= 0.03:
                    take_profit = entry_price - (risk_dist * 1.5) # RR 1.5 برای صید مطمئن‌تر وین‌ریت
                    
                    recent_lows = [x["low"] for x in sub[-25:-2]]
                    closest_support = min(recent_lows) if recent_lows else 0
                    if closest_support > take_profit and closest_support < entry_price:
                        continue

                    trade_won, trade_lost = False, False
                    end_idx = min(i + 20, len(candles) - 1)
                    
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
                    skip_until = end_idx + 4 # فاصله بین معاملات برای کنترل تعداد روزانه
                    trade_taken = True

                    if trade_won: 
                        wins += 1; balance += (risk_amount * 1.5)
                    elif trade_lost: 
                        losses += 1; balance -= risk_amount

            # --- لانگ با فیلتر سنگین روند و مومنتوم ---
            if not trade_taken:
                is_up_trend = (ema20 > ema50) and (c["close"] > ema200)
                is_strong_long = (c["close"] > c["open"]) and ((c["close"] - c["open"]) > (c["high"] - c["low"]) * 0.65) and (rsi > 58)

                if is_up_trend and is_strong_long:
                    entry_price = c["close"]
                    stop_loss = prev_c["low"] - (entry_price * 0.0015)
                    risk_dist = entry_price - stop_loss

                    if 0 < (risk_dist / entry_price) <= 0.03:
                        take_profit = entry_price + (risk_dist * 1.5)
                        
                        recent_highs = [x["high"] for x in sub[-25:-2]]
                        closest_resistance = max(recent_highs) if recent_highs else 999999
                        if closest_resistance < take_profit and closest_resistance > entry_price:
                            continue

                        trade_won, trade_lost = False, False
                        end_idx = min(i + 20, len(candles) - 1)
                        
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
                        skip_until = end_idx + 4

                        if trade_won: 
                            wins += 1; balance += (risk_amount * 1.5)
                        elif trade_lost: 
                            losses += 1; balance -= risk_amount

        total_wins += wins
        total_losses += losses
        grand_total_trades += symbol_trades
        print(f" > {symbol} -> Total: {symbol_trades} (Longs: {symbol_longs}, Shorts: {symbol_shorts}) | Wins: {wins} | Losses: {losses}")

    win_rate = (total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

    print("\n==================================================")
    print("      AGGREGATED STRICT CONTROL RESULTS           ")
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
