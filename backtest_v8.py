import json
import urllib.request

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
TIMEFRAME = "1h"
TARGET_RR = 2.0  # ریسک به ریوارد دقیقاً ۱ به ۲

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

def calculate_atr(candles, period=14):
    if len(candles) < period + 1:
        return candles[-1]["high"] - candles[-1]["low"]
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]
        l = candles[i]["low"]
        pc = candles[i-1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return sum(trs[-period:]) / period

def calculate_bollinger_bands(candles, period=20, std_dev=2.0):
    if len(candles) < period:
        cl = candles[-1]["close"]
        return cl, cl, cl
    subset = [c["close"] for c in candles[-period:]]
    sma = sum(subset) / period
    variance = sum((x - sma) ** 2 for x in subset) / period
    dev = std_dev * (variance ** 0.5)
    return sma + dev, sma, sma - dev

def _backtest():
    initial_balance = 1000.0
    balance = initial_balance
    risk_amount = 25.0
    
    total_wins = 0
    total_losses = 0
    grand_total_trades = 0
    grand_total_longs = 0
    grand_total_shorts = 0

    print("==================================================")
    print(" SCORE HUNTER PRO - BOLLINGER & RSI HIGH WIN 1:2  ")
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
            
            ema20 = calculate_ema(sub, 20)
            ema50 = calculate_ema(sub, 50)
            rsi = calculate_rsi(sub, 14)
            atr = calculate_atr(sub, 14)
            upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(sub, 20, 2.0)
            trade_taken = False

            if (c["high"] - c["low"]) == 0 or atr == 0:
                continue

            # --- شورت با بولینگر و RSI و ریوارد ۱ به ۲ ---
            is_down_trend = ema20 < ema50
            is_short_bb = c["high"] >= upper_bb or c["close"] > middle_bb # برخورد یا نزدیک باند بالا
            is_short = is_down_trend and (c["close"] < c["open"]) and (rsi < 52) and (c["close"] < ema20)

            if is_short:
                entry_price = c["close"]
                stop_loss = c["high"] + (atr * 0.5)  # حد ضرر مهندسی شده با ATR
                risk_dist = stop_loss - entry_price

                if 0 < (risk_dist / entry_price) <= 0.035:
                    take_profit = entry_price - (risk_dist * TARGET_RR)  # دقیقاً ۱ به ۲
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 14, len(candles) - 1)
                    
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
                    skip_until = end_idx - 1
                    trade_taken = True

                    if trade_won: 
                        wins += 1; balance += (risk_amount * TARGET_RR)
                    elif trade_lost: 
                        losses += 1; balance -= risk_amount

            # --- لانگ با بولینگر و RSI و ریوارد ۱ به ۲ ---
            if not trade_taken:
                is_up_trend = ema20 > ema50
                is_long = is_up_trend and (c["close"] > c["open"]) and (rsi > 48) and (c["close"] > ema20)

                if is_long:
                    entry_price = c["close"]
                    stop_loss = c["low"] - (atr * 0.5)  # حد ضرر مهندسی شده با ATR
                    risk_dist = entry_price - stop_loss

                    if 0 < (risk_dist / entry_price) <= 0.035:
                        take_profit = entry_price + (risk_dist * TARGET_RR)  # دقیقاً ۱ به ۲
                        trade_won, trade_lost = False, False
                        end_idx = min(i + 14, len(candles) - 1)
                        
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
                        skip_until = end_idx - 1
                        trade_taken = True

                        if trade_won: 
                            wins += 1; balance += (risk_amount * TARGET_RR)
                        elif trade_lost: 
                            losses += 1; balance -= risk_amount

        total_wins += wins
        total_losses += losses
        grand_total_trades += symbol_trades
        print(f" > {symbol} -> Total: {symbol_trades} (Longs: {symbol_longs}, Shorts: {symbol_shorts}) | Wins: {wins} | Losses: {losses}")

    win_rate = (total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

    print("\n==================================================")
    print("      AGGREGATED BOLLINGER STRATEGY RESULTS       ")
    print("==================================================\n")
    print(f"Total Trades       : {grand_total_trades}")
    print(f"  - Total Longs    : {grand_total_longs}")
    print(f"  - Total Shorts     : {grand_total_shorts}")
    print(f"Winning Trades     : {total_wins}")
    print(f"Losing Trades      : {total_losses}")
    print(f"Overall Win Rate   : {win_rate:.2f}%")
    print(f"Final Balance      : ${balance:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    _backtest()
