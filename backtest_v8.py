import json
import urllib.request

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
TIMEFRAME = "1h"
TARGET_RR = 1.3  # ریسک به ریوارد بهینه برای سیستم‌های برگشت به میانگین

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
            opens = quotes.get("open", [])
            highs = quotes.get("high", [])
            lows = quotes.get("low", [])
            closes = quotes.get("close", [])
            volumes = quotes.get("volume", [])
            
            candles = []
            for idx in range(len(timestamps)):
                op, hi, lo, cl, vol, ts = opens[idx], highs[idx], lows[idx], closes[idx], volumes[idx], timestamps[idx]
                if op is None or hi is None or lo is None or cl is None:
                    continue
                candles.append({
                    "timestamp": int(ts), 
                    "open": float(op), 
                    "high": float(hi), 
                    "low": float(lo), 
                    "close": float(cl),
                    "volume": float(vol) if vol is not None else 0.0
                })
            
            candles.sort(key=lambda x: x["timestamp"])
            if len(candles) > 100:
                return candles[-limit:]
    except Exception as e:
        print(f"[!] Error fetching {symbol}: {e}")
    raise ValueError(f"[!] Could not fetch data for {symbol}.")

def calculate_bollinger_bands(closes, period=20, num_std=2.0):
    if len(closes) < period:
        c = closes[-1]
        return c, c, c
    sub = closes[-period:]
    sma = sum(sub) / period
    variance = sum((x - sma) ** 2 for x in sub) / period
    std_dev = variance ** 0.5
    upper = sma + (num_std * std_dev)
    lower = sma - (num_std * std_dev)
    return upper, sma, lower

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
    risk_percentage = 0.02
    
    total_wins = 0
    total_losses = 0
    grand_total_trades = 0
    grand_total_longs = 0
    grand_total_shorts = 0

    print("==================================================")
    print(" SCORE HUNTER PRO - BOLLINGER MEAN REVERSION V25  ")
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

        for i in range(30, len(candles) - 15):
            if i < skip_until:
                continue

            sub = candles[:i+1]
            closes = [x["close"] for x in sub]
            c = sub[-2]       # کندل سیگنال
            prev_c = sub[-3]

            upper, sma, lower = calculate_bollinger_bands(closes, period=20, num_std=2.0)
            rsi = calculate_rsi(sub, 14)
            trade_taken = False

            candle_range = c["high"] - c["low"]
            if candle_range == 0:
                continue

            current_risk_amount = balance * risk_percentage

            # --- لانگ (خرید در کف باند بولینگر و اشباع فروش RSI) ---
            is_oversold = (c["low"] <= lower or c["close"] <= lower) and (rsi < 35) and (c["close"] > c["open"])

            if is_oversold:
                entry_price = c["close"]
                stop_loss = min(prev_c["low"], c["low"]) - (entry_price * 0.001)
                risk_dist = entry_price - stop_loss

                if 0 < (risk_dist / entry_price) <= 0.03:
                    take_profit = entry_price + (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 10, len(candles) - 1)  # بستن سریع معامله برای برگشت به میانگین
                    
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
                    trade_taken = True

                    if trade_won: 
                        wins += 1; balance += (current_risk_amount * TARGET_RR)
                    elif trade_lost: 
                        losses += 1; balance -= current_risk_amount

            # --- شورت (فروش در سقف باند بولینگر و اشباع خرید RSI) ---
            is_overbought = (c["high"] >= upper or c["close"] >= upper) and (rsi > 65) and (c["close"] < c["open"])

            if not trade_taken and is_overbought:
                entry_price = c["close"]
                stop_loss = max(prev_c["high"], c["high"]) + (entry_price * 0.001)
                risk_dist = stop_loss - entry_price

                if 0 < (risk_dist / entry_price) <= 0.03:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 10, len(candles) - 1)
                    
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

                    if trade_won: 
                        wins += 1; balance += (current_risk_amount * TARGET_RR)
                    elif trade_lost: 
                        losses += 1; balance -= current_risk_amount

        total_wins += wins
        total_losses += losses
        grand_total_trades += symbol_trades
        print(f" > {symbol} -> Total: {symbol_trades} (Longs: {symbol_longs}, Shorts: {symbol_shorts}) | Wins: {wins} | Losses: {losses}")

    win_rate = (total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

    print("\n==================================================")
    print("      AGGREGATED BOLLINGER REVERSION V25 RESULTS  ")
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
    run_backtest()
