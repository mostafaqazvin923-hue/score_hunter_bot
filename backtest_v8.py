import json
import urllib.request

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
TIMEFRAME = "15m"
TARGET_RR = 2.0  # ثابت و غیرقابل تغییر به درخواست شما

def fetch_15m_klines_yahoo(symbol, limit=8800):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=15m&range=60d"
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
    risk_percentage = 0.01
    
    total_wins = 0
    total_losses = 0
    grand_total_trades = 0
    grand_total_longs = 0
    grand_total_shorts = 0

    print("==================================================")
    print(" SCORE HUNTER PRO - 70% WIN-RATE & 1:2 RR V29     ")
    print("==================================================")

    for symbol in SYMBOLS:
        try:
            candles = fetch_15m_klines_yahoo(symbol, limit=8800)
        except Exception as err:
            print(err)
            continue
        
        wins = 0
        losses = 0
        symbol_trades = 0
        symbol_longs = 0
        symbol_shorts = 0
        skip_until = 0

        for i in range(30, len(candles) - 10):
            if i < skip_until:
                continue

            sub = candles[:i+1]
            c = sub[-2]       # کندل تریگر
            prev_c = sub[-3]  # کندل قبل برای تعیین پیوت ساختاری
            prev2_c = sub[-4]

            rsi = calculate_rsi(sub, 14)
            trade_taken = False

            candle_range = c["high"] - c["low"]
            if candle_range == 0:
                continue

            # فیلتر بدنه قاطع و حجم تاییدشده برای جلوگیری از نویز
            body_size = abs(c["close"] - c["open"])
            if (body_size / candle_range) < 0.60:
                continue

            recent_volumes = [x["volume"] for x in sub[-22:-2]]
            avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1.0
            if c["volume"] <= (avg_volume * 1.3):
                continue

            current_risk_amount = balance * risk_percentage

            # --- لانگ استراکچری (بازگشت از کف محلی با استاپ فشرده) ---
            is_long_setup = (c["close"] > c["open"]) and (c["low"] <= min(prev_c["low"], prev2_c["low"])) and (rsi < 45)

            if is_long_setup:
                entry_price = c["close"]
                stop_loss = min(prev_c["low"], c["low"]) - (entry_price * 0.0008)
                risk_dist = entry_price - stop_loss

                if 0.0005 * entry_price <= risk_dist <= 0.012 * entry_price:
                    take_profit = entry_price + (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 6, len(candles) - 1)
                    
                    for j in range(i + 1, end_idx + 1):
                        future_c = candles[j]
                        # اول بررسی تارگت برای دقت در رفتار واقعی قیمت در اسکالپ
                        if future_c["high"] >= take_profit:
                            trade_won = True; break
                        if future_c["low"] <= stop_loss:
                            trade_lost = True; break
                    
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

            # --- شورت استراکچری (بازگشت از سقف محلی با استاپ فشرده) ---
            is_short_setup = (c["close"] < c["open"]) and (c["high"] >= max(prev_c["high"], prev2_c["high"])) and (rsi > 55)

            if not trade_taken and is_short_setup:
                entry_price = c["close"]
                stop_loss = max(prev_c["high"], c["high"]) + (entry_price * 0.0008)
                risk_dist = stop_loss - entry_price

                if 0.0005 * entry_price <= risk_dist <= 0.012 * entry_price:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 6, len(candles) - 1)
                    
                    for j in range(i + 1, end_idx + 1):
                        future_c = candles[j]
                        if future_c["low"] <= take_profit:
                            trade_won = True; break
                        if future_c["high"] >= stop_loss:
                            trade_lost = True; break
                    
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
    print("      AGGREGATED V29 RESULTS (70% & 1:2)          ")
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
