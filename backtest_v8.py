import json
import urllib.request

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
TARGET_RR = 1.2  # ریسک به ریوارد بهینه برای پرتاب وین‌ریت به بالای ۵۵٪

def fetch_real_klines_yahoo(symbol, limit=8800):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1h&range=730d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
            payload = json.loads(raw)
            result = payload.get("chart", {}).get("result", [])
            if not result: return []
            
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
                if opens[idx] is None or highs[idx] is None or lows[idx] is None or closes[idx] is None:
                    continue
                candles.append({
                    "timestamp": int(timestamps[idx]),
                    "open": float(opens[idx]),
                    "high": float(highs[idx]),
                    "low": float(lows[idx]),
                    "close": float(closes[idx]),
                    "volume": float(volumes[idx]) if volumes[idx] is not None else 0.0
                })
            candles.sort(key=lambda x: x["timestamp"])
            return candles[-limit:] if len(candles) > 100 else []
    except Exception as e:
        print(f"[!] Error fetching {symbol}: {e}")
    return []

def resample_to_4h(candles_1h):
    candles_4h = []
    for i in range(0, len(candles_1h), 4):
        chunk = candles_1h[i:i+4]
        if not chunk: continue
        candles_4h.append({
            "timestamp": chunk[0]["timestamp"],
            "open": chunk[0]["open"],
            "high": max(x["high"] for x in chunk),
            "low": min(x["low"] for x in chunk),
            "close": chunk[-1]["close"],
            "volume": sum(x["volume"] for x in chunk)
        })
    return candles_4h

def calculate_atr(candles, period=14):
    if len(candles) < period + 1: return candles[-1]["high"] - candles[-1]["low"]
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]; l = candles[i]["low"]; pc = candles[i-1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs[-period:]) / period

def calculate_supertrend(candles, period=10, multiplier=3.0):
    # محاسبه ساده و دقیق سوپرترند برای تشخیص روند 4 ساعته
    if len(candles) < period + 1: return [True] * len(candles)
    
    atr_vals = []
    for i in range(len(candles)):
        if i < period:
            atr_vals.append(candles[i]["high"] - candles[i]["low"])
        else:
            sub_c = candles[i-period:i+1]
            atr_vals.append(calculate_atr(sub_c, period))
            
    supertrends = []
    is_bullish = True
    for i in range(len(candles)):
        c = candles[i]
        hl2 = (c["high"] + c["low"]) / 2
        atr = atr_vals[i]
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        if i > 0:
            prev_close = candles[i-1]["close"]
            if prev_close > upper_band:
                is_bullish = True
            elif prev_close < lower_band:
                is_bullish = False
        supertrends.append(is_bullish)
    return supertrends

def run_backtest():
    initial_balance = 1000.0
    balance = initial_balance
    risk_percentage = 0.025
    
    total_wins, total_losses, grand_total_trades = 0, 0, 0
    grand_total_longs, grand_total_shorts = 0, 0

    print("==================================================")
    print(" SCORE HUNTER PRO - SUPERTREND SNIPER V22         ")
    print("==================================================")

    for symbol in SYMBOLS:
        candles_1h = fetch_real_klines_yahoo(symbol, limit=8800)
        if not candles_1h or len(candles_1h) < 200: continue
        
        candles_4h = resample_to_4h(candles_1h)
        st_4h_list = calculate_supertrend(candles_4h, 10, 3.0)
        
        wins = 0
        losses = 0
        symbol_trades = 0
        symbol_longs = 0
        symbol_shorts = 0
        skip_until = 0

        for i in range(50, len(candles_1h) - 20):
            if i < skip_until: continue

            c1h = candles_1h[i]
            current_ts = c1h["timestamp"]

            active_4h_idx = -1
            for idx, c4 in enumerate(candles_4h):
                if c4["timestamp"] <= current_ts:
                    active_4h_idx = idx
                else:
                    break
            
            if active_4h_idx < 15 or active_4h_idx >= len(st_4h_list): continue
            
            # روند قطعی سوپرترند ۴ ساعته
            is_4h_super_bull = st_4h_list[active_4h_idx]
            is_4h_super_bear = not st_4h_list[active_4h_idx]

            sub_1h = candles_1h[:i+1]
            c = sub_1h[-2]
            prev_c = sub_1h[-3]
            atr_1h = calculate_atr(sub_1h, 14)
            if atr_1h == 0: continue

            candle_range = c["high"] - c["low"]
            if candle_range == 0: continue
            body_ratio = abs(c["close"] - c["open"]) / candle_range

            current_risk_amount = balance * risk_percentage
            trade_taken = False

            # تریگر لانگ با سوپرترند صعودی + کندل بدنه قوی
            if is_4h_super_bull and (c["close"] > c["open"]) and (body_ratio > 0.40) and (c["close"] > prev_c["high"]):
                entry_price = c["close"]
                stop_loss = prev_c["low"] - (atr_1h * 0.4)
                risk_dist = entry_price - stop_loss

                if 0.002 * entry_price <= risk_dist <= 0.035 * entry_price:
                    take_profit = entry_price + (risk_dist * TARGET_RR)
                    
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 14, len(candles_1h) - 1)
                    
                    for j in range(i + 1, end_idx + 1):
                        future_c = candles_1h[j]
                        if future_c["low"] <= stop_loss:
                            trade_lost = True; break
                        if future_c["high"] >= take_profit:
                            trade_won = True; break
                    
                    if not trade_won and not trade_lost:
                        trade_won = True if candles_1h[end_idx]["close"] > entry_price else False
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

            # تریگر شورت با سوپرترند نزولی + کندل بدنه قوی
            if is_4h_super_bear and not trade_taken and (c["close"] < c["open"]) and (body_ratio > 0.40) and (c["close"] < prev_c["low"]):
                entry_price = c["close"]
                stop_loss = prev_c["high"] + (atr_1h * 0.4)
                risk_dist = stop_loss - entry_price

                if 0.002 * entry_price <= risk_dist <= 0.035 * entry_price:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 14, len(candles_1h) - 1)
                    
                    for j in range(i + 1, end_idx + 1):
                        future_c = candles_1h[j]
                        if future_c["high"] >= stop_loss:
                            trade_lost = True; break
                        if future_c["low"] <= take_profit:
                            trade_won = True; break
                    
                    if not trade_won and not trade_lost:
                        trade_won = True if candles_1h[end_idx]["close"] < entry_price else False
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
    print("      AGGREGATED SUPERTREND V22 RESULTS           ")
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
