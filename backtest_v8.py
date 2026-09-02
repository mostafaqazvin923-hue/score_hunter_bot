import json
import urllib.request

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
TARGET_RR = 2.0

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

def calculate_ema(closes, period):
    if len(closes) < period: return closes[-1]
    multiplier = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for p in closes[period:]:
        ema = (p - ema) * multiplier + ema
    return ema

def calculate_atr(candles, period=14):
    if len(candles) < period + 1: return candles[-1]["high"] - candles[-1]["low"]
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]; l = candles[i]["low"]; pc = candles[i-1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs[-period:]) / period

def run_backtest():
    initial_balance = 1000.0
    balance = initial_balance
    risk_percentage = 0.025  # ریسک ۲.۵ درصدی برای سود مرکب بهینه
    
    total_wins = 0
    total_losses = 0
    grand_total_trades = 0
    grand_total_longs = 0
    grand_total_shorts = 0

    print("==================================================")
    print(" SCORE HUNTER PRO - MTF 4H/1H + RUNWAY V20        ")
    print("==================================================")

    for symbol in SYMBOLS:
        candles_1h = fetch_real_klines_yahoo(symbol, limit=8800)
        if not candles_1h or len(candles_1h) < 200: continue
        
        candles_4h = resample_to_4h(candles_1h)
        
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

            # پیدا کردن جهت 4 ساعته متناظر
            active_4h_idx = -1
            for idx, c4 in enumerate(candles_4h):
                if c4["timestamp"] <= current_ts:
                    active_4h_idx = idx
                else:
                    break
            
            if active_4h_idx < 30: continue
            
            sub_4h = candles_4h[:active_4h_idx+1]
            closes_4h = [x["close"] for x in sub_4h]
            ema_50_4h = calculate_ema(closes_4h, 50)
            
            is_4h_bullish = closes_4h[-1] > ema_50_4h
            is_4h_bearish = closes_4h[-1] < ema_50_4h

            sub_1h = candles_1h[:i+1]
            c = sub_1h[-2]       # کندل سیگنال
            prev_c = sub_1h[-3]  # کندل قبل برای استاپ
            atr_1h = calculate_atr(sub_1h, 14)
            if atr_1h == 0: continue

            current_risk_amount = balance * risk_percentage
            trade_taken = False

            # بررسی فضای مسیر (Runway Check) بر اساس سوییگ‌های اخیر
            recent_highs = [x["high"] for x in sub_1h[-40:-2]]
            recent_lows = [x["low"] for x in sub_1h[-40:-2]]
            major_resistance = max(recent_highs) if recent_highs else c["high"] * 1.05
            major_support = min(recent_lows) if recent_lows else c["low"] * 0.95

            # تریگر لانگ (همراستا با روند صعودی ۴ ساعته)
            if is_4h_bullish and (c["close"] > c["open"]) and (c["close"] > prev_c["high"]):
                entry_price = c["close"]
                stop_loss = min(prev_c["low"], c["low"]) - (atr_1h * 0.3)
                risk_dist = entry_price - stop_loss

                if 0.002 * entry_price <= risk_dist <= 0.035 * entry_price:
                    take_profit = entry_price + (risk_dist * TARGET_RR)
                    
                    # فیلتر فضای مسیر: آیا تا TP مانع محکمی هست؟
                    if major_resistance >= (take_profit - (risk_dist * 0.1)):
                        trade_won, trade_lost = False, False
                        end_idx = min(i + 16, len(candles_1h) - 1)
                        
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

            # تریگر شورت (همراستا با روند نزولی ۴ ساعته)
            if is_4h_bearish and not trade_taken and (c["close"] < c["open"]) and (c["close"] < prev_c["low"]):
                entry_price = c["close"]
                stop_loss = max(prev_c["high"], c["high"]) + (atr_1h * 0.3)
                risk_dist = stop_loss - entry_price

                if 0.002 * entry_price <= risk_dist <= 0.035 * entry_price:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    
                    if major_support <= (take_profit + (risk_dist * 0.1)):
                        trade_won, trade_lost = False, False
                        end_idx = min(i + 16, len(candles_1h) - 1)
                        
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
    print("      AGGREGATED MTF V20 RESULTS                  ")
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
