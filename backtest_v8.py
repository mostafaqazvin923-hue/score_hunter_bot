import json
import urllib.request

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
TIMEFRAME = "1h"
TARGET_RR = 2.0  # ریسک به ریوارد ثابت ۱ به ۲

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
    if len(closes) < period:
        return closes[-1]
    multiplier = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for p in closes[period:]:
        ema = (p - ema) * multiplier + ema
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
    risk_percentage = 0.02
    
    total_wins = 0
    total_losses = 0
    grand_total_trades = 0
    grand_total_longs = 0
    grand_total_shorts = 0

    print("==================================================")
    print(" SCORE HUNTER PRO - SNIPER ELITE V24 (TARGET 70%) ")
    print("==================================================")

    for symbol in SYMBOLS:
        try:
            candles_1h = fetch_real_klines_yahoo(symbol, limit=8800)
        except Exception as err:
            print(err)
            continue
        
        if len(candles_1h) < 200:
            continue

        candles_4h = resample_to_4h(candles_1h)
        
        wins = 0
        losses = 0
        symbol_trades = 0
        symbol_longs = 0
        symbol_shorts = 0
        skip_until = 0

        for i in range(50, len(candles_1h) - 20):
            if i < skip_until:
                continue

            c1h = candles_1h[i]
            current_ts = c1h["timestamp"]

            # پیدا کردن کندل متناظر ۴ ساعته برای فیلتر روند کلان
            active_4h_idx = -1
            for idx, c4 in enumerate(candles_4h):
                if c4["timestamp"] <= current_ts:
                    active_4h_idx = idx
                else:
                    break
            
            if active_4h_idx < 30: 
                continue
            
            sub_4h = candles_4h[:active_4h_idx+1]
            closes_4h = [x["close"] for x in sub_4h]
            ema_20_4h = calculate_ema(closes_4h, 20)
            ema_50_4h = calculate_ema(closes_4h, 50)
            
            is_4h_bullish = ema_20_4h > ema_50_4h
            is_4h_bearish = ema_20_4h < ema_50_4h

            sub_1h = candles_1h[:i+1]
            c = sub_1h[-2]       # کندل سیگنال
            prev_c = sub_1h[-3]  # کندل قبل برای تعیین استاپ

            ema20_1h = calculate_ema([x["close"] for x in sub_1h], 20)
            ema50_1h = calculate_ema([x["close"] for x in sub_1h], 50)
            rsi = calculate_rsi(sub_1h, 14)
            trade_taken = False

            candle_range = c["high"] - c["low"]
            if candle_range == 0:
                continue

            # فیلتر بسیار سخت‌گیرانه مومنتوم بدنه (بدنه باید حداقل ۶۵٪ کل کندل باشد)
            body_size = abs(c["close"] - c["open"])
            body_ratio = body_size / candle_range
            if body_ratio < 0.65:
                continue

            # تاییدیه حجم معاملات قوی (باید حداقل ۵۰٪ بالاتر از میانگین باشد)
            recent_volumes = [x["volume"] for x in sub_1h[-22:-2]]
            avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1.0
            if c["volume"] <= (avg_volume * 1.5):
                continue

            current_risk_amount = balance * risk_percentage

            # --- شورت با فیلترهای استریل و هم‌راستایی ۴ ساعته و ۱ ساعته ---
            is_short = is_4h_bearish and (ema20_1h < ema50_1h) and (c["close"] < c["open"]) and (c["close"] < ema20_1h) and (rsi < 45)

            if is_short:
                entry_price = c["close"]
                stop_loss = prev_c["high"] + (entry_price * 0.001)
                risk_dist = stop_loss - entry_price

                if 0 < (risk_dist / entry_price) <= 0.025:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 8, len(candles_1h) - 1)  # بستن سریع‌تر پوزیشن در صورت عدم تحقق
                    
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
                    trade_taken = True

                    if trade_won: 
                        wins += 1; balance += (current_risk_amount * TARGET_RR)
                    elif trade_lost: 
                        losses += 1; balance -= current_risk_amount

            # --- لانگ با فیلترهای استریل و هم‌راستایی ۴ ساعته و ۱ ساعته ---
            if not trade_taken:
                is_long = is_4h_bullish and (ema20_1h > ema50_1h) and (c["close"] > c["open"]) and (c["close"] > ema20_1h) and (rsi > 55)

                if is_long:
                    entry_price = c["close"]
                    stop_loss = prev_c["low"] - (entry_price * 0.001)
                    risk_dist = entry_price - stop_loss

                    if 0 < (risk_dist / entry_price) <= 0.025:
                        take_profit = entry_price + (risk_dist * TARGET_RR)
                        trade_won, trade_lost = False, False
                        end_idx = min(i + 8, len(candles_1h) - 1)
                        
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
    print("      AGGREGATED V24 SNIPER ELITE RESULTS         ")
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
