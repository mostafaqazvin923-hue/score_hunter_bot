import json
import urllib.request
import time

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
TARGET_RR = 2.0  # قفل شده روی ۱ به ۲ به دستور شما

def fetch_1year_15m_auto(symbol):
    all_candles = {}
    ranges = ["59d", "118d", "177d", "236d", "295d", "354d"]
    
    for r in ranges:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=15m&range={r}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = response.read().decode("utf-8")
                payload = json.loads(raw)
                result = payload.get("chart", {}).get("result", [])
                if not result:
                    continue
                
                data = result[0]
                timestamps = data.get("timestamp", [])
                quotes = data.get("indicators", {}).get("quote", [{}])[0]
                opens = quotes.get("open", [])
                highs = quotes.get("high", [])
                lows = quotes.get("low", [])
                closes = quotes.get("close", [])
                volumes = quotes.get("volume", [])
                
                for idx in range(len(timestamps)):
                    ts = timestamps[idx]
                    op, hi, lo, cl, vol = opens[idx], highs[idx], lows[idx], closes[idx], volumes[idx]
                    if op is None or hi is None or lo is None or cl is None or ts is None:
                        continue
                    all_candles[int(ts)] = {
                        "timestamp": int(ts), 
                        "open": float(op), 
                        "high": float(hi), 
                        "low": float(lo), 
                        "close": float(cl),
                        "volume": float(vol) if vol is not None else 0.0
                    }
        except Exception:
            continue
        time.sleep(0.5)
        
    candles = list(all_candles.values())
    candles.sort(key=lambda x: x["timestamp"])
    return candles

def calculate_ema(data, period):
    emas = []
    multiplier = 2 / (period + 1)
    for i, val in enumerate(data):
        if i == 0:
            emas.append(val)
        else:
            emas.append((val * multiplier) + (emas[-1] * (1 - multiplier)))
    return emas

def calculate_atr(candles, period=14):
    atr_list = [0.0] * len(candles)
    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i-1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        if i <= period:
            atr_list[i] = (atr_list[i-1] * (i-1) + tr) / i if i > 1 else tr
        else:
            atr_list[i] = (atr_list[i-1] * (period - 1) + tr) / period
    return atr_list

def run_backtest():
    initial_balance = 1000.0
    balance = initial_balance
    risk_percentage = 0.01
    
    total_wins = 0
    total_losses = 0
    grand_total_trades = 0

    print("==================================================")
    print(" SCORE HUNTER PRO - V14.0 (INSTITUTIONAL SQUEEZE) ")
    print("==================================================")

    for symbol in SYMBOLS:
        try:
            candles = fetch_1year_15m_auto(symbol)
            if len(candles) < 200:
                continue
        except Exception:
            continue
        
        closes = [c["close"] for c in candles]
        volumes = [c["volume"] for c in candles]
        ema_150 = calculate_ema(closes, 150)
        atr_vals = calculate_atr(candles, 14)

        wins = 0
        losses = 0
        symbol_trades = 0
        skip_until = 0

        for i in range(150, len(candles) - 40):
            if i < skip_until:
                continue

            c = candles[i]
            prev = candles[i-1]
            prev2 = candles[i-2]
            trend = ema_150[i]
            atr = atr_vals[i]

            # میانگین حجم ۲۰ کندل گذشته برای تشخیص حجم نهادی
            avg_vol = sum(volumes[i-20:i]) / 20 if i >= 20 else volumes[i]

            current_risk_amount = balance * risk_percentage
            trade_taken = False

            # فیلتر سخت‌گیرانه فشردگی و انفجار صعودی (Bullish Squeeze Breakout)
            is_bullish_trend = c["close"] > trend
            is_volume_spike = c["volume"] > (avg_vol * 1.5) # حجم حداقل ۵۰ درصد بالاتر از میانگین
            is_price_expansion = (c["close"] - c["open"]) > (atr * 0.8) # کندل قدرتمند صعودی
            is_breakout = c["close"] > max(prev["high"], prev2["high"])

            if is_bullish_trend and is_volume_spike and is_price_expansion and is_breakout and atr > 0:
                entry_price = c["close"]
                stop_loss = entry_price - (atr * 1.2) # استاپ تایتل و مهندسی شده با ATR
                risk_dist = entry_price - stop_loss

                if risk_dist > 0:
                    take_profit = entry_price + (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 40, len(candles) - 1)

                    for j in range(i + 1, end_idx + 1):
                        fc = candles[j]
                        if fc["high"] >= take_profit:
                            trade_won = True; break
                        if fc["low"] <= stop_loss:
                            trade_lost = True; break

                    if trade_won or trade_lost:
                        symbol_trades += 1
                        skip_until = j + 15 # جلوگیری از اورتریدینگ و فیلتر سیگنال‌های اضافی
                        trade_taken = True
                        if trade_won:
                            wins += 1
                            balance += (current_risk_amount * TARGET_RR)
                        else:
                            losses += 1
                            balance -= current_risk_amount

            # فیلتر سخت‌گیرانه فشردگی و انفجار نزولی (Bearish Squeeze Breakdown)
            is_bearish_trend = c["close"] < trend
            is_bearish_expansion = (prev["open"] - c["close"]) > (atr * 0.8)
            is_bearish_breakout = c["close"] < min(prev["low"], prev2["low"])

            if not trade_taken and is_bearish_trend and is_volume_spike and is_bearish_expansion and is_bearish_breakout and atr > 0:
                entry_price = c["close"]
                stop_loss = entry_price + (atr * 1.2)
                risk_dist = stop_loss - entry_price

                if risk_dist > 0:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 40, len(candles) - 1)

                    for j in range(i + 1, end_idx + 1):
                        fc = candles[j]
                        if fc["low"] <= take_profit:
                            trade_won = True; break
                        if fc["high"] >= stop_loss:
                            trade_lost = True; break

                    if trade_won or trade_lost:
                        symbol_trades += 1
                        skip_until = j + 15
                        if trade_won:
                            wins += 1
                            balance += (current_risk_amount * TARGET_RR)
                        else:
                            losses += 1
                            balance -= current_risk_amount

        total_wins += wins
        total_losses += losses
        grand_total_trades += symbol_trades
        print(f" > {symbol} -> Trades: {symbol_trades} | Wins: {wins} | Losses: {losses}")

    win_rate = (total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

    print("\n==================================================")
    print("      AGGREGATED V14.0 RESULTS                    ")
    print("==================================================\n")
    print(f"Total Trades       : {grand_total_trades}")
    print(f"Winning Trades     : {total_wins}")
    print(f"Losing Trades      : {total_losses}")
    print(f"Overall Win Rate   : {win_rate:.2f}%")
    print(f"Final Balance      : ${balance:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_backtest()
