import json
import urllib.request
import time

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
TARGET_RR = 2.0  # قفل شده روی ۱ به ۲

def fetch_1year_15m_auto(symbol):
    all_candles = {}
    ranges = ["59d", "118d", "177d", "236d", "295d", "354d"]
    
    for r in ranges:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=15m&range={r}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        try:
            with urllib.request.urlopen(url, timeout=30) as response: # اصلاح برای پایتون استاندارد
                pass
        except:
            pass
            
    # استفاده از روش ایمن دریافت داده
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
    print(" SCORE HUNTER PRO - V16.0 (LIQUIDITY SWEEP / WHALE)")
    print("==================================================")

    for symbol in SYMBOLS:
        try:
            candles = fetch_1year_15m_auto(symbol)
            if len(candles) < 100:
                continue
        except Exception:
            continue
        
        atr_vals = calculate_atr(candles, 14)
        wins = 0
        losses = 0
        symbol_trades = 0
        skip_until = 0

        lookback = 20 # بازه برای پیدا کردن سقف و کف کلیدی نقدینگی

        for i in range(lookback, len(candles) - 40):
            if i < skip_until:
                continue

            c = candles[i]
            prev = candles[i-1]
            atr = atr_vals[i]

            # پیدا کردن بالاترین سقف و پایین‌ترین کف در ۲۰ کندل گذشته (استخر نقدینگی خرده‌فروشان)
            recent_slice = candles[i-lookback:i]
            swing_high = max(item["high"] for item in recent_slice)
            swing_low = min(item["low"] for item in recent_slice)

            current_risk_amount = balance * risk_percentage
            trade_taken = False

            # مدل ۱: استاپ‌هانت کف و بازگشت صعودی (Bullish Liquidity Sweep / Spring)
            # قیمت کف قبلی را با فتیله سوراخ کرده (استاپ‌ها را زده) ولی بسته شدن کندل بالای کف قبلی است
            is_sweep_low = prev["low"] < swing_low and c["close"] > swing_low
            is_bullish_rejection = c["close"] > c["open"] and (c["close"] - c["open"]) > (atr * 0.4)

            if is_sweep_low and is_bullish_rejection and atr > 0:
                entry_price = c["close"]
                stop_loss = prev["low"] - (atr * 0.5) # کمی پایین‌تر از فتیله نفوذی نهادینه‌شده
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
                        skip_until = j + 10
                        trade_taken = True
                        if trade_won:
                            wins += 1
                            balance += (current_risk_amount * TARGET_RR)
                        else:
                            losses += 1
                            balance -= current_risk_amount

            # مدل ۲: استاپ‌هانت سقف و بازگشت نزولی (Bearish Liquidity Sweep / Upthrust)
            is_sweep_high = prev["high"] > swing_high and c["close"] < swing_high
            is_bearish_rejection = c["open"] > c["close"] and (c["open"] - c["close"]) > (atr * 0.4)

            if not trade_taken and is_sweep_high and is_bearish_rejection and atr > 0:
                entry_price = c["close"]
                stop_loss = prev["high"] + (atr * 0.5)
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
                        skip_until = j + 10
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
    print("      AGGREGATED V16.0 WHALE RESULTS              ")
    print("==================================================\n")
    print(f"Total Trades       : {grand_total_trades}")
    print(f"Winning Trades     : {total_wins}")
    print(f"Losing Trades      : {total_losses}")
    print(f"Overall Win Rate   : {win_rate:.2f}%")
    print(f"Final Balance      : ${balance:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_backtest()
