import json
import urllib.request
import time

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
TARGET_RR = 3.0  # ریسک به ریوارد طلایی سبک اسمارت مانی (۱ به ۳)

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

def run_backtest():
    initial_balance = 1000.0
    balance = initial_balance
    risk_percentage = 0.01
    
    total_wins = 0
    total_losses = 0
    grand_total_trades = 0

    print("==================================================")
    print(" SCORE HUNTER PRO - SMC / WHALE SETUP (1-YEAR)    ")
    print("==================================================")

    for symbol in SYMBOLS:
        try:
            candles = fetch_1year_15m_auto(symbol)
            if len(candles) < 200:
                continue
        except Exception:
            continue
        
        closes = [c["close"] for c in candles]
        emas_200 = calculate_ema(closes, 200) # فیلتر جهت روند کلان نهنگ‌ها

        wins = 0
        losses = 0
        symbol_trades = 0
        skip_until = 0

        for i in range(200, len(candles) - 30):
            if i < skip_until:
                continue

            c = candles[i]
            prev_c = candles[i-1]
            prev2_c = candles[i-2]
            trend_ema = emas_200[i]

            # پیدا کردن سقف و کف‌های محدوده نقدینگی (۲۰ کندل قبل)
            lookback = candles[i-20:i]
            liquidity_high = max([x["high"] for x in lookback])
            liquidity_low = min([x["low"] for x in lookback])

            current_risk_amount = balance * risk_percentage
            trade_taken = False

            # --- ستاپ لانگ (شکار نقدینگی کف + تاییدیه شکست ساختار صعودی + همراستا با روند) ---
            is_sweep_low = prev_c["low"] < liquidity_low
            is_mss_bullish = c["close"] > prev_c["high"] and c["close']" not in locals() and c["close"] > c["open"]
            is_above_trend = c["close"] > trend_ema

            if is_above_trend and is_sweep_low and is_mss_bullish:
                entry_price = c["close"]
                stop_loss = min(prev_c["low"], prev2_c["low"]) - (entry_price * 0.0002) # استاپ پشت نقدینگی جارو شده
                risk_dist = entry_price - stop_loss

                if risk_dist > 0:
                    take_profit = entry_price + (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 48, len(candles) - 1) # اجازه تنفس به قیمت تا ۱۲ ساعت

                    for j in range(i + 1, end_idx + 1):
                        fc = candles[j]
                        if fc["high"] >= take_profit:
                            trade_won = True; break
                        if fc["low"] <= stop_loss:
                            trade_lost = True; break

                    if trade_won or trade_lost:
                        symbol_trades += 1
                        skip_until = j
                        trade_taken = True
                        if trade_won:
                            wins += 1; balance += (current_risk_amount * TARGET_RR)
                        else:
                            losses += 1; balance -= current_risk_amount

            # --- ستاپ شورت (شکار نقدینگی سقف + تاییدیه شکست ساختار نزولی + همراستا با روند) ---
            is_sweep_high = prev_c["high"] > liquidity_high
            is_mss_bearish = c["close"] < prev_c["low"] and c["close"] < c["open"]
            is_below_trend = c["close']" not in locals() and c["close"] < trend_ema

            if not trade_taken and is_below_trend and is_sweep_high and is_mss_bearish:
                entry_price = c["close"]
                stop_loss = max(prev_c["high"], prev2_c["high"]) + (entry_price * 0.0002)
                risk_dist = stop_loss - entry_price

                if risk_dist > 0:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 48, len(candles) - 1)

                    for j in range(i + 1, end_idx + 1):
                        fc = candles[j]
                        if fc["low"] <= take_profit:
                            trade_won = True; break
                        if fc["high"] >= stop_loss:
                            trade_lost = True; break

                    if trade_won or trade_lost:
                        symbol_trades += 1
                        skip_until = j
                        if trade_won:
                            wins += 1; balance += (current_risk_amount * TARGET_RR)
                        else:
                            losses += 1; balance -= current_risk_amount

        total_wins += wins
        total_losses += losses
        grand_total_trades += symbol_trades
        print(f" > {symbol} -> Trades: {symbol_trades} | Wins: {wins} | Losses: {losses}")

    win_rate = (total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

    print("\n==================================================")
    print("      AGGREGATED SMC WHALE RESULTS                ")
    print("==================================================\n")
    print(f"Total Trades       : {grand_total_trades}")
    print(f"Winning Trades     : {total_wins}")
    print(f"Losing Trades      : {total_losses}")
    print(f"Overall Win Rate   : {win_rate:.2f}%")
    print(f"Final Balance      : ${balance:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_backtest()
