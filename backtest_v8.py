import json
import urllib.request
import math

SYMBOLS = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD", "XRPUSDT": "XRP-USD"}
TARGET_RR = 2.0

def fetch_yahoo_one_year(symbol_key):
    yahoo_symbol = SYMBOLS[symbol_key]
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?interval=1h&range=1y"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    try:
        print(f"[*] Downloading LSOB Market Data for {symbol_key}...")
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
            for i in range(len(timestamps)):
                if (
                    i < len(opens) and i < len(highs) and i < len(lows) and i < len(closes) and i < len(volumes) and
                    opens[i] is not None and highs[i] is not None and lows[i] is not None and closes[i] is not None
                ):
                    candles.append({
                        "timestamp": int(timestamps[i]) * 1000,
                        "open": float(opens[i]), "high": float(highs[i]),
                        "low": float(lows[i]), "close": float(closes[i]),
                        "volume": float(volumes[i]) if volumes[i] is not None else 0.0
                    })
            candles.sort(key=lambda x: x["timestamp"])
            return candles
    except Exception as e:
        print(f"[!] Error fetching data for {symbol_key}: {e}")
    return []

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
    risk_amount = 25.0
    total_wins, total_losses, grand_total_trades = 0, 0, 0

    print("==================================================")
    print(" LSOB V16 - INSTITUTIONAL ORDER BLOCK STRATEGY    ")
    print("==================================================")

    for symbol_key in SYMBOLS.keys():
        candles = fetch_yahoo_one_year(symbol_key)
        if not candles or len(candles) < 250: continue
        
        print(f"[*] Running LSOB V16 Backtest for {symbol_key}...")
        wins, losses, symbol_trades = 0, 0, 0
        in_position_until = 0

        for i in range(50, len(candles) - 30):
            if i < in_position_until: 
                continue

            sub = candles[:i+1]
            c = sub[-1]      # کندل جاری (تست بلوک سفارش)
            prev = sub[-2]   # کندل ایمپالس (حرکت ناگهانی نهنگی)
            prev2 = sub[-3]  # کندل شکار نقدینگی (Sweep)

            atr = calculate_atr(sub, 14)
            if atr == 0: continue

            # تعیین سقف و کف محلی برای تشخیص شکار نقدینگی
            recent_high = max(x["high"] for x in sub[-30:-3])
            recent_low = min(x["low"] for x in sub[-30:-3])

            avg_vol = sum(x["volume"] for x in sub[-20:-3]) / 17 if len(sub) >= 20 else 1.0
            volume_surge = prev["volume"] > (avg_vol * 1.5)

            # ۱. ستاپ لانگ: قیمت کفِ قبلی را جارو کرده (پایین‌تر از کف رفته و برگشته)، بعد با حجم بالا صعود کرده
            sweep_low = prev2["low"] < recent_low and prev2["close"] > recent_low
            bullish_mss = sweep_low and volume_surge and (prev["close"] > prev["open"])
            
            # ۲. ستاپ شورت: قیمت سقفِ قبلی را جارو کرده (بالاتر از سقف رفته و برگشته)، بعد با حجم بالا ریزش کرده
            sweep_high = prev2["high"] > recent_high and prev2["close"] < recent_high
            bearish_mss = sweep_high and volume_surge and (prev["close"] < prev["open"])

            trade_taken = False

            if bullish_mss:
                # ورود روی بلوک سفارش (Order Block - کندل نزولیِ قبل از حرکت صعودی)
                entry_price = c["close"]
                stop_loss = recent_low - (atr * 0.5)  # پشت سطح جارو شده با فاصله ایمن
                risk_dist = entry_price - stop_loss

                if 0.002 * entry_price <= risk_dist <= 0.05 * entry_price:
                    take_profit = entry_price + (risk_dist * TARGET_RR)
                    
                    trade_won, trade_lost = False, False
                    end_idx = len(candles) - 1
                    
                    for j in range(i + 1, len(candles)):
                        future_c = candles[j]
                        if future_c["high"] >= take_profit:
                            trade_won = True
                            end_idx = j
                            break
                        if future_c["low"] <= stop_loss:
                            trade_lost = True
                            end_idx = j
                            break

                    symbol_trades += 1
                    in_position_until = end_idx + 1
                    trade_taken = True
                    
                    if trade_won: 
                        wins += 1
                        balance += (risk_amount * TARGET_RR)
                    elif trade_lost: 
                        losses += 1
                        balance -= risk_amount

            elif bearish_mss and not trade_taken:
                entry_price = c["close"]
                stop_loss = recent_high + (atr * 0.5)
                risk_dist = stop_loss - entry_price

                if 0.002 * entry_price <= risk_dist <= 0.05 * entry_price:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    
                    trade_won, trade_lost = False, False
                    end_idx = len(candles) - 1
                    
                    for j in range(i + 1, len(candles)):
                        future_c = candles[j]
                        if future_c["low"] <= take_profit:
                            trade_won = True
                            end_idx = j
                            break
                        if future_c["high"] >= stop_loss:
                            trade_lost = True
                            end_idx = j
                            break

                    symbol_trades += 1
                    in_position_until = end_idx + 1
                    
                    if trade_won: 
                        wins += 1
                        balance += (risk_amount * TARGET_RR)
                    elif trade_lost: 
                        losses += 1
                        balance -= risk_amount

        total_wins += wins
        total_losses += losses
        grand_total_trades += symbol_trades
        print(f" > {symbol_key} -> Trades: {symbol_trades} | Wins: {wins} | Losses: {losses}")

    win_rate = (total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

    print("\n==================================================")
    print("      AGGREGATED LSOB V16 RESULTS                 ")
    print("==================================================")
    print(f"Total Trades       : {grand_total_trades}")
    print(f"Winning Trades     : {total_wins}")
    print(f"Losing Trades      : {total_losses}")
    print(f"Overall Win Rate   : {win_rate:.2f}%")
    print(f"Final Balance      : ${balance:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_backtest()
