import json
import urllib.request
import time

SYMBOLS = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD", "XRPUSDT": "XRP-USD"}
TARGET_RR = 2.5

def fetch_yahoo_one_year(symbol_key):
    yahoo_symbol = SYMBOLS[symbol_key]
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?interval=1h&range=1y"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    try:
        print(f"[*] Downloading 1-year Whale Liquidity data for {symbol_key}...")
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
    print("   SCORE HUNTER PRO - WHALE LIQUIDITY SWEEP       ")
    print("==================================================")

    for symbol_key in SYMBOLS.keys():
        candles = fetch_yahoo_one_year(symbol_key)
        if not candles or len(candles) < 100: continue
        
        print(f"[*] Running Liquidity Sweep backtest for {symbol_key}...")
        wins, losses, symbol_trades, skip_until = 0, 0, 0, 0

        for i in range(30, len(candles) - 30):
            if i < skip_until: continue

            sub = candles[:i+1]
            c = sub[-2]       # کندل فعلی تاییدیه
            prev_c = sub[-3]  # کندل نفوذ و شکار نقدینگی
            
            atr = calculate_atr(sub, 14)
            if atr == 0: continue

            # تعیین سقف و کف مهم در 20 کندل گذشته (محل استاپ ریتیل‌ها)
            recent_swing_high = max(x["high"] for x in sub[-22:-2])
            recent_swing_low = min(x["low"] for x in sub[-22:-2])

            vol_avg = sum(x["volume"] for x in sub[-15:-2]) / 14 if len(sub) >= 15 else 1.0
            volume_spike = prev_c["volume"] > (vol_avg * 1.5) # حجم سنگین نهنگی در لحظه شکار

            # 1. شکار کف (Bullish Sweep / Stop Hunt Long):
            # کندل قبل به زیر کف مهم نفوذ کرده (استاپ‌ها را زده)، اما قیمت به شدت پس‌زده شده و کندل فعلی صعودی و با حجم بالاست.
            bullish_sweep = (prev_c["low"] < recent_swing_low) and (c["close"] > recent_swing_low) and (c["close"] > c["open"]) and volume_spike

            # 2. شکار سقف (Bearish Sweep / Stop Hunt Short):
            # کندل قبل به بالای سقف مهم نفوذ کرده (استاپ شورت‌ها را زده)، اما قیمت پس‌زده شده و کندل فعلی نزولی و با حجم بالاست.
            bearish_sweep = (prev_c["high"] > recent_swing_high) and (c["close"] < recent_swing_high) and (c["close"] < c["open"]) and volume_spike

            trade_taken = False

            if bullish_sweep:
                entry_price = c["close"]
                stop_loss = prev_c["low"] - (atr * 0.2) # استاپ پشت پایین‌ترین نقطه شکار نقدینگی
                risk_dist = entry_price - stop_loss

                if 0 < (risk_dist / entry_price) <= 0.035:
                    take_profit = entry_price + (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 48, len(candles) - 1)
                    
                    for j in range(i + 1, end_idx + 1):
                        future_c = candles[j]
                        if future_c["low"] <= stop_loss: trade_lost = True; end_idx = j; break
                        if future_c["high"] >= take_profit: trade_won = True; end_idx = j; break
                    
                    if not trade_won and not trade_lost:
                        trade_won = True if candles[end_idx]["close"] > entry_price else False
                        trade_lost = not trade_won

                    symbol_trades += 1
                    skip_until = end_idx + 10
                    trade_taken = True
                    if trade_won: wins += 1; balance += (risk_amount * TARGET_RR)
                    elif trade_lost: losses += 1; balance -= risk_amount

            if not trade_taken and bearish_sweep:
                entry_price = c["close"]
                stop_loss = prev_c["high"] + (atr * 0.2) # استاپ پشت بالاترین نقطه شکار نقدینگی
                risk_dist = stop_loss - entry_price

                if 0 < (risk_dist / entry_price) <= 0.035:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 48, len(candles) - 1)
                    
                    for j in range(i + 1, end_idx + 1):
                        future_c = candles[j]
                        if future_c["high"] >= stop_loss: trade_lost = True; end_idx = j; break
                        if future_c["low"] <= take_profit: trade_won = True; end_idx = j; break
                    
                    if not trade_won and not trade_lost:
                        trade_won = True if candles[end_idx]["close"] < entry_price else False
                        trade_lost = not trade_won

                    symbol_trades += 1
                    skip_until = end_idx + 10
                    if trade_won: wins += 1; balance += (risk_amount * TARGET_RR)
                    elif trade_lost: losses += 1; balance -= risk_amount

        total_wins += wins
        total_losses += losses
        grand_total_trades += symbol_trades
        print(f" > {symbol_key} -> Trades: {symbol_trades} | Wins: {wins} | Losses: {losses}")

    win_rate = (total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

    print("\n==================================================")
    print("      AGGREGATED WHALE SWEEP RESULTS              ")
    print("==================================================")
    print(f"Total Trades       : {grand_total_trades}")
    print(f"Winning Trades     : {total_wins}")
    print(f"Losing Trades      : {total_losses}")
    print(f"Overall Win Rate   : {win_rate:.2f}%")
    print(f"Final Balance      : ${balance:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_backtest()
