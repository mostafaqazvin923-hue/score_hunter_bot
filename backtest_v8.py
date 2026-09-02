import json
import urllib.request
import math

SYMBOLS = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD", "XRPUSDT": "XRP-USD"}
TARGET_RR = 2.0  # دقیقاً دو برابر حد ضرر

def fetch_yahoo_one_year(symbol_key):
    yahoo_symbol = SYMBOLS[symbol_key]
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?interval=1h&range=1y"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    try:
        print(f"[*] Downloading DVVE Market Data for {symbol_key}...")
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

def calculate_bollinger_bands(closes, period=20, std_dev=2.0):
    if len(closes) < period:
        return closes[-1], closes[-1], closes[-1]
    sub = closes[-period:]
    sma = sum(sub) / period
    variance = sum((x - sma) ** 2 for x in sub) / period
    stdev = math.sqrt(variance)
    upper = sma + (std_dev * stdev)
    lower = sma - (std_dev * stdev)
    return upper, sma, lower

def run_backtest():
    initial_balance = 1000.0
    balance = initial_balance
    risk_amount = 25.0
    total_wins, total_losses, grand_total_trades = 0, 0, 0

    print("==================================================")
    print(" DVVE V14.1 - STRATEGY WITH 2R & RUNWAY CHECK     ")
    print("==================================================")

    for symbol_key in SYMBOLS.keys():
        candles = fetch_yahoo_one_year(symbol_key)
        if not candles or len(candles) < 250: continue
        
        print(f"[*] Running DVVE V14.1 Backtest for {symbol_key}...")
        wins, losses, symbol_trades = 0, 0, 0
        in_position_until = 0

        for i in range(100, len(candles) - 30):
            if i < in_position_until: 
                continue

            sub = candles[:i+1]
            closes = [x["close"] for x in sub]
            volumes = [x["volume"] for x in sub]
            
            c = sub[-1]
            
            atr = calculate_atr(sub, 14)
            upper_bb, mid_bb, lower_bb = calculate_bollinger_bands(closes, 20, 2.0)
            bb_width = (upper_bb - lower_bb) / mid_bb

            if atr == 0: continue

            # تشخیص فشردگی باندهای بولینگر
            is_squeezed = bb_width < 0.035

            recent_highs = [x["high"] for x in sub[-50:-2]]
            recent_lows = [x["low"] for x in sub[-50:-2]]
            major_resistance = max(recent_highs) if recent_highs else c["high"] * 1.05
            major_support = min(recent_lows) if recent_lows else c["low"] * 0.95

            avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else 1.0
            volume_spike = c["volume"] > (avg_vol * 1.2)

            long_breakout = is_squeezed and volume_spike and (c["close"] > upper_bb) and (c["close"] > c["open"])
            short_breakout = is_squeezed and volume_spike and (c["close"] < lower_bb) and (c["close"] < c["open"])

            trade_taken = False

            if long_breakout:
                entry_price = c["close"]
                stop_loss = entry_price - (atr * 1.5)
                risk_dist = entry_price - stop_loss

                if 0.002 * entry_price <= risk_dist <= 0.05 * entry_price:
                    take_profit = entry_price + (risk_dist * TARGET_RR)
                    
                    # فیلتر فضای مسیر (آیا هدف ۲ برابری تا مقاومت فاصله کافی دارد؟)
                    has_enough_room = major_resistance >= (take_profit - (risk_dist * 0.2))

                    if has_enough_room:
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

            elif short_breakout and not trade_taken:
                entry_price = c["close"]
                stop_loss = entry_price + (atr * 1.5)
                risk_dist = stop_loss - entry_price

                if 0.002 * entry_price <= risk_dist <= 0.05 * entry_price:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    
                    # فیلتر فضای مسیر برای شورت
                    has_enough_room = major_support <= (take_profit + (risk_dist * 0.2))

                    if has_enough_room:
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
    print("      AGGREGATED DVVE V14.1 RESULTS               ")
    print("==================================================")
    print(f"Total Trades       : {grand_total_trades}")
    print(f"Winning Trades     : {total_wins}")
    print(f"Losing Trades      : {total_losses}")
    print(f"Overall Win Rate   : {win_rate:.2f}%")
    print(f"Final Balance      : ${balance:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_backtest()
