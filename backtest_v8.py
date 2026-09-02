import json
import urllib.request
import time

SYMBOLS = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD", "XRPUSDT": "XRP-USD"}
TARGET_RR = 2.0

def fetch_yahoo_one_year(symbol_key):
    yahoo_symbol = SYMBOLS[symbol_key]
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?interval=1h&range=1y"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    try:
        print(f"[*] Downloading 1-year Balanced Data for {symbol_key}...")
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

def calculate_ema(closes, period=50):
    if len(closes) < period: return closes[-1]
    multiplier = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for price in closes[period:]:
        ema = (price - ema) * multiplier + ema
    return ema

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1: return 50.0
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        diff = closes[-i] - closes[-i-1]
        if diff >= 0: gains += diff
        else: losses -= diff
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def run_backtest():
    initial_balance = 1000.0
    balance = initial_balance
    risk_amount = 25.0
    total_wins, total_losses, grand_total_trades = 0, 0, 0

    print("==================================================")
    print("   SCORE HUNTER PRO - BALANCED V7 (OPTIMIZED)     ")
    print("==================================================")

    for symbol_key in SYMBOLS.keys():
        candles = fetch_yahoo_one_year(symbol_key)
        if not candles or len(candles) < 100: continue
        
        print(f"[*] Running V7 Balanced backtest for {symbol_key}...")
        wins, losses, symbol_trades = 0, 0, 0
        in_position_until = 0

        for i in range(50, len(candles) - 30):
            # قانون قفل معامله روی هر ارز تا روشن شدن تکلیف پوزیشن قبلی همان ارز
            if i < in_position_until: 
                continue

            sub = candles[:i+1]
            closes = [x["close"] for x in sub]
            
            c = sub[-1]
            prev_c = sub[-2]
            
            atr = calculate_atr(sub, 14)
            ema_50 = calculate_ema(closes, 50)
            rsi = calculate_rsi(closes, 14)
            
            if atr == 0: continue

            # تعیین روند با EMA 50
            is_uptrend = c["close"] > ema_50
            is_downtrend = c["close"] < ema_50

            # اسوینگ‌های پویای کوتاه‌مدت برای افزایش تعداد معاملات به حد استاندارد روزانه
            recent_swing_high = max(x["high"] for x in sub[-10:-1])
            recent_swing_low = min(x["low"] for x in sub[-10:-1])

            # شرایط بهینه‌شده ورود (توازن بین فرکانس بالا و وین‌ریت مناسب)
            buy_signal = is_uptrend and (prev_c["low"] <= recent_swing_low) and (c["close"] > c["open"]) and (rsi < 50)
            sell_signal = is_downtrend and (prev_c["high"] >= recent_swing_high) and (c["close"] < c["open"]) and (rsi > 50)

            trade_taken = False

            if buy_signal:
                entry_price = c["close"]
                stop_loss = recent_swing_low - (atr * 0.4)
                risk_dist = entry_price - stop_loss

                if 0.001 * entry_price <= risk_dist <= 0.04 * entry_price:
                    take_profit = entry_price + (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = len(candles) - 1
                    
                    for j in range(i + 1, len(candles)):
                        future_c = candles[j]
                        if future_c["low"] <= stop_loss:
                            trade_lost = True
                            end_idx = j
                            break
                        if future_c["high"] >= take_profit:
                            trade_won = True
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

            elif sell_signal and not trade_taken:
                entry_price = c["close"]
                stop_loss = recent_swing_high + (atr * 0.4)
                risk_dist = stop_loss - entry_price

                if 0.001 * entry_price <= risk_dist <= 0.04 * entry_price:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = len(candles) - 1
                    
                    for j in range(i + 1, len(candles)):
                        future_c = candles[j]
                        if future_c["high"] >= stop_loss:
                            trade_lost = True
                            end_idx = j
                            break
                        if future_c["low"] <= take_profit:
                            trade_won = True
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
    print("      AGGREGATED BALANCED V7 RESULTS              ")
    print("==================================================")
    print(f"Total Trades       : {grand_total_trades}")
    print(f"Winning Trades     : {total_wins}")
    print(f"Losing Trades      : {total_losses}")
    print(f"Overall Win Rate   : {win_rate:.2f}%")
    print(f"Final Balance      : ${balance:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_backtest()
