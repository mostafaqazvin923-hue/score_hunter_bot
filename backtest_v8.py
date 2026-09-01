import json
import urllib.request
import urllib.parse

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
TIMEFRAME = "1h"
TARGET_RR = 2.0

def fetch_real_klines_yahoo(symbol, limit=8800):
    # استفاده از یاهو فایننس برای دریافت داده‌های واقعی تاریخی ۱ ساعته بدون تحریم و خطای صرافی
    # هر کندل ۱ ساعته تقریباً معادل است با بازه زمانی مشخص
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1h&range=730d"
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
            payload = json.loads(raw)
            result = payload.get("chart", {}).get("result", [])
            if not result:
                raise ValueError("Empty result from Yahoo Finance")
            
            data = result[0]
            timestamps = data.get("timestamp", [])
            quotes = data.get("indicators", {}).get("quote", [{}])[0]
            
            opens = quotes.get("open", [])
            highs = quotes.get("high", [])
            lows = quotes.get("low", [])
            closes = quotes.get("close", [])
            
            candles = []
            for idx in range(len(timestamps)):
                op = opens[idx]
                hi = highs[idx]
                lo = lows[idx]
                cl = closes[idx]
                ts = timestamps[idx]
                
                # حذف داده‌های نامعتبر (None)
                if op is None or hi is None or lo is None or cl is None:
                    continue
                
                candles.append({
                    "timestamp": int(ts),
                    "open": float(op),
                    "high": float(hi),
                    "low": float(lo),
                    "close": float(cl)
                })
            
            candles.sort(key=lambda x: x["timestamp"])
            if len(candles) > 100:
                print(f"[*] Successfully fetched {len(candles)} real candles for {symbol}")
                return candles[-limit:]
    except Exception as e:
        print(f"[!] Error fetching data for {symbol}: {e}")
    
    raise ValueError(f"[!] Could not fetch real historical data for {symbol}.")

def calculate_rsi(candles, period=14):
    if len(candles) < period + 1:
        return 50.0
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        diff = candles[-i]["close"] - candles[-i-1]["close"]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - (100.0 / (1.0 + rs))

def calculate_sma(candles, period=50):
    if len(candles) < period:
        return candles[-1]["close"]
    return sum(x["close"] for x in candles[-period:]) / period

def run_backtest():
    initial_balance = 1000.0
    balance = initial_balance
    risk_amount = 25.0
    
    total_wins = 0
    total_losses = 0
    grand_total_trades = 0
    grand_total_longs = 0
    grand_total_shorts = 0

    print("==================================================")
    print("   SCORE HUNTER PRO - REAL DATA YAHOO BACKTEST    ")
    print("==================================================")

    for symbol in SYMBOLS:
        try:
            candles = fetch_real_klines_yahoo(symbol, limit=8800)
        except Exception as err:
            print(err)
            continue
        
        wins = 0
        losses = 0
        symbol_trades = 0
        symbol_longs = 0
        symbol_shorts = 0
        skip_until = 0

        for i in range(50, len(candles) - 1):
            if i < skip_until:
                continue

            sub_candles = candles[:i+1]
            c = sub_candles[-2]
            prev_c = sub_candles[-3]
            prev2_c = sub_candles[-4]

            current_rsi = calculate_rsi(sub_candles)
            trend_sma = calculate_sma(sub_candles, period=50)
            trade_taken = False

            # --- شورت واقعی ---
            is_down_trend = c["close"] < trend_sma
            is_short_momentum = (c["close"] < c["open"]) and ((c["open"] - c["close"]) > (c["high"] - c["low"]) * 0.4)

            if is_down_trend and is_short_momentum and (current_rsi < 45):
                entry_price = c["close"]
                stop_loss = max(prev_c["high"], prev2_c["high"]) + (entry_price * 0.001)
                risk_dist = stop_loss - entry_price

                if risk_dist > 0 and (risk_dist / entry_price) <= 0.04:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    trade_won = False
                    trade_lost = False
                    end_idx = min(i + 24, len(candles) - 1)
                    
                    for j in range(i + 1, end_idx + 1):
                        future_c = candles[j]
                        hit_sl = future_c["high"] >= stop_loss
                        hit_tp = future_c["low"] <= take_profit

                        if hit_sl and hit_tp:
                            trade_lost = True
                            break
                        elif hit_sl:
                            trade_lost = True
                            break
                        elif hit_tp:
                            trade_won = True
                            break
                    
                    if not trade_won and not trade_lost:
                        if candles[end_idx]["close"] < entry_price:
                            trade_won = True
                        else:
                            trade_lost = True

                    symbol_trades += 1
                    symbol_shorts += 1
                    grand_total_shorts += 1
                    skip_until = end_idx
                    trade_taken = True

                    if trade_won:
                        wins += 1
                        balance += (risk_amount * TARGET_RR)
                    elif trade_lost:
                        losses += 1
                        balance -= risk_amount

            # --- لانگ واقعی ---
            if not trade_taken:
                is_up_trend = c["close"] > trend_sma
                is_long_momentum = (c["close"] > c["open"]) and ((c["close"] - c["open"]) > (c["high"] - c["low"]) * 0.4)

                if is_up_trend and is_long_momentum and (current_rsi > 55):
                    entry_price = c["close"]
                    stop_loss = min(prev_c["low"], prev2_c["low"]) - (entry_price * 0.001)
                    risk_dist = entry_price - stop_loss

                    if risk_dist > 0 and (risk_dist / entry_price) <= 0.04:
                        take_profit = entry_price + (risk_dist * TARGET_RR)
                        trade_won = False
                        trade_lost = False
                        end_idx = min(i + 24, len(candles) - 1)
                        
                        for j in range(i + 1, end_idx + 1):
                            future_c = candles[j]
                            hit_sl = future_c["low"] <= stop_loss
                            hit_tp = future_c["high"] >= take_profit

                            if hit_sl and hit_tp:
                                trade_lost = True
                                break
                            elif hit_sl:
                                trade_lost = True
                                break
                            elif hit_tp:
                                trade_won = True
                                break
                        
                        if not trade_won and not trade_lost:
                            if candles[end_idx]["close"] > entry_price:
                                trade_won = True
                            else:
                                trade_lost = True

                        symbol_trades += 1
                        symbol_longs += 1
                        grand_total_longs += 1
                        skip_until = end_idx

                        if trade_won:
                            wins += 1
                            balance += (risk_amount * TARGET_RR)
                        elif trade_lost:
                            losses += 1
                            balance -= risk_amount

        total_wins += wins
        total_losses += losses
        grand_total_trades += symbol_trades
        print(f" > {symbol} -> Total: {symbol_trades} (Longs: {symbol_longs}, Shorts: {symbol_shorts}) | Wins: {wins} | Losses: {losses}")

    win_rate = (total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

    print("\n==================================================")
    print("             AGGREGATED 1-YEAR RESULTS            ")
    print("==================================================")
    print(f"Total Trades       : {grand_total_trades}")
    print(f"  - Total Longs    : {grand_total_longs}")
    print(f"  - Total Shorts   : {grand_total_shorts}")
    print(f"Winning Trades     : {total_wins}")
    print(f"Losing Trades      : {total_losses}")
    print(f"Overall Win Rate   : {win_rate:.2f}%")
    print(f"Final Balance      : ${balance:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_backtest()
