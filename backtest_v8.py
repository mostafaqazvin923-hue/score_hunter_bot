import json
import urllib.request
import math

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
TIMEFRAME = "1h"
TARGET_RR = 2.0

def fetch_historical_klines(symbol, limit=8800):
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
                op = opens[idx]
                hi = highs[idx]
                lo = lows[idx]
                cl = closes[idx]
                ts = timestamps[idx]
                vol = volumes[idx] if idx < len(volumes) and volumes[idx] is not None else 1000.0
                
                if op is None or hi is None or lo is None or cl is None:
                    continue
                candles.append({
                    "timestamp": int(ts), 
                    "open": float(op), 
                    "high": float(hi), 
                    "low": float(lo), 
                    "close": float(cl),
                    "volume": float(vol)
                })
            
            candles.sort(key=lambda x: x["timestamp"])
            if len(candles) > 0:
                return candles[-limit:]
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
    
    raise ValueError(f"Could not fetch real historical data for {symbol}.")

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

def run_backtest():
    initial_balance = 1000.0
    balance = initial_balance
    risk_amount = 25.0
    
    total_wins = 0
    total_losses = 0
    grand_total_trades = 0

    print("==================================================")
    print("   SCORE HUNTER PRO - STRICT SMC & NO-OVERLAP     ")
    print("==================================================")

    for symbol in SYMBOLS:
        print(f"\n[*] Running strict institutional backtest for {symbol}...")
        try:
            candles = fetch_historical_klines(symbol, limit=8800)
        except Exception as err:
            print(err)
            continue
        
        wins = 0
        losses = 0
        symbol_trades = 0
        skip_until = 0  # جلوگیری از باز کردن پوزیشن همزمان و سیگنال‌های تکراری روی یک کندل

        for i in range(25, len(candles) - 1):
            if i < skip_until:
                continue

            sub_candles = candles[:i+1]
            c = sub_candles[-2]
            prev_c = sub_candles[-3]
            prev2_c = sub_candles[-4]

            # فیلتر حجم برای تایید ورود واقعی
            avg_vol = sum(x["volume"] for x in sub_candles[-15:-2]) / 14 if len(sub_candles) >= 15 else 1.0
            is_volume_confirmed = c["volume"] > (avg_vol * 1.05) if avg_vol > 0 else True

            # ساختار BOS و FVG صعودی
            recent_highs = max(x["high"] for x in sub_candles[-15:-2])
            is_bullish_bos = c["close"] > recent_highs and (c["close"] - c["open"]) > (c["high"] - c["low"]) * 0.45
            has_bullish_fvg = prev2_c["high"] < c["low"]

            # ساختار BOS و FVG نزولی
            recent_lows = min(x["low"] for x in sub_candles[-15:-2])
            is_bearish_bos = c["close"] < recent_lows and (c["open"] - c["close"]) > (c["high"] - c["low"]) * 0.45
            has_bearish_fvg = prev2_c["low"] > c["high"]

            current_rsi = calculate_rsi(sub_candles)
            trade_taken = False

            # بررسی پوزیشن LONG
            if is_bullish_bos and has_bullish_fvg and is_volume_confirmed and (45 < current_rsi < 68):
                entry_price = c["close"]
                stop_loss = min(prev_c["low"], prev2_c["low"]) - (entry_price * 0.0015)
                risk_dist = entry_price - stop_loss

                if 0 < (risk_dist / entry_price) <= 0.025:
                    take_profit = entry_price + (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 24, len(candles) - 1)
                    
                    for j in range(i + 1, end_idx + 1):
                        future_c = candles[j]
                        if future_c["low"] <= stop_loss:
                            trade_lost = True
                            end_idx = j
                            break
                        if future_c["high"] >= take_profit:
                            trade_won = True
                            end_idx = j
                            break
                    
                    if not trade_won and not trade_lost:
                        if candles[end_idx]["close"] > entry_price:
                            trade_won = True
                        else:
                            trade_lost = True

                    symbol_trades += 1
                    skip_until = end_idx  # قفل کردن لوپ تا بسته شدن کامل این معامله
                    trade_taken = True

                    if trade_won:
                        wins += 1
                        balance += (risk_amount * TARGET_RR)
                    elif trade_lost:
                        losses += 1
                        balance -= risk_amount

            # بررسی پوزیشن SHORT (اگر لانگ گرفته نشده باشد)
            if not trade_taken and is_bearish_bos and has_bearish_fvg and is_volume_confirmed and (32 < current_rsi < 55):
                entry_price = c["close"]
                stop_loss = max(prev_c["high"], prev2_c["high"]) + (entry_price * 0.0015)
                risk_dist = stop_loss - entry_price

                if 0 < (risk_dist / entry_price) <= 0.025:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 24, len(candles) - 1)
                    
                    for j in range(i + 1, end_idx + 1):
                        future_c = candles[j]
                        if future_c["high"] >= stop_loss:
                            trade_lost = True
                            end_idx = j
                            break
                        if future_c["low"] <= take_profit:
                            trade_won = True
                            end_idx = j
                            break
                    
                    if not trade_won and not trade_lost:
                        if candles[end_idx]["close"] < entry_price:
                            trade_won = True
                        else:
                            trade_lost = True

                    symbol_trades += 1
                    skip_until = end_idx  # قفل کردن لوپ تا بسته شدن کامل این معامله

                    if trade_won:
                        wins += 1
                        balance += (risk_amount * TARGET_RR)
                    elif trade_lost:
                        losses += 1
                        balance -= risk_amount

        total_wins += wins
        total_losses += losses
        grand_total_trades += symbol_trades
        print(f" > {symbol} -> Trades: {symbol_trades} | Wins: {wins} | Losses: {losses}")

    win_rate = (total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

    print("\n==================================================")
    print("         AGGREGATED STRICT SMC RESULTS            ")
    print("==================================================")
    print(f"Total Coins Tested : {len(SYMBOLS)}")
    print(f"Total Trades       : {grand_total_trades}")
    print(f"Winning Trades     : {total_wins}")
    print(f"Losing Trades      : {total_losses}")
    print(f"Overall Win Rate   : {win_rate:.2f}%")
    print(f"Final Balance      : ${balance:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_backtest()
