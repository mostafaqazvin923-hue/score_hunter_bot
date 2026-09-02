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
                op, hi, lo, cl, ts = opens[idx], highs[idx], lows[idx], closes[idx], timestamps[idx]
                vol = volumes[idx] if idx < len(volumes) and volumes[idx] is not None else 1000.0
                if op is None or hi is None or lo is None or cl is None:
                    continue
                candles.append({
                    "timestamp": int(ts), "open": float(op), "high": float(hi), 
                    "low": float(lo), "close": float(cl), "volume": float(vol)
                })
            candles.sort(key=lambda x: x["timestamp"])
            if len(candles) > 0:
                return candles[-limit:]
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
    raise ValueError(f"Could not fetch real historical data for {symbol}.")

def calculate_ema(candles, period=200):
    if len(candles) < period:
        return candles[-1]["close"]
    multiplier = 2 / (period + 1)
    ema = sum(x["close"] for x in candles[:period]) / period
    for candle in candles[period:]:
        ema = (candle["close"] - ema) * multiplier + ema
    return ema

def calculate_rsi(candles, period=14):
    if len(candles) < period + 1:
        return 50.0
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        diff = candles[-i]["close"] - candles[-i-1]["close"]
        if diff >= 0: gains += diff
        else: losses -= diff
    if losses == 0: return 100.0
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
    print("   SCORE HUNTER PRO - TREND FILTERED INSTITUTIONAL")
    print("==================================================")

    for symbol in SYMBOLS:
        print(f"\n[*] Running trend-filtered backtest for {symbol}...")
        try:
            candles = fetch_historical_klines(symbol, limit=8800)
        except Exception as err:
            print(err)
            continue
        
        wins = 0
        losses = 0
        symbol_trades = 0
        skip_until = 0

        for i in range(200, len(candles) - 1):
            if i < skip_until:
                continue

            sub_candles = candles[:i+1]
            c = sub_candles[-2]
            prev_c = sub_candles[-3]
            prev2_c = sub_candles[-4]

            ema_200 = calculate_ema(sub_candles, 200)
            current_rsi = calculate_rsi(sub_candles)
            
            avg_vol = sum(x["volume"] for x in sub_candles[-15:-2]) / 14
            is_volume_confirmed = c["volume"] > (avg_vol * 1.1)

            recent_highs = max(x["high"] for x in sub_candles[-20:-2])
            recent_lows = min(x["low"] for x in sub_candles[-20:-2])
            
            is_bullish_bos = c["close"] > recent_highs and (c["close"] - c["open"]) > (c["high"] - c["low"]) * 0.5
            is_bearish_bos = c["close"] < recent_lows and (c["open"] - c["close"]) > (c["high"] - c["low"]) * 0.5

            trade_taken = False

            # فقط در روند صعودی (بالای EMA 200) لانگ می‌گیریم
            if c["close"] > ema_200 and is_bullish_bos and is_volume_confirmed and (45 < current_rsi < 68):
                entry_price = c["close"]
                stop_loss = min(prev_c["low"], prev2_c["low"]) - (entry_price * 0.002)
                risk_dist = entry_price - stop_loss

                if 0 < (risk_dist / entry_price) <= 0.03:
                    take_profit = entry_price + (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 36, len(candles) - 1)
                    
                    for j in range(i + 1, end_idx + 1):
                        future_c = candles[j]
                        if future_c["low"] <= stop_loss:
                            trade_lost = True; end_idx = j; break
                        if future_c["high"] >= take_profit:
                            trade_won = True; end_idx = j; break
                    
                    if not trade_won and not trade_lost:
                        trade_won = True if candles[end_idx]["close"] > entry_price else False
                        trade_lost = not trade_won

                    symbol_trades += 1
                    skip_until = end_idx
                    trade_taken = True

                    if trade_won: wins += 1; balance += (risk_amount * TARGET_RR)
                    elif trade_lost: losses += 1; balance -= risk_amount

            # فقط در روند نزولی (پایین EMA 200) شورت می‌گیریم
            if not trade_taken and c["close"] < ema_200 and is_bearish_bos and is_volume_confirmed and (32 < current_rsi < 55):
                entry_price = c["close"]
                stop_loss = max(prev_c["high"], prev2_c["high"]) + (entry_price * 0.002)
                risk_dist = stop_loss - entry_price

                if 0 < (risk_dist / entry_price) <= 0.03:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 36, len(candles) - 1)
                    
                    for j in range(i + 1, end_idx + 1):
                        future_c = candles[j]
                        if future_c["high"] >= stop_loss:
                            trade_lost = True; end_idx = j; break
                        if future_c["low"] <= take_profit:
                            trade_won = True; end_idx = j; break
                    
                    if not trade_won and not trade_lost:
                        trade_won = True if candles[end_idx]["close"] < entry_price else False
                        trade_lost = not trade_won

                    symbol_trades += 1
                    skip_until = end_idx

                    if trade_won: wins += 1; balance += (risk_amount * TARGET_RR)
                    elif trade_lost: losses += 1; balance -= risk_amount

        total_wins += wins
        total_losses += losses
        grand_total_trades += symbol_trades
        print(f" > {symbol} -> Trades: {symbol_trades} | Wins: {wins} | Losses: {losses}")

    win_rate = (total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

    print("\n==================================================")
    print("      AGGREGATED TREND-FILTERED RESULTS           ")
    print("==================================================")
    print(f"Total Trades       : {grand_total_trades}")
    print(f"Winning Trades     : {total_wins}")
    print(f"Losing Trades      : {total_losses}")
    print(f"Overall Win Rate   : {win_rate:.2f}%")
    print(f"Final Balance      : ${balance:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_backtest()
