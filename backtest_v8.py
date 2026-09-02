import json
import urllib.request
import math

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
TIMEFRAME = "1hour"
TARGET_RR = 2.0

def fetch_coinex_klines(symbol, limit=1000):
    # استفاده از اندپوینت پایدار و رسمی V1 صرافی کوینکس
    url = f"https://api.coinex.com/v1/market/kline?market={symbol}&type={TIMEFRAME}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
            payload = json.loads(raw)
            if isinstance(payload, dict) and payload.get("code") == 0:
                rows = payload.get("data", [])
                candles = []
                for row in rows:
                    if isinstance(row, list) and len(row) >= 6:
                        ts = int(float(row[0]))
                        op = float(row[1])
                        cl = float(row[2])
                        hi = float(row[3])
                        lo = float(row[4])
                        vol = float(row[5]) if len(row) > 5 else 0.0
                        candles.append({
                            "timestamp": ts, "open": op, "high": hi, 
                            "low": lo, "close": cl, "volume": vol
                        })
                candles.sort(key=lambda x: x["timestamp"])
                if len(candles) > 50:
                    return candles
    except Exception as e:
        print(f"[!] Error fetching Coinex data for {symbol}: {e}")
    raise ValueError(f"[!] Could not fetch real data from Coinex for {symbol}.")

def calculate_ema(candles, period):
    if len(candles) < period:
        return candles[-1]["close"]
    multiplier = 2 / (period + 1)
    ema = sum(x["close"] for x in candles[:period]) / period
    for c in candles[period:]:
        ema = (c["close"] - ema) * multiplier + ema
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

def calculate_atr(candles, period=14):
    if len(candles) < period + 1:
        return candles[-1]["high"] - candles[-1]["low"]
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]
        l = candles[i]["low"]
        pc = candles[i-1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return sum(trs[-period:]) / period

def calculate_adx_proxy(candles, period=14):
    if len(candles) < period + 2:
        return 25.0
    moves = []
    for i in range(1, len(candles[-period:])):
        diff = candles[-i]["close"] - candles[-i-1]["close"]
        moves.append(abs(diff))
    avg_move = sum(moves) / len(moves) if moves else 1.0
    total_range = candles[-1]["high"] - candles[-1]["low"]
    return min(100.0, max(10.0, (avg_move / (total_range if total_range > 0 else 1.0)) * 50 + 20))

def run_backtest():
    initial_balance = 1000.0
    balance = initial_balance
    risk_amount = 25.0
    
    total_wins = 0
    total_losses = 0
    grand_total_trades = 0

    print("==================================================")
    print("   SCORE HUNTER PRO - 3-LAYER WHALE SYSTEM (COINEX)")
    print("==================================================")

    for symbol in SYMBOLS:
        print(f"\n[*] Running backtest for {symbol} on Coinex...")
        try:
            candles = fetch_coinex_klines(symbol, limit=1000)
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

            sub = candles[:i+1]
            c = sub[-2]
            prev_c = sub[-3]
            prev2_c = sub[-4]

            # لایه ۱: روند (EMA 50 / 200 + ساختار)
            ema50 = calculate_ema(sub, 50)
            ema200 = calculate_ema(sub, 200)
            
            # لایه ۲: مومنتوم، قدرت و نوسان (RSI, ADX, ATR)
            rsi = calculate_rsi(sub, 14)
            adx = calculate_adx_proxy(sub, 14)
            atr = calculate_atr(sub, 14)

            # لایه ۳: جریان سفارشات و نقدینگی (CVD & Volume Proxy)
            candle_range = c["high"] - c["low"]
            if candle_range == 0 or atr == 0:
                continue
            
            buying_pressure = (c["close"] - c["open"]) / candle_range
            vol_avg = sum(x["volume"] for x in sub[-15:-2]) / 14 if len(sub) >= 15 else 1.0
            cvd_confirmed = c["volume"] > (vol_avg * 1.05)

            recent_highs = max(x["high"] for x in sub[-20:-2])
            recent_lows = min(x["low"] for x in sub[-20:-2])

            is_long = (c["close"] > ema50) and (ema50 > ema200) and (adx > 22) and (45 < rsi < 70) and (buying_pressure > 0.35) and cvd_confirmed and (c["close"] > recent_highs)
            is_short = (c["close"] < ema50) and (ema50 < ema200) and (adx > 22) and (30 < rsi < 55) and (buying_pressure < -0.35) and cvd_confirmed and (c["close"] < recent_lows)

            trade_taken = False

            if is_long:
                entry_price = c["close"]
                stop_loss = min(prev_c["low"], prev2_c["low"]) - (atr * 0.3)
                risk_dist = entry_price - stop_loss

                if 0 < (risk_dist / entry_price) <= 0.035:
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
                    skip_until = end_idx  # قفل کامل معامله تا تعیین تکلیف نهایی
                    trade_taken = True

                    if trade_won: wins += 1; balance += (risk_amount * TARGET_RR)
                    elif trade_lost: losses += 1; balance -= risk_amount

            if not trade_taken and is_short:
                entry_price = c["close"]
                stop_loss = max(prev_c["high"], prev2_c["high"]) + (atr * 0.3)
                risk_dist = stop_loss - entry_price

                if 0 < (risk_dist / entry_price) <= 0.035:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 36, len(candles) - 1)
                    
                    for j in range(i + 1, end_idx + 1):
                        future_c = candles[j]
                        if future_c["high"] >= stop_loss:
                            trade_lost = True; end_idx = j; break
                        if future_c["low"] >= take_profit:
                            trade_won = True; end_idx = j; break
                    
                    if not trade_won and not trade_lost:
                        trade_won = True if candles[end_idx]["close"] < entry_price else False
                        trade_lost = not trade_won

                    symbol_trades += 1
                    skip_until = end_idx  # قفل کامل معامله تا تعیین تکلیف نهایی

                    if trade_won: wins += 1; balance += (risk_amount * TARGET_RR)
                    elif trade_lost: losses += 1; balance -= risk_amount

        total_wins += wins
        total_losses += losses
        grand_total_trades += symbol_trades
        print(f" > {symbol} -> Trades: {symbol_trades} | Wins: {wins} | Losses: {losses}")

    win_rate = (total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

    print("\n==================================================")
    print("      AGGREGATED 3-LAYER WHALE BACKTEST RESULTS   ")
    print("==================================================")
    print(f"Total Trades       : {grand_total_trades}")
    print(f"Winning Trades     : {total_wins}")
    print(f"Losing Trades      : {total_losses}")
    print(f"Overall Win Rate   : {win_rate:.2f}%")
    print(f"Final Balance      : ${balance:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_backtest()
