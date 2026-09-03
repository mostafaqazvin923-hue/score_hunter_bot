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

def calculate_rsi(closes, period=14):
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    rsi_list = [50.0] * len(closes)
    seed = deltas[:period]
    up = sum(s for s in seed if s > 0) / period
    down = -sum(s for s in seed if s < 0) / period
    if down != 0:
        rs = up / down
        rsi_list[period] = 100 - (100 / (1 + rs))
    
    for i in range(period + 1, len(closes)):
        delta = deltas[i - 1]
        up = (up * (period - 1) + (delta if delta > 0 else 0)) / period
        down = (down * (period - 1) + (-delta if delta < 0 else 0)) / period
        if down == 0:
            rsi_list[i] = 100
        else:
            rs = up / down
            rsi_list[i] = 100 - (100 / (1 + rs))
    return rsi_list

def calculate_macd(closes, fast=12, slow=26, signal=9):
    exp1 = calculate_ema(closes, fast)
    exp2 = calculate_ema(closes, slow)
    macd_line = [exp1[i] - exp2[i] for i in range(len(closes))]
    signal_line = calculate_ema(macd_line, signal)
    return macd_line, signal_line

def run_backtest():
    initial_balance = 1000.0
    balance = initial_balance
    risk_percentage = 0.01
    
    total_wins = 0
    total_losses = 0
    grand_total_trades = 0

    print("==================================================")
    print(" SCORE HUNTER PRO - V12.1 (CLEAN & FIXED)         ")
    print("==================================================")

    for symbol in SYMBOLS:
        try:
            candles = fetch_1year_15m_auto(symbol)
            if len(candles) < 100:
                continue
        except Exception:
            continue
        
        closes = [c["close"] for c in candles]
        ema_50 = calculate_ema(closes, 50)
        rsi_vals = calculate_rsi(closes, 14)
        macd_line, signal_line = calculate_macd(closes)

        wins = 0
        losses = 0
        symbol_trades = 0
        skip_until = 0

        for i in range(50, len(candles) - 30):
            if i < skip_until:
                continue

            c = candles[i]
            prev_c = candles[i-1]
            trend_val = ema_50[i]
            current_rsi = rsi_vals[i]
            m_line = macd_line[i]
            s_line = signal_line[i]
            prev_m = macd_line[i-1]
            prev_s = signal_line[i-1]

            current_risk_amount = balance * risk_percentage
            trade_taken = False

            # --- استراتژی لانگ ---
            is_uptrend = c["close"] > trend_val
            is_macd_cross_up = prev_m <= prev_s and m_line > s_line
            is_rsi_buy = 40 <= current_rsi <= 65

            if is_uptrend and is_macd_cross_up and is_rsi_buy:
                entry_price = c["close"]
                stop_loss = min(prev_c["low"], candles[i-2]["low"]) - (entry_price * 0.0005)
                risk_dist = entry_price - stop_loss

                if risk_dist > 0:
                    take_profit = entry_price + (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 36, len(candles) - 1)

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
                            wins += 1
                            balance += (current_risk_amount * TARGET_RR)
                        else:
                            losses += 1
                            balance -= current_risk_amount

            # --- استراتژی شورت ---
            is_downtrend = c["close"] < trend_val
            is_macd_cross_down = prev_m >= prev_s and m_line < s_line
            is_rsi_sell = 35 <= current_rsi <= 60

            if not trade_taken and is_downtrend and is_macd_cross_down and is_rsi_sell:
                entry_price = c["close"]
                stop_loss = max(prev_c["high"], candles[i-2]["high"]) + (entry_price * 0.0005)
                risk_dist = stop_loss - entry_price

                if risk_dist > 0:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 36, len(candles) - 1)

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
    print("      AGGREGATED V12.1 RESULTS                    ")
    print("==================================================\n")
    print(f"Total Trades       : {grand_total_trades}")
    print(f"Winning Trades     : {total_wins}")
    print(f"Losing Trades      : {total_losses}")
    print(f"Overall Win Rate   : {win_rate:.2f}%")
    print(f"Final Balance      : ${balance:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_backtest()
