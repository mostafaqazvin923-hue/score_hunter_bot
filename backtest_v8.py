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
        print(f"[*] Downloading V13 Sniper Data for {symbol_key}...")
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

def calculate_ema(closes, period):
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

def calculate_adx(candles, period=14):
    if len(candles) < period * 2: return 25.0
    tr_list, plus_dm_list, minus_dm_list = [], [], []
    
    for i in range(1, len(candles)):
        h = candles[i]["high"]; l = candles[i]["low"]; pc = candles[i-1]["close"]
        ph = candles[i-1]["high"]; pl = candles[i-1]["low"]
        
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_list.append(tr)
        
        up_move = h - ph
        down_move = pl - l
        
        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0
        
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)
        
    if len(tr_list) < period: return 25.0
    atr_smooth = sum(tr_list[-period:]) / period
    plus_di = (sum(plus_dm_list[-period:]) / atr_smooth) * 100 if atr_smooth > 0 else 0
    minus_di = (sum(minus_dm_list[-period:]) / atr_smooth) * 100 if atr_smooth > 0 else 0
    
    di_sum = plus_di + minus_di
    if di_sum == 0: return 25.0
    return (abs(plus_di - minus_di) / di_sum) * 100

def run_backtest():
    initial_balance = 1000.0
    balance = initial_balance
    risk_amount = 25.0
    total_wins, total_losses, grand_total_trades = 0, 0, 0

    print("==================================================")
    print(" SCORE HUNTER PRO - SNIPER HIGH WIN-RATE V13      ")
    print("==================================================")

    for symbol_key in SYMBOLS.keys():
        candles = fetch_yahoo_one_year(symbol_key)
        if not candles or len(candles) < 250: continue
        
        print(f"[*] Running V13 Sniper Backtest for {symbol_key}...")
        wins, losses, symbol_trades = 0, 0, 0
        in_position_until = 0

        for i in range(200, len(candles) - 30):
            if i < in_position_until: 
                continue

            sub = candles[:i+1]
            closes = [x["close"] for x in sub]
            volumes = [x["volume"] for x in sub]
            
            c = sub[-1]
            prev_c = sub[-2]
            
            atr = calculate_atr(sub, 14)
            ema_200 = calculate_ema(closes, 200)
            ema_50 = calculate_ema(closes, 50)
            ema_20 = calculate_ema(closes, 20)
            rsi = calculate_rsi(closes, 14)
            adx = calculate_adx(sub, 14)
            
            if atr == 0: continue

            # فیلتر رژیم بسیار سنگین (فقط روندهای خیلی قوی ADX > 30)
            if adx < 30: continue

            # تلاقی کامل و بی‌نقص روندها
            is_strong_bullish = (c["close"] > ema_200) and (ema_20 > ema_50) and (ema_50 > ema_200)
            is_strong_bearish = (c["close"] < ema_200) and (ema_20 < ema_50) and (ema_50 < ema_200)

            avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else 1.0
            volume_spike = c["volume"] > (avg_vol * 1.4)

            # شرایط تک‌تیرانداز با سخت‌گیری بالا
            sniper_buy = is_strong_bullish and volume_spike and (prev_c["close"] < ema_20) and (c["close"] > c["open"]) and (rsi > 55)
            sniper_sell = is_strong_bearish and volume_spike and (prev_c["close"] > ema_20) and (c["close"] < c["open"]) and (rsi < 45)

            trade_taken = False

            if sniper_buy:
                entry_price = c["close"]
                stop_loss = entry_price - (atr * 1.8)  # استاپ ایمن‌تر برای جلوگیری از اسکرپ
                risk_dist = entry_price - stop_loss

                if 0.003 * entry_price <= risk_dist <= 0.04 * entry_price:
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

            elif sniper_sell and not trade_taken:
                entry_price = c["close"]
                stop_loss = entry_price + (atr * 1.8)
                risk_dist = stop_loss - entry_price

                if 0.003 * entry_price <= risk_dist <= 0.04 * entry_price:
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
    print("      AGGREGATED V13 SNIPER RESULTS               ")
    print("==================================================")
    print(f"Total Trades       : {grand_total_trades}")
    print(f"Winning Trades     : {total_wins}")
    print(f"Losing Trades      : {total_losses}")
    print(f"Overall Win Rate   : {win_rate:.2f}%")
    print(f"Final Balance      : ${balance:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_backtest()
