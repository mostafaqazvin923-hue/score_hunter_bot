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
        print(f"[*] Downloading SMC V17 Market Data for {symbol_key}...")
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

def run_backtest():
    initial_balance = 1000.0
    balance = initial_balance
    risk_amount = 25.0
    total_wins, total_losses, grand_total_trades = 0, 0, 0

    print("==================================================")
    print(" SMC V17 - INSTITUTIONAL SNIPER & RUNWAY CHECK    ")
    print("==================================================")

    for symbol_key in SYMBOLS.keys():
        candles = fetch_yahoo_one_year(symbol_key)
        if not candles or len(candles) < 250: continue
        
        print(f"[*] Running SMC V17 Backtest for {symbol_key}...")
        wins, losses, symbol_trades = 0, 0, 0
        in_position_until = 0

        for i in range(200, len(candles) - 30):
            if i < in_position_until: 
                continue

            sub = candles[:i+1]
            closes = [x["close"] for x in sub]
            volumes = [x["volume"] for x in sub]
            
            c = sub[-1]      # کندل فعلی (ورود روی پولبک به بلوک سفارش)
            prev = sub[-2]   # کندل ایمپالس (شکست ساختار / BOS)
            prev2 = sub[-3]  # کندل شکار نقدینگی (Sweep)

            atr = calculate_atr(sub, 14)
            ema_200 = calculate_ema(closes, 200) # فیلتر روند کلان مؤسساتی
            if atr == 0: continue

            # تعیین سطوح ساختاری برای بررسی فضای مسیر (Runway)
            recent_highs = [x["high"] for x in sub[-60:-3]]
            recent_lows = [x["low"] for x in sub[-60:-3]]
            major_resistance = max(recent_highs) if recent_highs else c["high"] * 1.05
            major_support = min(recent_lows) if recent_lows else c["low"] * 0.95

            # تشخیص نواحی کلیدی نقدینگی محلی
            swing_low_market = min(x["low"] for x in sub[-40:-3])
            swing_high_market = max(x["high"] for x in sub[-40:-3])

            avg_vol = sum(volumes[-25:-3]) / 22 if len(volumes) >= 25 else 1.0
            volume_spike = prev["volume"] > (avg_vol * 1.6) # حجم نهنگی

            # 1. ستاپ اسمارت مانی لانگ: قیمت کف قبلی را جارو کرده، با حجم بالا به سمت بالا پولک زده و بالای روند 200 است
            sweep_low = prev2["low"] <= swing_low_market and prev2["close"] > swing_low_market
            bullish_smc = sweep_low and volume_spike and (prev["close"] > prev["open"]) and (c["close"] > ema_200)

            # 2. ستاپ اسمارت مانی شورت: قیمت سقف قبلی را جارو کرده، با حجم بالا ریخته و زیر روند 200 است
            sweep_high = prev2["high"] >= swing_high_market and prev2["close"] < swing_high_market
            bearish_smc = sweep_high and volume_spike and (prev["close"] < prev["open"]) and (c["close"] < ema_200)

            trade_taken = False

            if bullish_smc:
                entry_price = c["close"]
                stop_loss = swing_low_market - (atr * 0.3) # پشت استاپ‌های جاروشده
                risk_dist = entry_price - stop_loss

                if 0.003 * entry_price <= risk_dist <= 0.04 * entry_price:
                    take_profit = entry_price + (risk_dist * TARGET_RR)
                    
                    # فیلتر فضای مسیر (Runway Check): آیا مقاومت سنگینی مانع رسیدن به هدف 2R است؟
                    has_enough_room = major_resistance >= (take_profit - (risk_dist * 0.1))

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

            elif bearish_smc and not trade_taken:
                entry_price = c["close"]
                stop_loss = swing_high_market + (atr * 0.3)
                risk_dist = stop_loss - entry_price

                if 0.003 * entry_price <= risk_dist <= 0.04 * entry_price:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    
                    # فیلتر فضای مسیر برای شورت
                    has_enough_room = major_support <= (take_profit + (risk_dist * 0.1))

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
    print("      AGGREGATED SMC V17 SNIPER RESULTS           ")
    print("==================================================")
    print(f"Total Trades       : {grand_total_trades}")
    print(f"Winning Trades     : {total_wins}")
    print(f"Losing Trades      : {total_losses}")
    print(f"Overall Win Rate   : {win_rate:.2f}%")
    print(f"Final Balance      : ${balance:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_backtest()
