import json
import urllib.request
import math

SYMBOLS = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD", "XRPUSDT": "XRP-USD"}
TARGET_RR = 2.0  # ضریب ریسک به ریوارد مهندسی‌شده

def fetch_yahoo_one_year(symbol_key):
    yahoo_symbol = SYMBOLS[symbol_key]
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?interval=1h&range=1y"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    try:
        print(f"[*] Downloading 1H Market Data for {symbol_key}...")
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

def resample_to_4h(candles_1h):
    # تبدیل کندل‌های ۱ ساعته به ۴ ساعته برای تشخیص جهت کلان بازار
    candles_4h = []
    for i in range(0, len(candles_1h), 4):
        chunk = candles_1h[i:i+4]
        if not chunk: continue
        c4 = {
            "timestamp": chunk[0]["timestamp"],
            "open": chunk[0]["open"],
            "high": max(x["high"] for x in chunk),
            "low": min(x["low"] for x in chunk),
            "close": chunk[-1]["close"],
            "volume": sum(x["volume"] for x in chunk)
        }
        candles_4h.append(c4)
    return candles_4h

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
    print(" MTF V18 - 4H TREND + 1H SNIPER ENTRY & RUNWAY    ")
    print("==================================================")

    for symbol_key in SYMBOLS.keys():
        candles_1h = fetch_yahoo_one_year(symbol_key)
        if not candles_1h or len(candles_1h) < 300: continue
        
        candles_4h = resample_to_4h(candles_1h)
        print(f"[*] Running MTF V18 Backtest for {symbol_key}...")
        
        wins, losses, symbol_trades = 0, 0, 0
        in_position_until = 0

        # نقشه‌برداری زمانی برای دسترسی سریع به روند 4H از روی کندل 1H
        for i in range(50, len(candles_1h) - 30):
            if i < in_position_until: 
                continue

            c1h = candles_1h[i]
            current_timestamp = c1h["timestamp"]

            # پیدا کردن کندل 4H متناظر با زمان فعلی
            active_4h_index = -1
            for idx, c4 in enumerate(candles_4h):
                if c4["timestamp"] <= current_timestamp:
                    active_4h_index = idx
                else:
                    break
            
            if active_4h_index < 30: continue
            
            sub_4h = candles_4h[:active_4h_index+1]
            closes_4h = [x["close"] for x in sub_4h]
            ema_50_4h = calculate_ema(closes_4h, 50)
            
            # 1. تعیین جهت کلان در تایم فریم 4 ساعته
            is_4h_bullish = closes_4h[-1] > ema_50_4h
            is_4h_bearish = closes_4h[-1] < ema_50_4h

            # داده‌های 1 ساعته برای نقطه ورود
            sub_1h = candles_1h[:i+1]
            closes_1h = [x["close"] for x in sub_1h]
            volumes_1h = [x["volume"] for x in sub_1h]
            
            c = sub_1h[-1]
            prev = sub_1h[-2]
            atr_1h = calculate_atr(sub_1h, 14)
            if atr_1h == 0: continue

            # بررسی فضای مسیر (Runway & Obstacle Check) بر اساس سوییگ‌های 1 ساعته اخیر
            recent_highs = [x["high"] for x in sub_1h[-50:-2]]
            recent_lows = [x["low"] for x in sub_1h[-50:-2]]
            major_resistance = max(recent_highs) if recent_highs else c["high"] * 1.05
            major_support = min(recent_lows) if recent_lows else c["low"] * 0.95

            avg_vol = sum(volumes_1h[-20:]) / 20 if len(volumes_1h) >= 20 else 1.0
            volume_spike = c["volume"] > (avg_vol * 1.3)

            # 2. تریگر ورود در تایم‌فریم 1 ساعته همراستا با جهت 4 ساعته
            swing_low_1h = min(x["low"] for x in sub_1h[-20:-2])
            swing_high_1h = max(x["high"] for x in sub_1h[-20:-2])

            bullish_entry = is_4h_bullish and volume_spike and (prev["close"] < prev["open"]) and (c["close"] > c["open"]) and (c["close"] > prev["high"])
            bearish_entry = is_4h_bearish and volume_spike and (prev["close"] > prev["open"]) and (c["close"] < c["open"]) and (c["close"] < prev["low"])

            trade_taken = False

            if bullish_entry:
                # مهندسی حد ضرر (SL): پشت کف سوینگ 1 ساعته منهای بافر ATR
                entry_price = c["close"]
                stop_loss = swing_low_1h - (atr_1h * 0.5)
                risk_dist = entry_price - stop_loss

                if 0.002 * entry_price <= risk_dist <= 0.04 * entry_price:
                    take_profit = entry_price + (risk_dist * TARGET_RR)
                    
                    # بررسی فضای مسیر: آیا تا رسیدن به TP مانع مقاومتی بزرگی وجود دارد؟
                    has_enough_room = major_resistance >= (take_profit - (risk_dist * 0.1))

                    if has_enough_room:
                        trade_won, trade_lost = False, False
                        end_idx = len(candles_1h) - 1
                        
                        for j in range(i + 1, len(candles_1h)):
                            future_c = candles_1h[j]
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

            elif bearish_entry and not trade_taken:
                # مهندسی حد ضرر برای شورت: پشت سقف سوینگ 1 ساعته به‌اضافه بافر ATR
                entry_price = c["close"]
                stop_loss = swing_high_1h + (atr_1h * 0.5)
                risk_dist = stop_loss - entry_price

                if 0.002 * entry_price <= risk_dist <= 0.04 * entry_price:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    
                    # بررسی فضای مسیر برای شورت
                    has_enough_room = major_support <= (take_profit + (risk_dist * 0.1))

                    if has_enough_room:
                        trade_won, trade_lost = False, False
                        end_idx = len(candles_1h) - 1
                        
                        for j in range(i + 1, len(candles_1h)):
                            future_c = candles_1h[j]
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
    print("      AGGREGATED MTF V18 RESULTS                  ")
    print("==================================================")
    print(f"Total Trades       : {grand_total_trades}")
    print(f"Winning Trades     : {total_wins}")
    print(f"Losing Trades      : {total_losses}")
    print(f"Overall Win Rate   : {win_rate:.2f}%")
    print(f"Final Balance      : ${balance:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_backtest()
