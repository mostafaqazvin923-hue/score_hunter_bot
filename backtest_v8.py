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
        print(f"[*] Downloading 1-year Ichimoku data for {symbol_key}...")
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

def calculate_ichimoku_levels(sub_candles):
    # ایچیموکو استاندارد: Tenkan (9), Kijun (26), Senkou B (52)
    if len(sub_candles) < 52:
        return None
    
    # 9 کندل اخیر
    nine_highs = [c["high"] for c in sub_candles[-9:]]
    nine_lows = [c["low"] for c in sub_candles[-9:]]
    tenkan = (max(nine_highs) + min(nine_lows)) / 2
    
    # 26 کندل اخیر
    26_highs = [c["high"] for c in sub_candles[-26:]]
    26_lows = [c["low"] for c in sub_candles[-26:]]
    kijun = (max(26_highs) + min(26_lows)) / 2
    
    # 52 کندل اخیر برای لبه ابر
    52_highs = [c["high"] for c in sub_candles[-52:]]
    52_lows = [c["low"] for c in sub_candles[-52:]]
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (max(52_highs) + min(52_lows)) / 2
    
    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b
    }

def calculate_atr(candles, period=14):
    if len(candles) < period + 1: return candles[-1]["high"] - candles[-1]["low"]
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]; l = candles[i]["low"]; pc = candles[i-1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs[-period:]) / period

def run_backtest():
    initial_balance = 1000.0
    balance = initial_balance
    risk_amount = 25.0
    total_wins, total_losses, grand_total_trades = 0, 0, 0

    print("==================================================")
    print("   SCORE HUNTER PRO - ICHIMOKU PRO (INSTITUTIONAL)")
    print("==================================================")

    for symbol_key in SYMBOLS.keys():
        candles = fetch_yahoo_one_year(symbol_key)
        if not candles or len(candles) < 100: continue
        
        print(f"[*] Running Ichimoku Pro backtest for {symbol_key}...")
        wins, losses, symbol_trades, skip_until = 0, 0, 0, 0

        for i in range(60, len(candles) - 30):
            if i < skip_until: continue

            sub = candles[:i+1]
            c = sub[-2]
            prev_c = sub[-3]
            
            ichi = calculate_ichimoku_levels(sub)
            if not ichichi_valid := ichi: continue
            
            atr = calculate_atr(sub, 14)
            if atr == 0: continue

            tenkan = ichi["tenkan"]
            kijun = ichi["kijun"]
            s_a = ichi["senkou_a"]
            s_b = ichi["senkou_b"]
            
            cloud_top = max(s_a, s_b)
            cloud_bottom = min(s_a, s_b)

            # تاییدیه حرفه‌ای چیکو اسپن (Chikou Span Confirmation - مقایسه قیمت الان با ۲۶ کندل قبل)
            chikou_current_price = c["close"]
            price_26_ago = sub[-28]["close"] # حدود 26 کندل عقب‌تر
            chikou_bullish = chikou_current_price > price_26_ago
            chikou_bearish = chikou_current_price < price_26_ago

            # شرایط حرفه‌ای ایچیموکو برای لانگ:
            # ۱. قیمت بالای ابر (Cloud Top)
            # ۲. تنکان بالاتر از کیوجسن (TK Bullish Cross)
            # ۳. تاییدیه چیکو اسپن
            is_long = (c["close"] > cloud_top) and (tenkan > kijun) and chikou_bullish and (c["close"] > tenkan)

            # شرایط حرفه‌ای ایچیموکو برای شورت:
            # ۱. قیمت زیر ابر (Cloud Bottom)
            # ۲. تنکان پایین‌تر از کیوجسن (TK Bearish Cross)
            # ۳. تاییدیه چیکو اسپن
            is_short = (c["close"] < cloud_bottom) and (tenkan < kijun) and chikou_bearish and (c["close"] < tenkan)

            trade_taken = False

            if is_long:
                entry_price = c["close"]
                # استاپ لاس حرفه‌ای پشت کیوجسن یا لبه ابر
                stop_loss = min(kijun, cloud_bottom) - (atr * 0.2)
                risk_dist = entry_price - stop_loss

                if 0 < (risk_dist / entry_price) <= 0.04:
                    take_profit = entry_price + (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 48, len(candles) - 1)
                    
                    for j in range(i + 1, end_idx + 1):
                        future_c = candles[j]
                        if future_c["low"] <= stop_loss: trade_lost = True; end_idx = j; break
                        if future_c["high"] >= take_profit: trade_won = True; end_idx = j; break
                    
                    if not trade_won and not trade_lost:
                        trade_won = True if candles[end_idx]["close"] > entry_price else False
                        trade_lost = not trade_won

                    symbol_trades += 1
                    skip_until = end_idx
                    trade_taken = True
                    if trade_won: wins += 1; balance += (risk_amount * TARGET_RR)
                    elif trade_lost: losses += 1; balance -= risk_amount

            if not trade_taken and is_short:
                entry_price = c["close"]
                # استاپ لاس حرفه‌ای بالای کیوجسن یا لبه ابر
                stop_loss = max(kijun, cloud_top) + (atr * 0.2)
                risk_dist = stop_loss - entry_price

                if 0 < (risk_dist / entry_price) <= 0.04:
                    take_profit = entry_price - (risk_dist * TARGET_RR)
                    trade_won, trade_lost = False, False
                    end_idx = min(i + 48, len(candles) - 1)
                    
                    for j in range(i + 1, end_idx + 1):
                        future_c = candles[j]
                        if future_c["high"] >= stop_loss: trade_lost = True; end_idx = j; break
                        if future_c["low"] <= take_profit: trade_won = True; end_idx = j; break
                    
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
        print(f" > {symbol_key} -> Trades: {symbol_trades} | Wins: {wins} | Losses: {losses}")

    win_rate = (total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

    print("\n==================================================")
    print("      AGGREGATED ICHIMOKU PRO RESULTS             ")
    print("==================================================")
    print(f"Total Trades       : {grand_total_trades}")
    print(f"Winning Trades     : {total_wins}")
    print(f"Losing Trades      : {total_losses}")
    print(f"Overall Win Rate   : {win_rate:.2f}%")
    print(f"Final Balance      : ${balance:.2f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_backtest()
