import ccxt
import pandas as pd
import numpy as np
import time

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
INITIAL_TOTAL_BALANCE = 1000.0
BALANCE_PER_COIN = INITIAL_TOTAL_BALANCE / len(SYMBOLS)
TARGET_RR = 2.0
RISK_PERCENTAGE = 0.01

exchange = ccxt.coinex({'enableRateLimit': True})

# ساخت بازه‌های ماهانه از سپتامبر 2025 تا سپتامبر 2026
months = pd.date_range(start='2025-09-01', end='2026-09-01', freq='MS')

grand_total_trades = 0
grand_total_wins = 0
grand_total_losses = 0
current_total_balance = INITIAL_TOTAL_BALANCE

print("============================================================")
print("WHALE PULLBACK - MONTH-BY-MONTH REAL COINEX BACKTEST (1h)")
print("============================================================")

for symbol in SYMBOLS:
    print(f"\n🔹 شروع بررسی نماد: {symbol}")
    symbol_balance = BALANCE_PER_COIN
    
    for i in range(len(months) - 1):
        start_date = months[i]
        end_date = months[i+1]
        
        since = exchange.parse8601(start_date.strftime('%Y-%m-%dT%H:%M:%SZ'))
        end_ts = exchange.parse8601(end_date.strftime('%Y-%m-%dT%H:%M:%SZ'))
        
        all_ohlcv = []
        current_since = since
        
        # دریافت داده‌های ماهانه با تایم‌فریم 1h برای پوشش کامل تاریخچه
        while current_since < end_ts:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=current_since, limit=1000)
                if not ohlcv:
                    break
                
                filtered_ohlcv = [c for c in ohlcv if c[0] < end_ts]
                if not filtered_ohlcv:
                    break
                    
                all_ohlcv.extend(filtered_ohlcv)
                current_since = ohlcv[-1][0] + 1
                
                if len(ohlcv) < 1000:
                    break
                time.sleep(exchange.rateLimit / 1000)
            except Exception as e:
                time.sleep(3)
                continue
                
        if len(all_ohlcv) < 200:
            continue
            
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        opens = df['open'].values
        volumes = df['volume'].values
        
        # اندیکاتورها
        close_series = pd.Series(closes)
        ema_20 = close_series.ewm(span=20, adjust=False).mean().values
        ema_50 = close_series.ewm(span=50, adjust=False).mean().values
        ema_200 = close_series.ewm(span=200, adjust=False).mean().values
        
        tr = np.maximum(highs - lows, np.maximum(abs(highs - np.roll(closes, 1)), abs(lows - np.roll(closes, 1))))
        atr = pd.Series(tr).rolling(window=14).mean().fillna(value=0).values
        vol_sma = pd.Series(volumes).rolling(window=20).mean().values
        
        m_wins, m_losses, m_trades = 0, 0, 0
        i_idx = 200
        cooldown = 0

        while i_idx < len(df) - 30:
            if cooldown > 0:
                cooldown -= 1
                i_idx += 1
                continue

            c_close, c_open, c_high, c_low = closes[i_idx], opens[i_idx], highs[i_idx], lows[i_idx]
            c_vol, c_vol_avg = volumes[i_idx], vol_sma[i_idx]
            c_ema20, c_ema50, c_ema200 = ema_20[i_idx], ema_50[i_idx], ema_200[i_idx]
            c_atr = atr[i_idx]

            if c_atr == 0 or np.isnan(c_vol_avg):
                i_idx += 1
                continue

            is_uptrend = (c_close > c_ema200) and (c_ema20 > c_ema50)
            is_downtrend = (c_close < c_ema200) and (c_ema20 < c_ema50)

            if not is_uptrend and not is_downtrend:
                i_idx += 1
                continue

            lookback = 10
            recent_high = max(highs[i_idx-lookback:i_idx])
            recent_low = min(lows[i_idx-lookback:i_idx])
            trade_executed = False

            # لانگ
            if is_uptrend and (c_close > recent_high) and (c_vol > c_vol_avg):
                entry = c_close
                sl = min(lows[i_idx-3:i_idx+1]) - (c_atr * 0.5)
                risk_dist = entry - sl
                if risk_dist > 0:
                    tp = entry + (risk_dist * TARGET_RR)
                    won, lost = False, False
                    for j in range(i_idx + 1, min(i_idx + 24, len(df))):
                        if highs[j] >= tp: won = True; break
                        if lows[j] <= sl: lost = True; break
                    if won or lost:
                        m_trades += 1
                        grand_total_trades += 1
                        risk_amount = symbol_balance * RISK_PERCENTAGE
                        if won:
                            m_wins += 1; grand_total_wins += 1
                            symbol_balance += (risk_amount * TARGET_RR)
                        else:
                            m_losses += 1; grand_total_losses += 1
                            symbol_balance -= risk_amount
                        i_idx = j; cooldown = 3; trade_executed = True

            # شورت
            elif is_downtrend and (c_close < recent_low) and (c_vol > c_vol_avg) and not trade_executed:
                entry = c_close
                sl = max(highs[i_idx-3:i_idx+1]) + (c_atr * 0.5)
                risk_dist = sl - entry
                if risk_dist > 0:
                    tp = entry - (risk_dist * TARGET_RR)
                    won, lost = False, False
                    for j in range(i_idx + 1, min(i_idx + 24, len(df))):
                        if lows[j] <= tp: won = True; break
                        if highs[j] >= sl: lost = True; break
                    if won or lost:
                        m_trades += 1
                        grand_total_trades += 1
                        risk_amount = symbol_balance * RISK_PERCENTAGE
                        if won:
                            m_wins += 1; grand_total_wins += 1
                            symbol_balance += (risk_amount * TARGET_RR)
                        else:
                            m_losses += 1; grand_total_losses += 1
                            symbol_balance -= risk_amount
                        i_idx = j; cooldown = 3; trade_executed = True

            if not trade_executed:
                i_idx += 1

        if m_trades > 0:
            print(f"    ماه {start_date.strftime('%Y-%m')} -> معاملات: {m_trades} | برد: {m_wins} | باخت: {m_losses}")

overall_win_rate = (grand_total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

print("\n" + "="*60)
print("FINAL 1-YEAR AGGREGATED RESULT (1h TIMEFRAME)")
print("="*60)
print(f"TOTAL TRADES : {grand_total_trades}")
print(f"TOTAL WINS   : {grand_total_wins}")
print(f"TOTAL LOSSES : {grand_total_losses}")
print(f"WIN RATE     : {overall_win_rate:.2f}%")
print("="*60)
