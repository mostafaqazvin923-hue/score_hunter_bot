import yfinance as yf
import pandas as pd
import numpy as np

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
INITIAL_TOTAL_BALANCE = 1000.0
BALANCE_PER_COIN = INITIAL_TOTAL_BALANCE / len(SYMBOLS)
TARGET_RR = 2.0
RISK_PERCENTAGE = 0.01

months = pd.date_range(start='2025-09-01', end='2026-09-01', freq='MS')

grand_total_trades = 0
grand_total_wins = 0
grand_total_losses = 0

print("============================================================")
print("WHALE PULLBACK - OPTIMIZED EMA PULLBACK & ADX STRATEGY")
print("============================================================")

for symbol in SYMBOLS:
    print(f"\n🔹 شروع بررسی نماد: {symbol}")
    symbol_balance = BALANCE_PER_COIN
    
    for i in range(len(months) - 1):
        start_date = months[i]
        end_date = months[i+1]
        
        try:
            df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1h', progress=False)
            
            if df.empty or len(df) < 50:
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
                
            closes = df['Close'].values
            highs = df['High'].values
            lows = df['Low'].values
            opens = df['Open'].values
            volumes = df['Volume'].values
            
            # محاسبه اندیکاتورها
            close_series = pd.Series(closes)
            ema_20 = close_series.ewm(span=20, adjust=False).mean().values
            ema_50 = close_series.ewm(span=50, adjust=False).mean().values
            ema_200 = close_series.ewm(span=200, adjust=False).mean().values
            
            tr = np.maximum(highs - lows, np.maximum(abs(highs - np.roll(closes, 1)), abs(lows - np.roll(closes, 1))))
            atr = pd.Series(tr).rolling(window=14).mean().fillna(value=0).values
            vol_sma = pd.Series(volumes).rolling(window=20).mean().values
            
            # محاسبه ADX برای سنجش قدرت روند
            df_adx = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
            df_adx['tr'] = tr
            df_adx['hd'] = df_adx['high'] - df_adx['high'].shift(1)
            df_adx['ld'] = df_adx['low'].shift(1) - df_adx['low']
            df_adx['pdm'] = np.where((df_adx['hd'] > df_adx['ld']) & (df_adx['hd'] > 0), df_adx['hd'], 0)
            df_adx['mdm'] = np.where((df_adx['ld'] > df_adx['hd']) & (df_adx['ld'] > 0), df_adx['ld'], 0)
            pdi = (pd.Series(df_adx['pdm']).rolling(14).sum() / (pd.Series(df_adx['tr']).rolling(14).sum() + 1e-9)) * 100
            mdi = (pd.Series(df_adx['mdm']).rolling(14).sum() / (pd.Series(df_adx['tr']).rolling(14).sum() + 1e-9)) * 100
            dx = (abs(pdi - mdi) / (pdi + mdi + 1e-9)) * 100
            adx = dx.rolling(14).mean().fillna(20).values

            m_wins, m_losses, m_trades = 0, 0, 0
            idx = 200
            cooldown = 0

            while idx < len(df) - 24:
                if cooldown > 0:
                    cooldown -= 1
                    idx += 1
                    continue

                c_close, c_open, c_high, c_low = closes[idx], opens[idx], highs[idx], lows[idx]
                c_vol, c_vol_avg = volumes[idx], vol_sma[idx]
                c_ema20, c_ema50, c_ema200 = ema_20[idx], ema_50[idx], ema_200[idx]
                c_atr, c_adx = atr[idx], adx[idx]

                if c_atr == 0 or np.isnan(c_vol_avg) or c_adx < 22:
                    idx += 1
                    continue

                # تشخیص روند اصلی
                is_uptrend = (c_close > c_ema200) and (c_ema20 > c_ema50)
                is_downtrend = (c_close < c_ema200) and (c_ema20 < c_ema50)

                if not is_uptrend and not is_downtrend:
                    idx += 1
                    continue

                trade_executed = False

                # منطق پولبک صعودی: قیمت به نزدیکی EMA 20 اصلاح کرده و کندل برگشتی زده است
                if is_uptrend and (c_low <= c_ema20 * 1.005) and (c_close > c_open) and (c_vol > c_vol_avg * 0.8):
                    entry = c_close
                    sl = c_low - (c_atr * 0.8)
                    risk_dist = entry - sl
                    if risk_dist > 0:
                        tp = entry + (risk_dist * TARGET_RR)
                        won, lost = False, False
                        for j in range(idx + 1, min(idx + 24, len(df))):
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
                            idx = j; cooldown = 4; trade_executed = True

                # منطق پولبک نزولی: قیمت به نزدیکی EMA 20 پولبک زده و کندل نزولی زده است
                elif is_downtrend and (c_high >= c_ema20 * 0.995) and (c_open > c_close) and (c_vol > c_vol_avg * 0.8) and not trade_executed:
                    entry = c_close
                    sl = c_high + (c_atr * 0.8)
                    risk_dist = sl - entry
                    if risk_dist > 0:
                        tp = entry - (risk_dist * TARGET_RR)
                        won, lost = False, False
                        for j in range(idx + 1, min(idx + 24, len(df))):
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
                            idx = j; cooldown = 4; trade_executed = True

                if not trade_executed:
                    idx += 1

            if m_trades > 0:
                print(f"    ماه {start_date.strftime('%Y-%m')} -> معاملات: {m_trades} | برد: {m_wins} | باخت: {m_losses}")

        except Exception as e:
            continue

overall_win_rate = (grand_total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

print("\n" + "="*60)
print("FINAL OPTIMIZED BACKTEST RESULT")
print("="*60)
print(f"TOTAL TRADES : {grand_total_trades}")
print(f"TOTAL WINS   : {grand_total_wins}")
print(f"TOTAL LOSSES : {grand_total_losses}")
print(f"WIN RATE     : {overall_win_rate:.2f}%")
print("="*60)
