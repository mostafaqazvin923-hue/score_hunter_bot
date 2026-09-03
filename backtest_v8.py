import yfinance as yf
import pandas as pd
import numpy as np

# نمادها در یاهو فایننس با ساختار USD هستند
SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
INITIAL_TOTAL_BALANCE = 1000.0
BALANCE_PER_COIN = INITIAL_TOTAL_BALANCE / len(SYMBOLS)
TARGET_RR = 2.0
RISK_PERCENTAGE = 0.01

# ساخت بازه‌های ماهانه از سپتامبر 2025 تا سپتامبر 2026
months = pd.date_range(start='2025-09-01', end='2026-09-01', freq='MS')

grand_total_trades = 0
grand_total_wins = 0
grand_total_losses = 0

print("============================================================")
print("WHALE PULLBACK - MONTH-BY-MONTH YAHOO FINANCE BACKTEST")
print("============================================================")

for symbol in SYMBOLS:
    print(f"\n🔹 شروع بررسی نماد: {symbol}")
    symbol_balance = BALANCE_PER_COIN
    
    for i in range(len(months) - 1):
        start_date = months[i]
        end_date = months[i+1]
        
        try:
            # دانلود داده‌های ساعتی یا روزانه با بازه مشخص از یاهو
            df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1h', progress=False)
            
            if df.empty or len(df) < 50:
                continue
                
            # پاکسازی فرمت ستون‌ها در صورت چندسطحی بودن در yfinance جدید
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
                
            closes = df['Close'].values
            highs = df['High'].values
            lows = df['Low'].values
            opens = df['Open'].values
            volumes = df['Volume'].values
            
            # اندیکاتورها
            close_series = pd.Series(closes)
            ema_20 = close_series.ewm(span=20, adjust=False).mean().values
            ema_50 = close_series.ewm(span=50, adjust=False).mean().values
            ema_200 = close_series.ewm(span=200, adjust=False).mean().values
            
            tr = np.maximum(highs - lows, np.maximum(abs(highs - np.roll(closes, 1)), abs(lows - np.roll(closes, 1))))
            atr = pd.Series(tr).rolling(window=14).mean().fillna(value=0).values
            vol_sma = pd.Series(volumes).rolling(window=20).mean().values
            
            m_wins, m_losses, m_trades = 0, 0, 0
            idx = 50
            cooldown = 0

            while idx < len(df) - 24:
                if cooldown > 0:
                    cooldown -= 1
                    idx += 1
                    continue

                c_close, c_open, c_high, c_low = closes[idx], opens[idx], highs[idx], lows[idx]
                c_vol, c_vol_avg = volumes[idx], vol_sma[idx]
                c_ema20, c_ema50, c_ema200 = ema_20[idx], ema_50[idx], ema_200[idx]
                c_atr = atr[idx]

                if c_atr == 0 or np.isnan(c_vol_avg):
                    idx += 1
                    continue

                # شرایط روند (همان منطق روانِ تست اولیه)
                is_uptrend = (c_close > c_ema200) and (c_ema20 > c_ema50)
                is_downtrend = (c_close < c_ema200) and (c_ema20 < c_ema50)

                if not is_uptrend and not is_downtrend:
                    idx += 1
                    continue

                lookback = 10
                recent_high = max(highs[idx-lookback:idx])
                recent_low = min(lows[idx-lookback:idx])
                trade_executed = False

                # لانگ
                if is_uptrend and (c_close > recent_high) and (c_vol > c_vol_avg):
                    entry = c_close
                    sl = min(lows[idx-3:idx+1]) - (c_atr * 0.5)
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
                            idx = j; cooldown = 3; trade_executed = True

                # شورت
                elif is_downtrend and (c_close < recent_low) and (c_vol > c_vol_avg) and not trade_executed:
                    entry = c_close
                    sl = max(highs[idx-3:idx+1]) + (c_atr * 0.5)
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
                            idx = j; cooldown = 3; trade_executed = True

                if not trade_executed:
                    idx += 1

            if m_trades > 0:
                print(f"    ماه {start_date.strftime('%Y-%m')} -> معاملات: {m_trades} | برد: {m_wins} | باخت: {m_losses}")

        except Exception as e:
            continue

overall_win_rate = (grand_total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

print("\n" + "="*60)
print("FINAL 1-YEAR AGGREGATED RESULT (YAHOO FINANCE)")
print("="*60)
print(f"TOTAL TRADES : {grand_total_trades}")
print(f"TOTAL WINS   : {grand_total_wins}")
print(f"TOTAL LOSSES : {grand_total_losses}")
print(f"WIN RATE     : {overall_win_rate:.2f}%")
print("="*60)
