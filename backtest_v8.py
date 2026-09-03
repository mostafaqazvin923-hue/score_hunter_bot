import yfinance as yf
import pandas as pd
import numpy as np

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
INITIAL_TOTAL_BALANCE = 1000.0
BALANCE_PER_COIN = INITIAL_TOTAL_BALANCE / len(SYMBOLS)
TARGET_RR = 2.0  # نسبت ریسک به ریوارد ثابت ۱ به ۲
RISK_PERCENTAGE = 0.01

months = pd.date_range(start='2025-09-01', end='2026-09-01', freq='MS')

grand_total_trades = 0
grand_total_wins = 0
grand_total_losses = 0
grand_total_breakeven = 0

print("============================================================")
print("WHALE TRAP WITH RISK-FREE (BREAKEVEN) MANAGEMENT (RR 1:2)")
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
            
            tr = np.maximum(highs - lows, np.maximum(abs(highs - np.roll(closes, 1)), abs(lows - np.roll(closes, 1))))
            atr = pd.Series(tr).rolling(window=14).mean().fillna(value=0).values
            vol_sma = pd.Series(volumes).rolling(window=20).mean().values

            m_wins, m_losses, m_be, m_trades = 0, 0, 0, 0
            idx = 25
            cooldown = 0

            while idx < len(df) - 24:
                if cooldown > 0:
                    cooldown -= 1
                    idx += 1
                    continue

                c_close, c_open, c_high, c_low = closes[idx], opens[idx], highs[idx], lows[idx]
                c_vol, c_vol_avg = volumes[idx], vol_sma[idx]
                c_atr = atr[idx]

                if c_atr == 0 or np.isnan(c_vol_avg):
                    idx += 1
                    continue

                lookback = 25
                swing_high = max(highs[idx-lookback:idx])
                swing_low = min(lows[idx-lookback:idx])

                trade_executed = False

                # 1. تله نهنگ صعودی (Long) با مدیریت ریسک‌فری
                if (c_low < swing_low) and (c_close > swing_low) and (c_close > c_open) and (c_vol > c_vol_avg * 1.3):
                    entry = c_close
                    sl = c_low - (c_atr * 0.3)
                    risk_dist = entry - sl
                    
                    if risk_dist > 0:
                        tp = entry + (risk_dist * TARGET_RR)
                        halfway_tp = entry + risk_dist  # نقطه نصف راه (RR 1:1) برای ریسک‌فری کردن
                        
                        won, lost, be = False, False, False
                        current_sl = sl
                        
                        for j in range(idx + 1, min(idx + 36, len(df))):
                            # چک کردن رسیدن به نصف راه برای انتقال حد ضرر به نقطه ورود (Risk-Free)
                            if highs[j] >= halfway_tp and current_sl == sl:
                                current_sl = entry  # ریسک فری شدن
                            
                            # چک کردن برخورد به TP نهایی
                            if highs[j] >= tp:
                                won = True
                                break
                            
                            # چک کردن برخورد به حد ضرر (که ممکن است روی نقطه ورود یا استاپ اولیه باشد)
                            if lows[j] <= current_sl:
                                if current_sl == entry:
                                    be = True  # سر به سر
                                else:
                                    lost = True  # باخت کامل
                                break
                                
                        if won or lost or be:
                            m_trades += 1
                            grand_total_trades += 1
                            risk_amount = symbol_balance * RISK_PERCENTAGE
                            
                            if won:
                                m_wins += 1
                                grand_total_wins += 1
                                symbol_balance += (risk_amount * TARGET_RR)
                            elif lost:
                                m_losses += 1
                                grand_total_losses += 1
                                symbol_balance -= risk_amount
                            elif be:
                                m_be += 1
                                grand_total_breakeven += 1
                                # سرمایه تغییری نمی‌کند (سر به سر)
                                
                            idx = j
                            cooldown = 5
                            trade_executed = True

                # 2. تله نهنگ نزولی (Short) با مدیریت ریسک‌فری
                elif (c_high > swing_high) and (c_close < swing_high) and (c_open > c_close) and (c_vol > c_vol_avg * 1.3) and not trade_executed:
                    entry = c_close
                    sl = c_high + (c_atr * 0.3)
                    risk_dist = sl - entry
                    
                    if risk_dist > 0:
                        tp = entry - (risk_dist * TARGET_RR)
                        halfway_tp = entry - risk_dist  # نقطه نصف راه برای ریسک‌فری
                        
                        won, lost, be = False, False, False
                        current_sl = sl
                        
                        for j in range(idx + 1, min(idx + 36, len(df))):
                            if lows[j] <= halfway_tp and current_sl == sl:
                                current_sl = entry  # ریسک فری شدن
                                
                            if lows[j] <= tp:
                                won = True
                                break
                            
                            if highs[j] >= current_sl:
                                if current_sl == entry:
                                    be = True
                                else:
                                    lost = True
                                break
                                
                        if won or lost or be:
                            m_trades += 1
                            grand_total_trades += 1
                            risk_amount = symbol_balance * RISK_PERCENTAGE
                            
                            if won:
                                m_wins += 1
                                grand_total_wins += 1
                                symbol_balance += (risk_amount * TARGET_RR)
                            elif lost:
                                m_losses += 1
                                grand_total_losses += 1
                                symbol_balance -= risk_amount
                            elif be:
                                m_be += 1
                                grand_total_breakeven += 1
                                
                            idx = j
                            cooldown = 5
                            trade_executed = True

                if not trade_executed:
                    idx += 1

            if m_trades > 0:
                print(f"    ماه {start_date.strftime('%Y-%m')} -> معاملات: {m_trades} | برد: {m_wins} | باخت: {m_losses} | سر به سر: {m_be}")

        except Exception as e:
            continue

# محاسبه وین‌ریت خالص (فقط بر اساس برد و باخت، بدون احتساب معاملات سر به سر در مخرج یا صورتِ برد)
decided_trades = grand_total_wins + grand_total_losses
overall_win_rate = (grand_total_wins / decided_trades * 100) if decided_trades > 0 else 0

print("\n" + "="*60)
print("FINAL RISK-FREE (BREAKEVEN) BACKTEST RESULT")
print("="*60)
print(f"TOTAL TRADES     : {grand_total_trades}")
print(f"TOTAL WINS       : {grand_total_wins}")
print(f"TOTAL LOSSES     : {grand_total_losses}")
print(f"BREAKEVEN (BE)   : {grand_total_breakeven}")
print(f"NET WIN RATE     : {overall_win_rate:.2f}%")
print("="*60)
