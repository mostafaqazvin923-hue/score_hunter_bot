import yfinance as yf
import pandas as pd
import numpy as np

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
INITIAL_TOTAL_BALANCE = 1000.0
BALANCE_PER_COIN = INITIAL_TOTAL_BALANCE / len(SYMBOLS)
TARGET_RR = 2.0  # قفل روی نسبت ریسک به ریوارد ۱ به ۲ دقیق
RISK_PERCENTAGE = 0.01

months = pd.date_range(start='2025-09-01', end='2026-09-01', freq='MS')

grand_total_trades = 0
grand_total_wins = 0
grand_total_losses = 0

print("============================================================")
print("WHALE TRAP & LIQUIDITY SWEEP STRATEGY (RR = 1:2)")
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

            m_wins, m_losses, m_trades = 0, 0, 0
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

                # تعیین سقف و کف نوسانی اخیر (۲۵ کندل گذشته برای تشخیص نقدینگی)
                lookback = 25
                swing_high = max(highs[idx-lookback:idx])
                swing_low = min(lows[idx-lookback:idx])

                trade_executed = False

                # 1. تله نهنگ صعودی (Bullish Liquidity Sweep / Long)
                # قیمت کفِ قبلی را جارو کرده (پایین‌تر رفته) اما قدرت بسته شده و داخل بازه برگشته با حجم بالا
                if (c_low < swing_low) and (c_close > swing_low) and (c_close > c_open) and (c_vol > c_vol_avg * 1.3):
                    entry = c_close
                    sl = c_low - (c_atr * 0.3)  # حد ضرر ایمن پشت نفوذ
                    risk_dist = entry - sl
                    
                    if risk_dist > 0:
                        tp = entry + (risk_dist * TARGET_RR)
                        won, lost = False, False
                        
                        for j in range(idx + 1, min(idx + 36, len(df))):
                            if highs[j] >= tp:
                                won = True
                                break
                            if lows[j] <= sl:
                                lost = True
                                break
                                
                        if won or lost:
                            m_trades += 1
                            grand_total_trades += 1
                            risk_amount = symbol_balance * RISK_PERCENTAGE
                            if won:
                                m_wins += 1
                                grand_total_wins += 1
                                symbol_balance += (risk_amount * TARGET_RR)
                            else:
                                m_losses += 1
                                grand_total_losses += 1
                                symbol_balance -= risk_amount
                            idx = j
                            cooldown = 5
                            trade_executed = True

                # 2. تله نهنگ نزولی (Bearish Liquidity Sweep / Short)
                elif (c_high > swing_high) and (c_close < swing_high) and (c_open > c_close) and (c_vol > c_vol_avg * 1.3) and not trade_executed:
                    entry = c_close
                    sl = c_high + (c_atr * 0.3)
                    risk_dist = sl - entry
                    
                    if risk_dist > 0:
                        tp = entry - (risk_dist * TARGET_RR)
                        won, lost = False, False
                        
                        for j in range(idx + 1, min(idx + 36, len(df))):
                            if lows[j] <= tp:
                                won = True
                                break
                            if highs[j] >= sl:
                                lost = True
                                break
                                
                        if won or lost:
                            m_trades += 1
                            grand_total_trades += 1
                            risk_amount = symbol_balance * RISK_PERCENTAGE
                            if won:
                                m_wins += 1
                                grand_total_wins += 1
                                symbol_balance += (risk_amount * TARGET_RR)
                            else:
                                m_losses += 1
                                grand_total_losses += 1
                                symbol_balance -= risk_amount
                            idx = j
                            cooldown = 5
                            trade_executed = True

                if not trade_executed:
                    idx += 1

            if m_trades > 0:
                print(f"    ماه {start_date.strftime('%Y-%m')} -> معاملات: {m_trades} | برد: {m_wins} | باخت: {m_losses}")

        except Exception as e:
            continue

overall_win_rate = (grand_total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

print("\n" + "="*60)
print("FINAL WHALE TRAP BACKTEST RESULT (RR 1:2)")
print("="*60)
print(f"TOTAL TRADES : {grand_total_trades}")
print(f"TOTAL WINS   : {grand_total_wins}")
print(f"TOTAL LOSSES : {grand_total_losses}")
print(f"WIN RATE     : {overall_win_rate:.2f}%")
print("="*60)
