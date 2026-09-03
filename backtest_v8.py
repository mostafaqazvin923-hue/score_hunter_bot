import yfinance as yf
import pandas as pd
import numpy as np

SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
INITIAL_TOTAL_BALANCE = 1000.0
BALANCE_PER_COIN = INITIAL_TOTAL_BALANCE / len(SYMBOLS)
RISK_PERCENTAGE = 0.01

months = pd.date_range(start='2025-09-01', end='2026-09-01', freq='MS')

grand_total_trades = 0
grand_total_wins = 0
grand_total_losses = 0

print("============================================================")
print("WHALE PULLBACK - DYNAMIC MOMENTUM EXIT STRATEGY (1h)")
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
            
            close_series = pd.Series(closes)
            ema_9 = close_series.ewm(span=9, adjust=False).mean().values
            ema_20 = close_series.ewm(span=20, adjust=False).mean().values
            ema_200 = close_series.ewm(span=200, adjust=False).mean().values
            
            tr = np.maximum(highs - lows, np.maximum(abs(highs - np.roll(closes, 1)), abs(lows - np.roll(closes, 1))))
            atr = pd.Series(tr).rolling(window=14).mean().fillna(value=0).values
            vol_sma = pd.Series(volumes).rolling(window=20).mean().values

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
                c_ema9, c_ema20, c_ema200 = ema_9[idx], ema_20[idx], ema_200[idx]
                c_atr = atr[idx]

                if c_atr == 0 or np.isnan(c_vol_avg):
                    idx += 1
                    continue

                is_uptrend = (c_close > c_ema200) and (c_ema9 > c_ema20)
                is_downtrend = (c_close < c_ema200) and (c_ema9 < c_ema20)

                if not is_uptrend and not is_downtrend:
                    idx += 1
                    continue

                trade_executed = False

                # سیستم خروج پویا برای پوزیشن خرید (Long)
                if is_uptrend and (c_low <= c_ema20 * 1.01) and (c_close > c_open) and (c_vol > c_vol_avg * 0.8):
                    entry = c_close
                    sl = c_low - (c_atr * 0.8)
                    risk_amount = symbol_balance * RISK_PERCENTAGE
                    
                    if entry > sl:
                        won = False
                        lost = False
                        exit_price = entry
                        
                        # پیمايش کندل‌ها به جلو تا زمان برخورد به SL یا خروج مومنتومی
                        for j in range(idx + 1, min(idx + 30, len(df))):
                            # بررسی برخورد حد ضرر
                            if lows[j] <= sl:
                                lost = True
                                exit_price = sl
                                idx = j
                                break
                            
                            # خروج پویا: اگر کندل معکوسِ پرقدرت یا افت مومنتوم زیر EMA 9 رخ داد
                            if closes[j] < ema_9[j] and closes[j] < opens[j]:
                                if closes[j] > entry:
                                    won = True
                                    exit_price = closes[j]
                                else:
                                    lost = True
                                    exit_price = closes[j]
                                idx = j
                                break
                        
                        # اگر تا انتهایبازه بسته نشد، با قیمت آخرین کندل ببند
                        if not won and not lost:
                            idx = min(idx + 15, len(df) - 1)
                            if closes[idx] > entry:
                                won = True
                            else:
                                lost = True
                            exit_price = closes[idx]

                        m_trades += 1
                        grand_total_trades += 1
                        
                        if won:
                            m_wins += 1
                            grand_total_wins += 1
                            profit_pct = (exit_price - entry) / (entry - sl)
                            symbol_balance += (risk_amount * max(profit_pct, 1.0))
                        else:
                            m_losses += 1
                            grand_total_losses += 1
                            symbol_balance -= risk_amount
                            
                        cooldown = 3
                        trade_executed = True

                # سیستم خروج پویا برای پوزیشن فروش (Short)
                elif is_downtrend and (c_high >= c_ema20 * 0.99) and (c_open > c_close) and (c_vol > c_vol_avg * 0.8) and not trade_executed:
                    entry = c_close
                    sl = c_high + (c_atr * 0.8)
                    risk_amount = symbol_balance * RISK_PERCENTAGE
                    
                    if sl > entry:
                        won = False
                        lost = False
                        exit_price = entry
                        
                        for j in range(idx + 1, min(idx + 30, len(df))):
                            if highs[j] >= sl:
                                lost = True
                                exit_price = sl
                                idx = j
                                break
                            
                            if closes[j] > ema_9[j] and closes[j] > opens[j]:
                                if closes[j] < entry:
                                    won = True
                                    exit_price = closes[j]
                                else:
                                    lost = True
                                    exit_price = closes[j]
                                idx = j
                                break
                        
                        if not won and not lost:
                            idx = min(idx + 15, len(df) - 1)
                            if closes[idx] < entry:
                                won = True
                            else:
                                lost = True
                            exit_price = closes[idx]

                        m_trades += 1
                        grand_total_trades += 1
                        
                        if won:
                            m_wins += 1
                            grand_total_wins += 1
                            profit_pct = (entry - exit_price) / (sl - entry)
                            symbol_balance += (risk_amount * max(profit_pct, 1.0))
                        else:
                            m_losses += 1
                            grand_total_losses += 1
                            symbol_balance -= risk_amount
                            
                        cooldown = 3
                        trade_executed = True

                if not trade_executed:
                    idx += 1

            if m_trades > 0:
                print(f"    ماه {start_date.strftime('%Y-%m')} -> معاملات: {m_trades} | برد: {m_wins} | باخت: {m_losses}")

        except Exception as e:
            continue

overall_win_rate = (grand_total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

print("\n" + "="*60)
print("FINAL DYNAMIC EXIT BACKTEST RESULT")
print("="*60)
print(f"TOTAL TRADES : {grand_total_trades}")
print(f"TOTAL WINS   : {grand_total_wins}")
print(f"TOTAL LOSSES : {grand_total_losses}")
print(f"WIN RATE     : {overall_win_rate:.2f}%")
print("="*60)
