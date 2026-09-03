import os
import yfinance as yf
import pandas as pd
import numpy as np

# دیکشنری نمادها و نام فایل‌های محلی
FILES = {
    "BTC-USD": "BTC_15m.csv",
    "ETH-USD": "ETH_15m.csv",
    "SOL-USD": "SOL_15m.csv",
    "XRP-USD": "XRP_15m.csv"
}

INITIAL_TOTAL_BALANCE = 1000.0
BALANCE_PER_COIN = INITIAL_TOTAL_BALANCE / len(FILES)
TARGET_RR = 2.0  # ریسک به ریوارد ۱ به ۲ ثابت
RISK_PERCENTAGE = 0.01

print("============================================================")
print("WHALE TRAP 15m BACKTEST (AUTO-DOWNLOAD LOCAL CSV & RUN)")
print("============================================================")

# مرحله ۱: بررسی و دانلود خودکار فایل‌ها اگر وجود نداشته باشند
for symbol, filename in FILES.items():
    if not os.path.exists(filename):
        print(f"📥 در حال دانلود داده‌های ۱۵ دقیقه‌ای برای {symbol}...")
        try:
            # دانلود داده‌های ۶۰ روز گذشته در تایم‌فریم ۱۵ دقیقه (حداکثر محدودیت یاهو برای ۱۵ دقیقه)
            df_dl = yf.download(symbol, period="60d", interval="15m", progress=False)
            if not df_dl.empty:
                if isinstance(df_dl.columns, pd.MultiIndex):
                    df_dl.columns = df_dl.columns.droplevel(1)
                df_dl.to_csv(filename)
                print(f"✅ فایل {filename} با موفقیت ذخیره شد.")
            else:
                print(f"⚠️ نتوانست داده‌ای برای {symbol} دانلود کند.")
        except Exception as e:
            print(f"❌ خطا در دانلود {symbol}: {e}")

grand_total_trades = 0
grand_total_wins = 0
grand_total_losses = 0

# مرحله ۲: اجرای بک‌تست روی فایل‌های محلی
for symbol, filename in FILES.items():
    if not os.path.exists(filename):
        continue
        
    print(f"\n🔹 در حال پردازش بک‌تست برای: {symbol}")
    
    try:
        df = pd.read_csv(filename)
        df.columns = [c.capitalize() for c in df.columns]
        
        if df.empty or len(df) < 50:
            continue
            
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        opens = df['Open'].values
        volumes = df['Volume'].values
        
        tr = np.maximum(highs - lows, np.maximum(abs(highs - np.roll(closes, 1)), abs(lows - np.roll(closes, 1))))
        atr = pd.Series(tr).rolling(window=14).mean().fillna(value=0).values
        vol_sma = pd.Series(volumes).rolling(window=20).mean().values

        symbol_balance = BALANCE_PER_COIN
        m_wins, m_losses, m_trades = 0, 0, 0
        idx = 50
        cooldown = 0

        while idx < len(df) - 50:
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

            lookback = 50
            swing_high = max(highs[idx-lookback:idx])
            swing_low = min(lows[idx-lookback:idx])

            trade_executed = False

            # 1. تله نهنگ صعودی (Long)
            if (c_low < swing_low) and (c_close > swing_low) and (c_close > c_open) and (c_vol > c_vol_avg * 1.3):
                entry = c_close
                sl = c_low - (c_atr * 0.3)
                risk_dist = entry - sl
                
                if risk_dist > 0:
                    tp = entry + (risk_dist * TARGET_RR)
                    won, lost = False, False
                    
                    for j in range(idx + 1, min(idx + 70, len(df))):
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
                        cooldown = 12
                        trade_executed = True

            # 2. تله نهنگ نزولی (Short)
            elif (c_high > swing_high) and (c_close < swing_high) and (c_open > c_close) and (c_vol > c_vol_avg * 1.3) and not trade_executed:
                entry = c_close
                sl = c_high + (c_atr * 0.3)
                risk_dist = sl - entry
                
                if risk_dist > 0:
                    tp = entry - (risk_dist * TARGET_RR)
                    won, lost = False, False
                    
                    for j in range(idx + 1, min(idx + 70, len(df))):
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
                        cooldown = 12
                        trade_executed = True

            if not trade_executed:
                idx += 1

        print(f"    📊 نتایج {symbol} -> معاملات: {m_trades} | برد: {m_wins} | باخت: {m_losses}")

    except Exception as e:
        print(f"    ❌ خطا در پردازش فایل {filename}: {e}")

overall_win_rate = (grand_total_wins / grand_total_trades * 100) if grand_total_trades > 0 else 0

print("\n" + "="*60)
print("FINAL 15m AUTOMATED BACKTEST RESULT (RR 1:2)")
print("="*60)
print(f"TOTAL TRADES : {grand_total_trades}")
print(f"TOTAL WINS   : {grand_total_wins}")
print(f"TOTAL LOSSES : {grand_total_losses}")
print(f"WIN RATE     : {overall_win_rate:.2f}%")
print("="*60)
