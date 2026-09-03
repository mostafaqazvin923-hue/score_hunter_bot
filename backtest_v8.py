import pandas as pd
import numpy as np
import os

SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]

print("============================================================")
print("🚀 اجرای موتور بک‌تست استراتژی تله‌ی نهنگ + فیلتر روند")
print("============================================================")

for symbol in SYMBOLS:
    filename = f"{symbol}_15m.csv"
    if not os.path.exists(filename):
        print(f"\n⚠️ فایل {filename} یافت نشد.")
        continue
        
    print(f"\n🔹 تحلیل و بک‌تست روی {symbol}...")
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 1. محاسبه فیلتر روند (میانگین متحرک 200 کندلی)
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # 2. محاسبه سقف و کف‌های 20 کندل اخیر (برای شناسایی نقدینگی)
    df['Prev_High'] = df['High'].shift(1).rolling(window=20).max()
    df['Prev_Low'] = df['Low'].shift(1).rolling(window=20).min()
    
    trades = []
    
    # تنظیمات ریسک به ریوارد (توقف ضرر 1.5٪ و حد سود 3٪ -> R:R = 1:2)
    sl_pct = 0.015
    tp_pct = 0.030
    
    for i in range(200, len(df) - 50):
        current = df.iloc[i]
        close = current['Close']
        sma = current['SMA_200']
        
        # اگر مقادیر ناقص باشند رد شو
        if pd.isna(sma) or pd.isna(current['Prev_High']) or pd.isna(current['Prev_Low']):
            continue
            
        # سیگنال خرید (Long Whale Trap): قیمت کفِ قبلی را جارو کرده اما بسته شده بالا تر از آن + روند صعودی
        is_long_trap = (current['Low'] < current['Prev_Low']) and (close > current['Prev_Low']) and (close > sma)
        
        # سیگنال فروش (Short Whale Trap): قیمت سقفِ قبلی را جارو کرده اما بسته شده پایین‌تر از آن + روند نزولی
        is_short_trap = (current['High'] > current['Prev_High']) and (close < current['Prev_High']) and (close < sma)
        
        if is_long_trap or is_short_trap:
            entry_price = close
            entry_time = current['Date']
            
            if is_long_trap:
                sl = entry_price * (1 - sl_pct)
                tp = entry_price * (1 + tp_pct)
                side = 'LONG'
            else:
                sl = entry_price * (1 + sl_pct)
                tp = entry_price * (1 - tp_pct)
                side = 'SHORT'
                
            # شبیه‌سازی نتیجه معامله در کندل‌های آینده
            outcome = 'OPEN'
            for j in range(i + 1, min(i + 50, len(df))):
                future_candle = df.iloc[j]
                high = future_candle['High']
                low = future_candle['Low']
                
                if side == 'LONG':
                    if low <= sl:
                        outcome = 'LOSS'
                        break
                    elif high >= tp:
                        outcome = 'WIN'
                        break
                else: # SHORT
                    if high >= sl:
                        outcome = 'LOSS'
                        break
                    elif low <= tp:
                        outcome = 'WIN'
                        break
            
            if outcome in ['WIN', 'LOSS']:
                trades.append({
                    'Time': entry_time,
                    'Side': side,
                    'Entry': entry_price,
                    'Outcome': outcome
                })
                
    # محاسبه آمار نهایی
    if trades:
        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['Outcome'] == 'WIN'])
        losses = len(trades_df[trades_df['Outcome'] == 'LOSS'])
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        
        # محاسبه سود خالص تقریبی با فرض ریسک به ریوارد 1:2
        net_profit = (wins * 2) - losses
        
        print(f"📊 نتایج نهایی برای {symbol}:")
        print(f"   - تعداد کل معاملات معتبر: {total_trades}")
        print(f"   - معاملات برنده (WIN): {wins}")
        print(f"   - معاملات بازنده (LOSS): {losses}")
        print(f"   - **وین‌ریت نهایی (Win Rate):** {win_rate:.2f}%")
        print(f"   - امتیاز عملکرد (Profit Score): {net_profit}")
    else:
        print(f"⚠️ هیچ معامله‌ای با شرایط فیلتر روند روی {symbol} ثبت نشد.")

print("\n✨ بک‌تست به اتمام رسید.")
