import pandas as pd
import numpy as np

def run_causal_backtest(df):
    """
    بک‌تست استاندارد و بدون نگاه به آینده (Causal) برای score_hunter_bot
    با استفاده از منطق تریگر شکست سقف/کف کندل مرجع برای جلوگیری از تاخیر زمانی.
    """
    trades = []
    capital = 100000.0
    position = None  # None, 'LONG', 'SHORT'
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    
    # فرض بر این است که دیتافریم دارای ستون‌های Open, High, Low, Close و Signal است
    # Signal روی کندل i-1 محاسبه شده است: 1 برای لانگ، -1 برای شورت، 0 برای خنثی
    
    for i in range(2, len(df)):
        current_open = df['Open'].iloc[i]
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]
        current_close = df['Close'].iloc[i]
        
        prev_high = df['High'].iloc[i-1]
        prev_low = df['Low'].iloc[i-1]
        signal = df['Signal'].iloc[i-1]  # تصمیم گرفته شده بر اساس گذشته (بدون تقلب)
        
        # ۱. مدیریت پوزیشن‌های باز (بررسی برخورد با استاپ‌لاس یا تیک‌پرافیت در کندل جاری)
        if position == 'LONG':
            if current_low <= stop_loss:
                # خروج با ضرر
                pnl = (stop_loss - entry_price) * position_size
                capital += pnl
                trades.append({'type': 'LONG', 'result': 'LOSS', 'pnl': pnl})
                position = None
                continue
            elif current_high >= take_profit:
                # خروج با سود
                pnl = (take_profit - entry_price) * position_size
                capital += pnl
                trades.append({'type': 'LONG', 'result': 'WIN', 'pnl': pnl})
                position = None
                continue
                
        elif position == 'SHORT':
            if current_high >= stop_loss:
                pnl = (entry_price - stop_loss) * position_size
                capital += pnl
                trades.append({'type': 'SHORT', 'result': 'LOSS', 'pnl': pnl})
                position = None
                continue
            elif current_low <= take_profit:
                pnl = (entry_price - take_profit) * position_size
                capital += pnl
                trades.append({'type': 'SHORT', 'result': 'WIN', 'pnl': pnl})
                position = None
                continue

        # ۲. ورود جدید (اگر پوزیشنی نداریم) - استفاده از تریگر Breakout برای جبران تاخیر
        if position is None and signal != 0:
            position_size = (capital * 0.02) / (current_open * 0.01) # مدیریت ریسک ۲ درصدی
            
            if signal == 1:  # سیگنال خرید
                # تریگر: اگر قیمت از سقف کندل قبل بالاتر رفت ورود کن
                trigger_price = prev_high
                if current_high >= trigger_price:
                    position = 'LONG'
                    entry_price = max(current_open, trigger_price) # قیمت اجرایی واقعی
                    stop_loss = prev_low  # حد ضرر کف کندل قبل
                    take_profit = entry_price + (entry_price - stop_loss) * 2.0  # R:R = 1:2
                    
            elif signal == -1:  # سیگنال فروش
                trigger_price = prev_low
                if current_low <= trigger_price:
                    position = 'SHORT'
                    entry_price = min(current_open, trigger_price)
                    stop_loss = prev_high
                    take_profit = entry_price - (stop_loss - entry_price) * 2.0

    # خروجی نهایی متریک‌ها
    total_trades = len(trades)
    winning_trades = [t for t in trades if t['result'] == 'WIN']
    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
    total_pnl = sum(t['pnl'] for t in trades)
    
    print(f"--- نتیجه بک‌تست مهندسی‌شده (بدون تقلب و بدون تاخیر مخرب) ---")
    print(f"Trades = {total_trades} | Win Rate = {win_rate:.2f}% | Total PnL = ${total_pnl:,.2f}")
    
    return trades

if __name__ == "__main__":
    # تست نمونه روی ساختار داده
    pass
