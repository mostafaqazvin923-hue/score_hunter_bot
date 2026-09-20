import os
import glob
import pandas as pd
import numpy as np

def load_repository_data():
    """جستجوی خودکار فایل دیتا در مخزن برای جلوگیری از خطای مسیر"""
    possible_paths = [
        "data.csv",
        "dataset.csv",
        "historical_data.csv",
        "data/*.csv",
        "*.csv"
    ]
    
    for pattern in possible_paths:
        files = glob.glob(pattern)
        if files:
            print(f"-> فایل دیتا پیدا شد: {files[0]}")
            return pd.read_csv(files[0])
            
    raise FileNotFoundError("هیچ فایل دیتایی (CSV) در مخزن پیدا نشد! لطفاً مسیر دیتا را بررسی کنید.")

def run_score_hunter_backtest(df):
    """
    بک‌تست استاندارد و Causal (بدون نگاه به آینده) برای score_hunter_bot
    """
    initial_capital = 100000.0
    capital = initial_capital
    peak_capital = initial_capital
    max_drawdown = 0.0
    
    position = None 
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    position_size = 0.0
    
    trades = []
    equity_curve = [initial_capital]
    
    # اطمینان از وجود ستون‌های اصلی
    required_columns = ['Open', 'High', 'Low', 'Close', 'Signal']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"ستون ضروری '{col}' در دیتافریم موجود نیست!")

    for i in range(1, len(df)):
        current_open = df['Open'].iloc[i]
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]
        current_close = df['Close'].iloc[i]
        
        prev_open = df['Open'].iloc[i-1]
        prev_close = df['Close'].iloc[i-1]
        prev_high = df['High'].iloc[i-1]
        prev_low = df['Low'].iloc[i-1]
        signal = df['Signal'].iloc[i-1] # استفاده از سیگنال کندل قبلی (بدون تقلب)
        
        # ۱. مدیریت پوزیشن‌های باز
        if position == 'LONG':
            if current_low <= stop_loss:
                pnl = (stop_loss - entry_price) * position_size
                capital += pnl
                trades.append({'type': 'LONG', 'result': 'LOSS', 'pnl': pnl})
                position = None
            elif current_high >= take_profit:
                pnl = (take_profit - entry_price) * position_size
                capital += pnl
                trades.append({'type': 'LONG', 'result': 'WIN', 'pnl': pnl})
                position = None
                
        elif position == 'SHORT':
            if current_high >= stop_loss:
                pnl = (entry_price - stop_loss) * position_size
                capital += pnl
                trades.append({'type': 'SHORT', 'result': 'LOSS', 'pnl': pnl})
                position = None
            elif current_low <= take_profit:
                pnl = (entry_price - take_profit) * position_size
                capital += pnl
                trades.append({'type': 'SHORT', 'result': 'WIN', 'pnl': pnl})
                position = None

        # ۲. ورود جدید
        if position is None and signal != 0:
            risk_amount = capital * 0.02 
            
            if signal == 1:  # خرید
                if prev_close > prev_open:
                    position = 'LONG'
                    entry_price = current_open
                    stop_loss = prev_low
                    risk_per_unit = entry_price - stop_loss
                    if risk_per_unit > 0:
                        position_size = risk_amount / risk_per_unit
                        take_profit = entry_price + (risk_per_unit * 2.0)
                    
            elif signal == -1:  # فروش
                if prev_close < prev_open:
                    position = 'SHORT'
                    entry_price = current_open
                    stop_loss = prev_high
                    risk_per_unit = stop_loss - entry_price
                    if risk_per_unit > 0:
                        position_size = risk_amount / risk_per_unit
                        take_profit = entry_price - (risk_per_unit * 2.0)

        equity_curve.append(capital)
        if capital > peak_capital:
            peak_capital = capital
        
        current_drawdown = (peak_capital - capital) / peak_capital * 100
        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown

    total_trades = len(trades)
    winning_trades = [t for t in trades if t['result'] == 'WIN']
    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
    total_pnl = capital - initial_capital
    return_pct = (total_pnl / initial_capital) * 100

    print("=" * 50)
    print("گزارش نهایی بک‌تست (Score Hunter Bot - V8)")
    print("=" * 50)
    print(f"کل معاملات: {total_trades}")
    print(f"وین ریت (Win Rate): {win_rate:.2f}%")
    print(f"سود خالص کل: ${total_pnl:,.2f} ({return_pct:.2f}%)")
    print(f"حداکثر افت سرمایه (Max Drawdown): {max_drawdown:.2f}%")
    print("=" * 50)

    return trades, equity_curve

if __name__ == "__main__":
    df_data = load_repository_data()
    run_score_hunter_backtest(df_data)
