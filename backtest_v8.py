import pandas as pd
import numpy as np

def run_score_hunter_backtest(df):
    """
    کد کامل و یکپارچه بک‌تست v8 برای score_hunter_bot
    - کاملا Causal (بدون نگاه به آینده)
    - دارای منطق تریگر Breakout برای جلوگیری از تاخیر ورود
    - مجهز به محاسبه دقیق Max Drawdown و مدیریت ریسک
    """
    initial_capital = 100000.0
    capital = initial_capital
    peak_capital = initial_capital
    max_drawdown = 0.0
    
    position = None  # None, 'LONG', 'SHORT'
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    position_size = 0.0
    
    trades = []
    equity_curve = [initial_capital]
    
    # فرض بر این است که دیتافریم دارای ستون‌های Open, High, Low, Close و Signal است
    # ستون Signal روی کندل i-1 تولید شده است (1 برای خرید، -1 برای فروش، 0 برای خنثی)
    
    for i in range(2, len(df)):
        current_open = df['Open'].iloc[i]
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]
        current_close = df['Close'].iloc[i]
        
        prev_high = df['High'].iloc[i-1]
        prev_low = df['Low'].iloc[i-1]
        signal = df['Signal'].iloc[i-1]  # استفاده از سیگنال گذشته (بدون تقلب)
        
        # ۱. مدیریت پوزیشن‌های باز (بررسی برخورد با استاپ‌لاس یا تیک‌پرافیت در کندل جاری)
        if position == 'LONG':
            if current_low <= stop_loss:
                # خروج با ضرر (استاپ لاس)
                pnl = (stop_loss - entry_price) * position_size
                capital += pnl
                trades.append({'type': 'LONG', 'result': 'LOSS', 'pnl': pnl})
                position = None
            elif current_high >= take_profit:
                # خروج با سود (تیک پرافیت)
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

        # ۲. ورود جدید (اگر پوزیشنی نداریم) - استفاده از تریگر Breakout
        if position is None and signal != 0:
            risk_amount = capital * 0.02  # ریسک ۲ درصد از سرمایه در هر معامله
            
            if signal == 1:  # سیگنال خرید
                trigger_price = prev_high
                if current_high >= trigger_price:
                    position = 'LONG'
                    entry_price = max(current_open, trigger_price)
                    stop_loss = prev_low  # حد ضرر کف کندل قبل
                    risk_per_unit = entry_price - stop_loss
                    if risk_per_unit > 0:
                        position_size = risk_amount / risk_per_unit
                        take_profit = entry_price + (risk_per_unit * 2.0)  # ریوارد به ریسک 1 به 2
                    
            elif signal == -1:  # سیگنال فروش
                trigger_price = prev_low
                if current_low <= trigger_price:
                    position = 'SHORT'
                    entry_price = min(current_open, trigger_price)
                    stop_loss = prev_high  # حد ضرر سقف کندل قبل
                    risk_per_unit = stop_loss - entry_price
                    if risk_per_unit > 0:
                        position_size = risk_amount / risk_per_unit
                        take_profit = entry_price - (risk_per_unit * 2.0)

        # به‌روزرسانی سرمایه و محاسبه Max Drawdown لحظه‌ای
        equity_curve.append(capital)
        if capital > peak_capital:
            peak_capital = capital
        
        current_drawdown = (peak_capital - capital) / peak_capital * 100
        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown

    # محاسبه گزارش نهایی متریک‌ها
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
    # نمونه برای تست ساختار داده‌ی تستی
    dates = pd.date_range(start='2026-01-01', periods=100, freq='4h')
    np.random.seed(42)
    prices = 10000 + np.cumsum(np.random.randn(100) * 50)
    df_test = pd.DataFrame({
        'Open': prices + np.random.randn(100) * 5,
        'High': prices + abs(np.random.randn(100) * 10),
        'Low': prices - abs(np.random.randn(100) * 10),
        'Close': prices + np.random.randn(100) * 5
    }, index=dates)
    
    # تولید ستون سیگنال تستی تصادفی
    df_test['Signal'] = np.random.choice([0, 1, -1], size=len(df_test))
    
    run_score_hunter_backtest(df_test)
