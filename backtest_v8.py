import pandas as pd
import numpy as np

def run_score_hunter_backtest(df):
    initial_capital = 1000.0
    capital = initial_capital
    fixed_margin = 100.0  # مارجین ثابت ۱۰۰ دلار برای هر معامله
    
    position = None 
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    
    trades = []
    equity_curve = [initial_capital]

    # محاسبه اندیکاتورها روی کل دیتابیس به صورت کائوسال (صرفاً از گذشته برای کندل i-1 استفاده می‌شود)
    df['EMA_Fast'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # محاسبه ساده RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    for i in range(21, len(df)):
        current_open = df['Open'].iloc[i]
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]
        
        prev_fast = df['EMA_Fast'].iloc[i-1]
        prev_slow = df['EMA_Slow'].iloc[i-1]
        prev_rsi = df['RSI'].iloc[i-1]
        prev_low = df['Low'].iloc[i-1]
        prev_high = df['High'].iloc[i-1]
        entry_close = df['Close'].iloc[i-1]
        
        # ۱. مدیریت پوزیشن‌های باز
        if position == 'LONG':
            if current_low <= stop_loss:
                loss_pct = (entry_price - stop_loss) / entry_price
                pnl = - (fixed_margin * loss_pct * 3) # اهرم ۳ برای استراتژی مومنتوم
                capital += pnl
                trades.append({'result': 'LOSS', 'pnl': pnl})
                position = None
            elif current_high >= take_profit:
                win_pct = (take_profit - entry_price) / entry_price
                pnl = fixed_margin * win_pct * 3
                capital += pnl
                trades.append({'result': 'WIN', 'pnl': pnl})
                position = None
                
        elif position == 'SHORT':
            if current_high >= stop_loss:
                loss_pct = (stop_loss - entry_price) / entry_price
                pnl = - (fixed_margin * loss_pct * 3)
                capital += pnl
                trades.append({'result': 'LOSS', 'pnl': pnl})
                position = None
            elif current_low <= take_profit:
                win_pct = (entry_price - take_profit) / entry_price
                pnl = fixed_margin * win_pct * 3
                capital += pnl
                trades.append({'result': 'WIN', 'pnl': pnl})
                position = None

        # ۲. ستاپ ورود جدید (کراس EMA به همراه تاییدیه RSI و ریسک به ریوارد 1 به 2)
        if position is None and capital >= fixed_margin:
            # سیگنال خرید: کراس صعودی و RSI بالای 50 (روند صعودی قوی)
            if prev_fast > prev_slow and prev_rsi > 50:
                position = 'LONG'
                entry_price = current_open
                stop_loss = prev_low
                if entry_price > stop_loss:
                    take_profit = entry_price + (entry_price - stop_loss) * 2.0  # R:R = 1:2
                else:
                    position = None
                    
            # سیگنال فروش: کراس نزولی و RSI زیر 50 (روند نزولی قوی)
            elif prev_fast < prev_slow and prev_rsi < 50:
                position = 'SHORT'
                entry_price = current_open
                stop_loss = prev_high
                if stop_loss > entry_price:
                    take_profit = entry_price - (stop_loss - entry_price) * 2.0  # R:R = 1:2
                else:
                    position = None

        equity_curve.append(capital)

    # محاسبه ضررهای متوالی (Loss Streaks)
    loss_streaks = []
    current_streak = 0
    for t in trades:
        if t['result'] == 'LOSS':
            current_streak += 1
        else:
            if current_streak > 0:
                loss_streaks.append(current_streak)
                current_streak = 0
    if current_streak > 0:
        loss_streaks.append(current_streak)

    total_trades = len(trades)
    winning_trades = [t for t in trades if t['result'] == 'WIN']
    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
    total_pnl = capital - initial_capital

    print("=" * 50)
    print("گزارش نهایی بک‌تست (ستتاپ EMA + RSI با R:R = 1:2)")
    print("=" * 50)
    print(f"تعداد کل معامله ها: {total_trades}")
    print(f"وین ریت کلی (Win Rate): {win_rate:.2f}%")
    print(f"سود خالص کل: ${total_pnl:,.2f}")
    print(f"لیست ضررهای متوالی (Loss Streaks): {loss_streaks}")
    print("=" * 50)

    return trades, equity_curve

if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range(start='2026-01-01', periods=500, freq='4h')
    prices = 50000 + np.cumsum(np.random.randn(500) * 120)
    
    df_test = pd.DataFrame({
        'Open': prices + np.random.randn(500) * 15,
        'High': prices + abs(np.random.randn(500) * 25),
        'Low': prices - abs(np.random.randn(500) * 25),
        'Close': prices + np.random.randn(500) * 15
    }, index=dates)
    
    run_score_hunter_backtest(df_test)
