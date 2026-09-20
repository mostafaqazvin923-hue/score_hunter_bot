import pandas as pd
import numpy as np

def run_score_hunter_backtest(df):
    initial_capital = 1000.0
    capital = initial_capital
    fixed_margin = 100.0  # مارجین ثابت ۱۰۰ دلار
    
    position = None 
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    
    trades = []
    equity_curve = [initial_capital]

    # فیلتر روند سریع برای افزایش تعداد معاملات و دقت
    df['EMA_Fast'] = df['Close'].ewm(span=5, adjust=False).mean()

    for i in range(5, len(df)):
        current_open = df['Open'].iloc[i]
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]
        
        prev_close = df['Close'].iloc[i-1]
        prev_open = df['Open'].iloc[i-1]
        prev_high = df['High'].iloc[i-1]
        prev_low = df['Low'].iloc[i-1]
        prev_ema = df['EMA_Fast'].iloc[i-1]
        
        # ۱. مدیریت پوزیشن‌های باز
        if position == 'LONG':
            if current_low <= stop_loss:
                loss_pct = (entry_price - stop_loss) / entry_price
                pnl = - (fixed_margin * loss_pct * 3)
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

        # ۲. ورود پرمعامله (مومنتوم شکست سقف/کف کندل قبل به همراه تاییدیه EMA سریع)
        if position is None and capital >= fixed_margin:
            # سیگنال خرید: قیمت بالاتر از EMA سریع و کندل قبلی صعودی بوده
            if prev_close > prev_ema and prev_close > prev_open:
                position = 'LONG'
                entry_price = current_open
                stop_loss = prev_low
                if entry_price > stop_loss:
                    take_profit = entry_price + (entry_price - stop_loss) * 2.0  # R:R = 1:2
                else:
                    position = None
                    
            # سیگنال فروش: قیمت پایین‌تر از EMA سریع و کندل قبلی نزولی بوده
            elif prev_close < prev_ema and prev_close < prev_open:
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
    print("گزارش نهایی بک‌تست (استراتژی پرمعامله مومنتوم با R:R = 1:2)")
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
