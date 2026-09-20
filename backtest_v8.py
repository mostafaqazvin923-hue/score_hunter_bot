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

    # ۱. محاسبه اندیکاتورهای ستاپ حرفه‌ای (EMA 50 برای روند و EMA 9 / VWAP شبیه‌سازی‌شده برای پولبک)
    df['EMA_Trend'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_Fast'] = df['Close'].ewm(span=9, adjust=False).mean()
    
    # محاسبه ساده VWAP برای تایم‌فریم ساعتی
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

    # شروع لوپ از کندل ۵۰ به بعد برای دقت کامل اندیکاتورها (به صورت کاملا Causal روی i-1)
    for i in range(50, len(df)):
        current_open = df['Open'].iloc[i]
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]
        
        prev_close = df['Close'].iloc[i-1]
        prev_open = df['Open'].iloc[i-1]
        prev_high = df['High'].iloc[i-1]
        prev_low = df['Low'].iloc[i-1]
        prev_trend = df['EMA_Trend'].iloc[i-1]
        prev_vwap = df['VWAP'].iloc[i-1]
        prev_fast = df['EMA_Fast'].iloc[i-1]
        
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

        # ۲. ورود به معامله با ستاپ پولبک به VWAP و روند (۳ تا ۴ معامله در روز در تایم‌فریم ساعتی)
        if position is None and capital >= fixed_margin:
            # سیگنال خرید (Long): روند صعودی و پولبک قیمت به محدوده VWAP یا میانگین سریع
            if prev_close > prev_trend and prev_close >= prev_vwap and prev_fast > prev_vwap:
                if prev_close > prev_open: # تاییدیه کندل صعودی
                    position = 'LONG'
                    entry_price = current_open
                    stop_loss = prev_low
                    if entry_price > stop_loss:
                        take_profit = entry_price + (entry_price - stop_loss) * 2.0  # R:R = 1:2
                    else:
                        position = None
                    
            # سیگنال فروش (Short): روند نزولی و پولبک قیمت به محدوده VWAP
            elif prev_close < prev_trend and prev_close <= prev_vwap and prev_fast < prev_vwap:
                if prev_close < prev_open: # تاییدیه کندل نزولی
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

    print("=" * 60)
    print("گزارش نهایی بک‌تست (ستاپ حرفه‌ای VWAP Pullback با R:R = 1:2)")
    print("=" * 60)
    print(f"تعداد کل معامله ها: {total_trades}")
    print(f"وین ریت کلی (Win Rate): {win_rate:.2f}%")
    print(f"سود خالص دلاری: ${total_pnl:,.2f}")
    print(f"سرمایه نهایی حساب: ${capital:,.2f}")
    print(f"لیست ضررهای متوالی (Loss Streaks): {loss_streaks}")
    print("=" * 60)

    return trades, equity_curve

if __name__ == "__main__":
    np.random.seed(42)
    # تست روی کندل‌های ۱ ساعته (برای تامین فرکانس بالای معاملات در روز)
    periods_count = 1500  # معادل حدود ۲ ماه دیتای ۱ ساعته
    dates = pd.date_range(start='2026-01-01', periods=periods_count, freq='1h')
    prices = 50000 + np.cumsum(np.random.randn(periods_count) * 60)
    volumes = np.random.randint(100, 1000, size=periods_count)
    
    df_test = pd.DataFrame({
        'Open': prices + np.random.randn(periods_count) * 8,
        'High': prices + abs(np.random.randn(periods_count) * 15),
        'Low': prices - abs(np.random.randn(periods_count) * 15),
        'Close': prices + np.random.randn(periods_count) * 8,
        'Volume': volumes
    }, index=dates)
    
    run_score_hunter_backtest(df_test)
