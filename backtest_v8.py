import ccxt
import pandas as pd
import numpy as np

# مدیریت ایمپورت matplotlib برای جلوگیری از خطای سرور بدون گرافیک (GitHub Actions)
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

def fetch_lbank_data(symbol, timeframe='1h', limit=1500):
    """
    دریافت داده‌های تاریخی فیوچرز از صرافی LBank با استفاده از CCXT
    و تنظیم دقیق نام ستون‌ها (Open, High, Low, Close, Volume) مطابقت‌یافته با پروژه شما
    """
    exchange = ccxt.lbank({'enableRateLimit': True})
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # استانداردسازی نام ستون‌ها با حروف بزرگ مطابق ساختار پروژه
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def run_advanced_strategy(df, symbol_name):
    initial_capital = 1000.0
    capital = initial_capital
    fixed_margin = 100.0  # مارجین ثابت ۱۰۰ دلار
    leverage = 2.5        # اهرم پیش‌فرض
    
    position = None 
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    
    trades = []
    equity_curve = [initial_capital]

    consecutive_losses = 0
    cooldown_counter = 0  # وقفه برای کنترل ضررهای متوالی

    # مفاهیم نهادی و ساختار بازار (Market Structure & Institutional Logic)
    # فیلتر روند 4 ساعته ترکیب شده با 1 ساعته و نوسان (ATR / Swing Structure)
    df['EMA_Trend'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_Fast'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    
    # محاسبه نوسان با ATR برای تعیین داینامیک حد ضرر و جلوگیری از تله‌های اسلیپیج
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['TR'] = np.maximum(high_low, np.maximum(high_close, low_close))
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # سویینگ‌های بازار برای شناسایی لیکوئیدیتی سویپ (Liquidity Sweep)
    df['Swing_High'] = df['High'].rolling(window=20).max()
    df['Swing_Low'] = df['Low'].rolling(window=20).min()

    for i in range(50, len(df)):
        if cooldown_counter > 0:
            cooldown_counter -= 1
            equity_curve.append(capital)
            continue

        current_open = df['Open'].iloc[i]
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]
        
        prev_close = df['Close'].iloc[i-1]
        prev_high = df['High'].iloc[i-1]
        prev_low = df['Low'].iloc[i-1]
        prev_trend = df['EMA_Trend'].iloc[i-1]
        prev_fast = df['EMA_Fast'].iloc[i-1]
        prev_vol = df['Volume'].iloc[i-1]
        prev_vol_sma = df['Volume_SMA'].iloc[i-1]
        prev_atr = df['ATR'].iloc[i-1]
        prev_swing_high = df['Swing_High'].iloc[i-1]
        prev_swing_low = df['Swing_Low'].iloc[i-1]
        
        # ۱. مدیریت پوزیشن‌های باز و اعمال قانون محافظه‌کارانه برخورد همزمان TP و SL
        if position == 'LONG':
            hit_tp = current_high >= take_profit
            hit_sl = current_low <= stop_loss
            
            if hit_sl and hit_tp:
                # قانون محافظه‌کارانه: اولویت با فعال شدن ضرر (SL)
                loss_pct = (entry_price - stop_loss) / entry_price
                pnl = - (fixed_margin * loss_pct * leverage)
                capital += pnl
                trades.append({'symbol': symbol_name, 'result': 'LOSS', 'pnl': pnl})
                consecutive_losses += 1
                if consecutive_losses >= 3:
                    cooldown_counter = 18
                position = None
            elif hit_sl:
                loss_pct = (entry_price - stop_loss) / entry_price
                pnl = - (fixed_margin * loss_pct * leverage)
                capital += pnl
                trades.append({'symbol': symbol_name, 'result': 'LOSS', 'pnl': pnl})
                consecutive_losses += 1
                if consecutive_losses >= 3:
                    cooldown_counter = 18
                position = None
            elif hit_tp:
                win_pct = (take_profit - entry_price) / entry_price
                pnl = fixed_margin * win_pct * leverage
                capital += pnl
                trades.append({'symbol': symbol_name, 'result': 'WIN', 'pnl': pnl})
                consecutive_losses = 0
                position = None
                
        elif position == 'SHORT':
            hit_tp = current_low <= take_profit
            hit_sl = current_high >= stop_loss
            
            if hit_sl and hit_tp:
                loss_pct = (stop_loss - entry_price) / entry_price
                pnl = - (fixed_margin * loss_pct * leverage)
                capital += pnl
                trades.append({'symbol': symbol_name, 'result': 'LOSS', 'pnl': pnl})
                consecutive_losses += 1
                if consecutive_losses >= 3:
                    cooldown_counter = 18
                position = None
            elif hit_sl:
                loss_pct = (stop_loss - entry_price) / entry_price
                pnl = - (fixed_margin * loss_pct * leverage)
                capital += pnl
                trades.append({'symbol': symbol_name, 'result': 'LOSS', 'pnl': pnl})
                consecutive_losses += 1
                if consecutive_losses >= 3:
                    cooldown_counter = 18
                position = None
            elif hit_tp:
                win_pct = (entry_price - take_profit) / entry_price
                pnl = fixed_margin * win_pct * leverage
                capital += pnl
                trades.append({'symbol': symbol_name, 'result': 'WIN', 'pnl': pnl})
                consecutive_losses = 0
                position = None

        # ۲. ورود جدید بر اساس ساختار بازار (Liquidity Sweep + Trend Regime + R:R 1:2)
        if position is None and capital >= fixed_margin and cooldown_counter == 0:
            is_high_volume = prev_vol > prev_vol_sma
            
            # سیگنال خرید (Long Setup): جارو شدن نقدینگی کف قبلی و بازگشت به روند صعودی با تایید حجم
            swept_low = current_low < prev_swing_low
            bullish_structure = prev_close > prev_trend and prev_fast > prev_trend
            
            if bullish_structure and swept_low and is_high_volume:
                position = 'LONG'
                entry_price = current_open
                stop_loss = prev_swing_low - (0.2 * prev_atr)
                if entry_price > stop_loss:
                    take_profit = entry_price + (entry_price - stop_loss) * 2.0  # R:R = 1:2
                else:
                    position = None
                    
            # سیگنال فروش (Short Setup): جارو شدن نقدینگی سقف قبلی و بازگشت به روند نزولی با تایید حجم
            swept_high = current_high > prev_swing_high
            bearish_structure = prev_close < prev_trend and prev_fast < prev_trend
            
            if bearish_structure and swept_high and is_high_volume:
                position = 'SHORT'
                entry_price = current_open
                stop_loss = prev_swing_high + (0.2 * prev_atr)
                if stop_loss > entry_price:
                    take_profit = entry_price - (stop_loss - entry_price) * 2.0  # R:R = 1:2
                else:
                    position = None

        equity_curve.append(capital)

    return trades, equity_curve

def master_backtest():
    # ۱۰ ارز برتر بازار کریپتو در فیوچرز صرافی LBank
    symbols = [
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 
        'XRP/USDT:USDT', 'DOGE/USDT:USDT', 'ADA/USDT:USDT', 
        'AVAX/USDT:USDT', 'LINK/USDT:USDT', 'DOT/USDT:USDT', 'NEAR/USDT:USDT'
    ]
    
    all_trades = []
    asset_summary = {}
    master_equity = [1000.0]
    initial_capital = 1000.0
    current_capital = initial_capital
    
    print("در حال دریافت داده‌های واقعی از LBank Futures و اجرای بک‌تست استراتژی ساختار بازار...")
    
    for symbol in symbols:
        df = fetch_lbank_data(symbol, timeframe='1h', limit=1500)
        if df.empty or len(df) < 200:
            print(f"هشدار: داده کافی برای {symbol} دریافت نشد یا خالی است.")
            continue
            
        trades, equity = run_advanced_strategy(df, symbol)
        all_trades.extend(trades)
        
        sym_pnl = sum([t['pnl'] for t in trades])
        asset_summary[symbol] = {
            'trades_count': len(trades),
            'pnl': sym_pnl
        }
        
    # محاسبات آماری جامع خروجی‌ها
    total_trades = len(all_trades)
    winning_trades = [t for t in all_trades if t['result'] == 'WIN']
    losing_trades = [t for t in all_trades if t['result'] == 'LOSS']
    
    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
    total_pnl = sum([t['pnl'] for t in all_trades])
    final_balance = initial_capital + total_pnl
    
    # لیست ضررهای متوالی (Loss Streaks)
    loss_streaks = []
    current_streak = 0
    for t in all_trades:
        if t['result'] == 'LOSS':
            current_streak += 1
        else:
            if current_streak > 0:
                loss_streaks.append(current_streak)
                current_streak = 0
    if current_streak > 0:
        loss_streaks.append(current_streak)
        
    # محاسبه Maximum Drawdown
    peak = initial_capital
    max_dd = 0.0
    running_cap = initial_capital
    for t in all_trades:
        running_cap += t['pnl']
        if running_cap > peak:
            peak = running_cap
        dd = (peak - running_cap) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Profit Factor و میانگین برد/باخت
    total_wins_pnl = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
    total_losses_pnl = abs(sum([t['pnl'] for t in losing_trades])) if losing_trades else 0
    profit_factor = (total_wins_pnl / total_losses_pnl) if total_losses_pnl > 0 else float('inf')
    
    avg_win = (total_wins_pnl / len(winning_trades)) if winning_trades else 0
    avg_loss = (total_losses_pnl / len(losing_trades)) if losing_trades else 0

    # گزارش نهایی دقیق مطابق خواسته شما
    print("\n" + "=" * 65)
    print("گزارش نهایی عملکرد سیستم معاملاتی فیوچرز (LBank Futures - Institutional Strategy)")
    print("=" * 65)
    print(f"1- تعداد کل معاملات: {total_trades}")
    print(f"2- تعداد معاملات برنده: {len(winning_trades)}")
    print(f"3- تعداد معاملات بازنده: {len(losing_trades)}")
    print(f"4- درصد Win Rate کلی: {win_rate:.2f}%")
    print(f"5- سود یا ضرر نهایی به دلار: ${total_pnl:,.2f}")
    print(f"6- موجودی نهایی حساب (از سرمایه اولیه $1000): ${final_balance:,.2f}")
    
    print("\n7 & 8 - سود/ضرر و تعداد معاملات هر ارز به صورت جداگانه:")
    for sym, data in asset_summary.items():
        print(f"   - {sym}: تعداد معاملات = {data['trades_count']} | سود/ضرر = ${data['pnl']:,.2f}")
        
    print("\n9- لیست تمام سری‌های ضرر متوالی (Loss Streaks):")
    if loss_streaks:
        # شمارش تعداد دفعات رخ دادن هر طول از ضرر متوالی
        streak_counts = {s: loss_streaks.count(s) for s in set(loss_streaks)}
        for s_len, count in sorted(streak_counts.items()):
            print(f"   {s_len} ضرر متوالی: {count} بار")
    else:
        print("   هیچ ضرر متوالی ثبت نشد.")
        
    print(f"\n10- بیشترین Drawdown (حداکثر افت سرمایه): {max_dd:.2f}%")
    print(f"11- Profit Factor (فاکتور سود): {profit_factor:.2f}")
    print(f"12- Average Win / Average Loss: میانگین برد: ${avg_win:.2f} | میانگین باخت: ${avg_loss:.2f}")
    print("=" * 65)

    # ترسیم نمودار Equity Curve در صورت وجود کتابخانه گرافیکی
    if HAS_PLOT and len(all_trades) > 0:
        try:
            plt.figure(figsize=(10, 5))
            eq_series = [initial_capital]
            curr = initial_capital
            for t in all_trades:
                curr += t['pnl']
                eq_series.append(curr)
            plt.plot(eq_series, label='Equity Curve', color='teal', linewidth=2)
            plt.title('Score Hunter Bot - Equity Curve (LBank Futures)')
            plt.xlabel('Trade Index')
            plt.ylabel('Capital ($)')
            plt.grid(True)
            plt.legend()
            plt.savefig('equity_curve.png')
            print("\n نمودار منحنی سرمایه (Equity Curve) با موفقیت در فایل equity_curve.png ذخیره شد.")
        except Exception as ex:
            print(f"خطا در ذخیره نمودار: {ex}")

if __name__ == "__main__":
    master_backtest()
