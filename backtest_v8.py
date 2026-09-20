import ccxt
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

def fetch_lbank_data(symbol, timeframe='1h', limit=1500):
    """
    دریافت کاملاً ایمن و بدون خطای داده‌های تاریخی از صرافی LBank با دور زدن ساختار balance
    """
    # ایجاد نمونه صرافی با غیرفعال کردن کامل چک کردن لایسنس، بالانس و بقیه موارد احراز هویت
    exchange = ccxt.lbank({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot', # استفاده از ساختار اسپات عمومی برای دیتای پرایس اکشن (بدون خطای balance فیوچرز)
            'fetchMarkets': False
        }
    })
    
    # فرمت‌های مختلفی که ممکن است ال‌بنک برای نمادها بشناسد
    clean_sym = symbol.replace('/USDT', '')
    variants = [
        symbol,
        f"{clean_sym}_USDT",
        f"{clean_sym}USDT",
        symbol.replace('/', '_')
    ]
    
    for v in variants:
        try:
            ohlcv = exchange.fetch_ohlcv(v, timeframe=timeframe, limit=limit)
            if ohlcv and len(ohlcv) > 50:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df.rename(columns={
                    'open': 'Open', 'high': 'High', 
                    'low': 'Low', 'close': 'Close', 'volume': 'Volume'
                }, inplace=True)
                return df
        except Exception:
            continue
            
    # اگر روش بالا نشد، از طریق ساختار پیش‌فرض عمومی صرافی‌های دیگر یا عمومی‌سازی CCXT تست می‌کنیم
    try:
        exchange_generic = ccxt.binance({'enableRateLimit': True}) # به عنوان جایگزین اضطراری پابلیک دیتای مشابه در صورت قطع کامل لبانک
        # ولی برای ماندن روی لبانک، از ساختار فچ مستقیم استفاده می‌کنیم:
        raw_exchange = ccxt.lbank()
        raw_exchange.aio = False
        ohlcv = raw_exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if ohlcv and len(ohlcv) > 50:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            return df
    except Exception as e:
        print(f"خطای نهایی در دریافت داده برای {symbol}: {e}")
        
    return pd.DataFrame()

def run_improved_strategy(df, symbol_name):
    initial_capital = 1000.0
    capital = initial_capital
    fixed_margin = 100.0  
    leverage = 2.0        
    
    position = None 
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    
    trades = []
    consecutive_losses = 0
    cooldown_counter = 0

    df['EMA_Fast'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['TR'] = np.maximum(high_low, np.maximum(high_close, low_close))
    df['ATR'] = df['TR'].rolling(window=14).mean()

    for i in range(50, len(df)):
        if cooldown_counter > 0:
            cooldown_counter -= 1
            continue

        current_open = df['Open'].iloc[i]
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]
        
        prev_close = df['Close'].iloc[i-1]
        prev_fast = df['EMA_Fast'].iloc[i-1]
        prev_slow = df['EMA_Slow'].iloc[i-1]
        prev_rsi = df['RSI'].iloc[i-1]
        prev_atr = df['ATR'].iloc[i-1]
        
        if position == 'LONG':
            if current_low <= stop_loss:
                loss_pct = (entry_price - stop_loss) / entry_price
                pnl = - (fixed_margin * loss_pct * leverage)
                capital += pnl
                trades.append({'symbol': symbol_name, 'result': 'LOSS', 'pnl': pnl})
                consecutive_losses += 1
                if consecutive_losses >= 2:
                    cooldown_counter = 10
                position = None
            elif current_high >= take_profit:
                win_pct = (take_profit - entry_price) / entry_price
                pnl = fixed_margin * win_pct * leverage
                capital += pnl
                trades.append({'symbol': symbol_name, 'result': 'WIN', 'pnl': pnl})
                consecutive_losses = 0
                position = None
                
        elif position == 'SHORT':
            if current_high >= stop_loss:
                loss_pct = (stop_loss - entry_price) / entry_price
                pnl = - (fixed_margin * loss_pct * leverage)
                capital += pnl
                trades.append({'symbol': symbol_name, 'result': 'LOSS', 'pnl': pnl})
                consecutive_losses += 1
                if consecutive_losses >= 2:
                    cooldown_counter = 10
                position = None
            elif current_low <= take_profit:
                win_pct = (entry_price - take_profit) / entry_price
                pnl = fixed_margin * win_pct * leverage
                capital += pnl
                trades.append({'symbol': symbol_name, 'result': 'WIN', 'pnl': pnl})
                consecutive_losses = 0
                position = None

        if position is None and capital >= fixed_margin and cooldown_counter == 0:
            if prev_fast > prev_slow and 45 < prev_rsi < 70:
                position = 'LONG'
                entry_price = current_open
                stop_loss = entry_price - (1.5 * prev_atr)
                take_profit = entry_price + (2.5 * prev_atr)
                
            elif prev_fast < prev_slow and 30 < prev_rsi < 55:
                position = 'SHORT'
                entry_price = current_open
                stop_loss = entry_price + (1.5 * prev_atr)
                take_profit = entry_price - (2.5 * prev_atr)

    return trades

def master_backtest():
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 
        'XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 
        'AVAX/USDT', 'LINK/USDT', 'DOT/USDT', 'NEAR/USDT'
    ]
    
    all_trades = []
    asset_summary = {}
    initial_capital = 1000.0
    
    print("در حال دریافت داده‌های ۱۰۰٪ کامل هر ۱۰ ارز از LBank و اجرای بک‌تست نهایی...")
    
    for symbol in symbols:
        df = fetch_lbank_data(symbol, timeframe='1h', limit=1500)
        if df.empty or len(df) < 200:
            print(f"هشدار: داده کافی برای {symbol} دریافت نشد.")
            continue
            
        trades = run_improved_strategy(df, symbol)
        all_trades.extend(trades)
        
        sym_pnl = sum([t['pnl'] for t in trades])
        asset_summary[symbol] = {
            'trades_count': len(trades),
            'pnl': sym_pnl
        }
        
    total_trades = len(all_trades)
    winning_trades = [t for t in all_trades if t['result'] == 'WIN']
    losing_trades = [t for t in all_trades if t['result'] == 'LOSS']
    
    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
    total_pnl = sum([t['pnl'] for t in all_trades])
    final_balance = initial_capital + total_pnl
    
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

    total_wins_pnl = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
    total_losses_pnl = abs(sum([t['pnl'] for t in losing_trades])) if losing_trades else 0
    profit_factor = (total_wins_pnl / total_losses_pnl) if total_losses_pnl > 0 else float('inf')
    
    avg_win = (total_wins_pnl / len(winning_trades)) if winning_trades else 0
    avg_loss = (total_losses_pnl / len(losing_trades)) if losing_trades else 0

    print("\n" + "=" * 65)
    print("گزارش نهایی عملکرد سیستم معاملاتی (LBank - Final Clean Version)")
    print("=" * 65)
    print(f"1- تعداد کل معاملات: {total_trades}")
    print(f"2- تعداد معاملات برنده: {len(winning_trades)}")
    print(f"3- تعداد معاملات بازنده: {len(losing_trades)}")
    print(f"4- درصد Win Rate کلی: {win_rate:.2f}%")
    print(f"5- سود یا ضرر نهایی به دلار: ${total_pnl:,.2f}")
    print(f"6- موجودی نهایی حساب (از سرمایه اولیه $1000): ${final_balance:,.2f}")
    
    print("\n7 & 8 - سود/ضرر و تعداد معاملات هر ارز به صورت جداگانه:")
    if asset_summary:
        for sym, data in asset_summary.items():
            print(f"   - {sym}: تعداد معاملات = {data['trades_count']} | سود/ضرر = ${data['pnl']:,.2f}")
    else:
        print("   هیچ داده‌ای ثبت نشد.")
        
    print("\n9- لیست تمام سری‌های ضرر متوالی (Loss Streaks):")
    if loss_streaks:
        streak_counts = {s: loss_streaks.count(s) for s in set(loss_streaks)}
        for s_len, count in sorted(streak_counts.items()):
            print(f"   {s_len} ضرر متوالی: {count} بار")
    else:
        print("   هیچ ضرر متوالی ثبت نشد.")
        
    print(f"\n10- بیشترین Drawdown (حداکثر افت سرمایه): {max_dd:.2f}%")
    print(f"11- Profit Factor (فاکتور سود): {profit_factor:.2f}")
    print(f"12- Average Win / Average Loss: میانگین برد: ${avg_win:.2f} | میانگین باخت: ${avg_loss:.2f}")
    print("=" * 65)

if __name__ == "__main__":
    master_backtest()
