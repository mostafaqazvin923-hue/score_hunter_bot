import pandas as pd
import numpy as np

symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']

total_portfolio_trades = 0
total_portfolio_wins = 0
total_portfolio_losses = 0
initial_balance_per_coin = 1000.0  # سرمایه اولیه ۱۰۰۰ دلار برای هر ارز
initial_total_balance = len(symbols) * initial_balance_per_coin
current_total_balance = initial_total_balance

for symbol in symbols:
    print(f"\n⏳ در حال اجرای نسخه بهینه‌شده Score Hunter Pro v8.16 برای {symbol}...")
    
    np.random.seed(hash(symbol) % 2026)
    n_candles = 8760
    base_price = 65000 if 'BTC' in symbol else (3300 if 'ETH' in symbol else (160 if 'SOL' in symbol else 1.2))
    
    # تولید دیتای تاریخی یکساله (استفاده از 'h' به جای 'H' مطابق استاندارد جدید پانداس)
    trend_cycle = np.sin(np.linspace(0, 30, n_candles)) * 0.04
    returns = np.random.normal(0.00012, 0.013, n_candles) + trend_cycle / 40
    price_series = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2025-09-03', periods=n_candles, freq='h'),
        'open': price_series * (1 + np.random.normal(0, 0.001, n_candles)),
        'high': price_series * (1 + np.abs(np.random.normal(0.004, 0.002, n_candles))),
        'low': price_series * (1 - np.abs(np.random.normal(0.004, 0.002, n_candles))),
        'close': price_series,
        'volume': np.random.uniform(1000, 15000, n_candles)
    })

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values

    # محاسبه اندیکاتور ATR و RSI برای ورود دقیق
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))))
    atr = pd.Series(tr).rolling(window=14).mean().values

    delta = pd.Series(closes).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    balance = initial_balance_per_coin
    risk_percentage = 0.01
    target_rr = 2.5
    wins = 0
    losses = 0
    total_trades = 0
    skip_until = 0
    lookback = 15

    for i in range(lookback, len(df) - 30):
        if i < skip_until:
            continue

        c_close = closes[i]
        c_open = opens[i]
        c_high = highs[i]
        c_low = lows[i]
        c_atr = atr[i]
        c_rsi = rsi.iloc[i]

        if np.isnan(c_atr) or c_atr == 0 or np.isnan(c_rsi):
            continue

        current_risk_amount = balance * risk_percentage
        trade_taken = False

        recent_low = min(lows[i-lookback:i])
        recent_high = max(highs[i-lookback:i])

        # ستاپ لانگ اصلاح‌شده (انعطاف‌پذیر و پربازده)
        is_sweep_low = lows[i-1] <= recent_low
        is_bullish_trigger = c_close > c_open and c_rsi < 42

        if is_sweep_low and is_bullish_trigger:
            entry_price = c_close
            stop_loss = recent_low - (c_atr * 0.8)
            risk_dist = entry_price - stop_loss

            if risk_dist > 0:
                take_profit = entry_price + (risk_dist * target_rr)
                trade_won, trade_lost = False, False
                
                for j in range(i + 1, min(i + 35, len(df))):
                    if highs[j] >= take_profit:
                        trade_won = True; break
                    if lows[j] <= stop_loss:
                        trade_lost = True; break

                if trade_won or trade_lost:
                    total_trades += 1
                    skip_until = j + 1
                    trade_taken = True
                    if trade_won:
                        wins += 1
                        balance += (current_risk_amount * target_rr)
                    else:
                        losses += 1
                        balance -= current_risk_amount

        # ستاپ شورت اصلاح‌شده
        is_sweep_high = highs[i-1] >= recent_high
        is_bearish_trigger = c_open > c_close and c_rsi > 58

        if not trade_taken and is_sweep_high and is_bearish_trigger:
            entry_price = c_close
            stop_loss = recent_high + (c_atr * 0.8)
            risk_dist = stop_loss - entry_price

            if risk_dist > 0:
                take_profit = entry_price - (risk_dist * target_rr)
                trade_won, trade_lost = False, False
                
                for j in range(i + 1, min(i + 35, len(df))):
                    if lows[j] <= take_profit:
                        trade_won = True; break
                    if highs[j] >= stop_loss:
                        trade_lost = True; break

                if trade_won or trade_lost:
                    total_trades += 1
                    skip_until = j + 1
                    if trade_won:
                        wins += 1
                        balance += (current_risk_amount * target_rr)
                    else:
                        losses += 1
                        balance -= current_risk_amount

    total_portfolio_trades += total_trades
    total_portfolio_wins += wins
    total_portfolio_losses += losses
    current_total_balance += (balance - initial_balance_per_coin)

    sym_win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    print(f"[{symbol}] -> معاملات: {total_trades} | برد: {wins} | باخت: {losses} | وین‌ریت: {sym_win_rate:.2f}% | سود/زیان: ${balance - initial_balance_per_coin:.2f}")

portfolio_win_rate = (total_portfolio_wins / total_portfolio_trades * 100) if total_portfolio_trades > 0 else 0

print("\n" + "="*50)
print("📊 گزارش نهایی بک‌تست Score Hunter Pro v8.16")
print("="*50)
print(f"تعداد کل معاملات مجموع : {total_portfolio_trades}")
print(f"کل معاملات موفق (Wins) : {total_portfolio_wins}")
print(f"کل معاملات ناموفق (Loss): {total_portfolio_losses}")
print(f"وین‌ریت کلی پورتفیو     : {portfolio_win_rate:.2f}%")
print(f"سرمایه اولیه کل         : ${initial_total_balance:.2f}")
print(f"سرمایه نهایی کل         : ${current_total_balance:.2f}")
print(f"سود خالص کل پورتفیو    : ${current_total_balance - initial_total_balance:.2f}")
print("="*50)
