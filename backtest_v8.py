import pandas as pd
import numpy as np

symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']
total_portfolio_trades = 0
total_portfolio_wins = 0
total_portfolio_losses = 0
initial_total_balance = 1000.0  # کل سرمایه اولیه پورتفیو دقیقاً ۱۰۰۰ دلار
balance_per_coin = initial_total_balance / len(symbols)
current_total_balance = initial_total_balance

for symbol in symbols:
    print(f"\n⏳ در حال اجرای بک‌تست پرمعامله (هدف: ۲ تا ۳ معامله در روز) برای {symbol}...")
    
    # تولید دیتای یکساله ۱ ساعته (۸۷۶۰ کندل)
    np.random.seed(hash(symbol) % 2026)
    n_candles = 8760
    base_price = 65000 if 'BTC' in symbol else (3300 if 'ETH' in symbol else (160 if 'SOL' in symbol else 1.2))
    
    trend_cycle = np.sin(np.linspace(0, 40, n_candles)) * 0.06
    returns = np.random.normal(0.00015, 0.012, n_candles) + trend_cycle / 25
    price_series = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2025-09-03', periods=n_candles, freq='h'),
        'open': price_series * (1 + np.random.normal(0, 0.001, n_candles)),
        'high': price_series * (1 + np.abs(np.random.normal(0.003, 0.0015, n_candles))),
        'low': price_series * (1 - np.abs(np.random.normal(0.003, 0.0015, n_candles))),
        'close': price_series,
        'volume': np.random.uniform(1000, 15000, n_candles)
    })

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values

    balance = balance_per_coin
    risk_percentage = 0.01
    target_rr = 2.0
    wins = 0
    losses = 0
    total_trades = 0
    skip_until = 0
    lookback = 8  # بازه کوتاه‌تر برای تشخیص سریع‌تر سقف و کف‌ها و افزایش تعداد معاملات

    for i in range(lookback + 2, len(df) - 20):
        if i < skip_until:
            continue

        c_close = closes[i]
        c_open = opens[i]
        c_high = highs[i]
        c_low = lows[i]
        
        # محاسبه سریع RSI
        period = 14
        gains, losses_val = 0.0, 0.0
        for k in range(1, period + 1):
            diff = closes[i - k + 1] - closes[i - k]
            if diff >= 0:
                gains += diff
            else:
                losses_val -= diff
        current_rsi = 50.0 if losses_val == 0 else 100.0 - (100.0 / (1.0 + (gains / period) / (losses_val / period)))

        # منطق بهینه‌شده برای حجم بالای معاملات (BOS محلی + مومنتوم)
        recent_highs = max(highs[i-lookback:i-1])
        recent_lows = min(lows[i-lookback:i-1])
        
        is_bullish_bos = c_close > recent_highs and c_close > c_open
        is_bearish_bos = c_close < recent_lows and c_open > c_close

        trade_taken = False
        
        # بررسی پوزیشن لانگ
        if is_bullish_bos and (35 < current_rsi < 68):
            entry = c_close
            sl = recent_lows - (entry * 0.0015)
            risk_dist = entry - sl
            
            if risk_dist > 0 and (risk_dist / entry) <= 0.025:
                tp = entry + (risk_dist * target_rr)
                
                # ارزیابی آینده‌نگر واقعی
                trade_won, trade_lost = False, False
                for j in range(i + 1, min(i + 20, len(df))):
                    if highs[j] >= tp:
                        trade_won = True
                        break
                    if lows[j] <= sl:
                        trade_lost = True
                        break
                        
                if trade_won or trade_lost:
                    total_trades += 1
                    skip_until = j + 1
                    trade_taken = True
                    current_risk_amount = balance * risk_percentage
                    if trade_won:
                        wins += 1
                        balance += (current_risk_amount * target_rr)
                    else:
                        losses += 1
                        balance -= current_risk_amount

        # بررسی پوزیشن شورت
        if not trade_taken and is_bearish_bos and (32 < current_rsi < 65):
            entry = c_close
            sl = recent_highs + (entry * 0.0015)
            risk_dist = sl - entry
            
            if risk_dist > 0 and (risk_dist / entry) <= 0.025:
                tp = entry - (risk_dist * target_rr)
                
                trade_won, trade_lost = False, False
                for j in range(i + 1, min(i + 20, len(df))):
                    if lows[j] <= tp:
                        trade_won = True
                        break
                    if highs[j] >= sl:
                        trade_lost = True
                        break
                            
                    if trade_won or trade_lost:
                        total_trades += 1
                        skip_until = j + 1
                        current_risk_amount = balance * risk_percentage
                        if trade_won:
                            wins += 1
                            balance += (current_risk_amount * target_rr)
                        else:
                            losses += 1
                            balance -= current_risk_amount

    total_portfolio_trades += total_trades
    total_portfolio_wins += wins
    total_portfolio_losses += losses
    current_total_balance += (balance - balance_per_coin)

    sym_win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    print(f"[{symbol}] -> معاملات: {total_trades} | برد: {wins} | باخت: {losses} | وین‌ریت: {sym_win_rate:.2f}% | سود/زیان: ${balance - balance_per_coin:.2f}")

portfolio_win_rate = (total_portfolio_wins / total_portfolio_trades * 100) if total_portfolio_trades > 0 else 0

print("\n" + "="*50)
print("📊 گزارش نهایی بک‌تست پرمعامله (هدف ۸۰۰ تا ۱۰۰۰ معامله در سال)")
print("="*50)
print(f"تعداد کل معاملات مجموع : {total_portfolio_trades}")
print(f"کل معاملات موفق (Wins) : {total_portfolio_wins}")
print(f"کل معاملات ناموفق (Loss): {total_portfolio_losses}")
print(f"وین‌ریت کلی پورتفیو     : {portfolio_win_rate:.2f}%")
print(f"سرمایه اولیه کل         : ${initial_total_balance:.2f}")
print(f"سرمایه نهایی کل         : ${current_total_balance:.2f}")
print(f"سود خالص کل پورتفیو    : ${current_total_balance - initial_total_balance:.2f}")
print("="*50)
