import pandas as pd
import numpy as np

symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']
total_portfolio_trades = 0
total_portfolio_wins = 0
total_portfolio_losses = 0
initial_total_balance = 1000.0  # سرمایه کل دقیقا ۱۰۰۰ دلار
balance_per_coin = initial_total_balance / len(symbols)
current_total_balance = initial_total_balance

for symbol in symbols:
    print(f"\n⏳ در حال اجرای نسخه پرمعامله واقعی (Score Hunter Pro v8.20) برای {symbol}...")
    
    # تولید دیتای یکساله ۱ ساعته (۸۷۶۰ کندل)
    np.random.seed(hash(symbol) % 2026)
    n_candles = 8760
    base_price = 65000 if 'BTC' in symbol else (3300 if 'ETH' in symbol else (160 if 'SOL' in symbol else 1.2))
    
    trend_cycle = np.sin(np.linspace(0, 45, n_candles)) * 0.06
    returns = np.random.normal(0.0001, 0.012, n_candles) + trend_cycle / 25
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
    risk_percentage = 0.005  # ریسک ۰.۵ درصدی برای کنترل سرمایه در معاملات زیاد
    target_rr = 1.5          # ریوارد مناسب برای اسکلپ پردفعات
    wins = 0
    losses = 0
    total_trades = 0
    lookback = 5             # بازه بسیار کوتاه برای صید حداکثری سیگنال‌ها

    i = lookback + 2
    while i < len(df) - 15:
        c_close = closes[i]
        c_open = opens[i]
        c_high = highs[i]
        c_low = lows[i]
        
        # محاسبه سریع RSI
        period = 14
        if i >= period:
            gains, losses_val = 0.0, 0.0
            for k in range(1, period + 1):
                diff = closes[i - k + 1] - closes[i - k]
                if diff >= 0:
                    gains += diff
                else:
                    losses_val -= diff
            current_rsi = 50.0 if losses_val == 0 else 100.0 - (100.0 / (1.0 + (gains / period) / (losses_val / period)))
        else:
            current_rsi = 50.0

        recent_highs = max(highs[i-lookback:i])
        recent_lows = min(lows[i-lookback:i])
        
        # شرایط پرمعامله بر اساس شکست‌های سریع کوتاه-مدت
        is_bullish = c_close > recent_highs and current_rsi < 68
        is_bearish = c_close < recent_lows and current_rsi > 32

        trade_executed = False

        if is_bullish:
            entry = c_close
            sl = recent_lows - (entry * 0.001)
            risk_dist = entry - sl
            
            if risk_dist > 0:
                tp = entry + (risk_dist * target_rr)
                trade_won, trade_lost = False, False
                
                # ارزیابی آینده‌نگر واقعی در بازه ۱۲ کندل بعدی
                for j in range(i + 1, min(i + 12, len(df))):
                    if highs[j] >= tp:
                        trade_won = True
                        break
                    if lows[j] <= sl:
                        trade_lost = True
                        break
                        
                if trade_won or trade_lost:
                    total_trades += 1
                    current_risk_amount = balance * risk_percentage
                    if trade_won:
                        wins += 1
                        balance += (current_risk_amount * target_rr)
                    else:
                        losses += 1
                        balance -= current_risk_amount
                    
                    # پرش به کندل اتمام معامله برای ثبت پوزیشن‌های بعدی
                    i = j if (trade_won or trade_lost) else i + 1
                    trade_executed = True

        elif is_bearish and not trade_executed:
            entry = c_close
            sl = recent_highs + (entry * 0.001)
            risk_dist = sl - entry
            
            if risk_dist > 0:
                tp = entry - (risk_dist * target_rr)
                trade_won, trade_lost = False, False
                
                for j in range(i + 1, min(i + 12, len(df))):
                    if lows[j] <= tp:
                        trade_won = True
                        break
                    if highs[j] >= sl:
                        trade_lost = True
                        break
                            
                if trade_won or trade_lost:
                    total_trades += 1
                    current_risk_amount = balance * risk_percentage
                    if trade_won:
                        wins += 1
                        balance += (current_risk_amount * target_rr)
                    else:
                        losses += 1
                        balance -= current_risk_amount
                        
                    i = j if (trade_won or trade_lost) else i + 1
                    trade_executed = True

        if not trade_executed:
            i += 1

    total_portfolio_trades += total_trades
    total_portfolio_wins += wins
    total_portfolio_losses += losses
    current_total_balance += (balance - balance_per_coin)

    sym_win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    print(f"[{symbol}] -> معاملات: {total_trades} | برد: {wins} | باخت: {losses} | وین‌ریت: {sym_win_rate:.2f}% | سود/زیان: ${balance - balance_per_coin:.2f}")

portfolio_win_rate = (total_portfolio_wins / total_portfolio_trades * 100) if total_portfolio_trades > 0 else 0

print("\n" + "="*50)
print("📊 گزارش نهایی Score Hunter Pro v8.20 (پرمعامله و واقعی)")
print("="*50)
print(f"تعداد کل معاملات مجموع : {total_portfolio_trades}")
print(f"کل معاملات موفق (Wins) : {total_portfolio_wins}")
print(f"کل معاملات ناموفق (Loss): {total_portfolio_losses}")
print(f"وین‌ریت کلی پورتفیو     : {portfolio_win_rate:.2f}%")
print(f"سرمایه اولیه کل         : ${initial_total_balance:.2f}")
print(f"سرمایه نهایی کل         : ${current_total_balance:.2f}")
print(f"سود خالص کل پورتفیو    : ${current_total_balance - initial_total_balance:.2f}")
print("="*50)
