import subprocess
import sys

# نصب پکیج‌های تحلیل داده
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy"])

import pandas as pd
import numpy as np

symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']

total_portfolio_trades = 0
total_portfolio_wins = 0
total_portfolio_losses = 0
initial_total_balance = len(symbols) * 1000.0
current_total_balance = initial_total_balance

for symbol in symbols:
    print(f"\n⏳ در حال اجرای استراتژی اسمارت مانی (SMC & Order Block) برای {symbol}...")
    
    # تولید دیتای استاندارد یک‌ساله (۸۷۶۰ کندل ۱ ساعته) با نوسانات ساختاری واقعی بازار
    np.random.seed(hash(symbol) % 2026)
    n_candles = 8760
    base_price = 65000 if 'BTC' in symbol else (3300 if 'ETH' in symbol else (160 if 'SOL' in symbol else 1.2))
    
    # مدل‌سازی روندهای واقعی بازار (Trending + Mean Reversion ترکیبی)
    trend_cycle = np.sin(np.linspace(0, 25, n_candles)) * 0.03
    returns = np.random.normal(0.0001, 0.012, n_candles) + trend_cycle / 50
    price_series = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2025-09-03', periods=n_candles, freq='H'),
        'open': price_series * (1 + np.random.normal(0, 0.0015, n_candles)),
        'high': price_series * (1 + np.abs(np.random.normal(0.004, 0.002, n_candles))),
        'low': price_series * (1 - np.abs(np.random.normal(0.004, 0.002, n_candles))),
        'close': price_series,
        'volume': np.random.uniform(500, 10000, n_candles)
    })

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values

    # محاسبه ATR برای تعیین حد ضرر ساختاری
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))))
    atr = pd.Series(tr).rolling(window=14).mean().values

    balance = 1000.0
    risk_percentage = 0.01
    target_rr = 3.0  # ضریب پاداش به ریسک حرفه‌ای (1 به 3) برای جبران وین‌ریت
    wins = 0
    losses = 0
    total_trades = 0
    skip_until = 0
    lookback = 20

    for i in range(lookback, len(df) - 30):
        if i < skip_until:
            continue

        c_close = closes[i]
        c_open = opens[i]
        c_high = highs[i]
        c_low = lows[i]
        c_atr = atr[i]

        if np.isnan(c_atr) or c_atr == 0:
            continue

        current_risk_amount = balance * risk_percentage
        trade_taken = False

        # شناسایی ساختار بازار (Market Structure Break / BOS) و Order Block صعودی
        recent_high = max(highs[i-lookback:i])
        recent_low = min(lows[i-lookback:i])
        
        # تاییدیه اسمارت مانی: نفوذ به زیر کف قبلی (جمع‌آوری نقدینگی / Liquidity Sweep) و بازگشت سریع به داخل ساختار
        is_liquidity_sweep_low = lows[i-1] <= recent_low and c_close > recent_low
        is_bullish_order_block = c_close > c_open and (c_close - c_open) > (c_atr * 0.8)

        if is_liquidity_sweep_low and is_bullish_order_block:
            entry_price = c_close
            stop_loss = recent_low - (c_atr * 0.5) # حد ضرر پشت اردر بلاک بانک‌ها
            risk_dist = entry_price - stop_loss

            if risk_dist > 0:
                take_profit = entry_price + (risk_dist * target_rr)
                trade_won, trade_lost = False, False
                
                for j in range(i + 1, min(i + 40, len(df))):
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

        # شناسایی اردر بلاک نزولی (شارک شورت اسمارت مانی)
        is_liquidity_sweep_high = highs[i-1] >= recent_high and c_close < recent_high
        is_bearish_order_block = c_open > c_close and (c_open - c_close) > (c_atr * 0.8)

        if not trade_taken and is_liquidity_sweep_high and is_bearish_order_block:
            entry_price = c_close
            stop_loss = recent_high + (c_atr * 0.5)
            risk_dist = stop_loss - entry_price

            if risk_dist > 0:
                take_profit = entry_price - (risk_dist * target_rr)
                trade_won, trade_lost = False, False
                
                for j in range(i + 1, min(i + 40, len(df))):
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
    current_total_balance += (balance - 1000.0)

    sym_win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    print(f"[{symbol}] -> معاملات SMC: {total_trades} | برد: {wins} | باخت: {losses} | وین‌ریت: {sym_win_rate:.2f}% | سود/زیان: ${balance - 1000.0:.2f}")

portfolio_win_rate = (total_portfolio_wins / total_portfolio_trades * 100) if total_portfolio_trades > 0 else 0

print("\n" + "="*50)
print("📊 گزارش نهایی استراتژی نهادی اسمارت مانی (SMC & Order Block)")
print("="*50)
print(f"تعداد کل معاملات مجموع : {total_portfolio_trades}")
print(f"کل معاملات موفق (Wins) : {total_portfolio_wins}")
print(f"کل معاملات ناموفق (Loss): {total_portfolio_losses}")
print(f"وین‌ریت کلی پورتفیو     : {portfolio_win_rate:.2f}%")
print(f"سرمایه اولیه کل         : ${initial_total_balance:.2f}")
print(f"سرمایه نهایی کل         : ${current_total_balance:.2f}")
print(f"سود خالص کل پورتفیو    : ${current_total_balance - initial_total_balance:.2f}")
print("="*50)
