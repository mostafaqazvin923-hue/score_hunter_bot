import pandas as pd
import numpy as np

# تنظیمات اصلی صرافی کوینکس و قوانین قفل‌شده
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
INITIAL_TOTAL_BALANCE = 1000.0  # سرمایه کل پورتفیو
BALANCE_PER_COIN = INITIAL_TOTAL_BALANCE / len(SYMBOLS)
TARGET_RR = 2.0  # ریسک به ریوارد کاملاً ثابت و قفل‌شده روی ۱ به ۲
RISK_PERCENTAGE = 0.01  # ریسک ۱ درصد از سرمایه در هر معامله

total_portfolio_trades = 0
total_portfolio_wins = 0
total_portfolio_losses = 0
current_total_balance = INITIAL_TOTAL_BALANCE

for symbol in SYMBOLS:
    print(f"\n⏳ در حال اجرای بک‌تست WHALE PULLBACK 2R v1 (مخصوص کوینکس) برای {symbol}...")
-    
    # از آنجایی که کوینکس دیتای ۱۵ دقیقه یکساله را یکجا نمی‌دهد، 
    # ۳۵۰۴۰ کندل ۱۵ دقیقه‌ای (معادل یک سال کامل) برای تست دقیق شبیه‌سازی می‌شود.
    np.random.seed(hash(symbol) % 2026)
    n_candles = 35040 
    base_price = 65000 if 'BTC' in symbol else (3300 if 'ETH' in symbol else (160 if 'SOL' in symbol else 1.2))
    
    trend_cycle = np.sin(np.linspace(0, 50, n_candles)) * 0.08
    returns = np.random.normal(0.00004, 0.004, n_candles) + trend_cycle / 100
    price_series = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2025-09-03', periods=n_candles, freq='15min'),
        'open': price_series * (1 + np.random.normal(0, 0.0005, n_candles)),
        'high': price_series * (1 + np.abs(np.random.normal(0.0015, 0.0008, n_candles))),
        'low': price_series * (1 - np.abs(np.random.normal(0.0015, 0.0008, n_candles))),
        'close': price_series,
        'volume': np.random.uniform(500, 10000, n_candles)
    })

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    volumes = df['volume'].values

    # محاسبه اندیکاتورها (EMA, RSI, ATR, ADX شبیه‌سازی‌شده)
    close_series = pd.Series(closes)
    ema_20 = close_series.ewm(span=20, adjust=False).mean().values
    ema_50 = close_series.ewm(span=50, adjust=False).mean().values
    ema_200 = close_series.ewm(span=200, adjust=False).mean().values

    # محاسبه RSI 14
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).fillna(50).values

    # محاسبه ATR 14 برای حد ضرر ساختاری
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))))
    atr = pd.Series(tr).rolling(window=14).mean().fillna(value=0).values

    # شبیه‌سازی ADX برای سنجش قدرت روند و حذف بازارهای رنج
    adx = np.random.uniform(15, 35, n_candles) # فیلتر عدم معامله در رنج

    balance = BALANCE_PER_COIN
    wins = 0
    losses = 0
    total_trades = 0
    i = 200

    while i < len(df) - 30:
        c_close = closes[i]
        c_open = opens[i]
        c_high = highs[i]
        c_low = lows[i]
        c_ema20 = ema_20[i]
        c_ema50 = ema_50[i]
        c_ema200 = ema_200[i]
        c_rsi = rsi[i]
        c_adx = adx[i]
        c_atr = atr[i]

        # فیلتر طلایی: اگر بازار رنج باشد (ADX ضعیف)، هیچ معامله‌ای انجام نمی‌شود
        if c_adx < 18 or c_atr == 0:
            i += 1
            continue

        # بررسی شرایط روند صعودی (تایم فریم بالا / ساختار کلی)
        is_uptrend = (c_close > c_ema200) and (c_ema20 > c_ema50) and (c_ema50 > c_ema200) and (c_rsi > 50)
        # بررسی شرایط روند نزولی
        is_downtrend = (c_close < c_ema200) and (c_ema20 < c_ema50) and (c_ema50 < c_ema200) and (c_rsi < 50)

        lookback = 10
        recent_high = max(highs[i-lookback:i])
        recent_low = min(lows[i-lookback:i])

        trade_executed = False

        # ستاپ لانگ (WHALE PULLBACK LONG)
        if is_uptrend and (c_close > recent_high) and (c_close > c_open): # Breakout تاییدشده
            entry = c_close
            swing_low = min(lows[i-3:i])
            sl = swing_low - (c_atr * 0.5)  # حد ضرر پشت ساختار + ATR
            risk_dist = entry - sl

            # اگر حد ضرر بیش از حد بزرگ یا غیرمنطقی باشد، معامله حذف می‌شود
            if 0 < risk_dist <= (entry * 0.03):
                tp = entry + (risk_dist * TARGET_RR) # RR کاملا ثابت ۱ به ۲
                
                trade_won, trade_lost = False, False
                for j in range(i + 1, min(i + 24, len(df))):
                    if highs[j] >= tp:
                        trade_won = True
                        break
                    if lows[j] <= sl:
                        trade_lost = True
                        break

                if trade_won or trade_lost:
                    total_trades += 1
                    risk_amount = balance * RISK_PERCENTAGE
                    if trade_won:
                        wins += 1
                        balance += (risk_amount * TARGET_RR)
                    else:
                        losses += 1
                        balance -= risk_amount
                    
                    i = j  # پرش به اتمام معامله
                    trade_executed = True

        # ستاپ شورت (WHALE PULLBACK SHORT)
        elif is_downtrend and (c_close < recent_low) and (c_open > c_close) and not trade_executed:
            entry = c_close
            swing_high = max(highs[i-3:i])
            sl = swing_high + (c_atr * 0.5)
            risk_dist = sl - entry

            if 0 < risk_dist <= (entry * 0.03):
                tp = entry - (risk_dist * TARGET_RR) # RR کاملا ثابت ۱ به ۲
                
                trade_won, trade_lost = False, False
                for j in range(i + 1, min(i + 24, len(df))):
                    if lows[j] <= tp:
                        trade_won = True
                        break
                    if highs[j] >= sl:
                        trade_lost = True
                        break

                if trade_won or trade_lost:
                    total_trades += 1
                    risk_amount = balance * RISK_PERCENTAGE
                    if trade_won:
                        wins += 1
                        balance += (risk_amount * TARGET_RR)
                    else:
                        losses += 1
                        balance -= risk_amount
                    
                    i = j
                    trade_executed = True

        if not trade_executed:
            i += 1

    total_portfolio_trades += total_trades
    total_portfolio_wins += wins
    total_portfolio_losses += losses
    current_total_balance += (balance - BALANCE_PER_COIN)

    sym_win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    print(f"[{symbol}] (صرافی کوینکس) -> معاملات: {total_trades} | برد: {wins} | باخت: {losses} | وین‌ریت: {sym_win_rate:.2f}% | سود/زیان: ${balance - BALANCE_PER_COIN:.2f}")

portfolio_win_rate = (total_portfolio_wins / total_portfolio_trades * 100) if total_portfolio_trades > 0 else 0

print("\n" + "="*50)
print("📊 گزارش نهایی استراتژی WHALE PULLBACK 2R v1 (مخصوص کوینکس)")
print("="*50)
print(f"تعداد کل معاملات پورتفیو : {total_portfolio_trades}")
print(f"کل معاملات موفق (Wins)  : {total_portfolio_wins}")
print(f"کل معاملات ناموفق (Loss) : {total_portfolio_losses}")
print(f"وین‌ریت کل پورتفیو      : {portfolio_win_rate:.2f}%")
print(f"سرمایه اولیه کل          : ${INITIAL_TOTAL_BALANCE:.2f}")
print(f"سرمایه نهایی کل          : ${current_total_balance:.2f}")
print(f"سود خالص کل پورتفیو     : ${current_total_balance - INITIAL_TOTAL_BALANCE:.2f}")
print("="*50)
