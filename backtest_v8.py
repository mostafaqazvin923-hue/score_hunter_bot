import pandas as pd
import numpy as np

# تنظیمات صرافی کوینکس و قوانین قفل‌شده
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
INITIAL_TOTAL_BALANCE = 1000.0  # سرمایه کل پورتفیو
BALANCE_PER_COIN = INITIAL_TOTAL_BALANCE / len(SYMBOLS)
TARGET_RR = 2.0  # ریسک به ریوارد کاملاً ثابت و قفل‌شده روی ۱ به ۲
RISK_PERCENTAGE = 0.01  # ریسک ۱ درصد از سرمایه در هر معامله

total_portfolio_trades = 0
total_portfolio_wins = 0
total_portfolio_losses = 0
total_long_trades = 0
total_long_wins = 0
total_short_trades = 0
total_short_wins = 0
current_total_balance = INITIAL_TOTAL_BALANCE

print("============================================================")
print("WHALE PULLBACK v10 - BREAK-EVEN & FULL TARGET OPTIMized")
print("============================================================")

for symbol in SYMBOLS:
    print(f"\n⏳ در حال اجرای نسخه v10 (ایده ریسک‌فری هوشمند) برای {symbol}...")
    
    np.random.seed(hash(symbol) % 2026)
    n_candles = 35040  # یک سال کندل ۱۵ دقیقه‌ای
    base_price = 65000 if 'BTC' in symbol else (3300 if 'ETH' in symbol else (160 if 'SOL' in symbol else 1.2))
    
    trend_cycle = np.sin(np.linspace(0, 50, n_candles)) * 0.08
    returns = np.random.normal(0.00004, 0.004, n_candles) + trend_cycle / 100
    price_series = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2025-09-03', periods=n_candles, freq='15min'),
        'open': price_series * (1 + np.random.normal(0, 0.0005, n_candles)),
        'high': price_series * (1 + abs(np.random.normal(0.0015, 0.0008, n_candles))),
        'low': price_series * (1 - abs(np.random.normal(0.0015, 0.0008, n_candles))),
        'close': price_series,
        'volume': np.random.uniform(500, 10000, n_candles)
    })

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    volumes = df['volume'].values

    close_series = pd.Series(closes)
    ema_20 = close_series.ewm(span=20, adjust=False).mean().values
    ema_50 = close_series.ewm(span=50, adjust=False).mean().values
    ema_200 = close_series.ewm(span=200, adjust=False).mean().values

    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).fillna(50).values

    tr = np.maximum(highs - lows, np.maximum(abs(highs - np.roll(closes, 1)), abs(lows - np.roll(closes, 1))))
    atr = pd.Series(tr).rolling(window=14).mean().fillna(value=0).values

    vol_sma = pd.Series(volumes).rolling(window=20).mean().values
    adx = np.random.uniform(25, 50, n_candles)

    balance = BALANCE_PER_COIN
    wins = 0
    losses = 0
    total_trades = 0
    
    sym_long_trades = 0
    sym_long_wins = 0
    sym_short_trades = 0
    sym_short_wins = 0

    i = 200

    while i < len(df) - 30:
        c_close = closes[i]
        c_open = opens[i]
        c_high = highs[i]
        c_low = lows[i]
        c_vol = volumes[i]
        c_vol_avg = vol_sma[i]
        c_ema20 = ema_20[i]
        c_ema50 = ema_50[i]
        c_ema200 = ema_200[i]
        c_rsi = rsi[i]
        c_adx = adx[i]
        c_atr = atr[i]

        if c_adx < 25 or c_atr == 0 or np.isnan(c_vol_avg):
            i += 1
            continue

        is_uptrend = (c_close > c_ema200) and (c_ema20 > c_ema50) and (c_ema50 > c_ema200)
        is_downtrend = (c_close < c_ema200) and (c_ema20 < c_ema50) and (c_ema50 < c_ema200)

        score = 0
        if is_uptrend or is_downtrend: score += 3
        if is_uptrend and c_rsi > 58: score += 3
        elif is_downtrend and c_rsi < 42: score += 3
        if c_vol > (c_vol_avg * 1.3): score += 2

        if score < 8:
            i += 1
            continue

        lookback = 10
        recent_high = max(highs[i-lookback:i])
        recent_low = min(lows[i-lookback:i])

        trade_executed = False

        # ستاپ لانگ v10 (با ایده ریسک‌فری کامل در RR 1 و پوزیشن کامل تا RR 2)
        if is_uptrend and (c_close > recent_high) and (c_close > c_open):
            entry = c_close
            swing_low = min(lows[i-3:i])
            sl = swing_low - (c_atr * 0.4)
            risk_dist = entry - sl

            if 0 < risk_dist <= (entry * 0.02):
                tp_1 = entry + risk_dist         # هدف اول برای انتقال SL به نقطه ورود (Break-even)
                tp_2 = entry + (risk_dist * TARGET_RR) # هدف نهایی روی RR 2
                
                trade_won, trade_lost = False, False
                current_sl = sl
                
                for j in range(i + 1, min(i + 24, len(df))):
                    # بررسی برخورد با حد ضرر (یا نقطه ورود ریسک‌فری شده)
                    if lows[j] <= current_sl:
                        # اگر SL روی نقطه ورود بوده، معامله نه برد است و نه باخت (سربه-سر / رد می‌شود)
                        break
                    
                    # بررسی رسیدن به RR 1 برای ریسک‌فری کردن
                    if highs[j] >= tp_1 and current_sl < entry:
                        current_sl = entry  # انتقال دقیق حد ضرر به قیمت ورود
                    
                    # بررسی رسیدن به هدف نهایی RR 2
                    if highs[j] >= tp_2:
                        trade_won = True
                        break

                # ثبت نتیجه (اگر trade_won باشد سود کامل RR 2 وگرنه اگر ریسک‌فری شده باشد بدون ضرر رد می‌شود)
                if trade_won or current_sl == entry:
                    total_trades += 1
                    sym_long_trades += 1
                    total_long_trades += 1
                    risk_amount = balance * RISK_PERCENTAGE
                    if trade_won:
                        wins += 1
                        sym_long_wins += 1
                        total_long_wins += 1
                        balance += (risk_amount * TARGET_RR)  # سود کامل ۱ به ۲
                    # اگر ریسک‌فری شده باشد، balance تغییری نمی‌کند (نه سود، نه ضرر)
                    
                    i = j
                    trade_executed = True
                else:
                    # اگر به نقطه ورود یا RR1 نرسیده و مستقیم خورده به SL اولیه
                    total_trades += 1
                    sym_long_trades += 1
                    total_long_trades += 1
                    losses += 1
                    balance -= (balance * RISK_PERCENTAGE)
                    i = j
                    trade_executed = True

        # ستاپ شورت v10 (با ایده ریسک‌فری کامل در RR 1)
        elif is_downtrend and (c_close < recent_low) and (c_open > c_close) and not trade_executed:
            entry = c_close
            swing_high = max(highs[i-3:i])
            sl = swing_high + (c_atr * 0.4)
            risk_dist = sl - entry

            if 0 < risk_dist <= (entry * 0.02):
                tp_1 = entry - risk_dist
                tp_2 = entry - (risk_dist * TARGET_RR)
                
                trade_won, trade_lost = False, False
                current_sl = sl
                
                for j in range(i + 1, min(i + 24, len(df))):
                    if highs[j] >= current_sl:
                        break
                    
                    if lows[j] <= tp_1 and current_sl > entry:
                        current_sl = entry
                    
                    if lows[j] <= tp_2:
                        trade_won = True
                        break

                if trade_won or current_sl == entry:
                    total_trades += 1
                    sym_short_trades += 1
                    total_short_trades += 1
                    risk_amount = balance * RISK_PERCENTAGE
                    if trade_won:
                        wins += 1
                        sym_short_wins += 1
                        total_short_wins += 1
                        balance += (risk_amount * TARGET_RR)
                    
                    i = j
                    trade_executed = True
                else:
                    total_trades += 1
                    sym_short_trades += 1
                    total_short_trades += 1
                    losses += 1
                    balance -= (balance * RISK_PERCENTAGE)
                    i = j
                    trade_executed = True

        if not trade_executed:
            i += 1

    total_portfolio_trades += total_trades
    total_portfolio_wins += wins
    total_portfolio_losses += losses
    current_total_balance += (balance - BALANCE_PER_COIN)

    sym_win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    print(f"[{symbol}] (CoinEx - v10) -> معاملات: {total_trades} (لانگ: {sym_long_trades}, شورت: {sym_short_trades}) | وین‌ریت: {sym_win_rate:.2f}% | سود: ${balance - BALANCE_PER_COIN:.2f}")

portfolio_win_rate = (total_portfolio_wins / total_portfolio_trades * 100) if total_portfolio_trades > 0 else 0
long_win_rate = (total_long_wins / total_long_trades * 100) if total_long_trades > 0 else 0
short_win_rate = (total_short_wins / total_short_trades * 100) if total_short_trades > 0 else 0

print("\n" + "="*60)
print("FINAL RESULT - WHALE PULLBACK 2R v10 (BREAK-EVEN IDEAL)")
print("="*60)
print(f"TOTAL TRADES  : {total_portfolio_trades}")
print(f"  - LONG TRADES  : {total_long_trades} (موفق: {total_long_wins} | وین‌ریت: {long_win_rate:.2f}%)")
print(f"  - SHORT TRADES : {total_short_trades} (موفق: {total_short_wins} | وین‌ریت: {short_win_rate:.2f}%)")
print(f"TOTAL WINS    : {total_portfolio_wins}")
print(f"TOTAL LOSSES  : {total_portfolio_losses}")
print(f"OVERALL WIN   : {portfolio_win_rate:.2f}%")
print(f"INITIAL BAL   : ${INITIAL_TOTAL_BALANCE:.2f}")
print(f"FINAL BAL     : ${current_total_balance:.2f}")
print(f"NET PROFIT    : ${current_total_balance - INITIAL_TOTAL_BALANCE:.2f}")
print("="*60)
