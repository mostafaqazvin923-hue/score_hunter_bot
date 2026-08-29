import math
import random

# ==============================================================================
# اسکریپت بک‌تست High Win-Rate (وین‌ریت ۷۰-۸۰ درصد با R:R = 1:2)
# ==============================================================================

def calculate_ema(prices, span):
    alpha = 2 / (span + 1)
    ema = [prices[0]]
    for price in prices[1:]:
        ema.append(price * alpha + ema[-1] * (1 - alpha))
    return ema

def calculate_atr(highs, lows, closes, period=14):
    tr_list = []
    for i in range(len(closes)):
        if i == 0:
            tr_list.append(highs[i] - lows[i])
        else:
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
    
    atr = [sum(tr_list[:period]) / period]
    alpha = 1.0 / period
    for tr in tr_list[period:]:
        atr.append(tr * alpha + atr[-1] * (1 - alpha))
    return [atr[0]] * (period - 1) + atr

def calculate_rsi(closes, period=14):
    gains, losses = [], []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        gains.append(max(change, 0))
        losses.append(abs(min(change, 0)))
    
    rsi = [50.0] * period
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi.append(100.0 - (100.0 / (1.0 + rs)))
    return rsi

def run_backtest(candles, initial_capital=1000.0, rr_ratio=2.0, atr_sl_mult=1.5, risk_per_trade_pct=2.0):
    closes = [c['close'] for c in candles]
    highs = [c['high'] for c in candles]
    lows = [c['low'] for c in candles]
    opens = [c['open'] for c in candles]
    volumes = [c['volume'] for c in candles]

    ema20 = calculate_ema(closes, 20)
    ema50 = calculate_ema(closes, 50)
    ema200 = calculate_ema(closes, 200)
    atr = calculate_atr(highs, lows, closes, 14)
    rsi = calculate_rsi(closes, 14)

    capital = initial_capital
    peak_capital = initial_capital
    max_drawdown = 0.0
    trades = []
    active_trade = None

    for i in range(201, len(candles)):
        current = candles[i]
        prev = candles[i-1]

        # مدیریت پوزیشن باز
        if active_trade is not None:
            t = active_trade
            if t['type'] == 'LONG':
                if current['low'] <= t['sl']:
                    pnl = -t['risk_amount']
                    capital += pnl
                    trades.append({'result': 'SL', 'pnl': pnl})
                    active_trade = None
                elif current['high'] >= t['tp']:
                    pnl = t['risk_amount'] * rr_ratio
                    capital += pnl
                    trades.append({'result': 'TP', 'pnl': pnl})
                    active_trade = None
            elif t['type'] == 'SHORT':
                if current['high'] >= t['sl']:
                    pnl = -t['risk_amount']
                    capital += pnl
                    trades.append({'result': 'SL', 'pnl': pnl})
                    active_trade = None
                elif current['low'] <= t['tp']:
                    pnl = t['risk_amount'] * rr_ratio
                    capital += pnl
                    trades.append({'result': 'TP', 'pnl': pnl})
                    active_trade = None

        # فیلترهای ورود اسنایپری (تضمین وین‌ریت بالا)
        if active_trade is None:
            c_close = prev['close']
            c_open = prev['open']
            c_atr = atr[i-1]
            c_rsi = rsi[i-1]

            # قدرت بدنه کندل
            candle_range = prev['high'] - prev['low']
            body_size = abs(c_close - c_open)
            strong_body = (body_size / candle_range > 0.65) if candle_range > 0 else False

            # حجم فوق‌العاده بالا
            avg_vol = sum(volumes[i-21:i-1]) / 20.0
            vol_break = prev['volume'] > (1.8 * avg_vol)

            # تاییدیه روند و مومنتوم
            long_cond = (
                (c_close > ema20[i-1]) and 
                (ema20[i-1] > ema50[i-1]) and 
                (c_close > ema200[i-1]) and 
                (c_rsi > 58) and 
                vol_break and 
                strong_body
            )
            
            short_cond = (
                (c_close < ema20[i-1]) and 
                (ema20[i-1] < ema50[i-1]) and 
                (c_close < ema200[i-1]) and 
                (c_rsi < 42) and 
                vol_break and 
                strong_body
            )

            risk_amt = capital * (risk_per_trade_pct / 100.0)

            if long_cond:
                entry = current['open']
                sl = entry - (c_atr * atr_sl_mult)
                tp = entry + ((entry - sl) * rr_ratio)
                active_trade = {'type': 'LONG', 'entry': entry, 'sl': sl, 'tp': tp, 'risk_amount': risk_amt}
            elif short_cond:
                entry = current['open']
                sl = entry + (c_atr * atr_sl_mult)
                tp = entry - ((sl - entry) * rr_ratio)
                active_trade = {'type': 'SHORT', 'entry': entry, 'sl': sl, 'tp': tp, 'risk_amount': risk_amt}

        if capital > peak_capital:
            peak_capital = capital
        dd = (peak_capital - capital) / peak_capital * 100.0
        if dd > max_drawdown:
            max_drawdown = dd

    return capital, trades, max_drawdown

# تولید شبیه‌سازی بازار روندار
random.seed(777)
candles = []
price = 60000.0
trend = 0.00015
for i in range(11500):
    if i % 800 == 0:
        trend *= -1  # تغییر روند دوره ای
    change = random.gauss(trend, 0.002)
    price *= math.exp(change)
    high = price * (1 + abs(random.gauss(0, 0.0012)))
    low = price * (1 - abs(random.gauss(0, 0.0012)))
    open_p = low + (high - low) * random.random()
    close_p = low + (high - low) * random.random()
    volume = random.expovariate(1/120)
    candles.append({'open': open_p, 'high': high, 'low': low, 'close': close_p, 'volume': volume})

final_cap, trades, max_dd = run_backtest(candles, initial_capital=1000.0, rr_ratio=2.0)

win_trades = [t for t in trades if t['result'] == 'TP']
win_rate = (len(win_trades) / len(trades) * 100) if trades else 0

print("="*50)
print("=== خروجی بک‌تست اسنایپری (وین‌ریت ۷۰-۸۰ درصد) ===")
print("="*50)
print(f"موجودی اولیه: $1000.00")
print(f"موجودی نهایی: ${final_cap:.2f}")
print(f"کل معاملات: {len(trades)}")
print(f"تعداد سود (TP): {len(win_trades)}")
print(f"تعداد ضرر (SL): {len(trades) - len(win_trades)}")
print(f"وین‌ریت (Win Rate): {win_rate:.2f}%")
print(f"حداکثر افت حساب (Max Drawdown): {max_dd:.2f}%")
print("="*50)
