import math
import random
import datetime

# ==============================================================================
# اسکریپت بک‌تست پایتون خالص (بدون نیاز به pandas و numpy) - R:R = 1:2
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
    
    # Simple RMA/EMA for ATR
    atr = [sum(tr_list[:period]) / period]
    alpha = 1.0 / period
    for tr in tr_list[period:]:
        atr.append(tr * alpha + atr[-1] * (1 - alpha))
    return [atr[0]] * (period - 1) + atr

def run_backtest(candles, initial_capital=1000.0, rr_ratio=2.0, risk_per_trade_pct=2.0):
    closes = [c['close'] for c in candles]
    highs = [c['high'] for c in candles]
    lows = [c['low'] for c in candles]
    opens = [c['open'] for c in candles]
    volumes = [c['volume'] for c in candles]

    ema100 = calculate_ema(closes, 100)
    atr = calculate_atr(highs, lows, closes, 14)

    capital = initial_capital
    peak_capital = initial_capital
    max_drawdown = 0.0
    trades = []
    active_trade = None

    for i in range(101, len(candles)):
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
                if current['high'] <= t['sl']: # برای شورت بالاتر رفتن یعنی SL
                    pnl = -t['risk_amount']
                    capital += pnl
                    trades.append({'result': 'SL', 'pnl': pnl})
                    active_trade = None
                elif current['low'] <= t['tp']:
                    pnl = t['risk_amount'] * rr_ratio
                    capital += pnl
                    trades.append({'result': 'TP', 'pnl': pnl})
                    active_trade = None

        # سیگنال جدید
        if active_trade is None:
            c_close = prev['close']
            c_ema = ema100[i-1]
            c_atr = atr[i-1]

            # شکست سقف/کف ۱۰ کندل قبل
            recent_highs = [candles[j]['high'] for j in range(i-11, i-1)]
            recent_lows = [candles[j]['low'] for j in range(i-11, i-1)]
            upper_break = max(recent_highs)
            lower_break = min(recent_lows)

            # حجم
            avg_vol = sum(volumes[i-21:i-1]) / 20.0
            vol_break = prev['volume'] > (1.2 * avg_vol)

            # شرط لورج / ورود
            long_cond = (c_close > c_ema) and (c_close > upper_break) and vol_break
            short_cond = (c_close < c_ema) and (c_close < lower_break) and vol_break

            risk_amt = capital * (risk_per_trade_pct / 100.0)

            if long_cond:
                entry = current['open']
                sl = entry - (c_atr * 1.0)
                tp = entry + ((entry - sl) * rr_ratio)
                active_trade = {'type': 'LONG', 'entry': entry, 'sl': sl, 'tp': tp, 'risk_amount': risk_amt}
            elif short_cond:
                entry = current['open']
                sl = entry + (c_atr * 1.0)
                tp = entry - ((sl - entry) * rr_ratio)
                active_trade = {'type': 'SHORT', 'entry': entry, 'sl': sl, 'tp': tp, 'risk_amount': risk_amt}

        if capital > peak_capital:
            peak_capital = capital
        dd = (peak_capital - capital) / peak_capital * 100.0
        if dd > max_drawdown:
            max_drawdown = dd

    return capital, trades, max_drawdown

# تولید شبیه‌سازی قیمت ۴ ماه اخیر (۱۱,۵۰۰ کندل)
random.seed(42)
candles = []
price = 60000.0
for i in range(11500):
    change = random.gauss(0.00005, 0.003)
    price *= math.exp(change)
    high = price * (1 + abs(random.gauss(0, 0.002)))
    low = price * (1 - abs(random.gauss(0, 0.002)))
    open_p = low + (high - low) * random.random()
    close_p = low + (high - low) * random.random()
    volume = random.expovariate(1/100)
    candles.append({'open': open_p, 'high': high, 'low': low, 'close': close_p, 'volume': volume})

final_cap, trades, max_dd = run_backtest(candles, initial_capital=1000.0, rr_ratio=2.0)

win_trades = [t for t in trades if t['result'] == 'TP']
win_rate = (len(win_trades) / len(trades) * 100) if trades else 0

print("="*50)
print("=== خروجی بک‌تست استراتژی (ریسک به ریوارد ۱ به ۲) ===")
print("="*50)
print(f"موجودی اولیه: $1000.00")
print(f"موجودی نهایی: ${final_cap:.2f}")
print(f"کل معاملات: {len(trades)}")
print(f"تعداد سود (TP): {len(win_trades)}")
print(f"تعداد ضرر (SL): {len(trades) - len(win_trades)}")
print(f"وین‌ریت (Win Rate): {win_rate:.2f}%")
print(f"حداکثر افت حساب (Max Drawdown): {max_dd:.2f}%")
print("="*50)
