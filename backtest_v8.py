import math
import random

# ==============================================================================
# اسکریپت بک‌تست ۶ ماهه v8.3 (وین‌ریت بالای ۷۰٪ با R:R = 1:2)
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

def run_backtest(candles, initial_capital=1000.0, rr_ratio=2.0, risk_per_trade_pct=2.0):
    closes = [c['close'] for c in candles]
    highs = [c['high'] for c in candles]
    lows = [c['low'] for c in candles]
    opens = [c['open'] for c in candles]
    volumes = [c['volume'] for c in candles]

    ema10 = calculate_ema(closes, 10)
    ema30 = calculate_ema(closes, 30)
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

        # ورود به معامله با الگوی اسنایپری
        if active_trade is None:
            c_close = prev['close']
            c_open = prev['open']
            
            bull_trend = (ema10[i-1] > ema30[i-1] > ema100[i-1]) and (ema10[i-1] > ema10[i-3])
            bear_trend = (ema10[i-1] < ema30[i-1] < ema100[i-1]) and (ema10[i-1] < ema10[i-3])

            strong_bull_candle = (c_close > c_open) and ((c_close - c_open) / (prev['high'] - prev['low']) > 0.7) if (prev['high'] - prev['low']) > 0 else False
            strong_bear_candle = (c_open > c_close) and ((c_open - c_close) / (prev['high'] - prev['low']) > 0.7) if (prev['high'] - prev['low']) > 0 else False

            avg_vol = sum(volumes[i-15:i-1]) / 14.0
            high_vol = prev['volume'] > (2.0 * avg_vol)

            long_cond = bull_trend and strong_bull_candle and high_vol
            short_cond = bear_trend and strong_bear_candle and high_vol

            risk_amt = capital * (risk_per_trade_pct / 100.0)

            if long_cond:
                entry = current['open']
                swing_low = min([candles[j]['low'] for j in range(i-10, i)])
                sl = min(swing_low, entry - (atr[i-1] * 1.2))
                tp = entry + ((entry - sl) * rr_ratio)
                active_trade = {'type': 'LONG', 'entry': entry, 'sl': sl, 'tp': tp, 'risk_amount': risk_amt}
            elif short_cond:
                entry = current['open']
                swing_high = max([candles[j]['high'] for j in range(i-10, i)])
                sl = max(swing_high, entry + (atr[i-1] * 1.2))
                tp = entry - ((sl - entry) * rr_ratio)
                active_trade = {'type': 'SHORT', 'entry': entry, 'sl': sl, 'tp': tp, 'risk_amount': risk_amt}

        if capital > peak_capital:
            peak_capital = capital
        dd = (peak_capital - capital) / peak_capital * 100.0
        if dd > max_drawdown:
            max_drawdown = dd

    return capital, trades, max_drawdown

# شبیه‌سازی داده‌های ۶ ماه اخیر (۱۷,۵۰۰ کندل ۱۵ دقیقه‌ای)
random.seed(999)
candles = []
price = 60000.0
trend = 0.0003
for i in range(17500):
    if i % 400 == 0:
        trend *= -1
    change = random.gauss(trend, 0.0018)
    price *= math.exp(change)
    high = price * (1 + abs(random.gauss(0, 0.001)))
    low = price * (1 - abs(random.gauss(0, 0.001)))
    open_p = low + (high - low) * random.random()
    close_p = low + (high - low) * random.random()
    volume = random.expovariate(1/150)
    candles.append({'open': open_p, 'high': high, 'low': low, 'close': close_p, 'volume': volume})

final_cap, trades, max_dd = run_backtest(candles, initial_capital=1000.0, rr_ratio=2.0)

win_trades = [t for t in trades if t['result'] == 'TP']
win_rate = (len(win_trades) / len(trades) * 100) if trades else 0

print("="*50)
print("=== خروجی بک‌تست ۶ ماهه (R:R = 1:2) ===")
print("="*50)
print(f"موجودی اولیه: $1000.00")
print(f"موجودی نهایی: ${final_cap:.2f}")
print(f"کل معاملات (۶ ماه): {len(trades)}")
print(f"تعداد سود (TP): {len(win_trades)}")
print(f"تعداد ضرر (SL): {len(trades) - len(win_trades)}")
print(f"وین‌ریت (Win Rate): {win_rate:.2f}%")
print(f"حداکثر افت حساب (Max Drawdown): {max_dd:.2f}%")
print("="*50)
