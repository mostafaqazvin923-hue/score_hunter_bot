import time
import requests

# ==============================================================================
# دریافت بیشترین حد ممکن کندل از صرافی کوینکس (حداکثر ۱۰۰۰ کندل معادل حدود ۴۲ روز)
# ==============================================================================
def fetch_safe_candles(symbol, timeframe='1hour', limit=1000):
    market = symbol.replace('/', '')
    url = "https://api.coinex.com/v2/spot/kline"
    params = {"market": market, "limit": limit, "period": timeframe}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get('code') == 0 and data.get('data'):
            candles = []
            for c in data['data']:
                candles.append({
                    'timestamp': int(c.get('created_at', c.get('time', 0))),
                    'open': float(c['open']),
                    'high': float(c['high']),
                    'low': float(c['low']),
                    'close': float(c['close']),
                    'volume': float(c['volume'])
                })
            candles.sort(key=lambda x: x['timestamp'])
            return candles
        return []
    except Exception:
        return []

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
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            tr_list.append(tr)
    atr = [sum(tr_list[:period]) / period]
    alpha = 1.0 / period
    for tr in tr_list[period:]:
        atr.append(tr * alpha + atr[-1] * (1 - alpha))
    return [atr[0]] * (period - 1) + atr

# ==============================================================================
# موتور بک‌تست نسخه v15.2 (بازه بزرگ‌تر روی داده‌های واقعی)
# ==============================================================================
def run_v15_backtest(assets_data, initial_capital=1000.0, risk_per_trade_pct=1.5):
    capital = initial_capital
    peak_capital = initial_capital
    max_drawdown = 0.0
    all_trades = []

    for asset_name, candles in assets_data.items():
        if len(candles) < 100:
            continue

        closes = [c['close'] for c in candles]
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]

        ema20 = calculate_ema(closes, 20)
        ema50 = calculate_ema(closes, 50)
        atr = calculate_atr(highs, lows, closes, 14)

        active_trade = None

        for i in range(50, len(candles)):
            current = candles[i]
            prev = candles[i-1]
            prev2 = candles[i-2]

            if active_trade:
                t = active_trade
                if t['type'] == 'LONG':
                    if current['low'] <= t['sl']:
                        capital -= t['risk_amount']
                        t['result'] = 'SL'
                        all_trades.append(t)
                        active_trade = None
                    elif current['high'] >= t['tp']:
                        capital += t['risk_amount'] * 1.8
                        t['result'] = 'TP'
                        all_trades.append(t)
                        active_trade = None
                elif t['type'] == 'SHORT':
                    if current['high'] >= t['sl']:
                        capital -= t['risk_amount']
                        t['result'] = 'SL'
                        all_trades.append(t)
                        active_trade = None
                    elif current['low'] <= t['tp']:
                        capital += t['risk_amount'] * 1.8
                        t['result'] = 'TP'
                        all_trades.append(t)
                        active_trade = None

            if not active_trade:
                trend_up = ema20[i-1] > ema50[i-1]
                trend_down = ema20[i-1] < ema50[i-1]

                pullback_long = prev2['close'] < prev2['open'] and prev['close'] < prev['open'] and current['close'] > current['open']
                pullback_short = prev2['close'] > prev2['open'] and prev['close'] > prev['open'] and current['close'] < current['open']

                long_cond = trend_up and pullback_long
                short_cond = trend_down and pullback_short

                risk_amt = capital * (risk_per_trade_pct / 100.0)

                if long_cond:
                    entry = current['open']
                    sl = entry - (atr[i-1] * 1.5)
                    tp = entry + ((entry - sl) * 1.8)
                    active_trade = {'asset': asset_name, 'type': 'LONG', 'entry': entry, 'sl': sl, 'tp': tp, 'risk_amount': risk_amt}
                elif short_cond:
                    entry = current['open']
                    sl = entry + (atr[i-1] * 1.5)
                    tp = entry - ((sl - entry) * 1.8)
                    active_trade = {'asset': asset_name, 'type': 'SHORT', 'entry': entry, 'sl': sl, 'tp': tp, 'risk_amount': risk_amt}

            if capital > peak_capital:
                peak_capital = capital
            dd = (peak_capital - capital) / peak_capital * 100.0
            if dd > max_drawdown:
                max_drawdown = dd

    return capital, all_trades, max_drawdown

if __name__ == "__main__":
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']
    assets_data = {}

    print("در حال دریافت حداکثر داده‌های تاریخی از صرافی کوینکس...")
    for symbol in symbols:
        candles = fetch_safe_candles(symbol, timeframe='1hour', limit=1000)
        if candles:
            assets_data[symbol] = candles
        time.sleep(0.2)

    final_cap, trades, max_dd = run_v15_backtest(assets_data, initial_capital=1000.0)
    
    # اصلاح خطای پرانتز در این بخش
    win_trades = [t for t in trades if t.get('result') == 'TP']
    win_rate = (len(win_trades) / len(trades) * 100) if trades else 0

    print("="*50)
    print("=== گزارش بک‌تست ربات نسخه v15.2 (بازه طولانی‌تر) ===")
    print("="*50)
    print(f"موجودی اولیه: $1000.00")
    print(f"موجودی نهایی: ${final_cap:.2f}")
    print(f"کل معاملات انجام شده: {len(trades)}")
    print(f"تعداد برد (TP): {len(win_trades)}")
    print(f"تعداد باخت (SL): {len(trades) - len(win_trades)}")
    print(f"وین‌ریت واقعی: {win_rate:.2f}%")
    print(f"حداکثر افت سرمایه (Max Drawdown): {max_dd:.2f}%")
    print("="*50)
