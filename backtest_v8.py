import time
import requests

# ==============================================================================
# دریافت کندل استاندارد تایم‌فریم ۱۵ دقیقه
# ==============================================================================
def fetch_safe_candles(symbol, timeframe='15min', limit=500):
    market = symbol.replace('/', '')
    url = "https://api.coinex.com/v2/spot/kline"
    params = {"market": market, "limit": limit, "period": timeframe}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get('code'] == 0 and data.get('data'):
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

def calculate_sma(prices, period):
    sma = []
    for i in range(len(prices)):
        if i < period - 1:
            sma.append(sum(prices[:i+1]) / (i + 1))
        else:
            sma.append(sum(prices[i-period+1:i+1]) / period)
    return sma

def calculate_ema(prices, span):
    alpha = 2 / (span + 1)
    ema = [prices[0]]
    for price in prices[1:]:
        ema.append(price * alpha + ema[-1] * (1 - alpha))
    return ema

def calculate_rsi(closes, period=14):
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsi = [50] * period
    for i in range(period, len(deltas)):
        gain = gains[i]
        loss = losses[i]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            rsi.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi.append(100 - (100 / (1 + rs)))
    return [50] + rsi

def calculate_bollinger_bands(closes, period=20, std_dev=2.0):
    sma = calculate_sma(closes, period)
    upper_band = []
    lower_band = []
    for i in range(len(closes)):
        if i < period - 1:
            upper_band.append(closes[i])
            lower_band.append(closes[i])
        else:
            segment = closes[i-period+1:i+1]
            mean = sma[i]
            variance = sum((x - mean) ** 2 for x in segment) / period
            deviation = variance ** 0.5
            upper_band.append(mean + (std_dev * deviation))
            lower_band.append(mean - (std_dev * deviation))
    return upper_band, sma, lower_band

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
# موتور بک‌تست نسخه v13.0 (ترکیب بولینگر بند، RSI و روند کلان)
# ==============================================================================
def run_v13_backtest(assets_data, initial_capital=1000.0, risk_per_trade_pct=1.5):
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

        ema50 = calculate_ema(closes, 50)
        ema200 = calculate_ema(closes, 200) if len(closes) >= 200 else calculate_ema(closes, len(closes)-1)
        upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(closes, period=20, std_dev=2.0)
        rsi = calculate_rsi(closes, 14)
        atr = calculate_atr(highs, lows, closes, 14)

        active_trade = None

        for i in range(200, len(candles)):
            current = candles[i]
            prev = candles[i-1]

            if active_trade:
                t = active_trade
                if t['type'] == 'LONG':
                    if current['low'] <= t['sl']:
                        capital -= t['risk_amount']
                        t['result'] = 'SL'
                        all_trades.append(t)
                        active_trade = None
                    elif current['high'] >= t['tp']:
                        capital += t['risk_amount'] * 1.6
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
                        capital += t['risk_amount'] * 1.6
                        t['result'] = 'TP'
                        all_trades.append(t)
                        active_trade = None

            if not active_trade:
                # تشخیص روند کلان صعودی یا نزولی
                macro_up = ema50[i-1] > ema200[i-1] and closes[i-1] > ema50[i-1]
                macro_down = ema50[i-1] < ema200[i-1] and closes[i-1] < ema50[i-1]

                # شرایط ورود با بولینگر بند و RSI (خرید در کف باند پایینی در روند صعودی / فروش در سقف باند بالایی در روند نزولی)
                long_cond = macro_up and (prev['low'] <= lower_bb[i-1]) and (rsi[i-1] < 40) and (current['close'] > current['open'])
                short_cond = macro_down and (prev['high'] >= upper_bb[i-1]) and (rsi[i-1] > 60) and (current['close'] < current['open'])

                risk_amt = capital * (risk_per_trade_pct / 100.0)

                if long_cond:
                    entry = current['open']
                    sl = entry - (atr[i-1] * 1.2)
                    tp = entry + ((entry - sl) * 1.6)
                    active_trade = {'asset': asset_name, 'type': 'LONG', 'entry': entry, 'sl': sl, 'tp': tp, 'risk_amount': risk_amt}
                elif short_cond:
                    entry = current['open']
                    sl = entry + (atr[i-1] * 1.2)
                    tp = entry - ((sl - entry) * 1.6)
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

    print("در حال دریافت داده‌های تایم‌فریم ۱۵ دقیقه برای ربات نسخه v13.0 (بولینگر + RSI)...")
    for symbol in symbols:
        candles = fetch_safe_candles(symbol, timeframe='15min', limit=500)
        if candles:
            assets_data[symbol] = candles
        time.sleep(0.2)

    final_cap, trades, max_dd = run_v13_backtest(assets_data, initial_capital=1000.0)
    
    win_trades = [t for t in trades if t.get('result') == 'TP']
    win_rate = (len(win_trades) / len(trades) * 100) if trades else 0

    print("="*50)
    print("=== گزارش بک‌تست ربات نسخه v13.0 (سیستم بولینگر و RSI حرفه‌ای) ===")
    print("="*50)
    print(f"موجودی اولیه: $1000.00")
    print(f"موجودی نهایی: ${final_cap:.2f}")
    print(f"کل معاملات انجام شده: {len(trades)}")
    print(f"تعداد برد (TP): {len(win_trades)}")
    print(f"تعداد باخت (SL): {len(trades) - len(win_trades)}")
    print(f"وین‌ریت واقعی: {win_rate:.2f}%")
    print(f"حداکثر افت سرمایه (Max Drawdown): {max_dd:.2f}%")
    print("="*50)
