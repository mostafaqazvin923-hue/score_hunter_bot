import time
import requests
import math

# ==============================================================================
# دریافت کندل‌های استاندارد از صرافی کوینکس (تایم‌فریم ۱ ساعته، حداکثر ظرفیت تاریخچه)
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

def calculate_bollinger_bands(closes, period=20, num_std=2.0):
    middle, upper, lower, bandwidth = [], [], [], []
    for i in range(len(closes)):
        if i < period - 1:
            middle.append(closes[i])
            upper.append(closes[i])
            lower.append(closes[i])
            bandwidth.append(0.0)
        else:
            window = closes[i - period + 1 : i + 1]
            m = sum(window) / period
            variance = sum((x - m) ** 2 for x in window) / period
            std = math.sqrt(variance)
            u = m + (num_std * std)
            l = m - (num_std * std)
            bw = (u - l) / m if m > 0 else 0.0
            
            middle.append(m)
            upper.append(u)
            lower.append(l)
            bandwidth.append(bw)
    return middle, upper, lower, bandwidth

def calculate_sma(prices, period):
    sma = []
    for i in range(len(prices)):
        if i < period - 1:
            sma.append(sum(prices[:i+1]) / (i + 1))
        else:
            sma.append(sum(prices[i-period+1:i+1]) / period)
    return sma

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
# موتور بک‌تست نسخه v19.0 (استراتژی بولینگر سکیوئز و انفجار نوسان)
# ==============================================================================
def run_v19_squeeze_backtest(assets_data, initial_capital=1000.0, risk_per_trade_pct=1.5):
    capital = initial_capital
    peak_capital = initial_capital
    max_drawdown = 0.0
    all_trades = []

    for asset_name, candles in assets_data.items():
        if len(candles) < 50:
            continue

        closes = [c['close'] for c in candles]
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        volumes = [c['volume'] for c in candles]

        middle, upper, lower, bandwidth = calculate_bollinger_bands(closes, period=20, num_std=2.0)
        volume_sma = calculate_sma(volumes, 20)
        atr = calculate_atr(highs, lows, closes, 14)

        active_trade = None

        for i in range(25, len(candles)):
            current = candles[i]
            
            # مدیریت پوزیشن باز
            if active_trade:
                t = active_trade
                if t['type'] == 'LONG':
                    if current['low'] <= t['sl']:
                        capital -= t['risk_amount']
                        t['result'] = 'SL'
                        all_trades.append(t)
                        active_trade = None
                    elif current['high'] >= t['tp']:
                        capital += t['risk_amount'] * 2.2  # ریسک به ریوارد قدرتمند ۲.۲
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
                        capital += t['risk_amount'] * 2.2
                        t['result'] = 'TP'
                        all_trades.append(t)
                        active_trade = None

            # شرایط ورود: خروج قیمت از باندهای بولینگر همراه با حجم بالا بعد از یک دوره فشرده‌سازی
            if not active_trade:
                # بررسی اینکه پهنای باند در ۵ کندل قبل نسبتاً کم بوده (فشرده‌سازی)
                recent_bw = bandwidth[i-5:i]
                is_squeezed = bandwidth[i-1] <= min(recent_bw) * 1.1
                
                is_high_volume = volumes[i] > (volume_sma[i] * 1.4)

                long_signal = closes[i] > upper[i] and is_high_volume and is_squeezed
                short_signal = closes[i] < lower[i] and is_high_volume and is_squeezed

                risk_amt = capital * (risk_per_trade_pct / 100.0)

                if long_signal:
                    entry = current['close']
                    sl = entry - (atr[i] * 1.5)
                    tp = entry + ((entry - sl) * 2.2)
                    active_trade = {'asset': asset_name, 'type': 'LONG', 'entry': entry, 'sl': sl, 'tp': tp, 'risk_amount': risk_amt}
                elif short_signal:
                    entry = current['close']
                    sl = entry + (atr[i] * 1.5)
                    tp = entry - ((sl - entry) * 2.2)
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

    print("در حال دریافت داده‌ها و اجرای ربات نسخه v19.0 (بولینگر سکیوئز)...")
    for symbol in symbols:
        candles = fetch_safe_candles(symbol, timeframe='1hour', limit=1000)
        if candles:
            assets_data[symbol] = candles
        time.sleep(0.2)

    final_cap, trades, max_dd = run_v19_squeeze_backtest(assets_data, initial_capital=1000.0)
    
    win_trades = [t for t in trades if t.get('result') == 'TP']
    win_rate = (len(win_trades) / len(trades) * 100) if trades else 0

    print("="*50)
    print("=== گزارش بک‌تست ربات نسخه v19.0 (استراتژی بولینگر سکیوئز) ===")
    print("="*50)
    print(f"موجودی اولیه: $1000.00")
    print(f"موجودی نهایی: ${final_cap:.2f}")
    print(f"کل معاملات انجام شده: {len(trades)}")
    print(f"تعداد برد (TP): {len(win_trades)}")
    print(f"تعداد باخت (SL): {len(trades) - len(win_trades)}")
    print(f"وین‌ریت واقعی: {win_rate:.2f}%")
    print(f"حداکثر افت سرمایه (Max Drawdown): {max_dd:.2f}%")
    print("="*50)
