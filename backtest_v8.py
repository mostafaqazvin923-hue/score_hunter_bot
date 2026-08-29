import time
import requests

# ==============================================================================
# دریافت کندل‌های استاندارد از صرافی کوینکس (تایم‌فریم ۱ ساعته)
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
# موتور بک‌تست نسخه v18.0 (استراتژی تک‌تیرانداز / شکست محدوده با حجم بالا)
# ==============================================================================
def run_v18_sniper_backtest(assets_data, initial_capital=1000.0, risk_per_trade_pct=1.5):
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

        volume_sma = calculate_sma(volumes, 20)
        atr = calculate_atr(highs, lows, closes, 14)

        active_trade = None
        lookback_period = 20  # بررسی بالاترین و پایین‌ترین قیمت در ۲۰ کندل گذشته

        for i in range(lookback_period, len(candles)):
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
                        capital += t['risk_amount'] * 2.0  # ریسک به ریوارد طلایی ۲ به ۱
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
                        capital += t['risk_amount'] * 2.0
                        t['result'] = 'TP'
                        all_trades.append(t)
                        active_trade = None

            # شرایط شکار (تک‌تیرانداز): شکست سقف یا کف ۲۰ کندل قبل با حجم حداقل ۱.۵ برابر میانگین
            if not active_trade:
                recent_high = max(highs[i-lookback_period : i])
                recent_low = min(lows[i-lookback_period : i])

                # حجم باید به طور محسوسی بالا باشد تا فیک‌بریک‌اوت نشود
                is_high_volume = volumes[i] > (volume_sma[i] * 1.5)

                long_breakout = closes[i] > recent_high and is_high_volume
                short_breakout = closes[i] < recent_low and is_high_volume

                risk_amt = capital * (risk_per_trade_pct / 100.0)

                if long_breakout:
                    entry = current['close']
                    sl = entry - (atr[i] * 1.5)
                    tp = entry + ((entry - sl) * 2.0)
                    active_trade = {'asset': asset_name, 'type': 'LONG', 'entry': entry, 'sl': sl, 'tp': tp, 'risk_amount': risk_amt}
                elif short_breakout:
                    entry = current['close']
                    sl = entry + (atr[i] * 1.5)
                    tp = entry - ((sl - entry) * 2.0)
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

    print("در حال دریافت داده‌های تاریخی برای ربات نسخه v18.0 (مدل تک‌تیرانداز)...")
    for symbol in symbols:
        candles = fetch_safe_candles(symbol, timeframe='1hour', limit=1000)
        if candles:
            assets_data[symbol] = candles
        time.sleep(0.2)

    final_cap, trades, max_dd = run_v18_sniper_backtest(assets_data, initial_capital=1000.0)
    
    win_trades = [t for t in trades if t.get('result') == 'TP']
    win_rate = (len(win_trades) / len(trades) * 100) if trades else 0

    print("="*50)
    print("=== گزارش بک‌تست ربات نسخه v18.0 (استراتژی تک‌تیرانداز) ===")
    print("="*50)
    print(f"موجودی اولیه: $1000.00")
    print(f"موجودی نهایی: ${final_cap:.2f}")
    print(f"کل معاملات انجام شده: {len(trades)}")
    print(f"تعداد برد (TP): {len(win_trades)}")
    print(f"تعداد باخت (SL): {len(trades) - len(win_trades)}")
    print(f"وین‌ریت واقعی: {win_rate:.2f}%")
    print(f"حداکثر افت سرمایه (Max Drawdown): {max_dd:.2f}%")
    print("="*50)
