import time
import requests

# ==============================================================================
# دریافت کندل‌های تاریخی واقعی از صرافی کوینکس (تنظیم شده با پارامترهای استاندارد v2)
# ==============================================================================
def fetch_historical_candles_coinex(symbol, timeframe='15min', limit=500):
    market = symbol.replace('/', '')
    url = "https://api.coinex.com/v2/spot/kline"
    
    # در API صرافی کوینکس پارامتر بازه زمانی به صورت period و با فرمت هایی مثل 15min ارسال می شود
    params = {
        "market": market,
        "limit": limit,
        "period": timeframe
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get('code') == 0 and data.get('data'):
            candles = []
            for c in data['data']:
                # سازگاری با ساختار پاسخ صرافی کوینکس (v2)
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
        
        # حالت دوم تست با پارامتر interval اگر period خطا داد
        params_alt = {"market": market, "limit": limit, "interval": timeframe}
        response_alt = requests.get(url, params=params_alt, timeout=10)
        data_alt = response_alt.json()
        
        if data_alt.get('code') == 0 and data_alt.get('data'):
            candles = []
            for c in data_alt['data']:
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

        print(f"خطا در دریافت تاریخچه برای {symbol}: {data.get('message', 'Unknown error')}")
        return []
    except Exception as e:
        print(f"ارور اتصال به کوینکس برای {symbol}: {e}")
        return []

# ==============================================================================
# توابع محاسباتی اندیکاتورها (منطبق بر v8.7)
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

# ==============================================================================
# موتور بک‌تست روی داده‌های واقعی
# ==============================================================================
def run_real_backtest(assets_data, initial_capital=1000.0, rr_ratio=2.0, risk_per_trade_pct=1.5):
    capital = initial_capital
    peak_capital = initial_capital
    max_drawdown = 0.0
    all_trades = []

    for asset_name, candles in assets_data.items():
        if len(candles) < 110:
            print(f"هشدار: داده‌های کافی برای {asset_name} موجود نیست (تعداد: {len(candles)}).")
            continue

        closes = [c['close'] for c in candles]
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        opens = [c['open'] for c in candles]
        volumes = [c['volume'] for c in candles]

        ema10 = calculate_ema(closes, 10)
        ema30 = calculate_ema(closes, 30)
        ema100 = calculate_ema(closes, 100)
        atr = calculate_atr(highs, lows, closes, 14)

        active_trade = None

        for i in range(100, len(candles)):
            current = candles[i]
            prev = candles[i-1]

            if active_trade:
                t = active_trade
                if t['type'] == 'LONG':
                    if current['low'] <= t['sl']:
                        pnl = -t['risk_amount']
                        capital += pnl
                        t['result'] = 'SL'
                        all_trades.append(t)
                        active_trade = None
                    elif current['high'] >= t['tp']:
                        pnl = t['risk_amount'] * rr_ratio
                        capital += pnl
                        t['result'] = 'TP'
                        all_trades.append(t)
                        active_trade = None
                elif t['type'] == 'SHORT':
                    if current['high'] >= t['sl']:
                        pnl = -t['risk_amount']
                        capital += pnl
                        t['result'] = 'SL'
                        all_trades.append(t)
                        active_trade = None
                    elif current['low'] <= t['tp']:
                        pnl = t['risk_amount'] * rr_ratio
                        capital += pnl
                        t['result'] = 'TP'
                        all_trades.append(t)
                        active_trade = None

            if not active_trade:
                c_close = prev['close']
                c_open = prev['open']
                
                bull_trend = (ema10[i-1] > ema30[i-1] > ema100[i-1]) and (ema10[i-1] > ema10[i-3])
                bear_trend = (ema10[i-1] < ema30[i-1] < ema100[i-1]) and (ema10[i-1] < ema10[i-3])

                strong_bull = (c_close > c_open) and ((c_close - c_open) / (prev['high'] - prev['low']) > 0.75) if (prev['high'] - prev['low']) > 0 else False
                strong_bear = (c_open > c_close) and ((c_open - c_close) / (prev['high'] - prev['low']) > 0.75) if (prev['high'] - prev['low']) > 0 else False

                avg_vol = sum(volumes[i-15:i-1]) / 14.0
                high_vol = prev['volume'] > (2.0 * avg_vol)

                long_cond = bull_trend and strong_bull and high_vol
                short_cond = bear_trend and strong_bear and high_vol

                risk_amt = capital * (risk_per_trade_pct / 100.0)

                if long_cond:
                    entry = current['open']
                    swing_low = min([candles[j]['low'] for j in range(i-10, i)])
                    sl = min(swing_low, entry - (atr[i-1] * 1.2))
                    tp = entry + ((entry - sl) * rr_ratio)
                    active_trade = {
                        'asset': asset_name, 'type': 'LONG', 'entry': entry, 
                        'sl': sl, 'tp': tp, 'risk_amount': risk_amt
                    }
                elif short_cond:
                    entry = current['open']
                    swing_high = max([candles[j]['high'] for j in range(i-10, i)])
                    sl = max(swing_high, entry + (atr[i-1] * 1.2))
                    tp = entry - ((sl - entry) * rr_ratio)
                    active_trade = {
                        'asset': asset_name, 'type': 'SHORT', 'entry': entry, 
                        'sl': sl, 'tp': tp, 'risk_amount': risk_amt
                    }

            if capital > peak_capital:
                peak_capital = capital
            dd = (peak_capital - capital) / peak_capital * 100.0
            if dd > max_drawdown:
                max_drawdown = dd

    return capital, all_trades, max_drawdown

if __name__ == "__main__":
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'AVAX/USDT', 'LINK/USDT']
    assets_real_data = {}

    print("در حال دانلود داده‌های تاریخی واقعی از صرافی کوینکس...")
    for symbol in symbols:
        # استفاده از تایم‌فریم استاندارد صرافی مثل 15min
        candles = fetch_historical_candles_coinex(symbol, timeframe='15min', limit=500)
        if candles:
            assets_real_data[symbol] = candles
            print(f"دریافت {len(candles)} کندل واقعی برای {symbol} با موفقیت انجام شد.")
        time.sleep(0.5)

    print("\nدر حال اجرای بک‌تست روی داده‌های واقعی کوینکس...")
    final_cap, trades, max_dd = run_real_backtest(assets_real_data, initial_capital=1000.0, rr_ratio=2.0)

    win_trades = [t for t in trades if t.get('result'] == 'TP']
    win_rate = (len(win_trades) / len(trades) * 100) if trades else 0

    print("="*50)
    print("=== خروجی بک‌تست واقعی v8.7 (صرافی کوینکس) ===")
    print("="*50)
    print(f"موجودی اولیه: $1000.00")
    print(f"موجودی نهایی: ${final_cap:.2f}")
    print(f"کل معاملات انجام شده: {len(trades)}")
    print(f"تعداد سود (TP): {len(win_trades)}")
    print(f"تعداد ضرر (SL): {len(trades) - len(win_trades)}")
    print(f"وین‌ریت (Win Rate): {win_rate:.2f}%")
    print(f"حداکثر افت حساب (Max Drawdown): {max_dd:.2f}%")
    print("="*50)
