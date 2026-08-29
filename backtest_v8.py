import json
import urllib.request

def get_crypto_klines(symbol="BTC-USDT", bar="15m", limit=300):
    """دریافت کندل‌ها از API عمومی OKX (بدون محدودیت آی‌پی روی GitHub)"""
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={bar}&limit={limit}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode())
            if res_data.get('code') != '0' or 'data' not in res_data:
                print(f"خطا در دیتا {symbol}: {res_data.get('msg')}")
                return None
            
            raw_list = res_data['data']
            # OKX دیتا را از جدید به قدیم ارسال می‌کند؛ آن را مرتب می‌کنیم
            raw_list.reverse()
            
            klines = []
            for item in raw_list:
                # [ts, o, h, l, c, vol, ...]
                klines.append({
                    'timestamp': int(item[0]),
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4]),
                    'volume': float(item[5])
                })
            return klines
    except Exception as e:
        print(f"خطا در دریافت دیتای {symbol}: {e}")
        return None

def calculate_ema(prices, span):
    alpha = 2 / (span + 1)
    ema = [prices[0]]
    for p in prices[1:]:
        ema.append(p * alpha + ema[-1] * (1 - alpha))
    return ema

def run_multi_timeframe_backtest(symbol="BTC-USDT"):
    klines_15m = get_crypto_klines(symbol=symbol, bar="15m", limit=300)
    if not klines_15m or len(klines_15m) < 100:
        print(f"[{symbol}] دیتای کافی دریافت نشد.")
        return []

    closes = [k['close'] for k in klines_15m]
    highs = [k['high'] for k in klines_15m]
    lows = [k['low'] for k in klines_15m]

    # محاسبه ATR (14)
    atr = []
    for i in range(len(klines_15m)):
        if i == 0:
            atr.append(highs[0] - lows[0])
        else:
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            atr.append(tr)
    
    atr_smooth = []
    for i in range(len(atr)):
        if i < 14:
            atr_smooth.append(sum(atr[:i+1]) / (i+1))
        else:
            atr_smooth.append(sum(atr[i-13:i+1]) / 14)

    # محاسبه RSI (14)
    rsi = [50.0] * len(closes)
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i-1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
        if i >= 14:
            avg_gain = sum(gains[-14:]) / 14
            avg_loss = sum(losses[-14:]) / 14
            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    # شبیه‌سازی روند با EMA 100
    ema_htf = calculate_ema(closes, 100)

    rr_ratio = 1.5
    atr_sl_mult = 1.5

    trades = []
    in_position = False
    pos_type, entry_price, sl, tp = None, 0, 0, 0

    for i in range(100, len(klines_15m)):
        c_close = closes[i]
        c_high = highs[i]
        c_low = lows[i]
        c_atr = atr_smooth[i]
        htf_trend = ema_htf[i]

        upper_break = max(highs[i-10:i])
        lower_break = min(lows[i-10:i])

        if not in_position:
            # LONG
            if (c_close > htf_trend) and (c_close > upper_break) and (rsi[i] > 50):
                in_position = True
                pos_type = 'LONG'
                entry_price = c_close
                sl = entry_price - (c_atr * atr_sl_mult)
                tp = entry_price + ((entry_price - sl) * rr_ratio)

            # SHORT
            elif (c_close < htf_trend) and (c_close < lower_break) and (rsi[i] < 50):
                in_position = True
                pos_type = 'SHORT'
                entry_price = c_close
                sl = entry_price + (c_atr * atr_sl_mult)
                tp = entry_price - ((sl - entry_price) * rr_ratio)

        else:
            if pos_type == 'LONG':
                if c_low <= sl:
                    trades.append({'symbol': symbol, 'type': 'LONG', 'result': 'LOSS', 'pnl': -1.0})
                    in_position = False
                elif c_high >= tp:
                    trades.append({'symbol': symbol, 'type': 'LONG', 'result': 'WIN', 'pnl': rr_ratio})
                    in_position = False

            elif pos_type == 'SHORT':
                if c_high >= sl:
                    trades.append({'symbol': symbol, 'type': 'SHORT', 'result': 'LOSS', 'pnl': -1.0})
                    in_position = False
                elif c_low >= tp:
                    trades.append({'symbol': symbol, 'type': 'SHORT', 'result': 'WIN', 'pnl': rr_ratio})
                    in_position = False

    if not trades:
        print(f"[{symbol}] هیچ سیگنالی در این بازه صادر نشد.")
        return []

    total = len(trades)
    wins = len([t for t in trades if t['result'] == 'WIN'])
    losses = total - wins
    win_rate = (wins / total) * 100
    pnl_r = sum([t['pnl'] for t in trades])

    print(f"[{symbol}] تعداد سیگنال: {total} | برد: {wins} | باخت: {losses} | وین‌ریت: {win_rate:.2f}% | سود: {pnl_r:.2f}R")
    return trades

if __name__ == "__main__":
    # نمادها بر اساس فرمت OKX
    symbols = [
        "BTC-USDT",
        "ETH-USDT",
        "SOL-USDT",
        "DOGE-USDT",
        "DYDX-USDT",
        "LINK-USDT",
        "ADA-USDT",
        "XRP-USDT",
        "NEAR-USDT",
        "AVAX-USDT"
    ]

    print("=== شروع بک‌تست بر پایه API صرافی OKX ===\n")
    all_trades = []

    for s in symbols:
        res = run_multi_timeframe_backtest(symbol=s)
        if res:
            all_trades.extend(res)

    if all_trades:
        total_all = len(all_trades)
        wins_all = len([t for t in all_trades if t['result'] == 'WIN'])
        losses_all = total_all - wins_all
        total_winrate = (wins_all / total_all) * 100
        total_pnl = sum([t['pnl'] for t in all_trades])

        print("\n==========================================")
        print("          نتایج جمع‌بندی کل ارزها          ")
        print("==========================================")
        print(f"مجموع کل سیگنال‌ها: {total_all}")
        print(f"کل معاملات موفق (WIN): {wins_all}")
        print(f"کل معاملات ناموفق (LOSS): {losses_all}")
        print(f"وین‌ریت میانگین کل: {total_winrate:.2f}%")
        print(f"مجموع سود خالص (بر حسب R): {total_pnl:.2f}R")
        print("==========================================")
