import pandas as pd
import numpy as np
import requests

def get_crypto_klines(symbol="SOLUSDT", interval="15m", limit=1000):
    """دریافت دیتای کندل‌ها از API عمومی"""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        res = requests.get(url, timeout=10).json()
        if not isinstance(res, list):
            return None
        df = pd.DataFrame(res, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        print(f"خطا در دریافت دیتای {symbol}: {e}")
        return None

def run_multi_timeframe_backtest(symbol="SOLUSDT"):
    df_15m = get_crypto_klines(symbol=symbol, interval="15m", limit=1000)
    if df_15m is None or df_15m.empty:
        print(f"دیتایی برای {symbol} دریافت نشد.")
        return None

    # ساخت تایم‌فریم ۱ ساعته درون خود کد برای تشخیص روند اصلی
    df_15m.set_index('timestamp', inplace=True)
    df_1h = df_15m.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

    # روند ۱ ساعته با EMA 200
    df_1h['trend_ema'] = df_1h['close'].ewm(span=200, adjust=False).mean()
    df_15m['htf_trend'] = df_1h['trend_ema'].reindex(df_15m.index, method='ffill')

    # محاسبات تایم‌فریم ۱۵ دقیقه (ورود و خروج)
    high_low = df_15m['high'] - df_15m['low']
    high_close = np.abs(df_15m['high'] - df_15m['close'].shift())
    low_close = np.abs(df_15m['low'] - df_15m['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df_15m['atr'] = np.max(ranges, axis=1).rolling(14).mean()

    df_15m['upper_break'] = df_15m['high'].shift(1).rolling(15).max()
    df_15m['lower_break'] = df_15m['low'].shift(1).rolling(15).min()

    # RSI
    delta = df_15m['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df_15m['rsi'] = 100 - (100 / (1 + rs))

    rr_ratio = 1.5
    atr_sl_mult = 1.5

    trades = []
    in_position = False
    pos_type, entry_price, sl, tp = None, 0, 0, 0

    for i in range(200, len(df_15m)):
        row = df_15m.iloc[i]
        c_close = row['close']
        c_high = row['high']
        c_low = row['low']
        c_atr = row['atr']
        htf_ema = row['htf_trend']

        if pd.isna(htf_ema) or pd.isna(c_atr):
            continue

        if not in_position:
            # LONG
            if (c_close > htf_ema) and (c_close > row['upper_break']) and (row['rsi'] > 50):
                in_position = True
                pos_type = 'LONG'
                entry_price = c_close
                sl = entry_price - (c_atr * atr_sl_mult)
                tp = entry_price + ((entry_price - sl) * rr_ratio)

            # SHORT
            elif (c_close < htf_ema) and (c_close < row['lower_break']) and (row['rsi'] < 50):
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
                elif c_low <= tp:
                    trades.append({'symbol': symbol, 'type': 'SHORT', 'result': 'WIN', 'pnl': rr_ratio})
                    in_position = False

    df_results = pd.DataFrame(trades)
    if df_results.empty:
        print(f"[{symbol}] هیچ سیگنالی صادر نشد.")
        return []

    total = len(df_results)
    wins = len(df_results[df_results['result'] == 'WIN'])
    losses = len(df_results[df_results['result'] == 'LOSS'])
    win_rate = (wins / total) * 100
    pnl_r = df_results['pnl'].sum()

    print(f"[{symbol}] تعداد سیگنال: {total} | برد: {wins} | باخت: {losses} | وین‌ریت: {win_rate:.2f}% | سود: {pnl_r:.2f}R")
    return trades

if __name__ == "__main__":
    # لیست ارزهای مورد نظر شما + ارزهای پرحجم پیشنهادی
    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "DOGEUSDT",
        "DYDXUSDT",
        "LINKUSDT",
        "ADAUSDT",
        "XRPUSDT",
        "NEARUSDT",
        "AVAXUSDT"
    ]

    print("=== شروع بک‌تست روی لیست ارزهای انتخابی ===\n")
    all_trades = []

    for s in symbols:
        res = run_multi_timeframe_backtest(symbol=s)
        if res:
            all_trades.extend(res)

    # ارائه گزارش کلی از کل ارزها
    df_all = pd.DataFrame(all_trades)
    if not df_all.empty:
        total_all = len(df_all)
        wins_all = len(df_all[df_all['result'] == 'WIN'])
        losses_all = len(df_all[df_all['result'] == 'LOSS'])
        total_winrate = (wins_all / total_all) * 100
        total_pnl = df_all['pnl'].sum()

        print("\n==========================================")
        print("          نتایج جمع‌بندی کل ارزها          ")
        print("==========================================")
        print(f"مجموع کل سیگنال‌ها: {total_all}")
        print(f"کل معاملات موفق (WIN): {wins_all}")
        print(f"کل معاملات ناموفق (LOSS): {losses_all}")
        print(f"وین‌ریت میانگین کل: {total_winrate:.2f}%")
        print(f"مجموع سود خالص (بر حسب R): {total_pnl:.2f}R")
        print("==========================================")
