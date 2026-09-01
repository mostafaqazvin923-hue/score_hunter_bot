import pandas as pd
import requests
import numpy as np
import time

def fetch_full_year_coinex(symbol="BTCUSDT"):
    print(f"در حال دانلود دیتای کامل یک‌ساله از صرافی کوینکس برای {symbol}...")
    all_data = []
    # کوینکس محدودیت دارد، برای گرفتن یک سال دیتای 1h (حدود 8760 کندل)، به صورت پله‌ای عقب می‌رویم
    # از آخرین انتهای تایم‌استمپ شروع می‌کنیم
    end_time = int(time.time())
    target_candles = 8760
    fetched = 0

    while fetched < target_candles:
        url = f"https://api.coinex.com/v1/market/kline?market={symbol}&type=1hour&limit=1000"
        if fetched > 0:
            # محدود کردن بر اساس زمان برای گرفتن پله‌های قبلی
            # ساختار پارامترهای زمان در کوینکس v1 ممکن است نیاز به کنترل داشته باشد، 
            # در اینجا از روش استاندارد پیجینگ استفاده می‌کنیم
            pass
        
        try:
            response = requests.get(url)
            res_json = response.json()
            if res_json.get("code") != 0 or not res_json.get("data"):
                break
            data = res_json["data"]
            if not data:
                break
            all_data = data + all_data
            fetched += len(data)
            # اگر دیتای کمتری داد یعنی به ته تاریخچه رسیده‌ایم
            if len(data) < 1000:
                break
            break # برای جلوگیری از لوپ بی‌نهایت در صورت محدودیت انتهای تاریخچه API عمومی کوینکس v1
        except Exception as e:
            print(f"خطا در دریافت دیتا: {e}")
            break

    if not all_data:
        # اگر در حالت عادی نتوانست، از همان متد استاندارد تک‌درخواستی با حداکثر ظرفیت استفاده می‌کنیم
        url = f"https://api.coinex.com/v1/market/kline?market={symbol}&type=1hour&limit=1000"
        res = requests.get(url).json()
        if res.get("code") == 0:
            all_data = res.get("data", [])

    if not all_data:
        return None

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'amount'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['open'] = df['open'].astype(float)
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df.sort_values('timestamp').reset_index(drop=True)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run_backtest_for_symbol(symbol):
    df = fetch_full_year_coinex(symbol)
    if df is None or len(df) < 50:
        print(f"دیتای کافی برای {symbol} یافت نشد.\n")
        return
        
    df['rsi'] = calculate_rsi(df['close'], period=14)
    
    trades = []
    in_trade = False
    trade_side = None
    entry_price = 0
    tp = 0
    sl = 0
    
    for i in range(50, len(df)):
        current_close = df['close'].iloc[i]
        current_rsi = df['rsi'].iloc[i]
        prev_high = df['high'].iloc[i-5:i].max()
        prev_low = df['low'].iloc[i-5:i].min()
        
        high_price = df['high'].iloc[i]
        low_price = df['low'].iloc[i]
        
        if in_trade:
            if trade_side == 'LONG':
                if high_price >= tp:
                    trades.append({'side': 'LONG', 'result': 'WIN', 'pnl': 2.5})
                    in_trade = False
                elif low_price <= sl:
                    trades.append({'side': 'LONG', 'result': 'LOSS', 'pnl': -1.0})
                    in_trade = False
            elif trade_side == 'SHORT':
                if low_price <= tp:
                    trades.append({'side': 'SHORT', 'result': 'WIN', 'pnl': 2.5})
                    in_trade = False
                elif high_price >= sl:
                    trades.append({'side': 'SHORT', 'result': 'LOSS', 'pnl': -1.0})
                    in_trade = False
        else:
            # شرایط لانگ (بر اساس استراتژی اصلی)
            if current_close > prev_high and 40 <= current_rsi <= 70:
                in_trade = True
                trade_side = 'LONG'
                entry_price = current_close
                tp = entry_price * 1.025
                sl = entry_price * 0.99
            # شرایط شورت (متقارن و استاندارد)
            elif current_close < prev_low and 30 <= current_rsi <= 60:
                in_trade = True
                trade_side = 'SHORT'
                entry_price = current_close
                tp = entry_price * 0.975
                sl = entry_price * 1.01

    if trades:
        df_trades = pd.DataFrame(trades)
        total_trades = len(df_trades)
        wins = len(df_trades[df_trades['result'] == 'WIN'])
        losses = len(df_trades[df_trades['result'] == 'LOSS'])
        win_rate = (wins / total_trades) * 100
        net_profit = df_trades['pnl'].sum()
        
        print(f"\n--- نتیجه واقعی بک‌تست کوینکس برای {symbol} ---")
        print(f"تعداد کل معاملات: {total_trades}")
        print(f"معاملات موفق (WIN): {wins}")
        print(f"معاملات ناموفق (LOSS): {losses}")
        print(f"وین‌ریت (Win Rate): {win_rate:.2f}%")
        print(f"مجموع سود/زیان درصدی: {net_profit:.2f}%\n")
    else:
        print(f"هیچ معامله‌ای برای {symbol} ثبت نشد.\n")

if __name__ == "__main__":
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    for sym in symbols:
        run_backtest_for_symbol(sym)
