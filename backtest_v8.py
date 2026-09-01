import pandas as pd
import requests
import numpy as np
import time

def fetch_full_year_coinex(symbol="BTCUSDT"):
    print(f"در حال دانلود دیتای کامل یک‌ساله از صرافی کوینکس برای {symbol}...")
    all_data = []
    # آخرین زمان برای شروع پیمایش به سمت گذشته
    end_time = int(time.time())
    target_candles = 8760 # حدود یک سال کندل 1 ساعته
    
    while len(all_data) < target_candles:
        url = f"https://api.coinex.com/v1/market/kline?market={symbol}&type=1hour&limit=1000"
        if all_data:
            # عقب بردن زمان بر اساس قدیمی‌ترین کندلی که تاکنون گرفته‌ایم
            oldest_timestamp = all_data[0][0]
            url = f"https://api.coinex.com/v1/market/kline?market={symbol}&type=1hour&limit=1000&end_time={oldest_timestamp}"
            
        try:
            response = requests.get(url)
            res_json = response.json()
            if res_json.get("code") != 0 or not res_json.get("data"):
                break
            data = res_json["data"]
            if not data:
                break
            
            # اضافه کردن دیتا به ابتدای لیست
            all_data = data + all_data
            
            # اگر دیتای جدیدی نداد یعنی به ته تاریخچه رسیده‌ایم
            if len(data) < 1000:
                break
                
            # وقفه کوتاه برای بلاک نشدن توسط API
            time.sleep(0.2)
        except Exception as e:
            print(f"خطا در دریافت پله‌ای دیتا: {e}")
            break

    if not all_data:
        return None

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'amount'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['open'] = df['open'].astype(float)
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    # حذف تکراری‌ها و مرتب‌سازی
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run_backtest_for_symbol(symbol):
    df = fetch_full_year_coinex(symbol)
    if df is None or len(df) < 200:
        print(f"دیتای کافی برای {symbol} یافت نشد.\n")
        return
        
    df['rsi'] = calculate_rsi(df['close'], period=14)
    
    trades = []
    in_trade = False
    trade_side = None
    entry_price, tp, sl = 0, 0, 0
    
    for i in range(200, len(df)):
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
            # شرایط لانگ
            if current_close > prev_high and 40 <= current_rsi <= 70:
                in_trade = True
                trade_side = 'LONG'
                entry_price = current_close
                tp = entry_price * 1.025
                sl = entry_price * 0.99
            # شرایط شورت
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
        
        print(f"\n--- نتیجه واقعی بک‌تست یک‌ساله کوینکس برای {symbol} ---")
        print(f"تعداد کل معاملات: {total_trades}")
        print(f"معاملات موفق (WIN): {wins}")
        print(f"معاملات ناموفق (LOSS): {losses}")
        print(f"وین‌ریت (Win Rate): {win_rate:.2f}%")
        print(f"مجموع سود/زیان درصدی: {net_profit:.2f}%\n")
    else:
        print(f"هیچ معامله‌ای برای {symbol} ثبت نشد.\n")

if __name__ == "__main__":
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]:
        run_backtest_for_symbol(sym)
