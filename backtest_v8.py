import pandas as pd
import requests
import numpy as np

def fetch_one_year_data(symbol="BTCUSDT", interval="1h"):
    print(f"در حال دانلود ۱ سال دیتای ۱ ساعته برای {symbol} از بایننس...")
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    limit = 1000
    end_time = int(pd.Timestamp.now().timestamp() * 1000)
    fetched = 0
    target_candles = 8760  # حدود ۱ سال (24 ساعت * 365 روز)
    
    while fetched < target_candles:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "endTime": end_time
        }
        try:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                break
            data = response.json()
            if not data:
                break
            all_data = data + all_data
            end_time = data[0][0] - 1
            fetched += len(data)
            if len(data) < limit:
                break
        except Exception as e:
            print(f"خطا در دریافت دیتا: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run_backtest_for_symbol(symbol):
    df = fetch_one_year_data(symbol, interval="1h")
    if df.empty or len(df) < 100:
        print(f"دیتای کافی برای {symbol} یافت نشد.")
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

    # گزارش نتایج
    if trades:
        df_trades = pd.DataFrame(trades)
        total_trades = len(df_trades)
        wins = len(df_trades[df_trades['result'] == 'WIN'])
        losses = len(df_trades[df_trades['result'] == 'LOSS'])
        win_rate = (wins / total_trades) * 100
        net_profit = df_trades['pnl'].sum()
        
        print(f"\n--- نتیجه بک‌تست یک‌ساله برای {symbol} ---")
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
