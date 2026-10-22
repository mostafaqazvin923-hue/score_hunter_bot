import os
import sys
import time
from datetime import datetime, timezone
import ccxt
import pandas as pd
import numpy as np

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'SUIUSDT', 'AVAXUSDT', 
    'NEARUSDT', 'ADAUSDT', 'BNBUSDT', 'APTUSDT', 'CRVUSDT', 
    'ONDOUSDT', 'PENDLEUSDT', 'ICPUSDT', 'WIFUSDT'
]
TIMEFRAME = '1h'
DAYS_TO_FETCH = 365 + 60
DATA_DIR = 'data_cache'

def get_exchange():
    # استفاده از تنظیمات پابلیک و غیرفعال کردن چک‌های اضافی برای دور زدن محدودیت‌های جغرافیایی سرور CI
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
            'adjustForTimeDifference': True
        }
    })
    
    # تنظیم هاست جایگزین عمومی بایننس فیوچرز در صورت مسدود بودن api.binance.com
    # صرافی بایننس اندپوینت‌های عمومی دیگری نیز برای فیوچرز دارد
    exchange.urls['api'] = {
        'public': 'https://fapi.binance.com/fapi/v1',
        'private': 'https://fapi.binance.com/fapi/v1',
        'v1': 'https://fapi.binance.com/api/v1',
        'v3': 'https://fapi.binance.com/api/v3',
    }
    return exchange

def fetch_ohlcv_data(exchange, symbol, since_ms):
    all_ohlcv = []
    current_since = since_ms
    now_ms = exchange.milliseconds()
    
    print(f"[{symbol}] در حال دریافت داده‌های OHLCV...")
    while current_since < now_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=current_since, limit=1500)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000.0)
        except Exception as e:
            print(f"[خطا] دریافت OHLCV برای {symbol} با شکست مواجه شد: {e}")
            sys.exit(1)
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def fetch_funding_rate_data(exchange, symbol, since_ms):
    all_fr = []
    current_since = since_ms
    now_ms = exchange.milliseconds()
    
    print(f"[{symbol}] در حال دریافت تاریخچه Funding Rate...")
    while current_since < now_ms:
        try:
            funding_rates = exchange.fetch_funding_rate_history(symbol, since=current_since, limit=1000)
            if not funding_rates:
                break
            for fr in funding_rates:
                all_fr.append({
                    'timestamp': pd.to_datetime(fr['timestamp'], unit='ms', utc=True),
                    'fundingRate': float(fr['fundingRate'])
                })
            current_since = funding_rates[-1]['timestamp'] + 1
            time.sleep(exchange.rateLimit / 1000.0)
        except Exception as e:
            print(f"[خطا] دریافت Funding Rate برای {symbol}: {e}")
            sys.exit(1)
            
    if not all_fr:
        print(f"[خطا بحرانی] هیچ داده فاندینگی برای {symbol} دریافت نشد!")
        sys.exit(1)
        
    df_fr = pd.DataFrame(all_fr)
    df_fr.drop_duplicates(subset=['timestamp'], inplace=True)
    df_fr.sort_values('timestamp', inplace=True)
    df_fr.reset_index(drop=True, inplace=True)
    return df_fr

def validate_and_clean_data(df_ohlcv, df_fr, symbol):
    print(f"[{symbol}] در حال اعتبارسنجی داده‌ها...")
    
    if df_ohlcv[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
        print(f"[خطا] داده‌های OHLCV برای {symbol} حاوی مقادیر NaN است.")
        sys.exit(1)
        
    invalid_prices = df_ohlcv[(df_ohlcv['high'] < df_ohlcv['low']) | 
                              (df_ohlcv['open'] <= 0) | 
                              (df_ohlcv['close'] <= 0)]
    if not invalid_prices.empty:
        print(f"[خطا] ناسازگاری قیمت در داده‌های {symbol} مشاهده شد.")
        sys.exit(1)

    if not df_ohlcv['timestamp'].is_monotonic_increasing:
        print(f"[خطا] ترتیب زمانی کندل‌های {symbol} نامعتبر است.")
        sys.exit(1)

    print(f"[{symbol}] اعتبارسنجی با موفقیت انجام شد. تعداد کل کندل‌ها: {len(df_ohlcv)}")
    return True

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    exchange = get_exchange()
    
    now = datetime.now(timezone.utc)
    since_dt = now - pd.Timedelta(days=DAYS_TO_FETCH)
    since_ms = int(since_dt.timestamp() * 1000)
    
    for symbol in SYMBOLS:
        print("\n" + "="*50)
        print(f"شروع پردازش نماد: {symbol}")
        print("="*50)
        
        df_ohlcv = fetch_ohlcv_data(exchange, symbol, since_ms)
        df_fr = fetch_funding_rate_data(exchange, symbol, since_ms)
        validate_and_clean_data(df_ohlcv, df_fr, symbol)
        
        ohlcv_path = os.path.join(DATA_DIR, f"{symbol}_OHLCV.csv")
        fr_path = os.path.join(DATA_DIR, f"{symbol}_FundingRate.csv")
        
        df_ohlcv.to_csv(ohlcv_path, index=False)
        df_fr.to_csv(fr_path, index=False)
        print(f"[{symbol}] داده‌ها با موفقیت ذخیره شدند.")

    print("\n[موفقیت] دریافت و اعتبارسنجی تمام نمادها با موفقیت پایان یافت.")

if __name__ == '__main__':
    main()
