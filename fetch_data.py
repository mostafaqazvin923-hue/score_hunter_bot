import os
import sys
import io
import zipfile
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'SUIUSDT', 'AVAXUSDT', 
    'NEARUSDT', 'ADAUSDT', 'BNBUSDT', 'APTUSDT', 'CRVUSDT', 
    'ONDOUSDT', 'PENDLEUSDT', 'ICPUSDT', 'WIFUSDT'
]
TIMEFRAME = '1h'
DAYS_TO_FETCH = 365 + 60  # ۳۶۵ روز تست + ۶۰ روز Warm-up
DATA_DIR = 'data_cache'

os.makedirs(DATA_DIR, exist_ok=True)

VISION_BASE = "https://data.binance.vision/data/futures/um"
FAPI_BASE = "https://fapi.binance.com/fapi/v1/fundingRate"

def get_target_date_range():
    now_utc = datetime.now(timezone.utc)
    current_hour_start = now_utc.replace(minute=0, second=0, microsecond=0)
    start_dt = current_hour_start - timedelta(days=DAYS_TO_FETCH)
    return start_dt, current_hour_start

def generate_months_and_days(start_dt, end_dt):
    months = set()
    days = []
    curr = start_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while curr <= end_dt:
        months.add((curr.year, curr.month))
        curr += timedelta(days=32)
        curr = curr.replace(day=1)
    
    d_curr = start_dt.date()
    d_end = end_dt.date()
    while d_curr <= d_end:
        days.append(d_curr)
        d_curr += timedelta(days=1)
    return sorted(list(months)), days

def parse_kline_dataframe(raw_df):
    """
    فایل‌های Binance Vision Klines بدون هیدر هستند و حداقل ۱۲ ستون استاندارد دارند:
    0: open_time, 1: open, 2: high, 3: low, 4: close, 5: volume, ...
    """
    if raw_df.shape[1] < 6:
        # اگر ستون‌ها کمتر از حد استاندارد باشند
        return pd.DataFrame()

    # انتخاب ۶ ستون اول به عنوان OHLCV و Timestamp
    df = raw_df.iloc[:, [0, 1, 2, 3, 4, 5]].copy()
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df.dropna(subset=['timestamp'], inplace=True)
    return df

def fetch_klines_for_symbol(symbol):
    start_dt, end_dt = get_target_date_range()
    months, days = generate_months_and_days(start_dt, end_dt)
    
    all_dfs = []
    success_requests = 0
    failed_requests = 0
    used_urls = []

    print(f"[{symbol}] شروع دریافت کندل‌های 1h از Binance Vision...")

    for year, month in months:
        m_str = f"{month:02d}"
        url = f"{VISION_BASE}/monthly/klines/{symbol}/1h/{symbol}-1h-{year}-{m_str}.zip"
        used_urls.append(url)
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                success_requests += 1
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    csv_filename = z.namelist()[0]
                    with z.open(csv_filename) as f:
                        df_m = pd.read_csv(f, header=None)
                        parsed_m = parse_kline_dataframe(df_m)
                        if not parsed_m.empty:
                            all_dfs.append(parsed_m)
            elif resp.status_code == 404:
                month_days = [d for d in days if d.year == year and d.month == month]
                for d in month_days:
                    d_str = d.strftime("%Y-%m-%d")
                    d_url = f"{VISION_BASE}/daily/klines/{symbol}/1h/{symbol}-1h-{d_str}.zip"
                    used_urls.append(d_url)
                    d_resp = requests.get(d_url, timeout=15)
                    if d_resp.status_code == 200:
                        success_requests += 1
                        with zipfile.ZipFile(io.BytesIO(d_resp.content)) as dz:
                            d_csv = dz.namelist()[0]
                            with dz.open(d_csv) as df_file:
                                df_d = pd.read_csv(df_file, header=None)
                                parsed_d = parse_kline_dataframe(df_d)
                                if not parsed_d.empty:
                                    all_dfs.append(parsed_d)
            else:
                failed_requests += 1
        except Exception as e:
            failed_requests += 1
            print(f"[هشدار] خطا در دریافت آرشیو برای {symbol} در ماه {year}-{m_str}: {e}")

    if not all_dfs:
        print(f"[خطا بحرانی] هیچ داده‌ای برای {symbol} از Binance Vision دریافت نشد!")
        sys.exit(1)

    df = pd.concat(all_dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df[(df['timestamp'] >= pd.Timestamp(start_dt)) & (df['timestamp'] < pd.Timestamp(end_dt))]
    
    if df.empty:
        print(f"[خطا بحرانی] دی‌اف کندل‌های {symbol} پس از فیلتر بازه زمانی خالی است!")
        sys.exit(1)
        
    return df, success_requests, failed_requests, used_urls

def fetch_funding_rate_for_symbol(symbol):
    start_dt, end_dt = get_target_date_range()
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    all_fr = []
    current_start = start_ms
    success_requests = 0
    failed_requests = 0
    used_url = FAPI_BASE

    print(f"[{symbol}] در حال دریافت تاریخچه Funding Rate از REST API بایننس...")
    
    api_failed = False
    while current_start < end_ms:
        params = {
            'symbol': symbol,
            'startTime': current_start,
            'endTime': end_ms,
            'limit': 1000
        }
        try:
            resp = requests.get(FAPI_BASE, params=params, timeout=15)
            if resp.status_code in [403, 451]:
                print(f"[هشدار] دسترسی به REST API فاندینگ با محدودیت مواجه شد ({resp.status_code}). استفاده از آرشیو Binance Vision...")
                api_failed = True
                break
            elif resp.status_code == 200:
                success_requests += 1
                data = resp.json()
                if not data:
                    break
                for item in data:
                    all_fr.append({
                        'timestamp': pd.to_datetime(item['fundingTime'], unit='ms', utc=True),
                        'fundingRate': float(item['fundingRate'])
                    })
                next_start = data[-1]['fundingTime'] + 1
                if next_start <= current_start:
                    break
                current_start = next_start
            else:
                failed_requests += 1
                break
        except Exception as e:
            api_failed = True
            break

    if api_failed or not all_fr:
        print(f"[{symbol}] استفاده از آرشیو ماهیانه Binance Vision برای Funding Rate...")
        months, _ = generate_months_and_days(start_dt, end_dt)
        for year, month in months:
            m_str = f"{month:02d}"
            fr_url = f"{VISION_BASE}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{year}-{m_str}.zip"
            try:
                resp = requests.get(fr_url, timeout=30)
                if resp.status_code == 200:
                    success_requests += 1
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                        csv_filename = z.namelist()[0]
                        with z.open(csv_filename) as f:
                            df_fr_m = pd.read_csv(f, header=None)
                            if df_fr_m.shape[1] >= 3:
                                for _, row in df_fr_m.iterrows():
                                    t_val = row[1]
                                    r_val = row[2]
                                    t_dt = pd.to_datetime(t_val, unit='ms', utc=True) if isinstance(t_val, (int, float)) else pd.to_datetime(t_val, utc=True)
                                    all_fr.append({
                                        'timestamp': t_dt,
                                        'fundingRate': float(r_val)
                                    })
                else:
                    failed_requests += 1
            except Exception as e:
                failed_requests += 1

    if not all_fr:
        print(f"[خطا بحرانی] هیچ رکورد فاندینگی برای {symbol} دریافت نشد!")
        sys.exit(1)

    df_fr = pd.DataFrame(all_fr)
    df_fr.drop_duplicates(subset=['timestamp'], inplace=True)
    df_fr.sort_values('timestamp', inplace=True)
    df_fr.reset_index(drop=True, inplace=True)
    
    df_fr = df_fr[(df_fr['timestamp'] >= pd.Timestamp(start_dt)) & (df_fr['timestamp'] < pd.Timestamp(end_dt))]
    return df_fr, success_requests, failed_requests, used_url

def validate_data(df_ohlcv, df_fr, symbol):
    print(f"[{symbol}] در حال اعتبارسنجی نهایی داده‌ها...")
    if df_ohlcv[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
        print(f"[خطا] مقادیر NaN در کندل‌های {symbol} وجود دارد.")
        sys.exit(1)
    if df_fr[['fundingRate']].isnull().any().any():
        print(f"[خطا] مقادیر NaN در فاندینگ ریت {symbol} وجود دارد.")
        sys.exit(1)

    invalid = df_ohlcv[(df_ohlcv['high'] < df_ohlcv['low']) | (df_ohlcv['open'] <= 0) | (df_ohlcv['close'] <= 0)]
    if not invalid.empty:
        print(f"[خطا] قیمت‌های نامعتبر در {symbol} شناسایی شد.")
        sys.exit(1)

    if df_ohlcv['timestamp'].dt.tz is None or df_fr['timestamp'].dt.tz is None:
        print(f"[خطا] تایم‌استمپ‌ها باید دارای منطقه زمانی UTC باشند.")
        sys.exit(1)

    if not df_ohlcv['timestamp'].is_monotonic_increasing:
        print(f"[خطا] ترتیب زمانی کندل‌های {symbol} صعودی نیست.")
        sys.exit(1)
    if df_ohlcv['timestamp'].duplicated().any():
        print(f"[خطا] تایم‌استمپ تکراری در کندل‌های {symbol} وجود دارد.")
        sys.exit(1)

    print(f"[{symbol}] اعتبارسنجی با موفقیت کامل انجام شد.")

def main():
    for symbol in SYMBOLS:
        print("\n" + "="*60)
        print(f"پردازش نماد: {symbol}")
        print("="*60)

        df_ohlcv, s_req_k, f_req_k, urls_k = fetch_klines_for_symbol(symbol)
        df_fr, s_req_f, f_req_f, url_f = fetch_funding_rate_for_symbol(symbol)

        validate_data(df_ohlcv, df_fr, symbol)

        print(f"--- گزارش دریافت داده برای {symbol} ---")
        print(f"منبع OHLCV: Binance Vision Archive")
        print(f"منبع Funding Rate: {url_f}")
        print(f"اولین کندل (UTC): {df_ohlcv['timestamp'].iloc[0]}")
        print(f"آخرین کندل (UTC): {df_ohlcv['timestamp'].iloc[-1]}")
        print(f"تعداد کل کندل‌های 1h: {len(df_ohlcv)}")
        print(f"تعداد رکوردهای Funding Rate: {len(df_fr)}")

        df_ohlcv.to_csv(os.path.join(DATA_DIR, f"{symbol}_OHLCV.csv"), index=False)
        df_fr.to_csv(os.path.join(DATA_DIR, f"{symbol}_FundingRate.csv"), index=False)
        print(f"فایل‌های {symbol} با موفقیت در {DATA_DIR} ذخیره شدند.")

    print("\n[موفقیت] تمام ۱۴ نماد با موفقیت از Binance Vision دریافت، اعتبارسنجی و ذخیره شدند.")

if __name__ == '__main__':
    main()
