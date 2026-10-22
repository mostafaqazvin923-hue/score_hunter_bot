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
    if raw_df.shape[1] < 6:
        return pd.DataFrame()
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
            else:
                failed_requests += 1
        except Exception as e:
            failed_requests += 1
            print(f"[هشدار] خطا در دریافت کندل {symbol} در ماه {year}-{m_str}: {e}")

    if not all_dfs:
        print(f"[خطا بحرانی] هیچ داده‌ای برای {symbol} از Binance Vision دریافت نشد!")
        sys.exit(1)

    df = pd.concat(all_dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[(df['timestamp'] >= pd.Timestamp(start_dt)) & (df['timestamp'] < pd.Timestamp(end_dt))]
    
    return df, success_requests, failed_requests, used_urls

def parse_funding_dataframe(df_raw):
    if df_raw.empty:
        return pd.DataFrame()
    try:
        first_val = str(df_raw.iloc[0, 0]).lower()
        if 'time' in first_val or 'calc' in first_val or 'symbol' in first_val:
            df = df_raw.iloc[1:].copy()
        else:
            df = df_raw.copy()
        
        time_col = None
        rate_col = None
        for col in df.columns:
            sample_val = str(df[col].iloc[0]) if len(df) > 0 else ""
            if sample_val.isdigit() and len(sample_val) >= 10:
                if time_col is None:
                    time_col = col
            try:
                float_val = float(sample_val)
                if -1.0 <= float_val <= 1.0 and float_val != 0.0:
                    if rate_col is None:
                        rate_col = col
            except:
                pass
                
        if time_col is None:
            time_col = 1 if df.shape[1] > 1 else 0
        if rate_col is None:
            rate_col = 2 if df.shape[1] > 2 else (1 if df.shape[1] > 1 else 0)
            
        res = pd.DataFrame()
        res['timestamp'] = pd.to_numeric(df[time_col], errors='coerce')
        res['fundingRate'] = pd.to_numeric(df[rate_col], errors='coerce')
        res.dropna(subset=['timestamp', 'fundingRate'], inplace=True)
        return res
    except Exception as e:
        print(f"[خطا در پارس فاندینگ] {e}")
        return pd.DataFrame()

def fetch_funding_rate_for_symbol(symbol):
    start_dt, end_dt = get_target_date_range()
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    all_fr = []
    success_requests = 0
    failed_requests = 0
    used_url = FAPI_BASE

    print(f"[{symbol}] در حال دریافت تاریخچه Funding Rate از REST API بایننس...")
    api_failed = False
    current_start = start_ms
    
    while current_start < end_ms:
        params = {
            'symbol': symbol,
            'startTime': current_start,
            'endTime': end_ms,
            'limit': 1000
        }
        try:
            resp = requests.get(FAPI_BASE, params=params, timeout=15)
            print(f"[HTTP FAPI] GET {symbol} -> Status: {resp.status_code}")
            if resp.status_code in [403, 451]:
                print(f"[هشدار] دسترسی REST API فاندینگ مسدود است ({resp.status_code}). استفاده از آرشیو رسمی Binance Vision...")
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
                api_failed = True
                break
        except Exception as e:
            print(f"[هشدار] خطای ارتباطی REST API فاندینگ: {e}")
            api_failed = True
            break

    if api_failed or not all_fr:
        print(f"[{symbol}] استفاده از آرشیو رسمی Binance Vision برای Funding Rate...")
        months, days = generate_months_and_days(start_dt, end_dt)
        
        for year, month in months:
            m_str = f"{month:02d}"
            fr_url = f"{VISION_BASE}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{year}-{m_str}.zip"
            used_url = fr_url
            try:
                resp = requests.get(fr_url, timeout=30)
                print(f"[HTTP Vision FR] GET {fr_url} -> Status: {resp.status_code}, Size: {len(resp.content)} bytes")
                if resp.status_code == 200:
                    success_requests += 1
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                        csv_filename = z.namelist()[0]
                        with z.open(csv_filename) as f:
                            df_fr_m = pd.read_csv(f, header=None)
                            parsed_fr = parse_funding_dataframe(df_fr_m)
                            for _, row in parsed_fr.iterrows():
                                t_val = row['timestamp']
                                t_dt = pd.to_datetime(t_val, unit='ms', utc=True)
                                all_fr.append({
                                    'timestamp': t_dt,
                                    'fundingRate': float(row['fundingRate'])
                                })
                elif resp.status_code == 404:
                    month_days = [d for d in days if d.year == year and d.month == month]
                    for d in month_days:
                        d_str = d.strftime("%Y-%m-%d")
                        d_url = f"{VISION_BASE}/daily/fundingRate/{symbol}/{symbol}-fundingRate-{d_str}.zip"
                        d_resp = requests.get(d_url, timeout=15)
                        if d_resp.status_code == 200:
                            success_requests += 1
                            with zipfile.ZipFile(io.BytesIO(d_resp.content)) as dz:
                                d_csv = dz.namelist()[0]
                                with dz.open(d_csv) as df_file:
                                    df_d = pd.read_csv(df_file, header=None)
                                    parsed_d = parse_funding_dataframe(df_d)
                                    for _, row in parsed_d.iterrows():
                                        t_val = row['timestamp']
                                        t_dt = pd.to_datetime(t_val, unit='ms', utc=True)
                                        all_fr.append({
                                            'timestamp': t_dt,
                                            'fundingRate': float(row['fundingRate'])
                                        })
                        else:
                            failed_requests += 1
                else:
                    failed_requests += 1
            except Exception as e:
                failed_requests += 1
                print(f"[هشدار] خطا در دریافت فاندینگ ویژن برای {symbol} در ماه {year}-{m_str}: {e}")

    if not all_fr:
        print(f"[خطا بحرانی] هیچ رکورد فاندینگی برای {symbol} از منابع رسمی بایننس دریافت نشد!")
        sys.exit(1)

    df_fr = pd.DataFrame(all_fr)
    df_fr['timestamp'] = pd.to_datetime(df_fr['timestamp'], utc=True)
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

    print("\n[موفقیت] تمام ۱۴ نماد با موفقیت از منابع رسمی بایننس دریافت، اعتبارسنجی و ذخیره شدند.")

if __name__ == '__main__':
    main()
