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
    بررسی و استانداردسازی دی‌اف کندل‌های بایننس (مدیریت فایل‌های با هیدر یا بدون هیدر)
    """
    # اگر ستون اول یا مقدار سطر اول شامل حروف باشد (مثل open_time)، یعنی فایل هیدر دارد
    first_val = str(raw_df.iloc[0, 0]).lower()
    if 'time' in first_val or 'open' in first_val:
        # اگر هیدر دارد، از سطر دوم به عنوان داده استفاده می‌کنیم و نام ستون‌ها را تنظیم می‌کنیم
        raw_df.columns = [str(c).strip().lower() for c in raw_df.iloc]
        df = raw_df.iloc[1:].copy()
        
        # شناسایی نام ستون زمان و قیمت‌ها
        time_col = next((c for c in df.columns if 'time' in c or 'open_time' in c), df.columns[0])
        open_col = next((c for c in df.columns if c == 'open'), df.columns[1])
        high_col = next((c for c in df.columns if c == 'high'), df.columns[2])
        low_col = next((c for c in df.columns if c == 'low'), df.columns[3])
        close_col = next((c for c in df.columns if c == 'close'), df.columns[4])
        vol_col = next((c for c in df.columns if 'volume' in c), df.columns[5])
        
        df = df[[time_col, open_col, high_col, low_col, close_col, vol_col]]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    else:
        # بدون هیدر (استاندارد بایننس ویژن: 0:open_time, 1:open, 2:high, 3:low, 4:close, 5:volume)
        df = raw_df.iloc[:, [0, 1, 2, 3, 4, 5]].copy()
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    # تبدیل مقادیر به عددی و حذف سطرهای احتمالی نامعتبر
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
                        all_dfs.append(parsed_m)
            elif resp.status_code == 404:
                month_days = [d for d in days if d.year == year and d.month == month]
                for d in month_days:
                    d_str = d.strftime("%Y-%m-%d")
                    d_url = f"{VISION_BASE}/daily/klines/{symbol}/1h/{symbol}-1h-{d_str}.zip"
                    used_urls.append(d_url)
                    d_resp = requests.get(d_url, timeout=20)
                    if d_resp.status_code == 200:
                        success_requests += 1
                        with zipfile.ZipFile(io.BytesIO(d_resp.content)) as dz:
                            d_csv = dz.namelist()[0]
                            with dz.open(d_csv) as df_file:
                                df_d = pd.read_csv(df_file, header=None)
                                parsed_d = parse_kline_dataframe(df_d)
                                all_dfs.append(parsed_d)
                    else:
                        failed_requests += 1
            else:
                failed_requests += 1
        except Exception as e:
            failed_requests += 1
            print(f"[هشدار] خطا در دریافت آرشیو برای {symbol} در ماه {year}-{m_str}: {e}")

    if not all_dfs:
        print(f"[خطا بحرانی] هیچ داده‌ای برای {symbol} از Binance Vision دریافت نشد!")
        sys.exit(1)

    df = pd.concat(all_dfs, ignore_index=True)
    
    # تبدیل به تایم‌استمپ UTC
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # فیلتر بازه زمانی و حذف کندل جاری ناقص
    df = df[(df['timestamp'] >= pd.Timestamp(start_dt)) & (df['timestamp'] < pd.Timestamp(end_dt))]
