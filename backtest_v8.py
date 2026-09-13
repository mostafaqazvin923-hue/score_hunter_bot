import os
import subprocess
import sys
from datetime import datetime, timedelta

# نصب خودکار ccxt در صورت نیاز
try:
    import ccxt
except ImportError:
    print("📦 در حال نصب کتابخانه ccxt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd

# تنظیمات اتصال به صرافی LBank و سبد ۱۰ ارز
exchange = ccxt.lbank({'enableRateLimit': True})
SYMBOLS = {
    'BTC': 'BTC/USDT',
    'ETH': 'ETH/USDT',
    'SOL': 'SOL/USDT',
    'XRP': 'XRP/USDT',
    'ADA': 'ADA/USDT',
    'AVAX': 'AVAX/USDT',
    'LINK': 'LINK/USDT',
    'NEAR': 'NEAR/USDT',
    'SUI': 'SUI/USDT',
    'DOT': 'DOT/USDT',
}

# بازه زمانی: یک سال گذشته
start_date = datetime.now() - timedelta(days=365)
since_timestamp = int(start_date.timestamp() * 1000)

print('============================================================')
print(
    '📥 دریافت داده‌های 1 ساعته از صرافی LBank برای سبد 10 ارز (بدون نگاه به'
    ' آینده)'
)
print('============================================================')

data_1h = {}

for symbol, lbank_symbol in SYMBOLS.items():
  print(f'🔹 در حال دانلود دیتای 1 ساعته {symbol}...')
  all_ohlcv = []
  current_since = since_timestamp
  now_timestamp = exchange.milliseconds()

  while current_since < now_timestamp:
    try:
      ohlcv = exchange.fetch_ohlcv(
          lbank_symbol, timeframe='1h', since=current_since, limit=1000
      )
      if not ohlcv:
        break
      current_since = ohlcv[-1][0] + 1
      all_ohlcv.extend(ohlcv)
      if len(ohlcv) < 1000:
        break
    except Exception as e:
      print(f'  ❌ خطا در دریافت داده {symbol}: {e}')
      break

  if all_ohlcv:
    df1h = pd.DataFrame(
        all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    df1h['Date'] = pd.to_datetime(df1h['Timestamp'], unit='ms')
    df1h = df1h[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df1h.dropna(inplace=True)
    df1h.drop_duplicates(subset=['Date'], inplace=True)
    df1h.sort_values('Date', inplace=True)
    df1h.reset_index(drop=True, inplace=True)
    data_1h[symbol] = df1h
    print(f'  ✔️ دیتای {symbol} آماده شد (تعداد کندل: {len(df1h)})')
  else:
    print(f'  ❌ دیتایی برای {symbol} دریافت نشد.')


def calculate_indicators(df):
  """محاسبه اندیکاتورها با رعایت کامل عدم نشت اطلاعات آینده (Shifted)"""
  df = df.copy()

  # اندیکاتورهای 1 ساعته
  df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
  df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

  # کانال دانچین 20 دوره‌ای (شیفت شده برای عدم استفاده از کندل جاری)
  df['Donchian_High'] = df['High'].shift(1).rolling(window=20).max()
  df['Donchian_Low'] = df['Low'].shift(1).rolling(window=20).min()

  # محاسبه ATR (14)
  high_low = df['High'] - df['Low']
  high_close = np.abs(df['High'] - df['Close'].shift(1))
  low_close = np.abs(df['Low'] - df['Close'].shift(1))
  tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
  df['ATR'] = tr.rolling(window=14).mean()

  # حجم نسبی (RVOL) - شیفت شده
  vol_ma = df['Volume'].shift(1).rolling(window=20).mean()
  df['RVOL'] = df['Volume'] / (vol_ma + 1e-9)

  # فیلتر ADX ساده‌شده (14 دوره‌ای)
  plus_dm = df['High'].diff().clip(lower=0)
  minus_dm = (-df['Low'].diff()).clip(lower=0)
  tr14 = tr.rolling(window=14).mean()
  plus_di = 100 * (plus_dm.rolling(window=14).mean() / (tr14 + 1e-9))
  minus_di = 100 * (minus_dm.rolling(window=14).mean() / (tr14 + 1e-9))
  dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
  df['ADX'] = dx.rolling(window=14).mean().fillna(20)

  return df


print('\n============================================================')
print('🚀 اجرای موتور بک‌تست رویدادمحور استراتژی ATB-10 (بدون Lookahead Bias)')
print('============================================================')

all_portfolio_trades = []
FEE_RATE = 0.0006  # کارمزد Taker (0.06%)
SLIPPAGE = 0.0002  # اسلیپیج (0.02%)

for symbol, df1h in data_1h.items():
  if len(df1h) < 300:
    continue

  df1h = calculate_indicators(df1h)

  # ساخت تایم‌فریم 4 ساعته کاملاً ایمن و بدون نشت
  df4h = (
      df1h.set_index('Date')
      .resample('4H')
      .agg({
          'Open': 'first',
          'High': 'max',
          'Low': 'min',
          'Close': 'last',
          'Volume': 'sum',
      })
      .dropna()
      .reset_index()
  )

  df4h['EMA_50_4H'] = df4h['Close'].ewm(span=50, adjust=False).mean()
  df4h['EMA_200_4H'] = df4h['Close'].ewm(span=200, adjust=False).mean()

  # شیفت دادن 4H به اندازه یک کندل به عقب تا در هر ساعت، فقط کندل 4H *قبلاً بسته شده* قابل دسترسی باشد
  df4h['EMA_50_4H'] = df4h['EMA_50_4H'].shift(1)
  df4h['EMA_200_4H'] = df4h['EMA_200_4H'].shift(1)
  df4h['Close_4H'] = df4h['Close'].shift(1)

  df1h['Date_4H'] = df1h['Date'].dt.floor('4h')
  df4h_indexed = df4h.set_index('Date')

  # شبیه‌سازی گام‌به‌گام رویدادمحور
  position = None  # None, 'LONG', 'SHORT'
  entry_price = 0.0
  stop_loss = 0.0
  take_profit = 0.0

  for i in range(200, len(df1h)):
    c1h = df1h.iloc[i]
    prev_c1h = df1h.iloc[i - 1]  # اطلاعات کندل کاملاً بسته شده 1 ساعته
    t4h_time = c1h['Date_4H']

    if t4h_time not in df4h_indexed.index:
      continue
    r4h = df4h_indexed.loc[t4h_time]

    # --- 1. مدیریت پوزیشن باز (بررسی TP و SL در کندل جاری بدون نگاه به آینده) ---
    if position is not None:
      if position == 'LONG':
        if c1h['Low'] <= stop_loss:
          # خروج با ضرر
          all_portfolio_trades.append({
              'Symbol': symbol,
              'Side': 'LONG',
              'Outcome': 'LOSS',
              'Return': -1.0,
          })
          position = None
        elif c1h['High'] >= take_profit:
          # خروج با سود (R:R = 1:2)
          all_portfolio_trades.append({
              'Symbol': symbol,
              'Side': 'LONG',
              'Outcome': 'WIN',
              'Return': 2.0,
          })
          position = None
      elif position == 'SHORT':
        if c1h['High'] >= stop_loss:
          all_portfolio_trades.append({
              'Symbol': symbol,
              'Side': 'SHORT',
              'Outcome': 'LOSS',
              'Return': -1.0,
          })
          position = None
        elif c1h['Low'] <= take_profit:
          all_portfolio_trades.append({
              'Symbol': symbol,
              'Side': 'SHORT',
              'Outcome': 'WIN',
              'Return': 2.0,
          })
          position = None

    # --- 2. بررسی سیگنال جدید (فقط در صورت بسته بودن پوزیشن روی این ارز) ---
    if position is None:
      # شرایط روند 4H
      is_4h_bullish = (r4h['Close_4H'] > r4h['EMA_200_4H']) and (
          r4h['EMA_50_4H'] > r4h['EMA_200_4H']
      )
      is_4h_bearish = (r4h['Close_4H'] < r4h['EMA_200_4H']) and (
          r4h['EMA_50_4H'] < r4h['EMA_200_4H']
      )

      # شرایط شکست 1H (استفاده از مقادیر شیفت‌شده قبلی)
      is_breakout_long = prev_c1h['Close'] > prev_c1h['Donchian_High']
      is_breakout_short = prev_c1h['Close'] < prev_c1h['Donchian_Low']

      # فیلتر حجم و روند
      is_volume_ok = prev_c1h['RVOL'] >= 1.2
      is_adx_ok = prev_c1h['ADX'] > 22

      if is_4h_bullish and is_breakout_long and is_volume_ok and is_adx_ok:
        position = 'LONG'
        entry_price = c1h['Open'] * (
            1 + SLIPPAGE
        )  # ورود در قیمت بازگشایی کندل جدید
        stop_loss = prev_c1h['Low'] - (0.5 * prev_c1h['ATR'])
        risk = entry_price - stop_loss
        if risk > 0:
          take_profit = entry_price + (2.0 * risk)
        else:
          position = None  # لغو معامله در صورت ناهنجاری ریسک

      elif is_4h_bearish and is_breakout_short and is_volume_ok and is_adx_ok:
        position = 'SHORT'
        entry_price = c1h['Open'] * (1 - SLIPPAGE)
        stop_loss = prev_c1h['High'] + (0.5 * prev_c1h['ATR'])
        risk = stop_loss - entry_price
        if risk > 0:
          take_profit = entry_price - (2.0 * risk)
        else:
          position = None

print('\n============================================================')
print('📊 گزارش نهایی ارزیابی عملکرد پورتفوی (ATB-10)')
print('============================================================')

if all_portfolio_trades:
  pf_df = pd.DataFrame(all_portfolio_trades)
  total_trades = len(pf_df)
  total_wins = len(pf_df[pf_df['Outcome'] == 'WIN'])
  total_losses = len(pf_df[pf_df['Outcome'] == 'LOSS'])
  win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
  net_r = pf_df['Return'].sum()

  print(f'🔸 تعداد کل معاملات پورتفوی: {total_trades}')
  print(f'🔸 معاملات برنده (WIN): {total_wins}')
  print(f'🔸 معاملات بازنده (LOSS): {total_losses}')
  print(f'🎯 **وین‌ریت تجمیعی پورتفوی:** {win_rate:.2f}%')
  print(f'💰 **مجموع بازدهی خالص (بر حسب R):** {net_r:.2f}R')

  print('\nتفکیک عملکرد به تفکیک هر نماد:')
  print(pf_df.groupby('Symbol')['Outcome'].value_counts().unstack(fill_value=0))
else:
  print('⚠️ هیچ معامله‌ای با شرایط تعیین‌شده ثبت نشد.')

print('\n✨ بک‌تست به پایان رسید.')
