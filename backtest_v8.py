import os
import subprocess
import sys
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    print("📦 در حال نصب کتابخانه ccxt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd

# صرافی LBank و سبد ۱۰ ارز
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

start_date = datetime.now() - timedelta(days=365)
since_timestamp = int(start_date.timestamp() * 1000)

print('============================================================')
print('📥 دریافت داده‌های 1 ساعته از صرافی LBank (استراتژی Pullback)')
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

  # میانگین‌های متحرک برای تشخیص روند
  df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
  df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
  df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

  # RSI (14) - شیفت شده
  delta = df['Close'].diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs = gain / (loss + 1e-9)
  df['RSI'] = 100 - (100 / (1 + rs))
  df['RSI_Shift'] = df['RSI'].shift(1)

  # ATR (14)
  high_low = df['High'] - df['Low']
  high_close = np.abs(df['High'] - df['Close'].shift(1))
  low_close = np.abs(df['Low'] - df['Close'].shift(1))
  tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
  df['ATR'] = tr.rolling(window=14).mean()

  # حجم نسبی (RVOL) - شیفت شده
  vol_ma = df['Volume'].shift(1).rolling(window=20).mean()
  df['RVOL'] = df['Volume'] / (vol_ma + 1e-9)

  return df


print('\n============================================================')
print('🚀 اجرای موتور بک‌تست پورتفوی (Pullback Continuation)')
print('============================================================')

all_portfolio_trades = []
SLIPPAGE = 0.0002  # اسلیپیج (0.02%)

for symbol, df1h in data_1h.items():
  if len(df1h) < 300:
    continue

  df1h = calculate_indicators(df1h)
  position = None
  entry_price = 0.0
  stop_loss = 0.0
  take_profit = 0.0

  for i in range(200, len(df1h)):
    c1h = df1h.iloc[i]
    prev_c1h = df1h.iloc[i - 1]

    # --- 1. مدیریت پوزیشن باز ---
    if position is not None:
      if position == 'LONG':
        if c1h['Low'] <= stop_loss:
          all_portfolio_trades.append({
              'Symbol': symbol,
              'Side': 'LONG',
              'Outcome': 'LOSS',
              'Return': -1.0,
          })
          position = None
        elif c1h['High'] >= take_profit:
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

    # --- 2. بررسی سیگنال با منطق Pullback در روند ---
    if position is None:
      # روند صعودی قوی: قیمت بالای EMA 200 و EMA 20 بالای EMA 50
      is_uptrend = (
          prev_c1h['Close'] > prev_c1h['EMA_200']
          and prev_c1h['EMA_20'] > prev_c1h['EMA_50']
      )
      # روند نزولی قوی: قیمت پایین EMA 200 و EMA 20 پایین EMA 50
      is_downtrend = (
          prev_c1h['Close'] < prev_c1h['EMA_200']
          and prev_c1h['EMA_20'] < prev_c1h['EMA_50']
      )

      # شرایط اصلاح (Pullback): قیمت به محدوده EMA 20 نزدیک شده یا کمی به زیر آن رفته، و RSI اصلاح انجام داده است
      pullback_long = (
          prev_c1h['Low'] <= prev_c1h['EMA_20']
          and prev_c1h['RSI_Shift'] < 50
          and prev_c1h['Close'] > prev_c1h['Open']
          and prev_c1h['RVOL'] >= 1.1
      )
      pullback_short = (
          prev_c1h['High'] >= prev_c1h['EMA_20']
          and prev_c1h['RSI_Shift'] > 50
          and prev_c1h['Close'] < prev_c1h['Open']
          and prev_c1h['RVOL'] >= 1.1
      )

      if is_uptrend and pullback_long:
        position = 'LONG'
        entry_price = c1h['Open'] * (1 + SLIPPAGE)
        # حد ضرر پایین‌تر از کف کندل اصلاحی یا ATR
        stop_loss = prev_c1h['Low'] - (0.5 * prev_c1h['ATR'])
        risk = entry_price - stop_loss
        if risk > 0 and (risk / entry_price) <= 0.04:
          take_profit = entry_price + (2.0 * risk)
        else:
          position = None

      elif is_downtrend and pullback_short:
        position = 'SHORT'
        entry_price = c1h['Open'] * (1 - SLIPPAGE)
        stop_loss = prev_c1h['High'] + (0.5 * prev_c1h['ATR'])
        risk = stop_loss - entry_price
        if risk > 0 and (risk / entry_price) <= 0.04:
          take_profit = entry_price - (2.0 * risk)
        else:
          position = None

print('\n============================================================')
print('📊 گزارش نهایی عملکرد پورتفوی (Pullback Continuation)')
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
