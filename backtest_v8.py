import os
import subprocess
import sys
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd

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
print('📥 دریافت داده‌های 1 ساعته از صرافی LBank (نسخه بهینه‌شده Win-Rate)')
print('============================================================')

data_1h = {}

for symbol, lbank_symbol in SYMBOLS.items():
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
    except Exception:
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


def calculate_indicators(df):
  df = df.copy()
  df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
  df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
  df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

  delta = df['Close'].diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs = gain / (loss + 1e-9)
  df['RSI'] = 100 - (100 / (1 + rs))
  df['RSI_Shift'] = df['RSI'].shift(1)

  high_low = df['High'] - df['Low']
  high_close = np.abs(df['High'] - df['Close'].shift(1))
  low_close = np.abs(df['Low'] - df['Close'].shift(1))
  tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
  df['ATR'] = tr.rolling(window=14).mean()

  # فیلتر ADX برای قدرت روند
  plus_dm = df['High'].diff().clip(lower=0)
  minus_dm = (-df['Low'].diff()).clip(lower=0)
  tr14 = tr.rolling(window=14).mean()
  plus_di = 100 * (plus_dm.rolling(window=14).mean() / (tr14 + 1e-9))
  minus_di = 100 * (minus_dm.rolling(window=14).mean() / (tr14 + 1e-9))
  dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
  df['ADX'] = dx.rolling(window=14).mean().fillna(20)

  vol_ma = df['Volume'].shift(1).rolling(window=20).mean()
  df['RVOL'] = df['Volume'] / (vol_ma + 1e-9)

  return df


all_portfolio_trades = []
SLIPPAGE = 0.0002

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

    if position is None:
      is_uptrend = (
          prev_c1h['Close'] > prev_c1h['EMA_200']
          and prev_c1h['EMA_20'] > prev_c1h['EMA_50']
      )
      is_downtrend = (
          prev_c1h['Close'] < prev_c1h['EMA_200']
          and prev_c1h['EMA_20'] < prev_c1h['EMA_50']
      )

      # اضافه شدن فیلتر ADX و قدرت بدنه کندل برای افزایش وین‌ریت
      is_adx_strong = prev_c1h['ADX'] > 22

      pullback_long = (
          prev_c1h['Low'] <= prev_c1h['EMA_20']
          and prev_c1h['RSI_Shift'] < 48
          and prev_c1h['Close'] > prev_c1h['Open']
          and (prev_c1h['Close'] - prev_c1h['Open'])
          > (prev_c1h['High'] - prev_c1h['Low']) * 0.4
          and prev_c1h['RVOL'] >= 1.2
          and is_adx_strong
      )

      pullback_short = (
          prev_c1h['High'] >= prev_c1h['EMA_20']
          and prev_c1h['RSI_Shift'] > 52
          and prev_c1h['Close'] < prev_c1h['Open']
          and (prev_c1h['Open'] - prev_c1h['Close'])
          > (prev_c1h['High'] - prev_c1h['Low']) * 0.4
          and prev_c1h['RVOL'] >= 1.2
          and is_adx_strong
      )

      if is_uptrend and pullback_long:
        position = 'LONG'
        entry_price = c1h['Open'] * (1 + SLIPPAGE)
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
print('📊 گزارش نهایی پورتفوی (بهینه‌سازی شده)')
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
