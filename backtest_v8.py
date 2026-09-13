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
print('📥 دریافت داده‌ها برای نسخه بهینه‌شده HUNTER-X LBR')
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


def calculate_indicators_1h(df):
  df = df.copy()
  df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
  df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

  df['L20'] = df['Low'].shift(1).rolling(window=20).min()
  df['H20'] = df['High'].shift(1).rolling(window=20).max()
  df['H10'] = df['High'].shift(1).rolling(window=10).max()
  df['L10'] = df['Low'].shift(1).rolling(window=10).min()

  tr1 = df['High'] - df['Low']
  tr2 = np.abs(df['High'] - df['Close'].shift(1))
  tr3 = np.abs(df['Low'] - df['Close'].shift(1))
  df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
  df['ATR'] = df['TR'].rolling(window=14).mean()

  vol_ma = df['Volume'].shift(1).rolling(window=20).mean()
  df['RVOL'] = df['Volume'] / (vol_ma + 1e-9)

  delta = df['Close'].diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs = gain / (loss + 1e-9)
  df['RSI'] = 100 - (100 / (1 + rs))

  return df


processed_data = {}
for symbol, df1h in data_1h.items():
  if len(df1h) < 300:
    continue
  df1h = calculate_indicators_1h(df1h)

  df4h = (
      df1h.set_index('Date')
      .resample('4h')
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

  df4h['EMA_50'] = df4h['Close'].ewm(span=50, adjust=False).mean().shift(1)
  df4h['EMA_200'] = df4h['Close'].ewm(span=200, adjust=False).mean().shift(1)
  df4h['Close_4H'] = df4h['Close'].shift(1)

  df1h['Date_4H'] = df1h['Date'].dt.floor('4h')
  processed_data[symbol] = {'1h': df1h, '4h': df4h.set_index('Date')}

print('⚙️ شروع اجرای بک‌تست اصلاح‌شده...')

all_trades = []
SLIPPAGE = 0.0003
FEE_RATE = 0.0007

for symbol, dat in processed_data.items():
  df1h = dat['1h']
  df4h_idx = dat['4h']

  position = None
  entry_price = 0.0
  stop_loss = 0.0
  take_profit = 0.0
  entry_index = 0
  active_bos_level = 0.0
  candles_since_bos = 0
  waiting_for_retest = False
  retest_side = None

  for i in range(200, len(df1h)):
    c1h = df1h.iloc[i]
    prev = df1h.iloc[i - 1]
    t4h_time = c1h['Date_4H']

    if t4h_time not in df4h_idx.index:
      continue
    r4h = df4h_idx.loc[t4h_time]

    if position is not None:
      candles_held = i - entry_index
      if position == 'LONG':
        if candles_held >= 24 or c1h['Low'] <= stop_loss or c1h['High'] >= take_profit:
          outcome = 'LOSS' if c1h['Low'] <= stop_loss else 'WIN'
          r_real = (
              -1.0 - (FEE_RATE * 2)
              if outcome == 'LOSS'
              else 2.0 - (FEE_RATE * 2)
          )
          all_trades.append({
              'Symbol': symbol,
              'Side': 'LONG',
              'Outcome': outcome,
              'Return': r_real,
          })
          position = None
      elif position == 'SHORT':
        if candles_held >= 24 or c1h['High'] >= stop_loss or c1h['Low'] <= take_profit:
          outcome = 'LOSS' if c1h['High'] >= stop_loss else 'WIN'
          r_real = (
              -1.0 - (FEE_RATE * 2)
              if outcome == 'LOSS'
              else 2.0 - (FEE_RATE * 2)
          )
          all_trades.append({
              'Symbol': symbol,
              'Side': 'SHORT',
              'Outcome': outcome,
              'Return': r_real,
          })
          position = None

    if position is None:
      regime_long = (r4h['Close_4H'] > r4h['EMA_200']) and (
          r4h['EMA_50'] > r4h['EMA_200']
      )
      regime_short = (r4h['Close_4H'] < r4h['EMA_200']) and (
          r4h['EMA_50'] < r4h['EMA_200']
      )

      if waiting_for_retest:
        candles_since_bos += 1
        if candles_since_bos > 6:
          waiting_for_retest = False
        else:
          if retest_side == 'LONG':
            retest_zone_high = active_bos_level + (0.35 * prev['ATR'])
            if prev['Low'] <= retest_zone_high and prev['Close'] > active_bos_level:
              if (
                  (prev['Close'] > prev['Open'])
                  and (prev['RSI'] > 45)
                  and (prev['RSI'] < 75)
              ):
                entry_price = c1h['Open'] * (1 + SLIPPAGE)
                struct_low = min(prev['Low'], active_bos_level)
                stop_loss = struct_low - (0.20 * prev['ATR'])
                sl_dist_pct = (entry_price - stop_loss) / entry_price

                if 0.003 <= sl_dist_pct <= 0.04:
                  risk = entry_price - stop_loss
                  take_profit = entry_price + (2.0 * risk)
                  position = 'LONG'
                  entry_index = i
                  waiting_for_retest = False

          elif retest_side == 'SHORT':
            retest_zone_low = active_bos_level - (0.35 * prev['ATR'])
            if prev['High'] >= retest_zone_low and prev['Close'] < active_bos_level:
              if (
                  (prev['Close'] < prev['Open'])
                  and (prev['RSI'] < 55)
                  and (prev['RSI'] > 25)
              ):
                entry_price = c1h['Open'] * (1 - SLIPPAGE)
                struct_high = max(prev['High'], active_bos_level)
                stop_loss = struct_high + (0.20 * prev['ATR'])
                sl_dist_pct = (stop_loss - entry_price) / entry_price

                if 0.003 <= sl_dist_pct <= 0.04:
                  risk = stop_loss - entry_price
                  take_profit = entry_price - (2.0 * risk)
                  position = 'SHORT'
                  entry_index = i
                  waiting_for_retest = False

      if not waiting_for_retest:
        if regime_long:
          sweep_long = (prev['Low'] < prev['L20']) and (
              prev['Close'] > prev['L20']
          )
          bos_long = (
              sweep_long
              and (prev['Close'] > prev['H10'])
              and (prev['Close'] > prev['Open'])
              and (prev['RVOL'] >= 1.0)
          )
          if bos_long:
            active_bos_level = prev['H10']
            candles_since_bos = 0
            waiting_for_retest = True
            retest_side = 'LONG'

        elif regime_short:
          sweep_short = (prev['High'] > prev['H20']) and (
              prev['Close'] < prev['H20']
          )
          bos_short = (
              sweep_short
              and (prev['Close'] < prev['L10'])
              and (prev['Close'] < prev['Open'])
              and (prev['RVOL'] >= 1.0)
          )
          if bos_short:
            active_bos_level = prev['L10']
            candles_since_bos = 0
            waiting_for_retest = True
            retest_side = 'SHORT'

print('\n============================================================')
print('📊 گزارش نهایی عملکرد پورتفوی (نسخه بهینه‌شده)')
print('============================================================')

if all_trades:
  trades_df = pd.DataFrame(all_trades)
  tot_trades = len(trades_df)
  tot_wins = len(trades_df[trades_df['Outcome'] == 'WIN'])
  tot_losses = len(trades_df[trades_df['Outcome'] == 'LOSS'])
  win_rate = (tot_wins / tot_trades) * 100 if tot_trades > 0 else 0
  net_r = trades_df['Return'].sum()

  print(f'🔸 تعداد کل معاملات پورتفوی: {tot_trades}')
  print(f'🔸 معاملات برنده (WIN): {tot_wins}')
  print(f'🔸 معاملات بازنده (LOSS): {tot_losses}')
  print(f'🎯 **وین‌ریت تجمیعی پورتفوی:** {win_rate:.2f}%')
  print(f'💰 **مجموع بازدهی خالص:** {net_r:.2f}R')

  print('\nتفکیک عملکرد به تفکیک هر نماد:')
  print(
      trades_df.groupby('Symbol')['Outcome']
      .value_counts()
      .unstack(fill_value=0)
  )
else:
  print('⚠️ معامله‌ای ثبت نشد.')

print('\n✨ بک‌تست به پایان رسید.')
