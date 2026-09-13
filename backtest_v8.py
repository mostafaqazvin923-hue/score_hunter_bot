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
print('📥 دریافت داده‌ها برای استراتژی ایچیموکو (همراه با ریسک‌فری و فیوز توقف)')
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


def calculate_ichimoku(df):
  df = df.copy()
  period9_high = df['High'].rolling(window=9).max()
  period9_low = df['Low'].rolling(window=9).min()
  df['Tenkan'] = (period9_high + period9_low) / 2

  period26_high = df['High'].rolling(window=26).max()
  period26_low = df['Low'].rolling(window=26).min()
  df['Kijun'] = (period26_high + period26_low) / 2

  df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)

  period52_high = df['High'].rolling(window=52).max()
  period52_low = df['Low'].rolling(window=52).min()
  df['Senkou_B'] = ((period52_high + period52_low) / 2).shift(26)

  tr1 = df['High'] - df['Low']
  tr2 = np.abs(df['High'] - df['Close'].shift(1))
  tr3 = np.abs(df['Low'] - df['Close'].shift(1))
  df['ATR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

  return df


processed_data = {}
for symbol, df1h in data_1h.items():
  if len(df1h) < 100:
    continue
  df1h = calculate_ichimoku(df1h)

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
  df4h = calculate_ichimoku(df4h)
  df4h['Cloud_Top'] = df4h[['Senkou_A', 'Senkou_B']].max(axis=1)
  df4h['Cloud_Bottom'] = df4h[['Senkou_A', 'Senkou_B']].min(axis=1)
  df4h['Trend_Long'] = df4h['Close'] > df4h['Cloud_Top']
  df4h['Trend_Short'] = df4h['Close'] < df4h['Cloud_Bottom']

  for col in ['Trend_Long', 'Trend_Short', 'Cloud_Top', 'Cloud_Bottom']:
    df4h[col] = df4h[col].shift(1)

  df1h['Date_4H'] = df1h['Date'].dt.floor('4h')
  processed_data[symbol] = {'1h': df1h, '4h': df4h.set_index('Date')}

print('⚙️ شروع اجرای بک‌تست با قوانین جدید ریسک‌فری و فیوز توقف...')

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
  is_breakeven = False

  for i in range(60, len(df1h)):
    c1h = df1h.iloc[i]
    prev = df1h.iloc[i - 1]
    t4h_time = c1h['Date_4H']

    if t4h_time not in df4h_idx.index:
      continue
    r4h = df4h_idx.loc[t4h_time]

    # مدیریت پوزیشن باز همراه با چک کردن ریسک‌فری (۵۰٪ مسیر تا TP)
    if position is not None:
      candles_held = i - entry_index
      if position == 'LONG':
        # بررسی ریسک‌فری: اگر High به نصف فاصله TP رسید
        if not is_breakeven and c1h['High'] >= entry_price + (
            0.5 * (take_profit - entry_price)
        ):
          stop_loss = entry_price  # قفل شدن استاپ روی نقطه ورود
          is_breakeven = True

        if (
            candles_held >= 24
            or c1h['Low'] <= stop_loss
            or c1h['High'] >= take_profit
        ):
          if c1h['High'] >= take_profit:
            outcome = 'WIN'
            r_real = 2.0 - (FEE_RATE * 2)
          elif c1h['Low'] <= stop_loss:
            outcome = 'WIN' if is_breakeven and stop_loss == entry_price else 'LOSS'
            r_real = (
                0.0 - (FEE_RATE * 2)
                if (is_breakeven and stop_loss == entry_price)
                else -1.0 - (FEE_RATE * 2)
            )
          else:  # Time Stop
            ret_val = (c1h['Close'] - entry_price) / (
                entry_price - (stop_loss if stop_loss != entry_price else entry_price - 0.01)
            )
            outcome = 'WIN' if c1h['Close'] > entry_price else 'LOSS'
            r_real = (
                max(0.0, ret_val) - (FEE_RATE * 2)
                if outcome == 'WIN'
                else -1.0 - (FEE_RATE * 2)
            )

          all_trades.append({
              'Timestamp': c1h['Date'],
              'Symbol': symbol,
              'Side': 'LONG',
              'Outcome': outcome,
              'Return': r_real,
          })
          position = None

      elif position == 'SHORT':
        if not is_breakeven and c1h['Low'] <= entry_price - (
            0.5 * (entry_price - take_profit)
        ):
          stop_loss = entry_price
          is_breakeven = True

        if (
            candles_held >= 24
            or c1h['High'] >= stop_loss
            or c1h['Low'] <= take_profit
        ):
          if c1h['Low'] <= take_profit:
            outcome = 'WIN'
            r_real = 2.0 - (FEE_RATE * 2)
          elif c1h['High'] >= stop_loss:
            outcome = 'WIN' if is_breakeven and stop_loss == entry_price else 'LOSS'
            r_real = (
                0.0 - (FEE_RATE * 2)
                if (is_breakeven and stop_loss == entry_price)
                else -1.0 - (FEE_RATE * 2)
            )
          else:
            outcome = 'WIN' if c1h['Close'] < entry_price else 'LOSS'
            r_real = 0.0 - (FEE_RATE * 2) if outcome == 'WIN' else -1.0 - (FEE_RATE * 2)

          all_trades.append({
              'Timestamp': c1h['Date'],
              'Symbol': symbol,
              'Side': 'SHORT',
              'Outcome': outcome,
              'Return': r_real,
          })
          position = None

    # سیگنال‌های ورود جدید (بدون تغییر در منطق اصلی)
    if position is None:
      cloud_top_1h = max(prev['Senkou_A'], prev['Senkou_B'])
      cloud_bot_1h = min(prev['Senkou_A'], prev['Senkou_B'])

      if r4h.get('Trend_Long', False) and prev['Close'] > cloud_top_1h:
        tk_cross_long = (prev['Tenkan'] > prev['Kijun']) and (
            df1h.iloc[i - 2]['Tenkan'] <= df1h.iloc[i - 2]['Kijun']
        )
        if tk_cross_long or (
            prev['Low'] <= prev['Kijun'] and prev['Close'] > prev['Kijun']
        ):
          entry_price = c1h['Open'] * (1 + SLIPPAGE)
          stop_loss = min(prev['Low'], cloud_bot_1h) - 0.2 * prev['ATR']
          sl_dist_pct = (entry_price - stop_loss) / entry_price

          if 0.003 <= sl_dist_pct <= 0.04:
            risk = entry_price - stop_loss
            take_profit = entry_price + (2.0 * risk)
            position = 'LONG'
            entry_index = i
            is_breakeven = False

      elif r4h.get('Trend_Short', False) and prev['Close'] < cloud_bot_1h:
        tk_cross_short = (prev['Tenkan'] < prev['Kijun']) and (
            df1h.iloc[i - 2]['Tenkan'] >= df1h.iloc[i - 2]['Kijun']
        )
        if tk_cross_short or (
            prev['High'] >= prev['Kijun'] and prev['Close'] < prev['Kijun']
        ):
          entry_price = c1h['Open'] * (1 - SLIPPAGE)
          stop_loss = max(prev['High'], cloud_top_1h) + 0.2 * prev['ATR']
          sl_dist_pct = (stop_loss - entry_price) / entry_price

          if 0.003 <= sl_dist_pct <= 0.04:
            risk = stop_loss - entry_price
            take_profit = entry_price - (2.0 * risk)
            position = 'SHORT'
            entry_index = i
            is_breakeven = False

print('\n============================================================')
print('📊 گزارش نهایی (با اعمال ریسک‌فری و فیلتر ضرر متوالی)')
print('============================================================')

if all_trades:
  trades_df = pd.DataFrame(all_trades)
  trades_df.sort_values('Timestamp', inplace=True)

  # اعمال فیلتر توقف ۲۴ ساعته پس از ۳ باخت متوالی در کل پورتفوی
  filtered_trades = []
  consecutive_losses = 0
  pause_until = None

  for idx, row in trades_df.iterrows():
    current_time = row['Timestamp']

    # اگر در حالت توقف هستیم، بررسی کنیم که آیا زمان توقف به پایان رسیده یا نه
    if pause_until is not None:
      if current_time < pause_until:
        continue  # این معامله به دلیل قانون توقف رد می‌شود
      else:
        pause_until = None
        consecutive_losses = 0

    filtered_trades.append(row)

    if row['Outcome'] == 'LOSS':
      consecutive_losses += 1
      if consecutive_losses >= 3:
        pause_until = current_time + timedelta(hours=24)
        consecutive_losses = 0
    else:
      consecutive_losses = 0

  if filtered_trades:
    f_df = pd.DataFrame(filtered_trades)
    tot_trades = len(f_df)
    tot_wins = len(f_df[f_df['Outcome'] == 'WIN'])
    tot_losses = len(f_df[f_df['Outcome'] == 'LOSS'])
    win_rate = (tot_wins / tot_trades) * 100 if tot_trades > 0 else 0
    net_r = f_df['Return'].sum()

    # محاسبه استریک‌ها روی لیست فیلتر شده
    outcomes = f_df['Outcome'].tolist()
    max_wins = 0
    max_losses = 0
    curr_wins = 0
    curr_losses = 0

    for out in outcomes:
      if out == 'WIN':
        curr_wins += 1
        curr_losses = 0
        if curr_wins > max_wins:
          max_wins = curr_wins
      else:
        curr_losses += 1
        curr_wins = 0
        if curr_losses > max_losses:
          max_losses = curr_losses

    print(f'🔸 تعداد کل معاملات پس از اعمال فیلتر توقف: {tot_trades}')
    print(f'🔸 معاملات برنده (WIN): {tot_wins}')
    print(f'🔸 معاملات بازنده (LOSS): {tot_losses}')
    print(f'🔥 **حداکثر سودهای متوالی:** {max_wins}')
    print(f'❄️ **حداکثر ضررهای متوالی (کنترل شده با فیوز):** {max_losses}')
    print(f'🎯 **وین‌ریت تجمیعی پورتفوی:** {win_rate:.2f}%')
    print(f'💰 **مجموع بازدهی خالص:** {net_r:.2f}R')
  else:
    print('⚠️ تمامی معاملات توسط فیلتر توقف مسدود شدند.')
else:
  print('⚠️ معامله‌ای ثبت نشد.')

print('\n✨ بک‌تست به پایان رسید.')
