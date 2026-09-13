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
print('📥 دریافت داده‌ها برای استراتژی ایچیموکو (محدودیت پوزیشن همزمان)')
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

print('⚙️ شروع اجرای بک‌تست با اعمال محدودیت پوزیشن همزمان...')

# شبیه‌سازی گام‌به‌گام زمان‌محور در کل پورتفوی برای مدیریت سقف پوزیشن‌های همزمان
all_timestamps = set()
for dat in processed_data.values():
  all_timestamps.update(dat['1h']['Date'].tolist())
sorted_timestamps = sorted(list(all_timestamps))

active_positions = {}  # symbol: position_dict
all_trades = []
SLIPPAGE = 0.0003
FEE_RATE = 0.0007
MAX_CONCURRENT_POSITIONS = 3

# تبدیل داده‌ها به دیکشنری برای دسترسی سریع‌تر در لوپ زمانی
dfs_1h = {sym: dat['1h'].set_index('Date') for sym, dat in processed_data.items()}
dfs_4h = {sym: dat['4h'] for sym, sym in processed_data.keys()}

# ردیابی وضعیت مدارشکن (Circuit Breaker) کل سبد
consecutive_losses = 0
pause_until = None

for ts in sorted_timestamps:
  # ۱. مدیریت و بستن پوزیشن‌های باز در این ساعت
  symbols_to_close = []
  for symbol, pos in active_positions.items():
    if ts not in dfs_1h[symbol].index:
      continue
    c1h = dfs_1h[symbol].loc[ts]
    entry_index = pos['entry_index']
    # پیدا کردن ایندکس عددی فعلی
    df1h_local = processed_data[symbol]['1h']
    match_rows = df1h_local[df1h_local['Date'] == ts]
    if match_rows.empty:
      continue
    curr_i = match_rows.index[0]
    candles_held = curr_i - entry_index

    if pos['side'] == 'LONG':
      if (
          candles_held >= 24
          or c1h['Low'] <= pos['stop_loss']
          or c1h['High'] >= pos['take_profit']
      ):
        outcome = 'LOSS' if c1h['Low'] <= pos['stop_loss'] else 'WIN'
        r_real = (
            -1.0 - (FEE_RATE * 2) if outcome == 'LOSS' else 2.0 - (FEE_RATE * 2)
        )
        all_trades.append({
            'Timestamp': ts,
            'Symbol': symbol,
            'Side': 'LONG',
            'Outcome': outcome,
            'Return': r_real,
        })
        symbols_to_close.append(symbol)
    elif pos['side'] == 'SHORT':
      if (
          candles_held >= 24
          or c1h['High'] >= pos['stop_loss']
          or c1h['Low'] <= pos['take_profit']
      ):
        outcome = 'LOSS' if c1h['High'] >= pos['stop_loss'] else 'WIN'
        r_real = (
            -1.0 - (FEE_RATE * 2) if outcome == 'LOSS' else 2.0 - (FEE_RATE * 2)
        )
        all_trades.append({
            'Timestamp': ts,
            'Symbol': symbol,
            'Side': 'SHORT',
            'Outcome': outcome,
            'Return': r_real,
        })
        symbols_to_close.append(symbol)

  for sym in symbols_to_close:
    del active_positions[sym]

  # ۲. بررسی وضعیت توقف اضطراری (Circuit Breaker)
  if pause_until is not None and ts < pause_until:
    continue
  elif pause_until is not None and ts >= pause_until:
    pause_until = None
    consecutive_losses = 0

  # ۳. جستجوی سیگنال‌های جدید برای نمادهایی که پوزیشن ندارند
  for symbol, dat in processed_data.items():
    if symbol in active_positions:
      continue
    if len(active_positions) >= MAX_CONCURRENT_POSITIONS:
      break  # سقف پوزیشن‌های همزمان پر شده است

    df1h = dat['1h']
    if ts not in df1h['Date'].values:
      continue

    match_rows = df1h[df1h['Date'] == ts]
    if match_rows.empty:
      continue
    i = match_rows.index[0]
    if i < 2:
      continue

    c1h = df1h.iloc[i]
    prev = df1h.iloc[i - 1]
    t4h_time = c1h['Date_4H']
    df4h_idx = dat['4h']

    if t4h_time not in df4h_idx.index:
      continue
    r4h = df4h_idx.loc[t4h_time]

    cloud_top_1h = max(prev['Senkou_A'], prev['Senkou_B'])
    cloud_bot_1h = min(prev['Senkou_A'], prev['Senkou_B'])

    # سیگنال Long
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
          active_positions[symbol] = {
              'side': 'LONG',
              'entry_price': entry_price,
              'stop_loss': stop_loss,
              'take_profit': take_profit,
              'entry_index': i,
          }
          continue

    # سیگنال Short
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
          active_positions[symbol] = {
              'side': 'SHORT',
              'entry_price': entry_price,
              'stop_loss': stop_loss,
              'take_profit': take_profit,
              'entry_index': i,
          }
          continue

print('\n============================================================')
print(
    '📊 گزارش جامع پورتفوی (با سقف ۳ پوزیشن همزمان و مدارشکن ضرر‌های متوالی)'
)
print('============================================================')

if all_trades:
  trades_df = pd.DataFrame(all_trades)
  trades_df.sort_values('Timestamp', inplace=True)

  # اعمال فیلتر مدارشکن (Circuit Breaker) روی کل تاریخچه معاملات
  filtered_trades = []
  consec_losses = 0
  p_until = None

  for idx, row in trades_df.iterrows():
    ctime = row['Timestamp']
    if p_until is not None:
      if ctime < p_until:
        continue
      else:
        p_until = None
        consec_losses = 0

    filtered_trades.append(row)

    if row['Outcome'] == 'LOSS':
      consec_losses += 1
      if consec_losses >= 3:
        p_until = ctime + timedelta(hours=24)
        consec_losses = 0
    else:
      consec_losses = 0

  if filtered_trades:
    f_df = pd.DataFrame(filtered_trades)
    tot_trades = len(f_df)
    tot_wins = len(f_df[f_df['Outcome'] == 'WIN'])
    tot_losses = len(f_df[f_df['Outcome'] == 'LOSS'])
    win_rate = (tot_wins / tot_trades) * 100 if tot_trades > 0 else 0
    net_r = f_df['Return'].sum()

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

    print(f'🔸 تعداد کل معاملات سبد: {tot_trades}')
    print(f'🔸 معاملات برنده (WIN): {tot_wins}')
    print(f'🔸 معاملات بازنده (LOSS): {tot_losses}')
    print(f'🔥 **حداکثر سودهای متوالی:** {max_wins}')
    print(f'❄️ **حداکثر ضررهای متوالی (کنترل شده):** {max_losses}')
    print(f'🎯 **وین‌ریت تجمیعی پورتفوی:** {win_rate:.2f}%')
    print(f'💰 **مجموع بازدهی خالص کل:** {net_r:.2f}R\n')

    print(
        '------------------------------------------------------------'
    )
    print('📈 **گزارش تفکیک‌شده به تفکیک هر ارز:**')
    print(
        '------------------------------------------------------------'
    )

    symbol_summary = []
    for sym in SYMBOLS.keys():
      sym_trades = f_df[f_df['Symbol'] == sym]
      s_tot = len(sym_trades)
      if s_tot > 0:
        s_wins = len(sym_trades[sym_trades['Outcome'] == 'WIN'])
        s_loss = len(sym_trades[sym_trades['Outcome'] == 'LOSS'])
        s_wr = (s_wins / s_tot) * 100
        s_net_r = sym_trades['Return'].sum()
      else:
        s_wins, s_loss, s_wr, s_net_r = 0, 0, 0.0, 0.0

      symbol_summary.append({
          'Symbol': sym,
          'Trades': s_tot,
          'Wins': s_wins,
          'Losses': s_loss,
          'WinRate(%)': round(s_wr, 2),
          'Net_R': round(s_net_r, 2),
      })

    summary_df = pd.DataFrame(symbol_summary)
    print(summary_df.to_string(index=False))
  else:
    print('⚠️ تمامی معاملات توسط فیلتر مدارشکن مسدود شدند.')
else:
  print('⚠️ معامله‌ای ثبت نشد.')

print('\n✨ بک‌تست به پایان رسید.')
