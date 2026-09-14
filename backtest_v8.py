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

# سبد ۱۱ ارزی بهینه‌شده
SYMBOLS = {
    'BTC': 'BTC/USDT',
    'ETH': 'ETH/USDT',
    'SOL': 'SOL/USDT',
    'XRP': 'XRP/USDT',
    'ADA': 'ADA/USDT',
    'AVAX': 'AVAX/USDT',
    'DOGE': 'DOGE/USDT',
    'DOT': 'DOT/USDT',
    'LTC': 'LTC/USDT',
    'RENDER': 'RENDER/USDT',
    'ATOM': 'ATOM/USDT',
}

start_date = datetime.now() - timedelta(days=365)
since_timestamp = int(start_date.timestamp() * 1000)

print('============================================================')
print(
    '📥 دریافت داده‌ها (HUNTER-ICHIMOKU V12 - Trend & ADX Filter Validation)'
)
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


def calculate_ichimoku_and_adx(df):
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

  # محاسبه ADX (شاخص جهت‌دار میانگین) برای تشخیص قدرت روند
  plus_dm = df['High'].diff()
  minus_dm = df['Low'].diff()
  plus_dm = np.where(
      (plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0
  )
  minus_dm = np.where(
      (minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0
  )

  tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
  atr14 = tr.rolling(14).mean()

  plus_di = (
      100
      * pd.Series(plus_dm, index=df.index)
      .rolling(14)
      .mean()
      / (atr14 + 1e-9)
  )
  minus_di = (
      100
      * pd.Series(minus_dm, index=df.index)
      .rolling(14)
      .mean()
      / (atr14 + 1e-9)
  )
  dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
  df['ADX'] = dx.rolling(14).mean()

  return df


processed_data = {}
for symbol, df1h in data_1h.items():
  if len(df1h) < 100:
    continue
  df1h = calculate_ichimoku_and_adx(df1h)

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
  df4h = calculate_ichimoku_and_adx(df4h)
  df4h['Cloud_Top'] = df4h[['Senkou_A', 'Senkou_B']].max(axis=1)
  df4h['Cloud_Bottom'] = df4h[['Senkou_A', 'Senkou_B']].min(axis=1)

  df4h['Trend_Long'] = (
      (df4h['Close'] > df4h['Cloud_Top'])
      & (df4h['Tenkan'] > df4h['Kijun'])
      & (df4h['ADX'] > 22)
  )
  df4h['Trend_Short'] = (
      (df4h['Close'] < df4h['Cloud_Bottom'])
      & (df4h['Tenkan'] < df4h['Kijun'])
      & (df4h['ADX'] > 22)
  )

  for col in [
      'Trend_Long',
      'Trend_Short',
      'Cloud_Top',
      'Cloud_Bottom',
      'ADX',
  ]:
    df4h[col] = df4h[col].shift(1)  # جلوگیری از Lookahead مطلق

  df1h['Date_4H'] = df1h['Date'].dt.floor('4h')
  processed_data[symbol] = {'1h': df1h, '4h': df4h.set_index('Date')}

print('⚙️ شروع اجرای بک‌تست HUNTER-ICHIMOKU V12...')

all_timestamps = set()
for dat in processed_data.values():
  all_timestamps.update(dat['1h']['Date'].tolist())
sorted_timestamps = sorted(list(all_timestamps))

active_positions = {}
all_trades = []
SLIPPAGE = 0.0003
FEE_RATE = 0.0007
MAX_CONCURRENT_POSITIONS = 2  # کاهش همپوشانی برای مدیریت بهتر ریسک

dfs_1h = {sym: dat['1h'].set_index('Date') for sym, dat in processed_data.items()}
dfs_4h = {sym: dat['4h'] for sym, dat in processed_data.items()}

for ts in sorted_timestamps:
  symbols_to_close = []
  for symbol, pos in active_positions.items():
    if ts not in dfs_1h[symbol].index:
      continue
    c1h = dfs_1h[symbol].loc[ts]
    entry_index = pos['entry_index']
    df1h_local = processed_data[symbol]['1h']
    match_rows = df1h_local[df1h_local['Date'] == ts]
    if match_rows.empty:
      continue
    curr_i = match_rows.index[0]
    candles_held = curr_i - entry_index

    if pos['side'] == 'LONG':
      hit_sl = c1h['Low'] <= pos['stop_loss']
      hit_tp = c1h['High'] >= pos['take_profit']
      is_timeout = candles_held >= 24

      if hit_sl or hit_tp or is_timeout:
        if hit_sl:
          outcome = 'LOSS'
          r_real = -1.0 - (FEE_RATE * 2)
        elif hit_tp:
          outcome = 'WIN'
          r_real = 2.0 - (FEE_RATE * 2)
        else:
          risk = pos['entry_price'] - pos['stop_loss']
          if risk > 0:
            r_real = (c1h['Close'] - pos['entry_price']) / risk - (FEE_RATE * 2)
          else:
            r_real = 0.0
          outcome = 'WIN' if r_real > 0 else 'LOSS'

        all_trades.append({
            'Timestamp': ts,
            'Symbol': symbol,
            'Side': 'LONG',
            'Outcome': outcome,
            'Return': r_real,
        })
        symbols_to_close.append(symbol)

    elif pos['side'] == 'SHORT':
      hit_sl = c1h['High'] >= pos['stop_loss']
      hit_tp = c1h['Low'] <= pos['take_profit']
      is_timeout = candles_held >= 24

      if hit_sl or hit_tp or is_timeout:
        if hit_sl:
          outcome = 'LOSS'
          r_real = -1.0 - (FEE_RATE * 2)
        elif hit_tp:
          outcome = 'WIN'
          r_real = 2.0 - (FEE_RATE * 2)
        else:
          risk = pos['stop_loss'] - pos['entry_price']
          if risk > 0:
            r_real = (pos['entry_price'] - c1h['Close']) / risk - (FEE_RATE * 2)
          else:
            r_real = 0.0
          outcome = 'WIN' if r_real > 0 else 'LOSS'

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

  for symbol, dat in processed_data.items():
    if symbol in active_positions:
      continue
    if len(active_positions) >= MAX_CONCURRENT_POSITIONS:
      break

    df1h = dat['1h']
    if ts not in df1h['Date'].values:
      continue

    match_rows = df1h[df1h['Date'] == ts]
    if match_rows.empty:
      continue
    i = match_rows.index[0]
    if i < 40:
      continue

    c1h = df1h.iloc[i]
    prev = df1h.iloc[i - 1]
    t4h_time = c1h['Date_4H']
    df4h_idx = dat['4h']

    if t4h_time not in df4h_idx.index:
      continue
    r4h = df4h_idx.loc[t4h_time]

    tk_cross_long = prev['Tenkan'] > prev['Kijun'] and prev['ADX'] > 20
    tk_cross_short = prev['Tenkan'] < prev['Kijun'] and prev['ADX'] > 20
    chikou_long = prev['Close'] > df1h.iloc[i - 27]['Close']
    chikou_short = prev['Close'] < df1h.iloc[i - 27]['Close']

    # منطق ورود لانگ با فیلتر روند قدرتمند
    if r4h.get('Trend_Long', False) and tk_cross_long and chikou_long:
      if (
          prev['Low'] <= prev['Kijun'] * 1.002
          and prev['Close'] > prev['Kijun']
      ):
        entry_price = c1h['Open'] * (1 + SLIPPAGE)
        stop_loss = (
            min(
                df1h['Low'].iloc[i - 5 : i].min(),
                min(prev['Senkou_A'], prev['Senkou_B']),
            )
            - 0.2 * prev['ATR']
        )
        sl_dist_pct = (entry_price - stop_loss) / entry_price

        if 0.003 <= sl_dist_pct <= 0.03:
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

    # منطق ورود شورت با فیلتر روند قدرتمند
    elif r4h.get('Trend_Short', False) and tk_cross_short and chikou_short:
      if (
          prev['High'] >= prev['Kijun'] * 0.998
          and prev['Close'] < prev['Kijun']
      ):
        entry_price = c1h['Open'] * (1 - SLIPPAGE)
        stop_loss = (
            max(
                df1h['High'].iloc[i - 5 : i].max(),
                max(prev['Senkou_A'], prev['Senkou_B']),
            )
            + 0.2 * prev['ATR']
        )
        sl_dist_pct = (stop_loss - entry_price) / entry_price

        if 0.003 <= sl_dist_pct <= 0.03:
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
print('📊 گزارش نهایی HUNTER-ICHIMOKU V12 (ADX Filtered)')
print('============================================================')

if all_trades:
  trades_df = pd.DataFrame(all_trades)
  trades_df.sort_values('Timestamp', inplace=True)

  tot_trades = len(trades_df)
  tot_wins = len(trades_df[trades_df['Outcome'] == 'WIN'])
  tot_losses = len(trades_df[trades_df['Outcome'] == 'LOSS'])
  win_rate = (tot_wins / tot_trades) * 100 if tot_trades > 0 else 0
  net_r = trades_df['Return'].sum()

  outcomes = trades_df['Outcome'].tolist()
  max_wins = 0
  max_losses = 0
  curr_wins = 0
  curr_losses = 0

  loss_sequences = []
  temp_loss_seq = 0

  for out in outcomes:
    if out == 'WIN':
      curr_wins += 1
      curr_losses = 0
      if curr_wins > max_wins:
        max_wins = curr_wins
      if temp_loss_seq > 0:
        loss_sequences.append(temp_loss_seq)
        temp_loss_seq = 0
    else:
      curr_losses += 1
      curr_wins = 0
      temp_loss_seq += 1
      if curr_losses > max_losses:
        max_losses = curr_losses

  if temp_loss_seq > 0:
    loss_sequences.append(temp_loss_seq)

  print(f'🔸 تعداد کل معاملات سبد: {tot_trades}')
  print(f'🔸 معاملات برنده (WIN): {tot_wins}')
  print(f'🔸 معاملات بازنده (LOSS): {tot_losses}')
  print(f'🔥 **حداکثر سودهای متوالی:** {max_wins}')
  print(f'❄️ **حداکثر ضررهای متوالی:** {max_losses}')
  print(f'🎯 **وین‌ریت تجمیعی پورتفوی:** {win_rate:.2f}%')
  print(f'💰 **مجموع بازدهی خالص کل:** {net_r:.2f}R\n')

  print('------------------------------------------------------------')
  print('📉 **لیست کامل تعداد ضررهای متوالی ثبت‌شده (در تمام دوره‌ها):**')
  print('------------------------------------------------------------')
  if loss_sequences:
    print(', '.join(map(str, loss_sequences)))
  else:
    print('هیچ زنجیره ضرری ثبت نشد.')
  print('\n------------------------------------------------------------')
  print('📈 **گزارش تفکیک‌شده به تفکیک هر ارز:**')
  print('------------------------------------------------------------')

  symbol_summary = []
  for sym in SYMBOLS.keys():
    sym_trades = trades_df[trades_df['Symbol'] == sym]
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
  print('⚠️ معامله‌ای ثبت نشد.')

print('\n✨ بک‌تست به پایان رسید.')
