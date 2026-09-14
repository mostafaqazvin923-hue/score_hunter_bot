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
    'DOGE': 'DOGE/USDT',
    'DOT': 'DOT/USDT',
    'LTC': 'LTC/USDT',
    'RENDER': 'RENDER/USDT',
    'ATOM': 'ATOM/USDT',
}

start_date = datetime.now() - timedelta(days=365)
since_timestamp = int(start_date.timestamp() * 1000)

print('============================================================')
print('📥 دریافت داده‌ها (HUNTER-V22 - Daily Mean Reversion Pullback)')
print('============================================================')

data_daily = {}

for symbol, lbank_symbol in SYMBOLS.items():
  all_ohlcv = []
  current_since = since_timestamp
  now_timestamp = exchange.milliseconds()

  while current_since < now_timestamp:
    try:
      ohlcv = exchange.fetch_ohlcv(
          lbank_symbol, timeframe='1d', since=current_since, limit=1000
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
    df = pd.DataFrame(
        all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['Date'], inplace=True)
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    tr1 = df['High'] - df['Low']
    tr2 = np.abs(df['High'] - df['Close'].shift(1))
    tr3 = np.abs(df['Low'] - df['Close'].shift(1))
    df['ATR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    # Trend filter & Pullback core
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['RSI'] = 100 - (
        100
        / (
            1
            + df['Close']
            .diff()
            .clip(lower=0)
            .rolling(14)
            .mean()
            / df['Close'].diff().clip(upper=0).abs().rolling(14).mean()
        )
    )

    data_daily[symbol] = df

print('⚙️ شروع اجرای بک‌تست روزانه HUNTER-V22...')

all_timestamps = set()
for df in data_daily.values():
  all_timestamps.update(df['Date'].tolist())
sorted_timestamps = sorted(list(all_timestamps))

active_positions = {}
all_trades = []
SLIPPAGE = 0.0005
FEE_RATE = 0.0007

dfs_daily = {sym: df.set_index('Date') for sym, df in data_daily.items()}

for ts in sorted_timestamps:
  symbols_to_close = []
  for symbol, pos in active_positions.items():
    if ts not in dfs_daily[symbol].index:
      continue
    c_day = dfs_daily[symbol].loc[ts]

    if pos['side'] == 'LONG':
      hit_sl = c_day['Low'] <= pos['stop_loss']
      hit_tp = c_day['High'] >= pos['take_profit']

      if hit_sl or hit_tp:
        if hit_sl:
          outcome = 'LOSS'
          r_real = -1.0 - (FEE_RATE * 2)
        else:
          outcome = 'WIN'
          r_real = 1.8 - (FEE_RATE * 2)  # ریسک به ریوارد منطقی ۱.۸

        all_trades.append({
            'Timestamp': ts,
            'Symbol': symbol,
            'Side': 'LONG',
            'Outcome': outcome,
            'Return': r_real,
        })
        symbols_to_close.append(symbol)

    elif pos['side'] == 'SHORT':
      hit_sl = c_day['High'] >= pos['stop_loss']
      hit_tp = c_day['Low'] <= pos['take_profit']

      if hit_sl or hit_tp:
        if hit_sl:
          outcome = 'LOSS'
          r_real = -1.0 - (FEE_RATE * 2)
        else:
          outcome = 'WIN'
          r_real = 1.8 - (FEE_RATE * 2)

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

  for symbol, df in data_daily.items():
    if symbol in active_positions:
      continue

    if ts not in df['Date'].values:
      continue

    match_rows = df[df['Date'] == ts]
    if match_rows.empty:
      continue
    i = match_rows.index[0]
    if i < 200:
      continue

    c_day = df.iloc[i]
    prev = df.iloc[i - 1]

    # استراتژی: روند کلی صعودی است (قیمت بالای EMA50 و EMA200) و قیمت یک اصلاح (Pullback) به سمت EMA50 زده و RSI زیر ۴۵ آمده
    long_pullback = (
        prev['Close'] > prev['EMA200']
        and prev['Close'] > prev['EMA50']
        and prev['Low'] <= prev['EMA50']
        and prev['Close'] > prev['EMA50']
        and prev['RSI'] < 50
    )

    if long_pullback:
      entry_price = c_day['Open'] * (1 + SLIPPAGE)
      stop_loss = entry_price - (2.0 * prev['ATR'])
      risk = entry_price - stop_loss
      take_profit = entry_price + (1.8 * risk)

      active_positions[symbol] = {
          'side': 'LONG',
          'entry_price': entry_price,
          'stop_loss': stop_loss,
          'take_profit': take_profit,
      }
      continue

print('\n============================================================')
print('📊 گزارش نهایی HUNTER-V22 (Daily Mean Reversion)')
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
