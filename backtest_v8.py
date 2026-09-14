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
print('📥 دریافت داده‌ها (HUNTER-V23 - Multi-Timeframe 1D & 4H)')
print('============================================================')

processed_data = {}

for symbol, lbank_symbol in SYMBOLS.items():
  # دریافت داده‌های 4 ساعته
  all_ohlcv_4h = []
  current_since = since_timestamp
  now_timestamp = exchange.milliseconds()

  while current_since < now_timestamp:
    try:
      ohlcv = exchange.fetch_ohlcv(
          lbank_symbol, timeframe='4h', since=current_since, limit=1000
      )
      if not ohlcv:
        break
      current_since = ohlcv[-1][0] + 1
      all_ohlcv_4h.extend(ohlcv)
      if len(ohlcv) < 1000:
        break
    except Exception:
      break

  if not all_ohlcv_4h:
    continue

  df4h = pd.DataFrame(
      all_ohlcv_4h,
      columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'],
  )
  df4h['Date'] = pd.to_datetime(df4h['Timestamp'], unit='ms')
  df4h = df4h[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
  df4h.dropna(inplace=True)
  df4h.drop_duplicates(subset=['Date'], inplace=True)
  df4h.sort_values('Date', inplace=True)
  df4h.reset_index(drop=True, inplace=True)

  # اندیکاتورهای 4 ساعته
  tr1 = df4h['High'] - df4h['Low']
  tr2 = np.abs(df4h['High'] - df4h['Close'].shift(1))
  tr3 = np.abs(df4h['Low'] - df4h['Close'].shift(1))
  df4h['ATR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
  df4h['EMA20'] = df4h['Close'].ewm(span=20, adjust=False).mean()
  df4h['Volume_MA'] = df4h['Volume'].rolling(20).mean()

  # ساخت فریم روزانه از روی داده‌های 4 ساعته برای جهت بازار
  df4h['Date_Daily'] = df4h['Date'].dt.floor('1d')
  df_daily = (
      df4h.set_index('Date')
      .resample('1d')
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
  df_daily['EMA50_Daily'] = (
      df_daily['Close'].ewm(span=50, adjust=False).mean().shift(1)
  )

  # محاسبه ADX روزانه برای تشخیص رژیم روند
  tr_d1 = df_daily['High'] - df_daily['Low']
  tr_d2 = np.abs(df_daily['High'] - df_daily['Close'].shift(1))
  tr_d3 = np.abs(df_daily['Low'] - df_daily['Close'].shift(1))
  atr_d = pd.concat([tr_d1, tr_d2, tr_d3], axis=1).max(axis=1).rolling(14).mean()
  plus_dm = df_daily['High'].diff()
  minus_dm = df_daily['Low'].diff()
  plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
  minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
  plus_di = (
      100
      * pd.Series(plus_dm).rolling(14).mean()
      / (atr_d.replace(0, np.nan))
  )
  minus_di = (
      100
      * pd.Series(minus_dm).rolling(14).mean()
      / (atr_d.replace(0, np.nan))
  )
  dx = (
      100
      * np.abs(plus_di - minus_di)
      / (plus_di + minus_di).replace(0, np.nan)
  )
  df_daily['ADX_Daily'] = dx.rolling(14).mean()

  processed_data[symbol] = {
      '4h': df4h.set_index('Date'),
      'daily': df_daily.set_index('Date'),
  }

print('⚙️ شروع اجرای بک‌تست چندتایم‌فریمه HUNTER-V23...')

all_timestamps = set()
for dat in processed_data.values():
  all_timestamps.update(dat['4h'].index.tolist())
sorted_timestamps = sorted(list(all_timestamps))

active_positions = {}
all_trades = []
SLIPPAGE = 0.0003
FEE_RATE = 0.0007
MAX_CONCURRENT_POSITIONS = 2

dfs_4h = {sym: dat['4h'] for sym, dat in processed_data.items()}
dfs_daily = {sym: dat['daily'] for sym, dat in processed_data.items()}

for ts in sorted_timestamps:
  symbols_to_close = []
  for symbol, pos in active_positions.items():
    if ts not in dfs_4h[symbol].index:
      continue
    c4h = dfs_4h[symbol].loc[ts]
    entry_index = pos['entry_index']
    df4h_local = dfs_4h[symbol]
    match_rows = df4h_local.reset_index()
    match_rows = match_rows[match_rows['Date'] == ts]
    if match_rows.empty:
      continue
    curr_i = match_rows.index[0]
    candles_held = curr_i - pos['raw_index']

    if pos['side'] == 'LONG':
      hit_sl = c4h['Low'] <= pos['stop_loss']
      hit_tp = c4h['High'] >= pos['take_profit']
      is_timeout = candles_held >= 30  # حداکثر ۵ روز نگهداری در 4H

      if hit_sl or hit_tp or is_timeout:
        if hit_sl:
          outcome = 'LOSS'
          r_real = -1.0 - (FEE_RATE * 2)
        elif hit_tp:
          outcome = 'WIN'
          r_real = 2.2 - (FEE_RATE * 2)
        else:
          risk = pos['entry_price'] - pos['stop_loss']
          if risk > 0:
            r_real = (c4h['Close'] - pos['entry_price']) / risk - (FEE_RATE * 2)
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

  for sym in symbols_to_close:
    del active_positions[sym]

  for symbol, dat in processed_data.items():
    if symbol in active_positions:
      continue
    if len(active_positions) >= MAX_CONCURRENT_POSITIONS:
      break

    df4h = dat['4h']
    if ts not in df4h.index:
      continue

    df4h_reset = df4h.reset_index()
    match_rows = df4h_reset[df4h_reset['Date'] == ts]
    if match_rows.empty:
      continue
    i = match_rows.index[0]
    if i < 50:
      continue

    c4h = df4h.iloc[i]
    prev4h = df4h.iloc[i - 1]

    # بررسی جهت بازار در تایم‌فریم روزانه
    daily_time = pd.Timestamp(ts).floor('1d')
    df_d = dfs_daily[symbol]
    if daily_time not in df_d.index:
      continue

    d_row = df_d.loc[daily_time]
    macro_bull = d_row['Close'] > d_row.get(
        'EMA50_Daily', d_row['Close']
    ) and d_row.get('ADX_Daily', 30) > 22

    # ستاپ ورود ۴ ساعته: روند روزانه صعودی + پولبک به EMA20 در ۴ ساعته + تایید حجم
    long_signal = (
        macro_bull
        and prev4h['Low'] <= prev4h['EMA20']
        and prev4h['Close'] > prev4h['EMA20']
        and prev4h['Volume'] > (1.2 * prev4h['Volume_MA'])
    )

    if long_signal:
      entry_price = c4h['Open'] * (1 + SLIPPAGE)
      stop_loss = (
          df4h['Low'].iloc[i - 4 : i].min() - 0.2 * prev4h['ATR']
      )  # کف محلی 4 ساعته
      sl_dist_pct = (entry_price - stop_loss) / entry_price

      if 0.005 <= sl_dist_pct <= 0.03:
        risk = entry_price - stop_loss
        take_profit = entry_price + (2.2 * risk)
        active_positions[symbol] = {
            'side': 'LONG',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'raw_index': i,
        }
        continue

print('\n============================================================')
print('📊 گزارش نهایی HUNTER-V23 (Multi-Timeframe 1D + 4H)')
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
