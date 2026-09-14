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
print(
    '📥 دریافت داده‌ها (HUNTER-V19 - Institutional Donchian Breakout & ADX'
    ' Regime)'
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


def calculate_indicators(df):
  df = df.copy()
  tr1 = df['High'] - df['Low']
  tr2 = np.abs(df['High'] - df['Close'].shift(1))
  tr3 = np.abs(df['Low'] - df['Close'].shift(1))
  df['ATR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

  # Donchian Channels (20 periods) - Turtle Trading Core
  df['Donchian_High'] = df['High'].rolling(20).max().shift(1)
  df['Donchian_Low'] = df['Low'].rolling(20).min().shift(1)

  # ADX (Average Directional Index) for Market Regime
  plus_dm = df['High'].diff()
  minus_dm = df['Low'].diff()
  plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
  minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

  tr = df['ATR']
  plus_di = (
      100
      * pd.Series(plus_dm).rolling(14).mean()
      / (tr.replace(0, np.nan))
  )
  minus_di = (
      100
      * pd.Series(minus_dm).rolling(14).mean()
      / (tr.replace(0, np.nan))
  )
  dx = (
      100
      * np.abs(plus_di - minus_di)
      / (plus_di + minus_di).replace(0, np.nan)
  )
  df['ADX'] = dx.rolling(14).mean()

  df['Volume_MA'] = df['Volume'].rolling(20).mean()
  return df


processed_data = {}
for symbol, df1h in data_1h.items():
  if len(df1h) < 200:
    continue
  df1h = calculate_indicators(df1h)

  df_daily = (
      df1h.set_index('Date')
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
  df_daily['Daily_EMA50'] = (
      df_daily['Close'].ewm(span=50, adjust=False).mean().shift(1)
  )
  df1h['Date_Daily'] = df1h['Date'].dt.floor('1d')

  processed_data[symbol] = {
      '1h': df1h,
      'daily': df_daily.set_index('Date'),
  }

print('⚙️ شروع اجرای بک‌تست HUNTER-V19 (Institutional Model)...')

all_timestamps = set()
for dat in processed_data.values():
  all_timestamps.update(dat['1h']['Date'].tolist())
sorted_timestamps = sorted(list(all_timestamps))

active_positions = {}
all_trades = []
SLIPPAGE = 0.0003
FEE_RATE = 0.0007
MAX_CONCURRENT_POSITIONS = 3

dfs_1h = {sym: dat['1h'].set_index('Date') for sym, dat in processed_data.items()}

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
      is_timeout = candles_held >= 48  # هولد طولانی‌تر در ترندها

      if hit_sl or hit_tp or is_timeout:
        if hit_sl:
          outcome = 'LOSS'
          r_real = -1.0 - (FEE_RATE * 2)
        elif hit_tp:
          outcome = 'WIN'
          r_real = 3.5 - (
              FEE_RATE * 2
          )  # پاداش سنگین ۳.۵ برابری برای جبران وین‌ریت
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
      is_timeout = candles_held >= 48

      if hit_sl or hit_tp or is_timeout:
        if hit_sl:
          outcome = 'LOSS'
          r_real = -1.0 - (FEE_RATE * 2)
        elif hit_tp:
          outcome = 'WIN'
          r_real = 3.5 - (FEE_RATE * 2)
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
    if i < 50:
      continue

    c1h = df1h.iloc[i]
    prev = df1h.iloc[i - 1]

    # فیلتر رژیم بازار (ADX باید بالای ۲۲ باشد تا مشخص شود بازار رنج نیست)
    if pd.isna(prev['ADX']) or prev['ADX'] < 22:
      continue

    daily_time = c1h['Date_Daily']
    df_daily_idx = dat['daily']
    if daily_time not in df_daily_idx.index:
      continue
    macro_bull = prev['Close'] > df_daily_idx.loc[daily_time].get(
        'Daily_EMA50', prev['Close']
    )
    macro_bear = prev['Close'] < df_daily_idx.loc[daily_time].get(
        'Daily_EMA50', prev['Close']
    )

    # استراتژی کانال دونچیان (شکست سقف یا کف ۲۰ کندل گذشته همراه با حجم بالا)
    breakout_long = (
        macro_bull
        and prev['Close'] >= prev['Donchian_High']
        and prev['Volume'] > (1.3 * prev['Volume_MA'])
    )

    breakout_short = (
        macro_bear
        and prev['Close'] <= prev['Donchian_Low']
        and prev['Volume'] > (1.3 * prev['Volume_MA'])
    )

    if breakout_long:
      entry_price = c1h['Open'] * (1 + SLIPPAGE)
      stop_loss = entry_price - (2.0 * prev['ATR'])  # حد ضرر مبتنی بر نوسان ATR
      sl_dist_pct = (entry_price - stop_loss) / entry_price

      if 0.005 <= sl_dist_pct <= 0.05:
        risk = entry_price - stop_loss
        take_profit = entry_price + (
            3.5 * risk
        )  # پاداش ۳.۵ برابری (اصول تریدرهای بزرگ)
        active_positions[symbol] = {
            'side': 'LONG',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_index': i,
        }
        continue

    elif breakout_short:
      entry_price = c1h['Open'] * (1 - SLIPPAGE)
      stop_loss = entry_price + (2.0 * prev['ATR'])
      sl_dist_pct = (stop_loss - entry_price) / entry_price

      if 0.005 <= sl_dist_pct <= 0.05:
        risk = stop_loss - entry_price
        take_profit = entry_price - (3.5 * risk)
        active_positions[symbol] = {
            'side': 'SHORT',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_index': i,
        }
        continue

print('\n============================================================')
print('📊 گزارش نهایی HUNTER-V19 (Institutional Model)')
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
