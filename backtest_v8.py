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
    '📥 دریافت داده‌ها (HUNTER-V27 - Institutional CTA & Trailing Volatility)'
)
print('============================================================')

processed_data = {}

for symbol, lbank_symbol in SYMBOLS.items():
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

  # ATR و میانگین نوسان برای فیلتر بازار رِنج (جلوگیری از ضررهای متوالی فرسایشی)
  tr1 = df4h['High'] - df4h['Low']
  tr2 = np.abs(df4h['High'] - df4h['Close'].shift(1))
  tr3 = np.abs(df4h['Low'] - df4h['Close'].shift(1))
  df4h['ATR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
  df4h['ATR_MA'] = df4h['ATR'].rolling(30).mean()

  # مومنتوم چندلایه (Dual-Lookback TSMOM)
  df4h['Mom_Short'] = (df4h['Close'] - df4h['Close'].shift(12)) / df4h[
      'Close'
  ].shift(12)
  df4h['Mom_Long'] = (df4h['Close'] - df4h['Close'].shift(36)) / df4h[
      'Close'
  ].shift(36)

  processed_data[symbol] = df4h.set_index('Date')

print('⚙️ شروع اجرای بک‌تست حرفه‌ای HUNTER-V27...')

all_timestamps = set()
for df in processed_data.values():
  all_timestamps.update(df.index.tolist())
sorted_timestamps = sorted(list(all_timestamps))

active_positions = {}
all_trades = []
SLIPPAGE = 0.0003
FEE_RATE = 0.0007
MAX_POSITIONS = 3

for ts in sorted_timestamps:
  symbols_to_close = []
  for symbol, pos in active_positions.items():
    if ts not in processed_data[symbol].index:
      continue
    c4h = processed_data[symbol].loc[ts]

    # مدیریت استاپ لاس پویا (Trailing Stop بر اساس بالاترین قیمت ثبت شده)
    if c4h['High'] > pos['highest_price']:
      pos['highest_price'] = c4h['High']
      # تریل کردن استاپ لاس به سمت بالا برای حفظ سود
      new_trailing_sl = pos['highest_price'] - (2.2 * c4h['ATR'])
      if new_trailing_sl > pos['stop_loss']:
        pos['stop_loss'] = new_trailing_sl

    hit_sl = c4h['Low'] <= pos['stop_loss']
    df_local = processed_data[symbol].reset_index()
    match_rows = df_local[df_local['Date'] == ts]
    if match_rows.empty:
      continue
    curr_i = match_rows.index[0]
    candles_held = curr_i - pos['entry_index']
    is_timeout = candles_held >= 50  # حداکثر زمان نگهداری در روند

    if hit_sl or is_timeout:
      initial_risk = pos['initial_risk']
      if hit_sl:
        # اگر استاپ لاس خورد، محاسبه بازدهی بر اساس قیمت استاپ یا بسته شدن
        exit_p = min(pos['stop_loss'], c4h['Open'])
      else:
        exit_p = c4h['Close']

      r_real = (exit_p - pos['entry_price']) / initial_risk - (FEE_RATE * 2)
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

  # رتبه‌بندی مومنتوم بین تمام ارزها
  current_scores = {}
  for symbol, df in processed_data.items():
    if ts in df.index:
      val = df.loc[ts, 'Mom_Long']
      if not np.isnan(val):
        current_scores[symbol] = val

  if not current_scores:
    continue

  ranked_symbols = sorted(
      current_scores.keys(), key=lambda x: current_scores[x], reverse=True
  )

  for symbol in ranked_symbols:
    if len(active_positions) >= MAX_POSITIONS:
      break
    if symbol in active_positions:
      continue

    df = processed_data[symbol]
    if ts not in df.index:
      continue

    df_reset = df.reset_index()
    match_rows = df_reset[df_reset['Date'] == ts]
    if match_rows.empty:
      continue
    i = match_rows.index[0]
    if i < 40:
      continue

    c4h = df.iloc[i]

    # فیلترهای استراتژیک صندوق‌های بزرگ:
    # 1. مومنتوم کوتاه و بلند مثبت
    # 2. فیلتر نوسان (ATR بالاتر از میانگین برای دوری از بازار فرسایشی و سایدوی)
    valid_trend = (
        c4h['Mom_Short'] > 0.02
        and c4h['Mom_Long'] > 0.05
        and c4h['ATR'] > c4h['ATR_MA']
    )

    if valid_trend:
      entry_price = c4h['Open'] * (1 + SLIPPAGE)
      initial_sl = entry_price - (2.0 * c4h['ATR'])
      initial_risk = entry_price - initial_sl
      sl_dist_pct = initial_risk / entry_price

      if 0.01 <= sl_dist_pct <= 0.045:
        active_positions[symbol] = {
            'side': 'LONG',
            'entry_price': entry_price,
            'stop_loss': initial_sl,
            'highest_price': entry_price,
            'initial_risk': initial_risk,
            'entry_index': i,
        }

print('\n============================================================')
print('📊 گزارش نهایی HUNTER-V27 (Institutional CTA & Trailing)')
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
  print('📉 **لیست کامل تعداد ضررهای متوالی ثبت‌شده:**')
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
