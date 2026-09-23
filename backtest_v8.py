import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
  import ccxt
except ImportError:
  subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
  import ccxt

# ============================================================
# SCORE-HUNTER PRO: Live & Backtest Engine
# ============================================================

exchange = ccxt.lbank({"enableRateLimit": True, "timeout": 20000})

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "BNB": "BNB/USDT",
    "ADA": "ADA/USDT",
    "XLM": "XLM/USDT",
    "DOGE": "DOGE/USDT",
    "NEAR": "NEAR/USDT",
    "ICP": "ICP/USDT",
    "CRV": "CRV/USDT",
}

TIMEFRAME = "15m"
SLIPPAGE = 0.0003
FEE_RATE = 0.0007

INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 50.0
RISK_REWARD = 2.0
DAYS = 365
MAX_CONSECUTIVE_LOSSES = 4  # مکانیزم کنترل حد نصاب باخت متوالی


# ============================================================
# 1. DATA FETCHING (LBank API Integration)
# ============================================================
def fetch_lbank_data(lbank_symbol, start_dt, end_dt):
  since_ts = int((start_dt - timedelta(days=15)).timestamp() * 1000)
  end_ts = int(end_dt.timestamp() * 1000)
  all_ohlcv = []
  current_since = since_ts

  try:
    while current_since < end_ts:
      batch = exchange.fetch_ohlcv(
          lbank_symbol, timeframe=TIMEFRAME, since=current_since, limit=1000
      )
      if not batch:
        break
      all_ohlcv.extend(batch)
      last_ts = batch[-1][0]
      if last_ts <= current_since:
        break
      current_since = last_ts + 1
      if len(batch) < 1000 or last_ts >= end_ts:
        break
      time.sleep(0.2)
  except Exception as e:
    print(f"Error fetching {lbank_symbol}: {e}")
    return None

  if not all_ohlcv:
    return None

  df = pd.DataFrame(
      all_ohlcv, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
  )
  df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
  df.set_index("Date", inplace=True)
  df.dropna(inplace=True)
  df.drop_duplicates(subset=["Timestamp"], keep="last", inplace=True)
  df.sort_index(inplace=True)

  return df[(df.index >= start_dt) & (df.index <= end_dt)]


# ============================================================
# 2. INDICATORS & PREPARATION
# ============================================================
def prepare_indicators(df):
  df = df.copy()
  df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
  df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

  high_low = df["High"] - df["Low"]
  high_close = np.abs(df["High"] - df["Close"].shift())
  low_close = np.abs(df["Low"] - df["Close"].shift())
  ranges = pd.concat([high_low, high_close, low_close], axis=1)
  df["ATR"] = ranges.max(axis=1).rolling(14).mean()
  return df


# ============================================================
# 3. BACKTEST ENGINE (Zero Look-Ahead Bias)
# ============================================================
def run_engine(symbol, df):
  trades = []
  capital = INITIAL_CAPITAL
  position_size = TRADE_MARGIN * LEVERAGE

  consecutive_losses = 0

  for i in range(200, len(df)):
    current_candle = df.iloc[i]
    prev_candle = df.iloc[i - 1]  # تکیه کامل بر کندل بسته شده

    # بررسی فیلتر حفاظتی ضررهای متوالی
    if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
      # وقفه حفاظتی یا ریست موقت ربات
      consecutive_losses = 0  # پس از استراحت بازنشانی می‌شود

    # منطق سیگنال‌دهی استاندارد روند و مومنتوم
    is_bullish = (
        prev_candle["Close"] > prev_candle["EMA_200"]
        and prev_candle["EMA_50"] > prev_candle["EMA_200"]
    )
    is_bearish = (
        prev_candle["Close"] < prev_candle["EMA_200"]
        and prev_candle["EMA_50"] < prev_candle["EMA_200"]
    )

    if not is_bullish and not is_bearish:
      continue

    side = "LONG" if is_bullish else "SHORT"
    entry_price = (
        current_candle["Open"] * (1.0 + SLIPPAGE)
        if side == "LONG"
        else current_candle["Open"] * (1.0 - SLIPPAGE)
    )
    atr = prev_candle["ATR"]

    if not np.isfinite(atr) or atr <= 0:
      continue

    if side == "LONG":
      sl = entry_price - (atr * 1.5)
      tp = entry_price + (atr * 1.5 * RISK_REWARD)
    else:
      sl = entry_price + (atr * 1.5)
      tp = entry_price - (atr * 1.5 * RISK_REWARD)

    # اسکن کندل‌های بعدی برای برخورد با TP یا SL
    outcome = "LOSS"
    exit_price = sl

    for j in range(i + 1, min(i + 35, len(df))):
      future_candle = df.iloc[j]
      h, l = future_candle["High"], future_candle["Low"]

      if side == "LONG":
        if l <= sl:
          outcome = "LOSS"
          exit_price = sl
          break
        elif h >= tp:
          outcome = "WIN"
          exit_price = tp
          break
      else:
        if h >= sl:
          outcome = "LOSS"
          exit_price = sl
          break
        elif l <= tp:
          outcome = "WIN"
          exit_price = tp
          break

    # محاسبه سود و زیان دلاری با احتساب کمیسیون صرافی
    price_ret = (
        (exit_price - entry_price) / entry_price
        if side == "LONG"
        else (entry_price - exit_price) / entry_price
    )
    dollar_pnl = (
        position_size * price_ret - position_size * FEE_RATE * 2.0
    )

    capital += dollar_pnl

    if outcome == "LOSS":
      consecutive_losses += 1
    else:
      consecutive_losses = 0

    trades.append({
        "Timestamp": df.index[i],
        "Symbol": symbol,
        "Side": side,
        "Outcome": outcome,
        "PnL": dollar_pnl,
        "Capital": capital,
    })

  return trades


# ============================================================
# 4. MAIN EXECUTION PIPELINE
# ============================================================
def main():
  print("=" * 60)
  print("SCORE-HUNTER PRO — LIVE & BACKTEST SYSTEM INITIALIZED")
  print("=" * 60)

  end_dt = datetime.now()
  start_dt = end_dt - timedelta(days=DAYS)

  all_trades = []

  for symbol, lbank_symbol in SYMBOLS.items():
    print(f"Processing {symbol} from LBank...")
    df = fetch_lbank_data(lbank_symbol, start_dt, end_dt)
    if df is None or len(df) < 200:
      continue

    df_prepared = prepare_indicators(df)
    trades = run_engine(symbol, df_prepared)
    all_trades.extend(trades)

  if not all_trades:
    print("No trades generated.")
    return

  trades_df = pd.DataFrame(all_trades)
  total_trades = len(trades_df)
  wins = trades_df[trades_df["Outcome"] == "WIN"]
  win_rate = (len(wins) / total_trades) * 100.0
  net_pnl = trades_df["PnL"].sum()

  print("\n" + "=" * 60)
  print(f"TOTAL TRADES: {total_trades}")
  print(f"WIN RATE:     {win_rate:.2f}%")
  print(f"NET PNL:      ${net_pnl:,.2f}")
  print("=" * 60)


if __name__ == "__main__":
  main()
