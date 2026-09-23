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
# SCORE-HUNTER PRO — DYNAMIC TRAILING STOP ENGINE
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
MAX_CONSECUTIVE_LOSSES = 4


# ============================================================
# 1. DATA INGESTION (LBank API)
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
# 2. INDICATOR ENGINE (Zero Look-Ahead)
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

  df["Body"] = (df["Close"] - df["Open"]).abs()
  df["Avg_Body"] = df["Body"].rolling(20).mean()
  return df


# ============================================================
# 3. BACKTEST EXECUTION WITH TRAILING STOP
# ============================================================
def run_backtest_engine(symbol, df):
  trades = []
  capital = INITIAL_CAPITAL
  position_size = TRADE_MARGIN * LEVERAGE
  consecutive_losses = 0

  for i in range(200, len(df)):
    current_candle = df.iloc[i]
    prev_candle = df.iloc[i - 1]

    if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
      consecutive_losses = 0

    is_bullish_trend = (
        prev_candle["Close"] > prev_candle["EMA_200"]
        and prev_candle["EMA_50"] > prev_candle["EMA_200"]
    )
    is_bearish_trend = (
        prev_candle["Close"] < prev_candle["EMA_200"]
        and prev_candle["EMA_50"] < prev_candle["EMA_200"]
    )

    strong_momentum = (
        prev_candle["Body"] > 1.2 * prev_candle["Avg_Body"]
    ) if "Avg_Body" in prev_candle else True

    if not (is_bullish_trend or is_bearish_trend) or not strong_momentum:
      continue

    side = "LONG" if is_bullish_trend else "SHORT"
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

    outcome = "LOSS"
    exit_price = sl

    # اسکن کندل‌های آینده همراه با تریلینگ استاپ پویا
    for j in range(i + 1, min(i + 50, len(df))):
      future_candle = df.iloc[j]
      h, l = future_candle["High"], future_candle["Low"]

      if side == "LONG":
        # فعال‌سازی تریلینگ اگر قیمت به اندازه 1 * ATR صعود کند
        if h >= entry_price + atr:
          new_sl = h - atr
          if new_sl > sl:
            sl = new_sl  # قفل کردن سود

        if l <= sl:
          outcome = "WIN" if sl > entry_price else "LOSS"
          exit_price = sl
          break
        elif h >= tp:
          outcome = "WIN"
          exit_price = tp
          break
      else:
        if l <= entry_price - atr:
          new_sl = l + atr
          if new_sl < sl:
            sl = new_sl

        if h >= sl:
          outcome = "WIN" if sl < entry_price else "LOSS"
          exit_price = sl
          break
        elif l <= tp:
          outcome = "WIN"
          exit_price = tp
          break

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
# 4. REPORTING & EXECUTION
# ============================================================
def main():
  print("=" * 70)
  print(
      "SCORE-HUNTER PRO — DYNAMIC TRAILING BACKTEST SIMULATION IN PROGRESS..."
  )
  print("=" * 70)

  end_dt = datetime.now()
  start_dt = end_dt - timedelta(days=DAYS)

  all_trades = []

  for symbol, lbank_symbol in SYMBOLS.items():
    print(f"Fetching & Backtesting {symbol} ({lbank_symbol})...")
    df = fetch_lbank_data(lbank_symbol, start_dt, end_dt)
    if df is None or len(df) < 200:
      continue

    df_prepared = prepare_indicators(df)
    trades = run_backtest_engine(symbol, df_prepared)
    all_trades.extend(trades)

  if not all_trades:
    print("No trades generated.")
    return

  trades_df = pd.DataFrame(all_trades)
  total_trades = len(trades_df)
  wins = trades_df[trades_df["Outcome"] == "WIN"]

  win_rate = (len(wins) / total_trades) * 100.0 if total_trades > 0 else 0
  net_pnl = trades_df["PnL"].sum()

  loss_streaks, current_streak = [], 0
  for outcome in trades_df["Outcome"]:
    if outcome == "LOSS":
      current_streak += 1
    else:
      if current_streak > 0:
        loss_streaks.append(current_streak)
      current_streak = 0
  if current_streak > 0:
    loss_streaks.append(current_streak)
  max_streak = max(loss_streaks) if loss_streaks else 0

  print("\n" + "=" * 70)
  print("== SCORE-HUNTER PRO: OPTIMIZED TRAILING RESULTS (1 YEAR) ==")
  print("=" * 70)
  print(f"Total Trades:              {total_trades}")
  print(f"Win Rate:                  {win_rate:.2f}%")
  print(f"Net Profit (USD):          ${net_pnl:,.2f}")
  print(f"Max Consecutive Losses:    {max_streak}")
  print("=" * 70)


if __name__ == "__main__":
  main()
