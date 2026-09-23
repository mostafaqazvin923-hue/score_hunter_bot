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
# SCORE-HUNTER PRO — V12 STATISTICAL & VOLATILITY ENGINE
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

TIMEFRAME = "1h"
SLIPPAGE = 0.0003
FEE_RATE = 0.0007

INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 50.0
RISK_REWARD = 2.0
DAYS = 365
MAX_CONSECUTIVE_LOSSES = 3


def fetch_lbank_data(lbank_symbol, start_dt, end_dt):
  since_ts = int((start_dt - timedelta(days=20)).timestamp() * 1000)
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


def prepare_quantitative_indicators(df):
  df = df.copy()

  # 1. محاسبه Z-Score آماری برای سنجش انحراف قیمت از میانگین ۵۰ دوره‌ای
  window = 50
  rolling_mean = df["Close"].rolling(window).mean()
  rolling_std = df["Close"].rolling(window).std()
  df["Z_Score"] = (df["Close"] - rolling_mean) / rolling_std

  # 2. فیلتر فشردگی نوسان (Volatility Squeeze) با استفاده از ATR
  high_low = df["High"] - df["Low"]
  high_close = np.abs(df["High"] - df["Close"].shift())
  low_close = np.abs(df["Low"] - df["Close"].shift())
  ranges = pd.concat([high_low, high_close, low_close], axis=1)
  df["ATR"] = ranges.max(axis=1).rolling(14).mean()
  df["ATR_SMA"] = df["ATR"].rolling(20).mean()

  # 3. تاییدیه مومنتوم حجم و بدنه کندل
  df["Body"] = (df["Close"] - df["Open"]).abs()
  df["Avg_Body"] = df["Body"].rolling(20).mean()

  # 4. روند کلان ۴ ساعته برای فیلتر جهت کلی بازار
  df_4h = df.resample("4h").agg({
      "Open": "first",
      "High": "max",
      "Low": "min",
      "Close": "last",
      "Volume": "sum",
  })
  df_4h.dropna(inplace=True)
  df_4h["EMA_Macro_4h"] = df_4h["Close"].ewm(span=50, adjust=False).mean()

  df["Macro_Close"] = df_4h["Close"].reindex(df.index, method="ffill")
  df["Macro_EMA"] = df_4h["EMA_Macro_4h"].reindex(df.index, method="ffill")

  return df


def run_backtest_engine(symbol, df):
  trades = []
  capital = INITIAL_CAPITAL
  position_size = TRADE_MARGIN * LEVERAGE
  consecutive_losses = 0

  i = 60
  while i < len(df):
    current_candle = df.iloc[i]
    prev_candle = df.iloc[i - 1]

    if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
      i += 24  # استراحت کنترلی ۲۴ کندلی پس از ۳ باخت متوالی
      consecutive_losses = 0
      continue

    # فیلتر جهت کلان بازار (۴ ساعته)
    is_macro_bullish = prev_candle["Macro_Close"] > prev_candle["Macro_EMA"]
    is_macro_bearish = prev_candle["Macro_Close"] < prev_candle["Macro_EMA"]

    # فیلتر فشردگی نوسان (عبور ATR از میانگین خود برای تایید خروج از حالت سایدوی)
    is_volatility_expanded = prev_candle["ATR"] > prev_candle["ATR_SMA"]

    if not is_volatility_expanded:
      i += 1
      continue

    # ستاپ آماری Z-Score به همراه هم‌راستایی با روند کلان و مومنتوم بدنه کندل
    is_long_setup = (
        is_macro_bullish
        and prev_candle["Z_Score"] < -1.0  # اشباع فروش آماری در روند صعودی
        and prev_candle["Body"] > 1.2 * prev_candle["Avg_Body"]
    )

    is_short_setup = (
        is_macro_bearish
        and prev_candle["Z_Score"] > 1.0  # اشباع خرید آماری در روند نزولی
        and prev_candle["Body"] > 1.2 * prev_candle["Avg_Body"]
    )

    if not is_long_setup and not is_short_setup:
      i += 1
      continue

    side = "LONG" if is_long_setup else "SHORT"
    entry_price = (
        current_candle["Open"] * (1.0 + SLIPPAGE)
        if side == "LONG"
        else current_candle["Open"] * (1.0 - SLIPPAGE)
    )
    atr = prev_candle["ATR"]

    if not np.isfinite(atr) or atr <= 0:
      i += 1
      continue

    if side == "LONG":
      sl = entry_price - (atr * 1.5)
      tp = entry_price + (atr * 1.5 * RISK_REWARD)
    else:
      sl = entry_price + (atr * 1.5)
      tp = entry_price - (atr * 1.5 * RISK_REWARD)

    outcome = "LOSS"
    exit_price = sl
    exit_index = i + 1

    # اسکن کندل‌های آتی با رعایت کامل قفل همپوشانی (Overlap Lock)
    for j in range(i + 1, min(i + 60, len(df))):
      future_candle = df.iloc[j]
      h, l = future_candle["High"], future_candle["Low"]

      if side == "LONG":
        if l <= sl:
          outcome = "LOSS"
          exit_price = sl
          exit_index = j
          break
        elif h >= tp:
          outcome = "WIN"
          exit_price = tp
          exit_index = j
          break
      else:
        if h >= sl:
          outcome = "LOSS"
          exit_price = sl
          exit_index = j
          break
        elif l <= tp:
          outcome = "WIN"
          exit_price = tp
          exit_index = j
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

    # پرش دقیق به کندل پس از خروج قطعی پوزیشن (قفل همپوشانی ایمن)
    i = exit_index + 1

  return trades


def main():
  print("=" * 70)
  print(
      "SCORE-HUNTER PRO — V12 STATISTICAL ENGINE SIMULATION IN PROGRESS..."
  )
  print("=" * 70)

  end_dt = datetime.now()
  start_dt = end_dt - timedelta(days=DAYS)

  all_trades = []

  for symbol, lbank_symbol in SYMBOLS.items():
    print(f"Fetching & Backtesting {symbol} ({lbank_symbol})...")
    df = fetch_lbank_data(lbank_symbol, start_dt, end_dt)
    if df is None or len(df) < 100:
      continue

    df_prepared = prepare_quantitative_indicators(df)
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
  print("== SCORE-HUNTER PRO: V12 STATISTICAL RESULTS (1 YEAR) ==")
  print("=" * 70)
  print(f"Total Trades:              {total_trades}")
  print(f"Win Rate:                  {win_rate:.2f}%")
  print(f"Net Profit (USD):          ${net_pnl:,.2f}")
  print(f"Max Consecutive Losses:    {max_streak}")
  print("=" * 70)


if __name__ == "__main__":
  main()
