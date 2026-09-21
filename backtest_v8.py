import os
import subprocess
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

# ============================================================
# HUNTER-V109 — STRICT GLOBAL CAP & MULTI-TIMEFRAME ENGINE
# ============================================================

exchange = ccxt.lbank({"enableRateLimit": True})

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "LINK": "LINK/USDT",
    "UNI": "UNI/USDT",
    "ICP": "ICP/USDT",
    "INJ": "INJ/USDT",
    "ATOM": "ATOM/USDT",
    "RENDER": "RENDER/USDT",
    "XLM": "XLM/USDT",
    "AAVE": "AAVE/USDT",
    "WIF": "WIF/USDT",
    "ONDO": "ONDO/USDT",
    "DOGE": "DOGE/USDT",
    "BNB": "BNB/USDT",
    "ADA": "ADA/USDT",
}

LOOKBACK_DAYS = 90
EXEC_TIMEFRAME = "15m"
MACRO_TIMEFRAME = "4h"
MAX_DAILY_TRADES = 3  # محدودیت سخت‌گیرانه: حداکثر ۳ معامله در کل روز برای کل سیستم
MAX_POSITIONS = 2

SLIPPAGE = 0.0002
FEE_RATE = 0.0007

ATR_PERIOD = 14
INITIAL_ATR_MULTIPLIER = 1.6
TP_ATR_MULTIPLIER = 3.2  # ریسک به ریوارد دقیق 1 به 2
TIMEOUT_CANDLES = 20

INITIAL_CAPITAL = 2000.0
TRADE_MARGIN = 100.0
LEVERAGE = 20.0

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)

print("=" * 68)
print("HUNTER-V109 — STRICT GLOBAL CAP ENGINE INITIALIZED")
print("=" * 68)

processed_data = {}

def fetch_ohlcv_data(lbank_symbol, timeframe):
    all_ohlcv = []
    current_since = since_timestamp
    last_seen = None

    while current_since < exchange.milliseconds():
        batch = None
        for attempt in range(3):
            try:
                batch = exchange.fetch_ohlcv(
                    lbank_symbol, timeframe=timeframe, since=current_since, limit=1000
                )
                break
            except Exception:
                if attempt == 2:
                    return None
        if not batch:
            break

        last_ts = batch[-1][0]
        if last_seen is not None and last_ts <= last_seen:
            return None

        all_ohlcv.extend(batch)
        last_seen = last_ts
        current_since = last_ts + 1

        if len(batch) < 1000:
            break

    if not all_ohlcv:
        return None

    df = pd.DataFrame(all_ohlcv, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

for symbol, lbank_symbol in SYMBOLS.items():
    df_15m = fetch_ohlcv_data(lbank_symbol, EXEC_TIMEFRAME)
    df_4h = fetch_ohlcv_data(lbank_symbol, MACRO_TIMEFRAME)

    if df_15m is None or df_4h is None or len(df_15m) < 500 or len(df_4h) < 50:
        continue

    df_15m.set_index("Date", inplace=True)
    df_4h.set_index("Date", inplace=True)

    tr1 = df_15m["High"] - df_15m["Low"]
    tr2 = np.abs(df_15m["High"] - df_15m["Close"].shift(1))
    tr3 = np.abs(df_15m["Low"] - df_15m["Close"].shift(1))
    df_15m["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(ATR_PERIOD).mean()
    df_15m["EMA50"] = df_15m["Close"].ewm(span=50, adjust=False).mean()
    
    delta = df_15m["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df_15m["RSI"] = 100 - (100 / (1 + (gain / loss)))

    df_4h["EMA_Macro"] = df_4h["Close"].ewm(span=50, adjust=False).mean()
    df_15m["Macro_Trend"] = df_4h["EMA_Macro"].reindex(df_15m.index, method="ffill")
    df_15m["Macro_Close"] = df_4h["Close"].reindex(df_15m.index, method="ffill")

    df_15m.dropna(inplace=True)
    if len(df_15m) > 200:
        processed_data[symbol] = df_15m

print(f"Valid multi-timeframe symbols loaded: {len(processed_data)}")

def get_all_timestamps(data):
    return sorted({ts for df in data.values() for ts in df.index})

def run_backtest(processed_data):
    all_timestamps = get_all_timestamps(processed_data)
    active_positions = {}
    all_trades = []
    
    current_day = None
    daily_trade_count = 0
    loss_streaks_list = []
    current_loss_streak = 0

    for ts in all_timestamps:
        ts_date = ts.date()
        if current_day != ts_date:
            current_day = ts_date
            daily_trade_count = 0  # ریست شدن شمارشگر معاملات روزانه در شروع روز جدید

        symbols_to_close = []

        for symbol, pos in list(active_positions.items()):
            df = processed_data[symbol]
            if ts not in df.index:
                continue
            c15m = df.loc[ts]

            if pos["side"] == "LONG":
                hit_sl = c15m["Low"] <= pos["stop_loss"]
                hit_tp = c15m["High"] >= pos["take_profit"]
            else:
                hit_sl = c15m["High"] >= pos["stop_loss"]
                hit_tp = c15m["Low"] <= pos["take_profit"]

            curr_i = df.index.get_loc(ts)
            candles_held = curr_i - pos["entry_index"]
            is_timeout = candles_held >= TIMEOUT_CANDLES

            if not (hit_sl or hit_tp or is_timeout):
                continue

            initial_risk = pos["initial_risk"]
            if pos["side"] == "LONG":
                if hit_tp:
                    exit_p = pos["take_profit"]
                elif hit_sl:
                    exit_p = min(pos["stop_loss"], c15m["Open"])
                else:
                    exit_p = c15m["Close"]
                r_real = ((exit_p - pos["entry_price"]) / initial_risk) - (FEE_RATE * 2)
                price_return_pct = (exit_p - pos["entry_price"]) / pos["entry_price"]
            else:
                if hit_tp:
                    exit_p = pos["take_profit"]
                elif hit_sl:
                    exit_p = max(pos["stop_loss"], c15m["Open"])
                else:
                    exit_p = c15m["Close"]
                r_real = ((pos["entry_price"] - exit_p) / initial_risk) - (FEE_RATE * 2)
                price_return_pct = (pos["entry_price"] - exit_p) / pos["entry_price"]

            outcome = "WIN" if r_real > 0 else "LOSS"
            
            if outcome == "LOSS":
                current_loss_streak += 1
            else:
                if current_loss_streak > 0:
                    loss_streaks_list.append(current_loss_streak)
                current_loss_streak = 0

            position_notional = TRADE_MARGIN * LEVERAGE
            dollar_pnl = (position_notional * price_return_pct) - (position_notional * FEE_RATE * 2)

            all_trades.append({
                "Timestamp": ts,
                "Symbol": symbol,
                "Side": pos["side"],
                "Outcome": outcome,
                "Return": r_real,
                "Dollar_PnL": dollar_pnl,
            })
            
            symbols_to_close.append(symbol)

        for sym in symbols_to_close:
            del active_positions[sym]

        # اگر سهمیه ۳ معامله امروز پر شده است، ترید جدیدی باز نکن
        if daily_trade_count >= MAX_DAILY_TRADES:
            continue

        if len(active_positions) >= MAX_POSITIONS:
            continue

        for symbol, df in processed_data.items():
            if symbol in active_positions:
                continue
            if ts not in df.index:
                continue

            i = df.index.get_loc(ts)
            if i < 50:
                continue

            prev_c = df.iloc[i - 1]
            c15m = df.iloc[i]

            macro_bullish = prev_c["Macro_Close"] > prev_c["Macro_Trend"]
            macro_bearish = prev_c["Macro_Close"] < prev_c["Macro_Trend"]

            valid_long = macro_bullish and (prev_c["Close"] > prev_c["EMA50"]) and (53 < prev_c["RSI"] < 65) and (prev_c["Close"] > prev_c["Open"])
            valid_short = macro_bearish and (prev_c["Close"] < prev_c["EMA50"]) and (35 < prev_c["RSI"] < 47) and (prev_c["Close"] < prev_c["Open"])

            if not (valid_long or valid_short):
                continue

            side = "LONG" if valid_long else "SHORT"
            entry_price = c15m["Open"] * (1 + SLIPPAGE) if side == "LONG" else c15m["Open"] * (1 - SLIPPAGE)
            
            initial_atr = c15m["ATR"]
            if side == "LONG":
                initial_sl = entry_price - (INITIAL_ATR_MULTIPLIER * initial_atr)
                take_profit = entry_price + (TP_ATR_MULTIPLIER * initial_atr)
            else:
                initial_sl = entry_price + (INITIAL_ATR_MULTIPLIER * initial_atr)
                take_profit = entry_price - (TP_ATR_MULTIPLIER * initial_atr)

            initial_risk = abs(entry_price - initial_sl)
            sl_dist_pct = initial_risk / entry_price

            if not (0.008 <= sl_dist_pct <= 0.02):
                continue

            active_positions[symbol] = {
                "side": side,
                "entry_price": float(entry_price),
                "stop_loss": float(initial_sl),
                "take_profit": float(take_profit),
                "initial_risk": float(initial_risk),
                "entry_index": int(i),
            }

            daily_trade_count += 1
            break  # در هر کندل حداکثر یک پوزیشن باز شود تا کنترل کامل حفظ شود

    if current_loss_streak > 0:
        loss_streaks_list.append(current_loss_streak)

    return pd.DataFrame(all_trades), loss_streaks_list

if __name__ == "__main__":
    trades_df, loss_streaks = run_backtest(processed_data)
    n = len(trades_df)
    win_rate = (trades_df["Outcome"].eq("WIN").mean() * 100) if n > 0 else 0.0
    net_pnl = float(trades_df["Dollar_PnL"].sum()) if n > 0 else 0.0
    max_streak = max(loss_streaks) if loss_streaks else 0

    print("=" * 72)
    print("HUNTER-V109 BACKTEST RESULTS (STRICT GLOBAL DAILY CAP)")
    print("=" * 72)
    print(f"Total Trades: {n}")
    print(f"Overall Win Rate: {win_rate:.2f}%")
    print(f"Net Dollar PnL: ${net_pnl:,.2f}")
    print(f"Max Loss Streak: {max_streak}")
    print(f"List of Consecutive Loss Streaks (Occurrences): {loss_streaks}")
    print("=" * 72)
