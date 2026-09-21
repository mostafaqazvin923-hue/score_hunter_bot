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

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.ensemble import RandomForestClassifier

# ============================================================
# HUNTER-V110 — MACHINE LEARNING QUANT ENGINE (WALK-FORWARD)
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
MAX_DAILY_TRADES = 3

SLIPPAGE = 0.0002
FEE_RATE = 0.0007

ATR_PERIOD = 14
INITIAL_ATR_MULTIPLIER = 1.6
TP_ATR_MULTIPLIER = 3.2
TIMEOUT_CANDLES = 20

INITIAL_CAPITAL = 2000.0
TRADE_MARGIN = 100.0
LEVERAGE = 20.0

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)

print("=" * 68)
print("HUNTER-V110 — MACHINE LEARNING QUANT ENGINE INITIALIZED")
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

    # مهندسی ویژگی‌ها (Feature Engineering) برای یادگیری ماشین
    tr1 = df_15m["High"] - df_15m["Low"]
    tr2 = np.abs(df_15m["High"] - df_15m["Close"].shift(1))
    tr3 = np.abs(df_15m["Low"] - df_15m["Close"].shift(1))
    df_15m["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(ATR_PERIOD).mean()
    
    df_15m["Return_1"] = df_15m["Close"].pct_change(1)
    df_15m["Return_4"] = df_15m["Close"].pct_change(4)
    df_15m["Vol_Ratio"] = df_15m["Volume"] / df_15m["Volume"].rolling(24).mean()
    
    delta = df_15m["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df_15m["RSI"] = 100 - (100 / (1 + (gain / loss)))

    df_4h["EMA_Macro"] = df_4h["Close"].ewm(span=50, adjust=False).mean()
    df_15m["Macro_Trend_Diff"] = (df_4h["Close"] - df_4h["EMA_Macro"]).reindex(df_15m.index, method="ffill")

    # هدف یادگیری ماشین: آیا سود آینده بیشتر از کارمزد و ریسک است؟
    future_return = df_15m["Close"].shift(-4) / df_15m["Close"] - 1
    df_15m["Target_Long"] = (future_return > 0.003).astype(int)
    df_15m["Target_Short"] = (future_return < -0.003).astype(int)

    df_15m.dropna(inplace=True)
    if len(df_15m) > 200:
        processed_data[symbol] = df_15m

print(f"Valid ML symbols loaded: {len(processed_data)}")

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

    feature_cols = ["Return_1", "Return_4", "Vol_Ratio", "RSI", "Macro_Trend_Diff", "ATR"]

    # آموزش مدل به روش Walk-Forward (آموزش روی داده‌های گذشته، پیش‌بینی آینده)
    models_long = {}
    models_short = {}

    for symbol, df in processed_data.items():
        # آموزش اولیه روی ۵۰ درصد اول داده‌ها
        split_idx = int(len(df) * 0.4)
        train_df = df.iloc[:split_idx]
        
        if len(train_df) > 100:
            X_train = train_df[feature_cols]
            y_long = train_df["Target_Long"]
            y_short = train_df["Target_Short"]
            
            clf_l = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
            clf_l.fit(X_train, y_long)
            models_long[symbol] = (clf_l, split_idx)

            clf_s = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
            clf_s.fit(X_train, y_short)
            models_short[symbol] = (clf_s, split_idx)

    for ts in all_timestamps:
        ts_date = ts.date()
        if current_day != ts_date:
            current_day = ts_date
            daily_trade_count = 0

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

        if daily_trade_count >= MAX_DAILY_TRADES:
            continue

        if len(active_positions) >= 2:
            continue

        for symbol, df in processed_data.items():
            if symbol in active_positions:
                continue
            if ts not in df.index:
                continue
            if symbol not in models_long:
                continue

            i = df.index.get_loc(ts)
            split_idx = models_long[symbol][1]
            if i <= split_idx:
                continue  # فقط روی داده‌های تست بعد از مدل ترید کن

            c15m = df.iloc[i]
            features_vector = c15m[feature_cols].values.reshape(1, -1)

            pred_long = models_long[symbol][0].predict(features_vector)[0]
            pred_short = models_short[symbol][0].predict(features_vector)[0]

            if not (pred_long == 1 or pred_short == 1):
                continue

            side = "LONG" if pred_long == 1 else "SHORT"
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
            break

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
    print("HUNTER-V110 BACKTEST RESULTS (MACHINE LEARNING)")
    print("=" * 72)
    print(f"Total Trades: {n}")
    print(f"Overall Win Rate: {win_rate:.2f}%")
    print(f"Net Dollar PnL: ${net_pnl:,.2f}")
    print(f"Max Loss Streak: {max_streak}")
    print(f"List of Consecutive Loss Streaks (Occurrences): {loss_streaks}")
    print("=" * 72)
