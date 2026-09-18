import os
import subprocess
import sys
from datetime import datetime, timedelta
from collections import defaultdict, Counter

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd

# ============================================================
# HUNTER-V74 — CIRCUIT BREAKER & CORRELATION ENGINE (V14)
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

REMOVED_COINS = {
    "NEAR", "OP", "HYPE", "HBAR", "AVAX", "SUI", "PENDLE", "TIA",
    "FET", "SEI", "ARB", "DOT", "ETC", "SHIB", "STX", "RUNE",
    "MKR", "APT", "LTC", "AR", "IMX", "PEPE", "BONK",
}

SYMBOLS = {
    k: v for k, v in SYMBOLS.items()
    if k not in REMOVED_COINS
}

LOOKBACK_DAYS = 365
TIMEFRAME = "4h"
MAX_POSITIONS = 5

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

ATR_PERIOD = 14
TRAILING_ATR_MULTIPLIER = 2.0
INITIAL_ATR_MULTIPLIER = 1.8
TIMEOUT_CANDLES = 45
EMA_WARMUP = 200

INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 80.0

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)

print("=" * 68)
print("HUNTER-V74 — CIRCUIT BREAKER & CORRELATION ENGINE (V14)")
print("=" * 68)

processed_data = {}

def fetch_symbol_data(lbank_symbol):
    all_ohlcv = []
    current_since = since_timestamp
    last_seen = None

    while current_since < exchange.milliseconds():
        batch = None
        for attempt in range(3):
            try:
                batch = exchange.fetch_ohlcv(
                    lbank_symbol, timeframe=TIMEFRAME, since=current_since, limit=1000
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

    if len(df) >= 2:
        now_ms = exchange.milliseconds()
        last_ms = int(df.iloc[-1]["Date"].timestamp() * 1000)
        if last_ms + 4 * 60 * 60 * 1000 > now_ms:
            df = df.iloc[:-1].copy()

    if len(df) < EMA_WARMUP + 50:
        return None

    deltas = df["Date"].diff().dropna()
    if not deltas.empty and deltas.max() > pd.Timedelta(hours=4, minutes=10):
        return None

    tr1 = df["High"] - df["Low"]
    tr2 = np.abs(df["High"] - df["Close"].shift(1))
    tr3 = np.abs(df["Low"] - df["Close"].shift(1))

    df["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(ATR_PERIOD).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["Mom_Short"] = (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)
    df["Mom_Long"] = (df["Close"] - df["Close"].shift(30)) / df["Close"].shift(30)
    df.set_index("Date", inplace=True)
    return df

for symbol, lbank_symbol in SYMBOLS.items():
    df4h = fetch_symbol_data(lbank_symbol)
    if df4h is not None:
        processed_data[symbol] = df4h

print(f"Valid symbols: {len(processed_data)} / {len(SYMBOLS)}")

def get_all_timestamps(data):
    return sorted({ts for df in data.values() for ts in df.index})

def is_correlation_allowed(symbol, active_positions, processed_data, ts, threshold=0.75):
    if not active_positions:
        return True
    df_cand = processed_data[symbol]
    if ts not in df_cand.index:
        return False
    idx_cand = df_cand.index.get_loc(ts)
    if idx_cand < 30:
        return True
    
    cand_returns = df_cand['Close'].iloc[idx_cand-30:idx_cand+1].pct_change().dropna()

    for active_sym in active_positions:
        df_act = processed_data[active_sym]
        if ts not in df_act.index:
            continue
        idx_act = df_act.index.get_loc(ts)
        if idx_act < 30:
            continue
        act_returns = df_act['Close'].iloc[idx_act-30:idx_act+1].pct_change().dropna()

        aligned = pd.concat([cand_returns, act_returns], axis=1).dropna()
        if len(aligned) < 15:
            continue
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        if not np.isnan(corr) and corr > threshold:
            return False
    return True

def run_backtest_v14(processed_data, max_consecutive_losses_limit=2, cooldown_candles=6):
    all_timestamps = get_all_timestamps(processed_data)
    active_positions = {}
    all_trades = []
    consecutive_losses = 0
    cooldown_counter = 0

    for ts in all_timestamps:
        # کاهش تایمر انجماد در هر کندل جدید
        if cooldown_counter > 0:
            cooldown_counter -= 1

        symbols_to_close = []

        for symbol, pos in list(active_positions.items()):
            df = processed_data[symbol]
            if ts not in df.index:
                continue
            c4h = df.loc[ts]

            if pos["side"] == "LONG":
                if c4h["High"] > pos["highest_price"]:
                    pos["highest_price"] = c4h["High"]
                    new_trailing_sl = pos["highest_price"] - TRAILING_ATR_MULTIPLIER * c4h["ATR"]
                    if new_trailing_sl > pos["stop_loss"]:
                        pos["stop_loss"] = new_trailing_sl
                hit_sl = c4h["Low"] <= pos["stop_loss"]
            else:
                if c4h["Low"] < pos["lowest_price"]:
                    pos["lowest_price"] = c4h["Low"]
                    new_trailing_sl = pos["lowest_price"] + TRAILING_ATR_MULTIPLIER * c4h["ATR"]
                    if new_trailing_sl < pos["stop_loss"]:
                        pos["stop_loss"] = new_trailing_sl
                hit_sl = c4h["High"] >= pos["stop_loss"]

            curr_i = df.index.get_loc(ts)
            candles_held = curr_i - pos["entry_index"]
            is_timeout = candles_held >= TIMEOUT_CANDLES

            if not (hit_sl or is_timeout):
                continue

            initial_risk = pos["initial_risk"]
            if pos["side"] == "LONG":
                exit_p = min(pos["stop_loss"], c4h["Open"]) if hit_sl else c4h["Close"]
                r_real = ((exit_p - pos["entry_price"]) / initial_risk) - (FEE_RATE * 2)
                price_return_pct = (exit_p - pos["entry_price"]) / pos["entry_price"]
            else:
                exit_p = max(pos["stop_loss"], c4h["Open"]) if hit_sl else c4h["Close"]
                r_real = ((pos["entry_price"] - exit_p) / initial_risk) - (FEE_RATE * 2)
                price_return_pct = (pos["entry_price"] - exit_p) / pos["entry_price"]

            outcome = "WIN" if r_real > 0 else "LOSS"
            position_notional = TRADE_MARGIN * LEVERAGE
            dollar_pnl = (position_notional * price_return_pct) - (position_notional * FEE_RATE * 2)

            # آپدیت مدارشکن پس از بسته شدن معامله
            if outcome == "LOSS":
                consecutive_losses += 1
                if consecutive_losses >= max_consecutive_losses_limit:
                    cooldown_counter = cooldown_candles # ربات برای چند کندل آینده منجمد می‌شود
            else:
                consecutive_losses = 0

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

        # اگر مدارشکن فعال باشد، اجازه ثبت سیگنال جدید داده نمی‌شود
        if cooldown_counter > 0:
            continue

        market_bull = True
        if "BTC" in processed_data and ts in processed_data["BTC"].index:
            btc_c = processed_data["BTC"].loc[ts]
            market_bull = btc_c["Close"] > btc_c["EMA200"]

        bullish_count = 0
        total_active_syms = 0
        for symbol, df in processed_data.items():
            if ts not in df.index:
                continue
            total_active_syms += 1
            if df.loc[ts, "Close"] > df.loc[ts, "EMA200"]:
                bullish_count += 1

        market_breadth_ratio = bullish_count / total_active_syms if total_active_syms > 0 else 0.5
        allow_longs = market_breadth_ratio >= 0.35
        allow_shorts = market_breadth_ratio <= 0.65

        # استخراج نامزدهای ورود با فیلتر همبستگی
        current_scores = {}
        for symbol, df in processed_data.items():
            if ts not in df.index:
                continue
            val = df.loc[ts, "Mom_Long"]
            if not np.isnan(val):
                current_scores[symbol] = float(val)

        if not current_scores:
            continue

        rev_bool = bool(market_bull)
        ranked_symbols = sorted(current_scores.keys(), key=lambda x: current_scores[x], reverse=rev_bool)
        candidates = []

        for symbol in ranked_symbols:
            if symbol in active_positions:
                continue
            if not is_correlation_allowed(symbol, active_positions, processed_data, ts, threshold=0.75):
                continue

            df = processed_data[symbol]
            if ts not in df.index:
                continue

            i = df.index.get_loc(ts)
            if i < EMA_WARMUP + 1:
                continue

            c4h = df.iloc[i]
            prev_c = df.iloc[i - 1]

            if market_bull:
                if not allow_longs:
                    continue
                regime_ok = c4h["Close"] > c4h["EMA20"] and c4h["EMA20"] > c4h["EMA50"] and c4h["Close"] > c4h["EMA200"]
                pullback_ok = prev_c["Low"] <= prev_c["EMA20"] * 1.015
                valid_signal = regime_ok and pullback_ok and (c4h["Mom_Short"] > 0.012) and (c4h["Mom_Long"] > 0.035)
                side = "LONG"
            else:
                if not allow_shorts:
                    continue
                regime_ok = c4h["Close"] < c4h["EMA20"] and c4h["EMA20"] < c4h["EMA50"] and c4h["Close"] < c4h["EMA200"]
                pullback_ok = prev_c["High"] >= prev_c["EMA20"] * 0.985
                valid_signal = regime_ok and pullback_ok and (c4h["Mom_Short"] < -0.012) and (c4h["Mom_Long"] < -0.035)
                side = "SHORT"

            if not valid_signal:
                continue

            entry_price = c4h["Open"] * (1 + SLIPPAGE) if side == "LONG" else c4h["Open"] * (1 - SLIPPAGE)
            initial_sl = entry_price - INITIAL_ATR_MULTIPLIER * c4h["ATR"] if side == "LONG" else entry_price + INITIAL_ATR_MULTIPLIER * c4h["ATR"]
            initial_risk = abs(entry_price - initial_sl)
            sl_dist_pct = initial_risk / entry_price

            if not (0.01 <= sl_dist_pct <= 0.04):
                continue

            candidates.append({
                "symbol": symbol,
                "side": side,
                "entry_price": float(entry_price),
                "initial_sl": float(initial_sl),
                "initial_risk": float(initial_risk),
                "entry_index": int(i),
            })

        if not candidates:
            continue

        slots = MAX_POSITIONS - len(active_positions)
        if slots <= 0:
            continue

        selected = candidates[:slots]
        for candidate in selected:
            active_positions[candidate["symbol"]] = {
                "side": candidate["side"],
                "entry_price": candidate["entry_price"],
                "initial_stop": candidate["initial_sl"],
                "entry_timestamp": ts,
                "stop_loss": candidate["initial_sl"],
                "highest_price": candidate["entry_price"],
                "lowest_price": candidate["entry_price"],
                "initial_risk": candidate["initial_risk"],
                "entry_index": candidate["entry_index"],
            }

    return pd.DataFrame(all_trades)

if __name__ == "__main__":
    base_trades = run_backtest_v14(processed_data, max_consecutive_losses_limit=99, cooldown_candles=0)
    v14_trades = run_backtest_v14(processed_data, max_consecutive_losses_limit=2, cooldown_candles=6)

    def stats(df):
        n = len(df)
        wr = (df["Outcome"].eq("WIN").mean() * 100) if n else 0
        pnl = float(df["Dollar_PnL"].sum()) if n else 0
        cur = mx = 0
        for x in df["Outcome"]:
            if x == "LOSS":
                cur += 1
                mx = max(mx, cur)
            else:
                cur = 0
        return n, wr, pnl, mx

    a = stats(base_trades)
    b = stats(v14_trades)

    print("=" * 72)
    print("HUNTER-V74 — CIRCUIT BREAKER ENGINE RESULTS (V14)")
    print("=" * 72)
    print(f"V74 اصلی         | Trades={a[0]} | WR={a[1]:.2f}% | PnL=${a[2]:,.2f} | MaxLS={a[3]}")
    print(f"V14 مدارشکن هوشمند | Trades={b[0]} | WR={b[1]:.2f}% | PnL=${b[2]:,.2f} | MaxLS={b[3]}")
