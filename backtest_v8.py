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
# HUNTER-V85 — INSTITUTIONAL MARKET STRUCTURE & ORDER FLOW
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

LOOKBACK_DAYS = 365
TIMEFRAME = "4h"
MAX_POSITIONS = 5

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

ATR_PERIOD = 14
INITIAL_ATR_MULTIPLIER = 1.5
TP_ATR_MULTIPLIER = 3.0  # ریوارد ۳ به ۱ برای جبران وین‌ریت
TIMEOUT_CANDLES = 30
EMA_WARMUP = 200

TRADE_MARGIN = 100.0
LEVERAGE = 10.0

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)

print("=" * 68)
print("HUNTER-V85 — MARKET STRUCTURE & VOLUME FLOW ENGINE")
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
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    
    # شناسایی سقف و کف‌های محلی برای ساختار بازار
    df["Highest_20"] = df["High"].rolling(20).max()
    df["Lowest_20"] = df["Low"].rolling(20).min()
    
    # فیلتر حجم نوسانی نهنگ‌ها
    df["Volume_MA"] = df["Volume"].rolling(20).mean()
    df["Volume_Spike"] = df["Volume"] > (df["Volume_MA"] * 1.3)

    df.set_index("Date", inplace=True)
    return df

for symbol, lbank_symbol in SYMBOLS.items():
    df4h = fetch_symbol_data(lbank_symbol)
    if df4h is not None:
        processed_data[symbol] = df4h

print(f"Valid institutional symbols loaded: {len(processed_data)}")

def get_all_timestamps(data):
    return sorted({ts for df in data.values() for ts in df.index})

def run_market_structure_backtest(processed_data):
    all_timestamps = get_all_timestamps(processed_data)
    active_positions = {}
    all_trades = []
    
    consecutive_losses = 0
    cooldown_counter = 0

    for ts in all_timestamps:
        if cooldown_counter > 0:
            cooldown_counter -= 1
            continue

        symbols_to_close = []

        for symbol, pos in list(active_positions.items()):
            df = processed_data[symbol]
            if ts not in df.index:
                continue
            c4h = df.loc[ts]

            # بررسی برخورد با حد ضرر یا حد سود (TP/SL)
            if pos["side"] == "LONG":
                hit_sl = c4h["Low"] <= pos["stop_loss"]
                hit_tp = c4h["High"] >= pos["take_profit"]
            else:
                hit_sl = c4h["High"] >= pos["stop_loss"]
                hit_tp = c4h["Low"] <= pos["take_profit"]

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
                    exit_p = min(pos["stop_loss"], c4h["Open"])
                else:
                    exit_p = c4h["Close"]
                r_real = ((exit_p - pos["entry_price"]) / initial_risk) - (FEE_RATE * 2)
                price_return_pct = (exit_p - pos["entry_price"]) / pos["entry_price"]
            else:
                if hit_tp:
                    exit_p = pos["take_profit"]
                elif hit_sl:
                    exit_p = max(pos["stop_loss"], c4h["Open"])
                else:
                    exit_p = c4h["Close"]
                r_real = ((pos["entry_price"] - exit_p) / initial_risk) - (FEE_RATE * 2)
                price_return_pct = (pos["entry_price"] - exit_p) / pos["entry_price"]

            outcome = "WIN" if r_real > 0 else "LOSS"
            
            if outcome == "LOSS":
                consecutive_losses += 1
                if consecutive_losses >= 3:
                    cooldown_counter = 6
            else:
                consecutive_losses = 0

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

        if consecutive_losses >= 3:
            continue

        # سیگنال ساختار بازار و جریان سفارشات (بدون نگاه به آینده - روی کندل i-1)
        for symbol, df in processed_data.items():
            if symbol in active_positions:
                continue
            if ts not in df.index:
                continue

            i = df.index.get_loc(ts)
            if i < EMA_WARMUP + 1:
                continue

            prev_c = df.iloc[i - 1]
            c4h = df.iloc[i]

            # شرایط ساختار لانگ: روند صعودی میان‌مدت + حجم بالا + شکست مقاومت محلی
            valid_long = (
                prev_c["Close"] > prev_c["EMA50"] and
                prev_c["EMA50"] > prev_c["EMA200"] and
                prev_c["Volume_Spike"] and
                prev_c["Close"] >= prev_c["Highest_20"] * 0.99
            )

            # شرایط ساختار شورت: روند نزولی میان‌مدت + حجم بالا + شکست حمایت محلی
            valid_short = (
                prev_c["Close"] < prev_c["EMA50"] and
                prev_c["EMA50"] < prev_c["EMA200"] and
                prev_c["Volume_Spike"] and
                prev_c["Close"] <= prev_c["Lowest_20"] * 1.01
            )

            if not (valid_long or valid_short):
                continue

            side = "LONG" if valid_long else "SHORT"
            
            entry_price = c4h["Open"] * (1 + SLIPPAGE) if side == "LONG" else c4h["Open"] * (1 - SLIPPAGE)
            
            if side == "LONG":
                initial_sl = entry_price - INITIAL_ATR_MULTIPLIER * c4h["ATR"]
                take_profit = entry_price + (INITIAL_ATR_MULTIPLIER * TP_ATR_MULTIPLIER) * c4h["ATR"]
            else:
                initial_sl = entry_price + INITIAL_ATR_MULTIPLIER * c4h["ATR"]
                take_profit = entry_price - (INITIAL_ATR_MULTIPLIER * TP_ATR_MULTIPLIER) * c4h["ATR"]

            initial_risk = abs(entry_price - initial_sl)
            sl_dist_pct = initial_risk / entry_price

            if not (0.012 <= sl_dist_pct <= 0.055):
                continue

            if len(active_positions) >= MAX_POSITIONS:
                break

            active_positions[symbol] = {
                "side": side,
                "entry_price": float(entry_price),
                "stop_loss": float(initial_sl),
                "take_profit": float(take_profit),
                "initial_risk": float(initial_risk),
                "entry_index": int(i),
            }

    return pd.DataFrame(all_trades)

if __name__ == "__main__":
    trades_df = run_market_structure_backtest(processed_data)
    n = len(trades_df)
    wr = (trades_df["Outcome"].eq("WIN").mean() * 100) if n else 0
    pnl = float(trades_df["Dollar_PnL"].sum()) if n else 0
    
    longs_count = len(trades_df[trades_df["Side"] == "LONG"]) if n else 0
    shorts_count = len(trades_df[trades_df["Side"] == "SHORT"]) if n else 0

    mx = cur = 0
    if n > 0:
        for x in trades_df["Outcome"]:
            if x == "LOSS":
                cur += 1
                mx = max(mx, cur)
            else:
                cur = 0

    print("=" * 72)
    print("MARKET STRUCTURE ENGINE BACKTEST RESULTS")
    print("=" * 72)
    print(f"Total Trades: {n} (Longs: {longs_count} | Shorts: {shorts_count})")
    print(f"Win Rate: {wr:.2f}% | Net PnL: ${pnl:,.2f} | Max Loss Streak: {mx}")
