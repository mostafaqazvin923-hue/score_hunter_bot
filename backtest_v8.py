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
# HUNTER-V96 - INSTITUTIONAL QUANTITATIVE MOMENTUM ENGINE
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
TRAILING_ATR_MULTIPLIER = 2.4
INITIAL_ATR_MULTIPLIER = 2.0
TIMEOUT_CANDLES = 40
EMA_WARMUP = 200

INITIAL_CAPITAL = 1000.0
BASE_TRADE_MARGIN = 100.0
LEVERAGE = 80.0

OUTPUT_DIR = "hunter_v96_output"

start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)

print("=" * 68)
print("HUNTER-V96 - INSTITUTIONAL QUANTITATIVE MOMENTUM ENGINE")
print("=" * 68)


# ============================================================
# DATA & QUANTITATIVE INDICATORS
# ============================================================

def fetch_symbol_data(lbank_symbol):
    all_ohlcv = []
    current_since = since_timestamp
    last_seen = None

    while current_since < exchange.milliseconds():
        batch = None
        for attempt in range(3):
            try:
                batch = exchange.fetch_ohlcv(
                    lbank_symbol,
                    timeframe=TIMEFRAME,
                    since=current_since,
                    limit=1000,
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

    df = pd.DataFrame(
        all_ohlcv,
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
    )
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
    df["ATR_Pct"] = df["ATR"] / df["Close"]

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    df["Mom_Short"] = (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)
    df["Mom_Long"] = (df["Close"] - df["Close"].shift(30)) / df["Close"].shift(30)
    
    # اضافه شدن شاخص حجم نسبی برای فیلتر کردن فیک‌بریک‌اوت‌ها
    df["Volume_SMA"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]

    df.set_index("Date", inplace=True)
    return df


processed_data = {}
for symbol, lbank_symbol in SYMBOLS.items():
    df4h = fetch_symbol_data(lbank_symbol)
    if df4h is not None:
        processed_data[symbol] = df4h


# ============================================================
# BACKTEST ENGINE
# ============================================================

def get_all_timestamps(data):
    return sorted({ts for df in data.values() for ts in df.index})


def run_backtest(processed_data):
    all_timestamps = get_all_timestamps(processed_data)
    active_positions = {}
    all_trades = []
    equity_curve = []

    for ts in all_timestamps:
        symbols_to_close = []

        for symbol, pos in list(active_positions.items()):
            df = processed_data[symbol]
            if ts not in df.index:
                continue

            c4h = df.loc[ts]

            if pos["side"] == "LONG":
                if not pos["be_triggered"] and c4h["High"] >= pos["entry_price"] + (pos["initial_risk"] * 1.5):
                    pos["stop_loss"] = max(pos["stop_loss"], pos["entry_price"])
                    pos["be_triggered"] = True

                if c4h["High"] > pos["highest_price"]:
                    pos["highest_price"] = c4h["High"]
                    new_trailing_sl = pos["highest_price"] - TRAILING_ATR_MULTIPLIER * c4h["ATR"]
                    if new_trailing_sl > pos["stop_loss"]:
                        pos["stop_loss"] = new_trailing_sl
                hit_sl = c4h["Low"] <= pos["stop_loss"]
            else:
                if not pos["be_triggered"] and c4h["Low"] <= pos["entry_price"] - (pos["initial_risk"] * 1.5):
                    pos["stop_loss"] = min(pos["stop_loss"], pos["entry_price"])
                    pos["be_triggered"] = True

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
            margin = pos["margin"]

            if pos["side"] == "LONG":
                exit_p = min(pos["stop_loss"], c4h["Open"]) if hit_sl else c4h["Close"]
                r_real = ((exit_p - pos["entry_price"]) / initial_risk) - (FEE_RATE * 2)
                price_return_pct = (exit_p - pos["entry_price"]) / pos["entry_price"]
            else:
                exit_p = max(pos["stop_loss"], c4h["Open"]) if hit_sl else c4h["Close"]
                r_real = ((pos["entry_price"] - exit_p) / initial_risk) - (FEE_RATE * 2)
                price_return_pct = (pos["entry_price"] - exit_p) / pos["entry_price"]

            is_be_protected = pos["be_triggered"] and (abs(exit_p - pos["entry_price"]) / pos["entry_price"] < 0.003)
            outcome = "WIN" if r_real > 0 else ("BE" if is_be_protected else "LOSS")

            position_notional = margin * LEVERAGE
            dollar_pnl = (position_notional * price_return_pct) - (position_notional * FEE_RATE * 2)

            all_trades.append({
                "Timestamp": ts,
                "Symbol": symbol,
                "Side": pos["side"],
                "Outcome": outcome,
                "Return": r_real,
                "Dollar_PnL": dollar_pnl,
                "ExitOrder": len(all_trades),
                "EntryTimestamp": pos.get("entry_timestamp", pd.NaT),
                "EntryPrice": pos["entry_price"],
                "InitialStop": pos["initial_stop"],
                "ExitPrice": exit_p,
            })
            symbols_to_close.append(symbol)

        for sym in symbols_to_close:
            del active_positions[sym]

        market_bull = True
        if "BTC" in processed_data and ts in processed_data["BTC"].index:
            btc_c = processed_data["BTC"].loc[ts]
            market_bull = btc_c["Close"] > btc_c["EMA200"]

        current_scores = {}
        for symbol, df in processed_data.items():
            if ts not in df.index:
                continue
            val = df.loc[ts, "Mom_Long"]
            if not np.isnan(val):
                current_scores[symbol] = val

        if not current_scores:
            continue

        ranked_symbols = sorted(
            current_scores.keys(),
            key=lambda x: float(current_scores[x]),
            reverse=market_bull,
        )

        candidates = []
        for symbol in ranked_symbols:
            if symbol in active_positions:
                continue

            df = processed_data[symbol]
            if ts not in df.index:
                continue

            i = df.index.get_loc(ts)
            if i < EMA_WARMUP + 1:
                continue

            c4h = df.iloc[i]
            prev_c = df.iloc[i - 1]

            if c4h["ATR_Pct"] > 0.08 or c4h["ATR_Pct"] < 0.004:
                continue

            # فیلتر سخت‌گیرانه‌تر حجم برای کاهش استریک باخت در بازارهای کم‌عمق
            if c4h["Volume_Ratio"] < 0.7:
                continue

            if market_bull:
                regime_ok = (c4h["Close"] > c4h["EMA20"] and c4h["EMA20"] > c4h["EMA50"] and c4h["Close"] > c4h["EMA200"])
                pullback_ok = prev_c["Low"] <= prev_c["EMA20"] * 1.015
                valid_signal = regime_ok and pullback_ok and (c4h["Mom_Short"] > 0.015) and (c4h["Mom_Long"] > 0.04)
                side = "LONG"
            else:
                regime_ok = (c4h["Close"] < c4h["EMA20"] and c4h["EMA20"] < c4h["EMA50"] and c4h["Close"] < c4h["EMA200"])
                pullback_ok = prev_c["High"] >= prev_c["EMA20"] * 0.985
                valid_signal = regime_ok and pullback_ok and (c4h["Mom_Short"] < -0.015) and (c4h["Mom_Long"] < -0.04)
                side = "SHORT"

            if not valid_signal:
                continue

            entry_price = c4h["Open"] * (1 + SLIPPAGE) if side == "LONG" else c4h["Open"] * (1 - SLIPPAGE)
            initial_sl = entry_price - INITIAL_ATR_MULTIPLIER * c4h["ATR"] if side == "LONG" else entry_price + INITIAL_ATR_MULTIPLIER * c4h["ATR"]
            initial_risk = abs(entry_price - initial_sl)
            sl_dist_pct = initial_risk / entry_price

            if not (0.012 <= sl_dist_pct <= 0.045):
                continue

            target_volatility_benchmark = 0.03
            volatility_scalar = target_volatility_benchmark / max(c4h["ATR_Pct"], 0.01)
            volatility_scalar = np.clip(volatility_scalar, 0.5, 1.8)
            dynamic_margin = BASE_TRADE_MARGIN * volatility_scalar

            candidates.append({
                "symbol": symbol,
                "side": side,
                "entry_price": float(entry_price),
                "initial_sl": float(initial_sl),
                "initial_risk": float(initial_risk),
                "entry_index": int(i),
                "margin": float(dynamic_margin),
            })

        slots = MAX_POSITIONS - len(active_positions)
        if slots <= 0 or not candidates:
            continue

        selected = candidates[:slots]
        for candidate in selected:
            symbol = candidate["symbol"]
            active_positions[symbol] = {
                "side": candidate["side"],
                "entry_price": candidate["entry_price"],
                "initial_stop": candidate["initial_sl"],
                "entry_timestamp": ts,
                "stop_loss": candidate["initial_sl"],
                "highest_price": candidate["entry_price"],
                "lowest_price": candidate["entry_price"],
                "initial_risk": candidate["initial_risk"],
                "margin": candidate["margin"],
                "entry_index": candidate["entry_index"],
                "be_triggered": False,
            }

        closed_pnl = sum(trade["Dollar_PnL"] for trade in all_trades)
        unrealized = 0.0
        for pos_symbol, pos in active_positions.items():
            df = processed_data[pos_symbol]
            if ts not in df.index:
                continue
            close_price = df.loc[ts, "Close"]
            if pos["side"] == "LONG":
                unrealized += (close_price - pos["entry_price"]) * (pos["margin"] * LEVERAGE / pos["entry_price"])
            else:
                unrealized += (pos["entry_price"] - close_price) * (pos["margin"] * LEVERAGE / pos["entry_price"])

        equity_curve.append({
            "Timestamp": ts,
            "Equity": INITIAL_CAPITAL + closed_pnl + unrealized,
        })

    return pd.DataFrame(all_trades), pd.DataFrame(equity_curve)


# ============================================================
# REPORTING
# ============================================================

def calculate_loss_streaks(trades_df):
    if trades_df.empty:
        return 0, []
    ordered = trades_df.sort_values(["Timestamp", "ExitOrder"], kind="stable")
    current = 0
    maximum = 0
    sequences = []
    for outcome in ordered["Outcome"]:
        if outcome == "LOSS":
            current += 1
            maximum = max(maximum, current)
        else:
            if current > 0:
                sequences.append(current)
            current = 0
    if current > 0:
        sequences.append(current)
    return maximum, sequences


def calculate_drawdown(equity_df):
    if equity_df.empty:
        return 0.0, 0.0
    equity = equity_df["Equity"].astype(float)
    peak = equity.cummax()
    dd = equity - peak
    max_dd = float(dd.min())
    if max_dd >= 0:
        return 0.0, 0.0
    # اصلاح فرمول درودان درصدی بر اساس قله‌ی متحرک اکویتی
    dd_pct_series = (dd / peak) * 100.0
    max_dd_pct = float(dd_pct_series.min())
    return max_dd, max_dd_pct


def report(name, trades_df, equity_df):
    print("\n" + "=" * 68)
    print(name)
    print("=" * 68)
    if trades_df.empty:
        print("No trades.")
        return {}

    trades_df = trades_df.sort_values(["Timestamp", "ExitOrder"], kind="stable").reset_index(drop=True)
    total = len(trades_df)
    wins = int((trades_df["Outcome"] == "WIN").sum())
    be_count = int((trades_df["Outcome"] == "BE").sum())
    losses = int((trades_df["Outcome"] == "LOSS").sum())
    wr = (wins / total * 100.0) if total > 0 else 0.0

    net_r = float(trades_df["Return"].sum())
    pnl = float(trades_df["Dollar_PnL"].sum())
    final_capital = INITIAL_CAPITAL + pnl
    return_pct = (final_capital / INITIAL_CAPITAL - 1.0) * 100.0

    max_ls, _ = calculate_loss_streaks(trades_df)
    dd_dollar, dd_pct = calculate_drawdown(equity_df)

    print(f"Trades        : {total}")
    print(f"Wins          : {wins}")
    print(f"BE (BreakEven): {be_count}")
    print(f"Losses        : {losses}")
    print(f"Win Rate      : {wr:.2f}%")
    print(f"Net R         : {net_r:.2f}R")
    print(f"Net PnL       : ${pnl:,.2f}")
    print(f"Final Capital : ${final_capital:,.2f}")
    print(f"Return        : {return_pct:.2f}%")
    print(f"Max Drawdown  : ${dd_dollar:,.2f} ({dd_pct:.2f}%)")
    print(f"Max Loss Streak: {max_ls}")


if __name__ == "__main__":
    trades_df, equity_df = run_backtest(processed_data)
    report("HUNTER-V96 REPORT", trades_df, equity_df)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not trades_df.empty:
        trades_df.to_csv(os.path.join(OUTPUT_DIR, "optimized_trades.csv"), index=False)
    if not equity_df.empty:
        equity_df.to_csv(os.path.join(OUTPUT_DIR, "optimized_equity.csv"), index=False)
    print(f"\nOptimization Complete. Results saved in '{OUTPUT_DIR}/'.")
