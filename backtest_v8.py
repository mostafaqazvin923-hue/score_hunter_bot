import os
import subprocess
import sys
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "ccxt"
    ])
    import ccxt

import numpy as np
import pandas as pd


# ============================================================
# EXCHANGE & UNIVERSE SETTINGS
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True
})

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
    "NEAR", "OP", "HYPE", "HBAR", "AVAX",
    "SUI", "PENDLE", "TIA", "FET", "SEI",
    "ARB", "DOT", "ETC", "SHIB", "STX",
    "RUNE", "MKR", "APT", "LTC", "AR",
    "IMX", "PEPE", "BONK",
}

SYMBOLS = {
    symbol: market
    for symbol, market in SYMBOLS.items()
    if symbol not in REMOVED_COINS
}


# ============================================================
# BACKTEST & RISK SETTINGS
# ============================================================

LOOKBACK_DAYS = 365
TIMEFRAME = "4h"

MAX_POSITIONS = 5

SLIPPAGE = 0.0003
FEE_RATE = 0.0007

ATR_PERIOD = 14
INITIAL_ATR_MULTIPLIER = 1.8
TRAILING_ATR_MULTIPLIER = 2.0

TIMEOUT_CANDLES = 45
EMA_WARMUP = 200

INITIAL_CAPITAL = 1000.0
BASE_MARGIN_PCT = 0.02
MAX_TOTAL_MARGIN_PCT = 0.12
LEVERAGE = 30.0


start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)


print("=" * 68)
print("📥 دریافت داده‌ها - HUNTER-V74.3")
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
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["ATR"] = true_range.rolling(ATR_PERIOD).mean()
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


# ============================================================
# CAPITAL & RISK MANAGEMENT UTILS
# ============================================================

def calculate_trade_margin(equity, base_margin_pct, max_total_margin_pct, current_margin_used):
    if equity <= 0:
        return 0.0
    proposed_margin = equity * base_margin_pct
    max_total_margin = equity * max_total_margin_pct
    remaining_margin = max_total_margin - current_margin_used
    if remaining_margin <= 0:
        return 0.0
    return float(min(proposed_margin, remaining_margin))


def get_current_margin_used(active_positions):
    return sum(pos["margin"] for pos in active_positions.values())


def calculate_dollar_pnl(side, entry_price, exit_price, notional):
    if entry_price <= 0 or exit_price <= 0 or notional <= 0:
        return 0.0
    if side == "LONG":
        price_return_pct = (exit_price - entry_price) / entry_price
    else:
        price_return_pct = (entry_price - exit_price) / entry_price
    gross_pnl = notional * price_return_pct
    fees = notional * FEE_RATE * 2
    return float(gross_pnl - fees)


def calculate_r_result(side, entry_price, exit_price, initial_risk, sl_dist_pct):
    if entry_price <= 0 or exit_price <= 0 or initial_risk <= 0 or sl_dist_pct <= 0:
        return 0.0, 0.0, 0.0
    if side == "LONG":
        raw_r = (exit_price - entry_price) / initial_risk
    else:
        raw_r = (entry_price - exit_price) / initial_risk
    total_fee_pct = FEE_RATE * 2
    fee_r = total_fee_pct / sl_dist_pct
    net_r = raw_r - fee_r
    return float(net_r), float(raw_r), float(fee_r)


def validate_stop_distance(entry, stop):
    dist_pct = abs(entry - stop) / entry
    is_valid = 0.01 <= dist_pct <= 0.04
    return is_valid, dist_pct


def get_current_equity(realized_pnl):
    return float(INITIAL_CAPITAL + realized_pnl)


def capital_safety_check(equity):
    return equity > 0


def calculate_capital_statistics(trades_df):
    if trades_df.empty:
        return {"MaxDrawdownPct": 0.0}
    cum_pnl = trades_df["Dollar_PnL"].cumsum()
    equity = INITIAL_CAPITAL + cum_pnl
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max * 100
    max_dd_pct = dd.min() if not dd.empty else 0.0
    return {"MaxDrawdownPct": abs(max_dd_pct)}


def calculate_side_statistics(trades_df):
    sides = {}
    for side in ["LONG", "SHORT"]:
        sub = trades_df[trades_df["Side"] == side]
        if sub.empty:
            sides[side] = "No trades"
        else:
            wr = (sub["Outcome"] == "WIN").mean() * 100
            pnl = sub["Dollar_PnL"].sum()
            sides[side] = f"Trades: {len(sub)} | WR: {wr:.2f}% | PnL: ${pnl:,.2f}"
    return sides


def calculate_loss_streak(trades_df):
    if trades_df.empty:
        return 0
    outcomes = (trades_df["Outcome"] == "LOSS").astype(int)
    max_streak = 0
    current_streak = 0
    for val in outcomes:
        if val == 1:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            current_streak = 0
    return max_streak


def calculate_symbol_statistics(trades_df):
    if trades_df.empty:
        return pd.DataFrame()
    summary = (
        trades_df.groupby("Symbol")
        .agg(
            Trades=("Outcome", "count"),
            WinRate=("Outcome", lambda x: (x == "WIN").mean() * 100),
            Total_PnL=("Dollar_PnL", "sum"),
        )
        .sort_values(by="Total_PnL", ascending=False)
    )
    return summary


# ============================================================
# FINAL BACKTEST ENGINE
# ============================================================

def run_backtest(processed_data):
    all_timestamps = sorted({
        ts
        for df in processed_data.values()
        for ts in df.index
    })

    active_positions = {}
    all_trades = []
    realized_pnl = 0.0

    for ts in all_timestamps:
        # ====================================================
        # 1) مدیریت پوزیشن‌های باز
        # ====================================================
        symbols_to_close = []

        for symbol, pos in list(active_positions.items()):
            df = processed_data[symbol]
            if ts not in df.index:
                continue

            c4h = df.loc[ts]

            # Trailing Stop
            if pos["side"] == "LONG":
                if c4h["High"] > pos["highest_price"]:
                    pos["highest_price"] = c4h["High"]
                    new_sl = pos["highest_price"] - TRAILING_ATR_MULTIPLIER * c4h["ATR"]
                    if new_sl > pos["stop_loss"]:
                        pos["stop_loss"] = new_sl
                hit_sl = c4h["Low"] <= pos["stop_loss"]
            else:
                if c4h["Low"] < pos["lowest_price"]:
                    pos["lowest_price"] = c4h["Low"]
                    new_sl = pos["lowest_price"] + TRAILING_ATR_MULTIPLIER * c4h["ATR"]
                    if new_sl < pos["stop_loss"]:
                        pos["stop_loss"] = new_sl
                hit_sl = c4h["High"] >= pos["stop_loss"]

            current_index = df.index.get_loc(ts)
            candles_held = current_index - pos["entry_index"]
            timeout = candles_held >= TIMEOUT_CANDLES

            if hit_sl or timeout:
                if hit_sl:
                    if pos["side"] == "LONG":
                        exit_price = min(pos["stop_loss"], c4h["Open"])
                    else:
                        exit_price = max(pos["stop_loss"], c4h["Open"])
                else:
                    exit_price = c4h["Close"]

                r_real, raw_r, fee_r = calculate_r_result(
                    pos["side"],
                    pos["entry_price"],
                    exit_price,
                    pos["initial_risk"],
                    pos["sl_dist_pct"]
                )

                dollar_pnl = calculate_dollar_pnl(
                    pos["side"],
                    pos["entry_price"],
                    exit_price,
                    pos["notional"]
                )

                realized_pnl += dollar_pnl
                outcome = "WIN" if r_real > 0 else "LOSS"

                all_trades.append({
                    "Timestamp": ts,
                    "Symbol": symbol,
                    "Side": pos["side"],
                    "Outcome": outcome,
                    "Return": r_real,
                    "Raw_R": raw_r,
                    "Fee_R": fee_r,
                    "Dollar_PnL": dollar_pnl,
                    "Entry": pos["entry_price"],
                    "Exit": exit_price,
                    "Hold": candles_held,
                    "ExitOrder": len(all_trades),
                })
                symbols_to_close.append(symbol)

        for symbol in symbols_to_close:
            del active_positions[symbol]

        # ====================================================
        # 2) محاسبه Equity
        # ====================================================
        current_equity = get_current_equity(realized_pnl)
        if not capital_safety_check(current_equity):
            break

        # ====================================================
        # 3) وضعیت بازار BTC + Breadth
        # ====================================================
        market_bull = True
        if "BTC" in processed_data and ts in processed_data["BTC"].index:
            btc_c = processed_data["BTC"].loc[ts]
            market_bull = btc_c["Close"] > btc_c["EMA200"]

        bullish_count = 0
        total_symbols = 0
        for symbol, df in processed_data.items():
            if ts in df.index:
                total_symbols += 1
                if df.loc[ts, "Close"] > df.loc[ts, "EMA200"]:
                    bullish_count += 1

        breadth = bullish_count / total_symbols if total_symbols > 0 else 0.5
        allow_longs = breadth >= 0.35
        allow_shorts = breadth <= 0.65

        # ====================================================
        # 4) رتبه‌بندی Momentum
        # ====================================================
        scores = {}
        for symbol, df in processed_data.items():
            if ts in df.index:
                value = df.loc[ts, "Mom_Long"]
                if not np.isnan(value):
                    scores[symbol] = value

        if not scores:
            continue

        ranked = sorted(
            scores.keys(),
            key=lambda x: float(scores[x]),
            reverse=bool(market_bull)
        )

        # ====================================================
        # 5) ساخت Entry
        # ====================================================
        for symbol in ranked:
            if len(active_positions) >= MAX_POSITIONS:
                break
            if symbol in active_positions:
                continue

            df = processed_data[symbol]
            if ts not in df.index:
                continue

            i = df.index.get_loc(ts)
            if i < EMA_WARMUP + 1 or i + 1 >= len(df):
                continue

            c4h = df.iloc[i]
            prev = df.iloc[i - 1]

            if market_bull:
                if not allow_longs:
                    continue
                regime = (
                    c4h["Close"] > c4h["EMA20"]
                    and c4h["EMA20"] > c4h["EMA50"]
                    and c4h["Close"] > c4h["EMA200"]
                )
                pullback = prev["Low"] <= prev["EMA20"] * 1.015
                momentum = c4h["Mom_Short"] > 0.012 and c4h["Mom_Long"] > 0.035
                valid_signal = regime and pullback and momentum
                side = "LONG"
            else:
                if not allow_shorts:
                    continue
                regime = (
                    c4h["Close"] < c4h["EMA20"]
                    and c4h["EMA20"] < c4h["EMA50"]
                    and c4h["Close"] < c4h["EMA200"]
                )
                pullback = prev["High"] >= prev["EMA20"] * 0.985
                momentum = c4h["Mom_Short"] < -0.012 and c4h["Mom_Long"] < -0.035
                valid_signal = regime and pullback and momentum
                side = "SHORT"

            if not valid_signal:
                continue

            entry = (
                c4h["Open"] * (1 + SLIPPAGE)
                if side == "LONG"
                else c4h["Open"] * (1 - SLIPPAGE)
            )

            stop = (
                entry - INITIAL_ATR_MULTIPLIER * c4h["ATR"]
                if side == "LONG"
                else entry + INITIAL_ATR_MULTIPLIER * c4h["ATR"]
            )

            valid_stop, sl_pct = validate_stop_distance(entry, stop)
            if not valid_stop:
                continue

            margin = calculate_trade_margin(
                current_equity,
                BASE_MARGIN_PCT,
                MAX_TOTAL_MARGIN_PCT,
                get_current_margin_used(active_positions)
            )

            if margin <= 0:
                continue

            position = {
                "side": side,
                "entry_price": entry,
                "stop_loss": stop,
                "highest_price": entry,
                "lowest_price": entry,
                "initial_risk": abs(entry - stop),
                "sl_dist_pct": sl_pct,
                "entry_index": i,
                "margin": margin,
                "notional": margin * LEVERAGE,
            }

            active_positions[symbol] = position

    return pd.DataFrame(all_trades)


# ============================================================
# FINAL REPORT
# ============================================================

def summarize_result(trades_df):
    print("\n" + "=" * 70)
    print("📊 HUNTER-V74.3 FINAL REPORT")
    print("=" * 70)

    if trades_df.empty:
        print("هیچ معامله‌ای ثبت نشد")
        return

    total = len(trades_df)
    wins = int((trades_df["Outcome"] == "WIN").sum())
    losses = total - wins
    wr = wins / total * 100
    pnl = float(trades_df["Dollar_PnL"].sum())
    final = INITIAL_CAPITAL + pnl

    stats = calculate_capital_statistics(trades_df)
    sides = calculate_side_statistics(trades_df)

    print(f"Trades: {total}")
    print(f"Win Rate: {wr:.2f}%")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Net R: {trades_df['Return'].sum():.2f}")
    print(f"PnL: ${pnl:,.2f}")
    print(f"Final Capital: ${final:,.2f}")
    print(f"Max DD: {stats['MaxDrawdownPct']:.2f}%")
    print(f"Max Loss Streak: {calculate_loss_streak(trades_df)}")

    print("\nLONG:")
    print(sides["LONG"])

    print("\nSHORT:")
    print(sides["SHORT"])

    print("\nTop Symbols:")
    print(calculate_symbol_statistics(trades_df).head(10))


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    df_trades = run_backtest(processed_data)
    summarize_result(df_trades)
    print("\n✨ HUNTER-V74.3 FINISHED")
