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


# ============================================================
# HUNTER-V75
# V74 GOLDEN CORE + BTC REGIME SHIELD + LOSS CONTROL
# ============================================================


exchange = ccxt.lbank({
    "enableRateLimit": True
})


# =========================
# SYMBOLS
# =========================

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


# ATR

ATR_PERIOD = 14
INITIAL_ATR_MULTIPLIER = 1.8
TRAILING_ATR_MULTIPLIER = 2.0


# Exit

TIMEOUT_CANDLES = 45
EMA_WARMUP = 200


# Money Management

INITIAL_CAPITAL = 1000.0

TRADE_MARGIN = 100.0
LEVERAGE = 80.0


# =========================
# NEW RISK CONTROL
# =========================

LOSS_WINDOW_CANDLES = 6          # 24 ساعت
MAX_LOSS_WINDOW = 3

HARD_LOSS_WINDOW_CANDLES = 12   # 48 ساعت
HARD_MAX_LOSS = 5

COOLDOWN_CANDLES = 12           # توقف 48 ساعت

# محدودیت همبستگی

MAX_SAME_DIRECTION_ENTRIES = 2


start_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
since_timestamp = int(start_date.timestamp() * 1000)


print("=" * 68)
print("📥 دریافت داده‌ها - HUNTER-V75")
print("=" * 68)


processed_data = {}


# ============================================================
# FETCH DATA
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
                    limit=1000
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
        columns=[
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    )

    df["Date"] = pd.to_datetime(
        df["Timestamp"],
        unit="ms"
    )

    df = df[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    ]

    df.dropna(inplace=True)
    df.drop_duplicates(
        subset=["Date"],
        keep="last",
        inplace=True
    )

    df.sort_values(
        "Date",
        inplace=True
    )
    df.reset_index(
        drop=True,
        inplace=True
    )

    if len(df) >= 2:
        now_ms = exchange.milliseconds()
        last_ms = int(
            df.iloc[-1]["Date"].timestamp() * 1000
        )
        if last_ms + 4 * 60 * 60 * 1000 > now_ms:
            df = df.iloc[:-1].copy()

    if len(df) < EMA_WARMUP + 50:
        return None

    deltas = df["Date"].diff().dropna()
    if not deltas.empty:
        if deltas.max() > pd.Timedelta(
            hours=4,
            minutes=10
        ):
            return None

    tr1 = df["High"] - df["Low"]
    tr2 = abs(
        df["High"] -
        df["Close"].shift(1)
    )
    tr3 = abs(
        df["Low"] -
        df["Close"].shift(1)
    )

    df["ATR"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1).rolling(
        ATR_PERIOD
    ).mean()

    df["EMA20"] = df["Close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA50"] = df["Close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df["EMA200"] = df["Close"].ewm(
        span=200,
        adjust=False
    ).mean()

    df["Mom_Short"] = (
        df["Close"] -
        df["Close"].shift(10)
    ) / df["Close"].shift(10)

    df["Mom_Long"] = (
        df["Close"] -
        df["Close"].shift(30)
    ) / df["Close"].shift(30)

    df.set_index(
        "Date",
        inplace=True
    )

    return df


for symbol, lbank_symbol in SYMBOLS.items():
    df4h = fetch_symbol_data(lbank_symbol)
    if df4h is not None:
        processed_data[symbol] = df4h


print(
    f"✅ تعداد نمادهای معتبر: {len(processed_data)} از {len(SYMBOLS)}"
)
print(
    "⚙️ آماده‌سازی موتور HUNTER-V75..."
)


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(processed_data):
    all_timestamps = sorted({
        ts
        for df in processed_data.values()
        for ts in df.index
    })

    active_positions = {}
    all_trades = []

    consecutive_losses = 0
    current_margin = TRADE_MARGIN
    cooldown = 0
    recent_losses = []

    for ts in all_timestamps:

        if cooldown > 0:
            cooldown -= 1

        symbols_to_close = []

        # ======================================
        # MANAGE OPEN POSITIONS
        # ======================================
        for symbol, pos in list(active_positions.items()):
            df = processed_data[symbol]
            if ts not in df.index:
                continue

            c4h = df.loc[ts]

            if pos["side"] == "LONG":
                if c4h["High"] > pos["highest_price"]:
                    pos["highest_price"] = c4h["High"]
                    new_sl = (
                        pos["highest_price"]
                        -
                        TRAILING_ATR_MULTIPLIER *
                        c4h["ATR"]
                    )
                    if new_sl > pos["stop_loss"]:
                        pos["stop_loss"] = new_sl

                hit_sl = (
                    c4h["Low"]
                    <=
                    pos["stop_loss"]
                )
            else:
                if c4h["Low"] < pos["lowest_price"]:
                    pos["lowest_price"] = c4h["Low"]
                    new_sl = (
                        pos["lowest_price"]
                        +
                        TRAILING_ATR_MULTIPLIER *
                        c4h["ATR"]
                    )
                    if new_sl < pos["stop_loss"]:
                        pos["stop_loss"] = new_sl

                hit_sl = (
                    c4h["High"]
                    >=
                    pos["stop_loss"]
                )

            candles_held = (
                df.index.get_loc(ts)
                -
                pos["entry_index"]
            )

            timeout = (
                candles_held
                >=
                TIMEOUT_CANDLES
            )

            if hit_sl or timeout:
                if pos["side"] == "LONG":
                    exit_price = (
                        min(
                            pos["stop_loss"],
                            c4h["Open"]
                        )
                        if hit_sl
                        else
                        c4h["Close"]
                    )
                    raw_return = (
                        exit_price -
                        pos["entry_price"]
                    ) / pos["entry_price"]
                else:
                    exit_price = (
                        max(
                            pos["stop_loss"],
                            c4h["Open"]
                        )
                        if hit_sl
                        else
                        c4h["Close"]
                    )
                    raw_return = (
                        pos["entry_price"]
                        -
                        exit_price
                    ) / pos["entry_price"]

                pnl = (
                    current_margin *
                    LEVERAGE *
                    raw_return
                )
                pnl -= (
                    current_margin *
                    LEVERAGE *
                    FEE_RATE *
                    2
                )

                outcome = (
                    "WIN"
                    if pnl > 0
                    else
                    "LOSS"
                )

                if outcome == "LOSS":
                    consecutive_losses += 1
                    recent_losses.append(ts)
                    recent_losses = [
                        x
                        for x in recent_losses
                        if (
                            ts - x
                        ).total_seconds()
                        <=
                        LOSS_WINDOW_CANDLES * 4 * 3600
                    ]

                    if consecutive_losses >= 6:
                        current_margin = 25
                    elif consecutive_losses >= 4:
                        current_margin = 50
                    elif consecutive_losses >= 2:
                        current_margin = 70

                    if consecutive_losses >= 8:
                        cooldown = COOLDOWN_CANDLES
                else:
                    consecutive_losses = 0
                    current_margin = TRADE_MARGIN
                    recent_losses.clear()

                all_trades.append({
                    "Timestamp": ts,
                    "Symbol": symbol,
                    "Side": pos["side"],
                    "Outcome": outcome,
                    "PnL": pnl,
                    "Margin": current_margin
                })

                symbols_to_close.append(symbol)

        for sym in symbols_to_close:
            del active_positions[sym]

        # ======================================
        # BTC MARKET CONTROL
        # ======================================
        btc_bull = True
        btc_allow_long = True
        btc_allow_short = True

        if (
            "BTC"
            in processed_data
            and
            ts in processed_data["BTC"].index
        ):
            btc = processed_data["BTC"].loc[ts]
            btc_bull = (
                btc["Close"]
                >
                btc["EMA200"]
            )

            if btc["Close"] < btc["EMA200"]:
                btc_allow_long = False

            if (
                btc["Close"]
                >
                btc["EMA200"]
            ):
                btc_allow_short = False

        # ======================================
        # MARKET BREADTH FILTER
        # ======================================
        bullish_count = 0
        total_count = 0

        for symbol, df in processed_data.items():
            if ts in df.index:
                total_count += 1
                if (
                    df.loc[ts, "Close"]
                    >
                    df.loc[ts, "EMA200"]
                ):
                    bullish_count += 1

        breadth = (
            bullish_count / total_count
            if total_count > 0
            else 0.5
        )

        if breadth < 0.35:
            btc_allow_long = False

        if breadth > 0.65:
            btc_allow_short = False

        if cooldown > 0:
            continue

        # ======================================
        # SCORE RANKING
        # ======================================
        scores = {}

        for symbol, df in processed_data.items():
            if ts not in df.index:
                continue

            row = df.loc[ts]

            if np.isnan(row["Mom_Long"]):
                continue

            if abs(row["Mom_Long"]) < 0.015:
                continue

            scores[symbol] = row["Mom_Long"]

        if not scores:
            continue

        ranked_symbols = sorted(
            scores.keys(),
            key=lambda x: scores[x],
            reverse=btc_bull
        )

        # ======================================
        # ENTRY LOGIC (V74 CORE)
        # ======================================
        for symbol in ranked_symbols:
            if len(active_positions) >= MAX_POSITIONS:
                break

            if symbol in active_positions:
                continue

            df = processed_data[symbol]
            if ts not in df.index:
                continue

            i = df.index.get_loc(ts)
            if i < EMA_WARMUP + 1:
                continue

            c4h = df.iloc[i]
            prev = df.iloc[i - 1]

            # ------------------------------
            # LONG
            # ------------------------------
            if btc_allow_long:
                long_setup = (
                    c4h["Close"]
                    >
                    c4h["EMA20"]
                    and
                    c4h["EMA20"]
                    >
                    c4h["EMA50"]
                    and
                    c4h["Close"]
                    >
                    c4h["EMA200"]
                    and
                    prev["Low"]
                    <=
                    prev["EMA20"] * 1.015
                    and
                    c4h["Mom_Short"]
                    >
                    0.012
                    and
                    c4h["Mom_Long"]
                    >
                    0.035
                )

                if long_setup:
                    side = "LONG"
                else:
                    side = None
            else:
                side = None

            # ------------------------------
            # SHORT
            # ------------------------------
            if side is None and btc_allow_short:
                short_setup = (
                    c4h["Close"]
                    <
                    c4h["EMA20"]
                    and
                    c4h["EMA20"]
                    <
                    c4h["EMA50"]
                    and
                    c4h["Close"]
                    <
                    c4h["EMA200"]
                    and
                    prev["High"]
                    >=
                    prev["EMA20"] * 0.985
                    and
                    c4h["Mom_Short"]
                    <
                    -0.012
                    and
                    c4h["Mom_Long"]
                    <
                    -0.035
                )

                if short_setup:
                    side = "SHORT"

            if side is None:
                continue

            # ======================================
            # POSITION SIZE
            # ======================================
            entry_price = (
                c4h["Open"] *
                (1 + SLIPPAGE)
                if side == "LONG"
                else
                c4h["Open"] *
                (1 - SLIPPAGE)
            )

            stop_loss = (
                entry_price
                -
                INITIAL_ATR_MULTIPLIER *
                c4h["ATR"]
                if side == "LONG"
                else
                entry_price
                +
                INITIAL_ATR_MULTIPLIER *
                c4h["ATR"]
            )

            risk = abs(
                entry_price -
                stop_loss
            )

            sl_percent = (
                risk /
                entry_price
            )

            if not (
                0.01
                <=
                sl_percent
                <=
                0.04
            ):
                continue

            same_side = sum(
                1
                for p in active_positions.values()
                if p["side"] == side
            )

            if same_side >= MAX_SAME_DIRECTION_ENTRIES:
                continue

            active_positions[symbol] = {
                "side": side,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "highest_price": entry_price,
                "lowest_price": entry_price,
                "entry_index": i,
            }

    return pd.DataFrame(all_trades)


# ============================================================
# FINAL REPORT
# ============================================================

def summarize_result(trades_df):
    print("\n" + "=" * 70)
    print("📊 HUNTER-V75 FINAL REPORT")
    print("=" * 70)

    if trades_df.empty:
        print("⚠️ هیچ معامله‌ای ثبت نشد")
        return

    trades_df = trades_df.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    total = len(trades_df)
    wins = (
        trades_df["Outcome"]
        ==
        "WIN"
    ).sum()
    losses = (
        trades_df["Outcome"]
        ==
        "LOSS"
    ).sum()

    win_rate = (
        wins / total * 100
    )

    total_pnl = trades_df["PnL"].sum()

    final_capital = (
        INITIAL_CAPITAL
        +
        total_pnl
    )

    # ==================================
    # DRAW DOWN
    # ==================================
    trades_df["Equity"] = (
        INITIAL_CAPITAL
        +
        trades_df["PnL"].cumsum()
    )

    trades_df["Peak"] = (
        trades_df["Equity"]
        .cummax()
    )

    trades_df["DD"] = (
        trades_df["Equity"]
        -
        trades_df["Peak"]
    )

    max_dd = trades_df["DD"].min()

    max_dd_pct = (
        max_dd
        /
        trades_df["Peak"].max()
        *
        100
    )

    # ==================================
    # LOSS STREAK
    # ==================================
    max_loss_streak = 0
    current_loss = 0

    for result in trades_df["Outcome"]:
        if result == "LOSS":
            current_loss += 1
            max_loss_streak = max(
                max_loss_streak,
                current_loss
            )
        else:
            current_loss = 0

    print(f"🔸 Total Trades: {total}")
    print(f"✅ Wins: {wins}")
    print(f"❌ Losses: {losses}")
    print(f"🎯 Win Rate: {win_rate:.2f}%")
    print(f"💰 Net PnL: ${total_pnl:,.2f}")
    print(f"🏦 Final Capital: ${final_capital:,.2f}")
    print(f"📉 Max Drawdown: ${max_dd:,.2f} ({max_dd_pct:.2f}%)")
    print(f"❄️ Max Loss Streak: {max_loss_streak}")

    print("\n------------------------------")
    print("📈 LONG / SHORT")
    print("------------------------------")

    for side in ["LONG", "SHORT"]:
        temp = trades_df[
            trades_df["Side"] == side
        ]
        if len(temp) == 0:
            continue

        side_wr = (
            (
                temp["Outcome"]
                ==
                "WIN"
            ).sum()
            /
            len(temp)
            *
            100
        )

        print(
            f"{side}: "
            f"Trades={len(temp)} | "
            f"WR={side_wr:.2f}% | "
            f"PnL=${temp['PnL'].sum():,.2f}"
        )

    print("\n------------------------------")
    print("🏆 TOP SYMBOLS")
    print("------------------------------")

    symbol_report = trades_df.groupby(
        "Symbol"
    ).agg(
        Trades=("Symbol", "count"),
        WinRate=(
            "Outcome",
            lambda x:
            (
                x == "WIN"
            ).mean() * 100
        ),
        Total_PnL=(
            "PnL",
            "sum"
        )
    ).sort_values(
        "Total_PnL",
        ascending=False
    )

    print(
        symbol_report.head(10)
    )

    print("\n✨ HUNTER-V75 FINISHED")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    trades = run_backtest(
        processed_data
    )
    summarize_result(
        trades
    )
