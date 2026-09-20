"""
HUNTER_PRO_V3_BACKTEST.py

Causal Trend + Momentum Futures Engine
LBank Futures / CCXT

Integrity rules:
- No look-ahead
- No repainting
- Signal only after a CLOSED 1H candle
- Entry only on the NEXT 1H candle open
- 4H regime uses only the latest COMPLETED 4H candle
- One position per symbol
- Maximum 3 simultaneous positions
- No timeout exit
- Fixed RR = 1:2 before fees/slippage
- If SL and TP are both touched in one candle, SL wins
- Portfolio events are processed chronologically
"""

import sys
import subprocess
from datetime import datetime, timedelta

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd


exchange = ccxt.lbank({
    "enableRateLimit": True,
    "options": {"defaultType": "swap"}
})

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "SUI": "SUI/USDT",
    "ADA": "ADA/USDT",
    "LINK": "LINK/USDT",
    "AVAX": "AVAX/USDT",
    "DOT": "DOT/USDT",
    "NEAR": "NEAR/USDT",
}

LOOKBACK_DAYS = 365
INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0
MAX_POSITIONS = 3

FEE_RATE = 0.0006
SLIPPAGE = 0.0002

RR = 2.0

# V3 filters. These are explicit rules, not result-fitting.
ATR_MULTIPLIER = 1.5
RVOL_MIN = 1.20
MOMENTUM_THRESHOLD = 0.005
ADX_MIN = 20.0

EMA_FAST = 50
EMA_SLOW = 200
ATR_PERIOD = 14
RVOL_PERIOD = 20
MOMENTUM_PERIOD = 20
ADX_PERIOD = 14


def fetch_data(symbol, timeframe):
    since = int(
        (datetime.now() - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000
    )

    candles = []
    last_seen = None

    while since < exchange.milliseconds():
        batch = None

        for _ in range(3):
            try:
                batch = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=1000
                )
                break
            except Exception:
                batch = None

        if not batch:
            return None

        last_ts = batch[-1][0]

        if last_seen is not None and last_ts <= last_seen:
            return None

        candles.extend(batch)
        last_seen = last_ts
        since = last_ts + 1

        if len(batch) < 1000:
            break

    if not candles:
        return None

    df = pd.DataFrame(
        candles,
        columns=[
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    )

    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")

    df.drop_duplicates("Date", keep="last", inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Do not use a currently forming candle.
    now_ms = exchange.milliseconds()
    tf_ms = 60 * 60 * 1000 if timeframe == "1h" else 4 * 60 * 60 * 1000

    if len(df):
        last_open_ms = int(df.iloc[-1]["Timestamp"])

        if last_open_ms + tf_ms > now_ms:
            df = df.iloc[:-1].copy()

    if len(df) < EMA_SLOW + 50:
        return None

    return df.reset_index(drop=True)


def add_indicators(df):
    df = df.copy()

    df["EMA50"] = df["Close"].ewm(
        span=EMA_FAST,
        adjust=False,
        min_periods=EMA_FAST
    ).mean()

    df["EMA200"] = df["Close"].ewm(
        span=EMA_SLOW,
        adjust=False,
        min_periods=EMA_SLOW
    ).mean()

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1 / ATR_PERIOD,
        adjust=False,
        min_periods=ATR_PERIOD
    ).mean()

    df["RVOL"] = (
        df["Volume"] /
        df["Volume"].rolling(
            RVOL_PERIOD,
            min_periods=RVOL_PERIOD
        ).mean()
    )

    df["Momentum"] = (
        df["Close"] /
        df["Close"].shift(MOMENTUM_PERIOD)
        - 1.0
    )

    # Wilder-style ADX using causal rolling/EMA calculations.
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0
        ),
        index=df.index
    )

    atr_w = tr.ewm(
        alpha=1 / ADX_PERIOD,
        adjust=False,
        min_periods=ADX_PERIOD
    ).mean()

    plus_di = (
        100.0 *
        plus_dm.ewm(
            alpha=1 / ADX_PERIOD,
            adjust=False,
            min_periods=ADX_PERIOD
        ).mean() /
        atr_w
    )

    minus_di = (
        100.0 *
        minus_dm.ewm(
            alpha=1 / ADX_PERIOD,
            adjust=False,
            min_periods=ADX_PERIOD
        ).mean() /
        atr_w
    )

    dx = (
        100.0 *
        (plus_di - minus_di).abs() /
        (plus_di + minus_di).replace(0, np.nan)
    )

    df["ADX"] = dx.ewm(
        alpha=1 / ADX_PERIOD,
        adjust=False,
        min_periods=ADX_PERIOD
    ).mean()

    return df


def build_4h_regime_map(df4):
    """
    For every completed 4H candle, RegimeAvailableAt is its CLOSE time.
    A 1H signal at time T may only use a 4H candle whose close time <= T.
    """
    r = df4[
        [
            "Date",
            "EMA50",
            "EMA200",
            "Close",
            "ADX"
        ]
    ].copy()

    r["RegimeAvailableAt"] = (
        r["Date"] + pd.Timedelta(hours=4)
    )

    r["Regime"] = np.where(
        (
            (r["Close"] > r["EMA50"]) &
            (r["EMA50"] > r["EMA200"]) &
            (r["ADX"] >= ADX_MIN)
        ),
        "LONG",
        np.where(
            (
                (r["Close"] < r["EMA50"]) &
                (r["EMA50"] < r["EMA200"]) &
                (r["ADX"] >= ADX_MIN)
            ),
            "SHORT",
            "NEUTRAL"
        )
    )

    return r[
        ["RegimeAvailableAt", "Regime"]
    ].sort_values("RegimeAvailableAt").reset_index(drop=True)


def prepare_symbol(symbol, raw1, raw4):
    h1 = add_indicators(raw1)
    h4 = add_indicators(raw4)

    regime_map = build_4h_regime_map(h4)

    # Exact time-based causal merge:
    # signal candle timestamp must be >= completed 4H close time.
    h1 = pd.merge_asof(
        h1.sort_values("Date"),
        regime_map,
        left_on="Date",
        right_on="RegimeAvailableAt",
        direction="backward",
        allow_exact_matches=True
    )

    h1["Symbol"] = symbol

    return {
        "1h": h1.reset_index(drop=True),
        "4h": h4.reset_index(drop=True)
    }


def signal_on_closed_candle(row):
    """
    All values here belong to the CLOSED signal candle.
    No future candle is referenced.
    """
    if pd.isna(row["Regime"]) or pd.isna(row["ATR"]):
        return None

    if pd.isna(row["RVOL"]) or pd.isna(row["Momentum"]):
        return None

    if pd.isna(row["ADX"]):
        return None

    if row["RVOL"] < RVOL_MIN:
        return None

    if row["ADX"] < ADX_MIN:
        return None

    if row["Regime"] == "LONG":
        if row["Close"] <= row["Open"]:
            return None

        if row["Momentum"] < MOMENTUM_THRESHOLD:
            return None

        return "LONG"

    if row["Regime"] == "SHORT":
        if row["Close"] >= row["Open"]:
            return None

        if row["Momentum"] > -MOMENTUM_THRESHOLD:
            return None

        return "SHORT"

    return None


def calculate_trade(signal_row, next_row, side):
    """
    Entry is the NEXT candle's open.
    No part of next candle's high/low is used for entry.
    """

    raw_open = float(next_row["Open"])
    atr = float(signal_row["ATR"])

    if not np.isfinite(raw_open) or not np.isfinite(atr) or atr <= 0:
        return None

    if side == "LONG":
        entry = raw_open * (1.0 + SLIPPAGE)
        sl = entry - ATR_MULTIPLIER * atr
        risk = entry - sl
        tp = entry + RR * risk
    else:
        entry = raw_open * (1.0 - SLIPPAGE)
        sl = entry + ATR_MULTIPLIER * atr
        risk = sl - entry
        tp = entry - RR * risk

    if risk <= 0:
        return None

    return {
        "side": side,
        "entry": entry,
        "SL": sl,
        "TP": tp,
        "risk": risk
    }


def pnl_for_exit(pos, exit_price):
    if pos["side"] == "LONG":
        price_return = (
            exit_price - pos["entry"]
        ) / pos["entry"]
    else:
        price_return = (
            pos["entry"] - exit_price
        ) / pos["entry"]

    gross = TRADE_MARGIN * price_return
    fees = TRADE_MARGIN * FEE_RATE * 2.0

    return gross - fees


def max_drawdown(equity_values):
    if not equity_values:
        return 0.0

    peak = equity_values[0]
    max_dd = 0.0

    for value in equity_values:
        peak = max(peak, value)
        max_dd = max(max_dd, peak - value)

    return max_dd


def loss_streaks(trades):
    streaks = []
    current = 0

    for result in trades["Result"]:
        if result == "LOSS":
            current += 1
        else:
            if current:
                streaks.append(current)
                current = 0

    if current:
        streaks.append(current)

    return streaks


def run_backtest(data):
    """
    Event-driven portfolio backtest.

    Important:
    - Positions are checked for exits using the current CLOSED candle.
    - A position cannot generate a new signal on its exit candle.
    - Entry orders created from a signal candle are executed on the
      following candle's OPEN.
    """

    all_timestamps = sorted(
        set(
            ts
            for obj in data.values()
            for ts in obj["1h"]["Date"]
        )
    )

    active = {}
    pending_entries = {}
    trades = []

    equity = INITIAL_CAPITAL
    equity_points = []

    for ts in all_timestamps:

        # ---------------------------------------------------------
        # 1. Execute pending entries at THIS candle's open.
        # ---------------------------------------------------------
        for sym, order in list(pending_entries.items()):

            if sym in active:
                del pending_entries[sym]
                continue

            if len(active) >= MAX_POSITIONS:
                break

            df = data[sym]["1h"]
            rows = df[df["Date"] == ts]

            if rows.empty:
                continue

            c = rows.iloc[0]

            active[sym] = {
                "side": order["side"],
                "entry": order["entry"],
                "SL": order["SL"],
                "TP": order["TP"],
                "entry_time": ts,
                "entry_bar": int(rows.index[0])
            }

            del pending_entries[sym]

        # ---------------------------------------------------------
        # 2. Check exits on the current candle.
        # ---------------------------------------------------------
        closed_this_candle = set()

        for sym, pos in list(active.items()):

            df = data[sym]["1h"]
            rows = df[df["Date"] == ts]

            if rows.empty:
                continue

            c = rows.iloc[0]

            if pos["side"] == "LONG":
                hit_sl = c["Low"] <= pos["SL"]
                hit_tp = c["High"] >= pos["TP"]
            else:
                hit_sl = c["High"] >= pos["SL"]
                hit_tp = c["Low"] <= pos["TP"]

            if not (hit_sl or hit_tp):
                continue

            # Conservative intrabar rule.
            if hit_sl:
                exit_price = pos["SL"]
                result = "LOSS"
            else:
                exit_price = pos["TP"]
                result = "WIN"

            pnl = pnl_for_exit(pos, exit_price)
            equity += pnl

            trades.append({
                "ExitTime": ts,
                "Symbol": sym,
                "Side": pos["side"],
                "Entry": pos["entry"],
                "Exit": exit_price,
                "SL": pos["SL"],
                "TP": pos["TP"],
                "PnL": pnl,
                "Result": result
            })

            del active[sym]
            closed_this_candle.add(sym)

        # ---------------------------------------------------------
        # 3. Create signals from THIS CLOSED candle.
        #    Execution is NEXT candle, never this candle.
        # ---------------------------------------------------------
        available_slots = MAX_POSITIONS - len(active) - len(pending_entries)

        if available_slots > 0:

            candidates = []

            for sym, obj in data.items():

                if sym in active or sym in pending_entries:
                    continue

                if sym in closed_this_candle:
                    continue

                df = obj["1h"]
                rows = df[df["Date"] == ts]

                if rows.empty:
                    continue

                i = int(rows.index[0])

                if i + 1 >= len(df):
                    continue

                signal_row = df.iloc[i]
                next_row = df.iloc[i + 1]

                side = signal_on_closed_candle(signal_row)

                if side is None:
                    continue

                trade = calculate_trade(
                    signal_row,
                    next_row,
                    side
                )

                if trade is None:
                    continue

                candidates.append(
                    (sym, trade)
                )

            # Deterministic selection:
            # strongest absolute momentum first.
            candidates.sort(
                key=lambda x: abs(
                    float(
                        data[x[0]]["1h"].loc[
                            data[x[0]]["1h"]["Date"] == ts,
                            "Momentum"
                        ].iloc[0]
                    )
                ),
                reverse=True
            )

            for sym, trade in candidates[:available_slots]:

                pending_entries[sym] = trade

        equity_points.append({
            "Timestamp": ts,
            "Equity": equity
        })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_points)

    return trades_df, equity_df, equity


def print_results(trades, equity_df, final_equity):

    print("=" * 70)
    print("HUNTER PRO V3 — CAUSAL RESULTS")
    print("=" * 70)

    total = len(trades)
    wins = int((trades["Result"] == "WIN").sum()) if total else 0
    losses = int((trades["Result"] == "LOSS").sum()) if total else 0

    win_rate = wins / total * 100 if total else 0.0

    pnl = float(trades["PnL"].sum()) if total else 0.0

    gross_win = float(
        trades.loc[trades["PnL"] > 0, "PnL"].sum()
    ) if total else 0.0

    gross_loss = float(
        -trades.loc[trades["PnL"] < 0, "PnL"].sum()
    ) if total else 0.0

    profit_factor = (
        gross_win / gross_loss
        if gross_loss > 0 else float("inf")
    )

    avg_win = (
        float(trades.loc[trades["PnL"] > 0, "PnL"].mean())
        if wins else 0.0
    )

    avg_loss = (
        float(trades.loc[trades["PnL"] < 0, "PnL"].mean())
        if losses else 0.0
    )

    dd = max_drawdown(
        equity_df["Equity"].tolist()
        if len(equity_df) else [INITIAL_CAPITAL]
    )

    streaks = loss_streaks(trades) if total else []

    print(f"Trades: {total}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"PnL: ${pnl:,.2f}")
    print(f"Final Balance: ${final_equity:,.2f}")
    print(f"Profit Factor: {profit_factor:.3f}")
    print(f"Average Win: ${avg_win:,.2f}")
    print(f"Average Loss: ${avg_loss:,.2f}")
    print(f"Max Drawdown: ${dd:,.2f}")

    print()
    print("LOSS STREAKS")
    print("-" * 70)

    if streaks:
        for s in streaks:
            print(f"{s} loss" if s == 1 else f"{s} losses")
        print(
            f"Maximum Consecutive Losses: {max(streaks)}"
        )
    else:
        print("No loss streaks")

    print()
    print("PER SYMBOL")
    print("-" * 70)

    if total:
        for sym in SYMBOLS:

            x = trades[trades["Symbol"] == sym]

            if x.empty:
                print(
                    f"{sym:<6} Trades=0 | Wins=0 | Losses=0 | "
                    f"WR=0.00% | PnL=$0.00"
                )
                continue

            n = len(x)
            w = int((x["Result"] == "WIN").sum())
            l = int((x["Result"] == "LOSS").sum())
            wr = w / n * 100
            spnl = x["PnL"].sum()

            print(
                f"{sym:<6} Trades={n:<5} "
                f"Wins={w:<5} Losses={l:<5} "
                f"WR={wr:6.2f}% PnL=${spnl:,.2f}"
            )
    else:
        print("No trades")


if __name__ == "__main__":

    processed = {}

    for name, symbol in SYMBOLS.items():

        print("Loading", name)

        h1 = fetch_data(symbol, "1h")
        h4 = fetch_data(symbol, "4h")

        if h1 is None or h4 is None:
            print("  skipped: insufficient/failed data")
            continue

        processed[name] = prepare_symbol(
            name,
            h1,
            h4
        )

    print("Valid symbols:", len(processed))

    if not processed:
        raise RuntimeError("No valid symbols loaded.")

    trades, equity_df, final_equity = run_backtest(
        processed
    )

    trades.to_csv(
        "hunter_v3_trades.csv",
        index=False
    )

    equity_df.to_csv(
        "hunter_v3_equity.csv",
        index=False
    )

    print_results(
        trades,
        equity_df,
        final_equity
    )
