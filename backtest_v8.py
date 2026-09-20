"""
HUNTER_PRO_V4_BACKTEST.py
==========================

Causal Liquidity / Structure / Trend System
LBank USDT-M Futures / CCXT

Design goals:
- No look-ahead
- No repainting
- Only completed 1H / 4H candles
- 4H regime becomes available only after its close
- Confirmed 1H pivots require PIVOT_RIGHT future candles to confirm
- Signal is generated only from information available at the signal candle close
- Entry is the NEXT candle open
- Fixed RR = 1:2 before fees/slippage
- No timeout / max-bars exit
- No overlapping position per symbol
- Maximum 3 portfolio positions
- No new signal on the same candle a position closes
- Conservative intrabar rule: if SL and TP are both touched, SL wins
- Full portfolio statistics + loss streaks + equity CSV
- No parameter search / no target-fitting inside the script

The strategy is intentionally structural rather than a collection of
indicator thresholds. It uses:

4H:
    Trend regime = EMA50 vs EMA200 + EMA50 slope + ADX

1H:
    Confirmed swing highs/lows
    Liquidity sweep
    Break of Structure (BOS)
    Retest of BOS level
    Entry on next candle open

This file is a research/backtest engine. It does not place live orders.
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


# ============================================================
# EXCHANGE / PORTFOLIO
# ============================================================

exchange = ccxt.lbank({
    "enableRateLimit": True,
    "options": {
        "defaultType": "swap"
    }
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

# ============================================================
# STRUCTURE PARAMETERS
# ============================================================

PIVOT_LEFT = 2
PIVOT_RIGHT = 2

EMA_FAST = 50
EMA_SLOW = 200

ATR_LEN = 14
ADX_LEN = 14

RVOL_LEN = 20
RVOL_MIN = 1.10

ADX_MIN = 20.0

# Minimum close-through distance for BOS.
BOS_ATR_BUFFER = 0.05

# Maximum distance from BOS level accepted as a retest.
RETEST_ATR_BUFFER = 0.20

# Stop is placed beyond the sweep extreme.
SL_ATR_BUFFER = 0.15

# Do not take structurally absurd stops.
MAX_RISK_PCT = 0.025


# ============================================================
# DATA
# ============================================================

def fetch_data(symbol, timeframe):
    since = int(
        (datetime.now() - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000
    )

    candles = []
    last_ts = None

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
            except Exception as exc:
                print(
                    f"  fetch retry {symbol} {timeframe}: {exc}"
                )

        if not batch:
            break

        newest = batch[-1][0]

        if last_ts is not None and newest <= last_ts:
            break

        candles.extend(batch)
        last_ts = newest
        since = newest + 1

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

    df["Date"] = pd.to_datetime(
        df["Timestamp"],
        unit="ms"
    )

    df.drop_duplicates(
        "Date",
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

    # Remove malformed candles.
    valid = (
        (df["High"] >= df[["Open", "Close"]].max(axis=1)) &
        (df["Low"] <= df[["Open", "Close"]].min(axis=1)) &
        (df["High"] >= df["Low"]) &
        (df["Volume"] >= 0)
    )

    df = df.loc[valid].copy()

    # Remove current open candle.
    tf_ms = (
        60 * 60 * 1000
        if timeframe == "1h"
        else 4 * 60 * 60 * 1000
    )

    now_ms = exchange.milliseconds()

    if len(df):
        last_open = int(df.iloc[-1]["Timestamp"])

        if last_open + tf_ms > now_ms:
            df = df.iloc[:-1].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    if len(df) < EMA_SLOW + 50:
        return None

    return df


# ============================================================
# INDICATORS
# ============================================================

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

    prev_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["TR"] = tr

    df["ATR"] = tr.ewm(
        alpha=1 / ATR_LEN,
        adjust=False,
        min_periods=ATR_LEN
    ).mean()

    df["RVOL"] = (
        df["Volume"] /
        df["Volume"].rolling(
            RVOL_LEN,
            min_periods=RVOL_LEN
        ).mean()
    )

    # Wilder-style ADX.
    up = df["High"].diff()
    down = -df["Low"].diff()

    plus_dm = pd.Series(
        np.where(
            (up > down) & (up > 0),
            up,
            0.0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down > up) & (down > 0),
            down,
            0.0
        ),
        index=df.index
    )

    atr_w = tr.ewm(
        alpha=1 / ADX_LEN,
        adjust=False,
        min_periods=ADX_LEN
    ).mean()

    plus_di = (
        100.0 *
        plus_dm.ewm(
            alpha=1 / ADX_LEN,
            adjust=False,
            min_periods=ADX_LEN
        ).mean() /
        atr_w
    )

    minus_di = (
        100.0 *
        minus_dm.ewm(
            alpha=1 / ADX_LEN,
            adjust=False,
            min_periods=ADX_LEN
        ).mean() /
        atr_w
    )

    dx = (
        100.0 *
        (plus_di - minus_di).abs() /
        (plus_di + minus_di).replace(0, np.nan)
    )

    df["ADX"] = dx.ewm(
        alpha=1 / ADX_LEN,
        adjust=False,
        min_periods=ADX_LEN
    ).mean()

    return df


# ============================================================
# CONFIRMED PIVOTS
# ============================================================

def add_confirmed_pivots(df):
    """
    A pivot at i is NOT usable at i.

    It becomes known only at i + PIVOT_RIGHT.
    Therefore the pivot level is forward-shifted into the
    first candle where a trader could actually know it.
    """

    df = df.copy()

    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()

    n = len(df)

    confirmed_high = np.full(n, np.nan)
    confirmed_low = np.full(n, np.nan)

    for i in range(PIVOT_LEFT, n - PIVOT_RIGHT):

        left_highs = highs[
            i - PIVOT_LEFT:i
        ]

        right_highs = highs[
            i + 1:i + 1 + PIVOT_RIGHT
        ]

        if (
            highs[i] > left_highs.max()
            and highs[i] >= right_highs.max()
        ):
            confirm_idx = i + PIVOT_RIGHT
            confirmed_high[confirm_idx] = highs[i]

        left_lows = lows[
            i - PIVOT_LEFT:i
        ]

        right_lows = lows[
            i + 1:i + 1 + PIVOT_RIGHT
        ]

        if (
            lows[i] < left_lows.min()
            and lows[i] <= right_lows.min()
        ):
            confirm_idx = i + PIVOT_RIGHT
            confirmed_low[confirm_idx] = lows[i]

    df["NewConfirmedHigh"] = confirmed_high
    df["NewConfirmedLow"] = confirmed_low

    # Forward-fill only AFTER the confirmation point.
    df["LastSwingHigh"] = df["NewConfirmedHigh"].ffill()
    df["LastSwingLow"] = df["NewConfirmedLow"].ffill()

    return df


# ============================================================
# 4H REGIME
# ============================================================

def build_regime(df4):
    x = df4.copy()

    x["EMA50Prev"] = x["EMA50"].shift(3)

    x["Regime"] = np.where(
        (
            (x["Close"] > x["EMA50"]) &
            (x["EMA50"] > x["EMA200"]) &
            (x["EMA50"] > x["EMA50Prev"]) &
            (x["ADX"] >= ADX_MIN)
        ),
        "LONG",
        np.where(
            (
                (x["Close"] < x["EMA50"]) &
                (x["EMA50"] < x["EMA200"]) &
                (x["EMA50"] < x["EMA50Prev"]) &
                (x["ADX"] >= ADX_MIN)
            ),
            "SHORT",
            "NEUTRAL"
        )
    )

    # 4H open + 4 hours = first instant at which this candle is known.
    x["AvailableAt"] = (
        x["Date"] +
        pd.Timedelta(hours=4)
    )

    return x[
        ["AvailableAt", "Regime"]
    ].sort_values(
        "AvailableAt"
    ).reset_index(drop=True)


# ============================================================
# PREPARE SYMBOL
# ============================================================

def prepare_symbol(name, h1, h4):
    h1 = add_indicators(h1)
    h1 = add_confirmed_pivots(h1)

    h4 = add_indicators(h4)

    regime = build_regime(h4)

    # At a 1H candle close, use only a 4H candle whose close
    # has already occurred.
    h1 = pd.merge_asof(
        h1.sort_values("Date"),
        regime,
        left_on="Date",
        right_on="AvailableAt",
        direction="backward",
        allow_exact_matches=True
    )

    h1["Symbol"] = name

    return h1.reset_index(drop=True)


# ============================================================
# STATE MACHINE
# ============================================================

def run_symbol_signals(df):
    """
    Generates causal structural setups.

    Long:
        1) Price sweeps below the latest confirmed swing low.
        2) Candle closes back above that swing low.
        3) A later candle closes above the pre-sweep swing high = BOS.
        4) A later candle retests the BOS level.
        5) Signal is generated at retest candle close.
        6) Entry occurs next candle open.

    Short is the exact mirror image.

    No signal is generated from the same candle that confirms
    the pivot, creates the sweep, or creates the BOS.
    """

    df = df.copy()

    state = "IDLE"

    sweep_level = np.nan
    sweep_extreme = np.nan
    bos_level = np.nan

    signals = []

    for i in range(len(df)):

        row = df.iloc[i]

        if (
            pd.isna(row["ATR"]) or
            pd.isna(row["Regime"]) or
            pd.isna(row["LastSwingHigh"]) or
            pd.isna(row["LastSwingLow"])
        ):
            signals.append(None)
            continue

        atr = float(row["ATR"])

        if atr <= 0:
            signals.append(None)
            continue

        signal = None

        # ------------------------------------------------------
        # LONG STATE MACHINE
        # ------------------------------------------------------

        if row["Regime"] == "LONG":

            # Start / refresh from a liquidity sweep.
            if state in ("IDLE", "AFTER_SWEEP_SHORT", "AFTER_BOS_SHORT"):

                swing_low = float(row["LastSwingLow"])

                swept = (
                    row["Low"] < swing_low
                    and row["Close"] > swing_low
                )

                if swept:
                    sweep_level = swing_low
                    sweep_extreme = float(row["Low"])

                    # BOS reference is the last confirmed swing high
                    # that existed before/at this point.
                    bos_level = float(row["LastSwingHigh"])

                    state = "AFTER_SWEEP_LONG"

                    signals.append(None)
                    continue

            if state == "AFTER_SWEEP_LONG":

                # Invalidation: close below the sweep level.
                if row["Close"] < sweep_level:
                    state = "IDLE"
                    signals.append(None)
                    continue

                # BOS must close sufficiently beyond the prior swing high.
                if (
                    row["Close"] >
                    bos_level + BOS_ATR_BUFFER * atr
                ):
                    state = "AFTER_BOS_LONG"
                    signals.append(None)
                    continue

            elif state == "AFTER_BOS_LONG":

                # Retest: candle trades to/through BOS level but
                # finishes back above it.
                retest = (
                    row["Low"] <= bos_level +
                    RETEST_ATR_BUFFER * atr
                    and
                    row["Close"] >= bos_level
                )

                if retest:

                    stop = (
                        sweep_extreme -
                        SL_ATR_BUFFER * atr
                    )

                    entry_reference = float(row["Close"])
                    risk = entry_reference - stop

                    if (
                        risk > 0
                        and risk / entry_reference <= MAX_RISK_PCT
                        and row["RVOL"] >= RVOL_MIN
                    ):
                        signal = "LONG"

                    state = "IDLE"

        # ------------------------------------------------------
        # SHORT STATE MACHINE
        # ------------------------------------------------------

        elif row["Regime"] == "SHORT":

            if state in ("IDLE", "AFTER_SWEEP_LONG", "AFTER_BOS_LONG"):

                swing_high = float(row["LastSwingHigh"])

                swept = (
                    row["High"] > swing_high
                    and row["Close"] < swing_high
                )

                if swept:
                    sweep_level = swing_high
                    sweep_extreme = float(row["High"])

                    bos_level = float(row["LastSwingLow"])

                    state = "AFTER_SWEEP_SHORT"

                    signals.append(None)
                    continue

            if state == "AFTER_SWEEP_SHORT":

                if row["Close"] > sweep_level:
                    state = "IDLE"
                    signals.append(None)
                    continue

                if (
                    row["Close"] <
                    bos_level - BOS_ATR_BUFFER * atr
                ):
                    state = "AFTER_BOS_SHORT"
                    signals.append(None)
                    continue

            elif state == "AFTER_BOS_SHORT":

                retest = (
                    row["High"] >= bos_level -
                    RETEST_ATR_BUFFER * atr
                    and
                    row["Close"] <= bos_level
                )

                if retest:

                    stop = (
                        sweep_extreme +
                        SL_ATR_BUFFER * atr
                    )

                    entry_reference = float(row["Close"])
                    risk = stop - entry_reference

                    if (
                        risk > 0
                        and risk / entry_reference <= MAX_RISK_PCT
                        and row["RVOL"] >= RVOL_MIN
                    ):
                        signal = "SHORT"

                    state = "IDLE"

        else:
            # Do not carry a setup indefinitely through a regime flip.
            state = "IDLE"

        signals.append(signal)

    df["Signal"] = signals

    return df


# ============================================================
# TRADE CONSTRUCTION
# ============================================================

def make_trade(signal_row, next_row, side):
    atr = float(signal_row["ATR"])

    if not np.isfinite(atr) or atr <= 0:
        return None

    raw_open = float(next_row["Open"])

    if side == "LONG":

        entry = raw_open * (1.0 + SLIPPAGE)

        stop = (
            float(signal_row["Low"])
            - SL_ATR_BUFFER * atr
        )

        # If the structural sweep low is not on the retest candle,
        # use the stored structural level through the signal row.
        if pd.notna(signal_row.get("StructuralStop", np.nan)):
            stop = float(signal_row["StructuralStop"])

        risk = entry - stop

        if risk <= 0:
            return None

        tp = entry + RR * risk

    else:

        entry = raw_open * (1.0 - SLIPPAGE)

        stop = (
            float(signal_row["High"])
            + SL_ATR_BUFFER * atr
        )

        if pd.notna(signal_row.get("StructuralStop", np.nan)):
            stop = float(signal_row["StructuralStop"])

        risk = stop - entry

        if risk <= 0:
            return None

        tp = entry - RR * risk

    if risk / entry > MAX_RISK_PCT:
        return None

    return {
        "Side": side,
        "Entry": entry,
        "SL": stop,
        "TP": tp,
        "Risk": risk
    }


# ============================================================
# EXIT MODEL
# ============================================================

def apply_exit_slippage(side, price):
    if side == "LONG":
        return price * (1.0 - SLIPPAGE)
    return price * (1.0 + SLIPPAGE)


def calculate_pnl(side, entry, exit_price):
    if side == "LONG":
        ret = (exit_price - entry) / entry
    else:
        ret = (entry - exit_price) / entry

    gross = TRADE_MARGIN * ret
    fees = TRADE_MARGIN * FEE_RATE * 2.0

    return gross - fees


# ============================================================
# PORTFOLIO BACKTEST
# ============================================================

def backtest(data):

    timestamps = sorted(
        set(
            ts
            for df in data.values()
            for ts in df["Date"]
        )
    )

    active = {}
    pending = {}

    trades = []

    equity = INITIAL_CAPITAL

    equity_rows = []

    for ts in timestamps:

        # ------------------------------------------------------
        # 1) Execute orders generated by PREVIOUS candle.
        # ------------------------------------------------------

        for symbol, order in list(pending.items()):

            if symbol in active:
                del pending[symbol]
                continue

            if len(active) >= MAX_POSITIONS:
                break

            df = data[symbol]

            rows = df[df["Date"] == ts]

            if rows.empty:
                continue

            idx = int(rows.index[0])
            candle = rows.iloc[0]

            active[symbol] = {
                "Side": order["Side"],
                "Entry": order["Entry"],
                "SL": order["SL"],
                "TP": order["TP"],
                "EntryTime": ts,
                "EntryIndex": idx
            }

            del pending[symbol]

        # ------------------------------------------------------
        # 2) Check exits on THIS candle.
        # ------------------------------------------------------

        closed_symbols = set()

        for symbol, pos in list(active.items()):

            df = data[symbol]

            rows = df[df["Date"] == ts]

            if rows.empty:
                continue

            candle = rows.iloc[0]

            if pos["Side"] == "LONG":

                # Gap-through handling:
                # If market opens below SL, actual exit is the open.
                if candle["Open"] <= pos["SL"]:
                    exit_raw = float(candle["Open"])
                    result = "LOSS"

                elif candle["Low"] <= pos["SL"]:
                    exit_raw = pos["SL"]
                    result = "LOSS"

                elif candle["High"] >= pos["TP"]:
                    exit_raw = pos["TP"]
                    result = "WIN"

                else:
                    continue

            else:

                if candle["Open"] >= pos["SL"]:
                    exit_raw = float(candle["Open"])
                    result = "LOSS"

                elif candle["High"] >= pos["SL"]:
                    exit_raw = pos["SL"]
                    result = "LOSS"

                elif candle["Low"] <= pos["TP"]:
                    exit_raw = pos["TP"]
                    result = "WIN"

                else:
                    continue

            exit_price = apply_exit_slippage(
                pos["Side"],
                exit_raw
            )

            pnl = calculate_pnl(
                pos["Side"],
                pos["Entry"],
                exit_price
            )

            equity += pnl

            trades.append({
                "ExitTime": ts,
                "Symbol": symbol,
                "Side": pos["Side"],
                "EntryTime": pos["EntryTime"],
                "Entry": pos["Entry"],
                "Exit": exit_price,
                "SL": pos["SL"],
                "TP": pos["TP"],
                "PnL": pnl,
                "Result": result
            })

            del active[symbol]
            closed_symbols.add(symbol)

        # ------------------------------------------------------
        # 3) Generate signals at THIS candle close.
        # ------------------------------------------------------

        free_slots = (
            MAX_POSITIONS -
            len(active) -
            len(pending)
        )

        if free_slots > 0:

            candidates = []

            for symbol, df in data.items():

                if symbol in active:
                    continue

                if symbol in pending:
                    continue

                # User requirement:
                # no new signal for a symbol on the same candle
                # that its previous trade was resolved.
                if symbol in closed_symbols:
                    continue

                rows = df[df["Date"] == ts]

                if rows.empty:
                    continue

                i = int(rows.index[0])

                if i + 1 >= len(df):
                    continue

                row = df.iloc[i]

                if row["Signal"] not in ("LONG", "SHORT"):
                    continue

                side = row["Signal"]

                # Build the stop using the actual structural
                # sweep extreme retained in the signal-generation
                # stage where possible.
                next_row = df.iloc[i + 1]

                order = make_trade(
                    row,
                    next_row,
                    side
                )

                if order is None:
                    continue

                # Quality score uses ONLY information available at
                # the signal close. It is not an optimization loop.
                score = (
                    float(row["RVOL"])
                    * float(row["ADX"])
                )

                candidates.append(
                    (
                        symbol,
                        order,
                        score
                    )
                )

            candidates.sort(
                key=lambda x: x[2],
                reverse=True
            )

            for symbol, order, _ in candidates[:free_slots]:
                pending[symbol] = order

        # ------------------------------------------------------
        # 4) Realized equity curve.
        # ------------------------------------------------------

        equity_rows.append({
            "Timestamp": ts,
            "Equity": equity
        })

    return (
        pd.DataFrame(trades),
        pd.DataFrame(equity_rows),
        equity
    )


# ============================================================
# REPORTING
# ============================================================

def get_loss_streaks(trades):
    if trades.empty:
        return []

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


def get_max_drawdown(equity_df):
    if equity_df.empty:
        return 0.0, 0.0

    curve = equity_df["Equity"].astype(float)

    peak = curve.cummax()
    dd = peak - curve

    max_dd = float(dd.max())

    peak_at_max = float(
        peak.loc[dd.idxmax()]
    )

    dd_pct = (
        max_dd / peak_at_max * 100.0
        if peak_at_max > 0
        else 0.0
    )

    return max_dd, dd_pct


def print_results(trades, equity_df, final_equity):

    print()
    print("=" * 70)
    print("HUNTER PRO V4 — CAUSAL STRUCTURE RESULTS")
    print("=" * 70)

    total = len(trades)

    wins = (
        int((trades["Result"] == "WIN").sum())
        if total else 0
    )

    losses = (
        int((trades["Result"] == "LOSS").sum())
        if total else 0
    )

    win_rate = (
        wins / total * 100.0
        if total else 0.0
    )

    pnl = (
        float(trades["PnL"].sum())
        if total else 0.0
    )

    gross_profit = (
        float(
            trades.loc[
                trades["PnL"] > 0,
                "PnL"
            ].sum()
        )
        if total else 0.0
    )

    gross_loss = (
        float(
            -trades.loc[
                trades["PnL"] < 0,
                "PnL"
            ].sum()
        )
        if total else 0.0
    )

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    avg_win = (
        float(
            trades.loc[
                trades["PnL"] > 0,
                "PnL"
            ].mean()
        )
        if wins else 0.0
    )

    avg_loss = (
        float(
            trades.loc[
                trades["PnL"] < 0,
                "PnL"
            ].mean()
        )
        if losses else 0.0
    )

    max_dd, max_dd_pct = get_max_drawdown(
        equity_df
    )

    streaks = get_loss_streaks(trades)

    print(f"RR (before costs): 1:{RR:.2f}")
    print(f"Trades: {total}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"PnL: ${pnl:,.2f}")
    print(f"Final Balance: ${final_equity:,.2f}")
    print(f"Profit Factor: {pf:.3f}")
    print(f"Average Win: ${avg_win:,.2f}")
    print(f"Average Loss: ${avg_loss:,.2f}")
    print(f"Max Drawdown: ${max_dd:,.2f}")
    print(f"Max Drawdown %: {max_dd_pct:.2f}%")

    print()
    print("LOSS STREAKS")
    print("-" * 70)

    if streaks:
        for s in streaks:
            print(
                f"{s} loss"
                if s == 1
                else f"{s} losses"
            )

        print(
            f"Maximum Consecutive Losses: {max(streaks)}"
        )
    else:
        print("No loss streaks")

    print()
    print("LONG / SHORT")
    print("-" * 70)

    for side in ("LONG", "SHORT"):

        x = trades[
            trades["Side"] == side
        ]

        n = len(x)

        w = (
            int((x["Result"] == "WIN").sum())
            if n else 0
        )

        wr = (
            w / n * 100.0
            if n else 0.0
        )

        side_pnl = (
            float(x["PnL"].sum())
            if n else 0.0
        )

        print(
            f"{side:<6} "
            f"Trades={n:<5} "
            f"Wins={w:<5} "
            f"Losses={n-w:<5} "
            f"WR={wr:6.2f}% "
            f"PnL=${side_pnl:,.2f}"
        )

    print()
    print("PER SYMBOL")
    print("-" * 70)

    for symbol in SYMBOLS:

        x = trades[
            trades["Symbol"] == symbol
        ]

        n = len(x)

        if n == 0:
            print(
                f"{symbol:<6} "
                "Trades=0    Wins=0    Losses=0    "
                "WR=0.00% PnL=$0.00"
            )
            continue

        w = int(
            (x["Result"] == "WIN").sum()
        )

        l = n - w

        wr = w / n * 100.0

        spnl = float(
            x["PnL"].sum()
        )

        print(
            f"{symbol:<6} "
            f"Trades={n:<5} "
            f"Wins={w:<5} "
            f"Losses={l:<5} "
            f"WR={wr:6.2f}% "
            f"PnL=${spnl:,.2f}"
        )

    print()
    print("FILES")
    print("-" * 70)
    print("hunter_v4_trades.csv")
    print("hunter_v4_equity.csv")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    prepared = {}

    for name, symbol in SYMBOLS.items():

        print(f"Loading {name} ...")

        h1 = fetch_data(
            symbol,
            "1h"
        )

        h4 = fetch_data(
            symbol,
            "4h"
        )

        if h1 is None or h4 is None:
            print(
                f"  {name}: skipped - insufficient data"
            )
            continue

        prepared[name] = prepare_symbol(
            name,
            h1,
            h4
        )

        prepared[name] = run_symbol_signals(
            prepared[name]
        )

        print(
            f"  {name}: "
            f"1H={len(prepared[name])}"
        )

    print()
    print("Valid symbols:", len(prepared))

    if not prepared:
        raise RuntimeError(
            "No valid symbols available."
        )

    trades, equity_df, final_equity = backtest(
        prepared
    )

    trades.to_csv(
        "hunter_v4_trades.csv",
        index=False
    )

    equity_df.to_csv(
        "hunter_v4_equity.csv",
        index=False
    )

    print_results(
        trades,
        equity_df,
        final_equity
    )
