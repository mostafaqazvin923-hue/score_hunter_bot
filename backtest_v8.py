""" 
HUNTER V15 — PRECISION REGIME PULLBACK
Causal LBank USDT-M Futures backtester.

Core idea:
4H regime -> 1H impulse -> controlled pullback -> rejection/continuation.
The system uses a FIXED 1:2 risk/reward target. There is no trailing stop,
no timeout exit, no same-candle entry, and no future-looking information.

IMPORTANT:
- This program does NOT promise 60% win rate or <=4 consecutive losses.
- Those are acceptance targets. The market decides the result.
- Do not tune parameters after seeing the test result and call that validation.
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

import numpy as np
import pandas as pd


# =========================
# CONFIGURATION
# =========================
LOOKBACK_DAYS = 365
WARMUP_DAYS = 70

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "SUI/USDT",
    "ADA/USDT", "LINK/USDT", "AVAX/USDT", "DOT/USDT", "NEAR/USDT",
]

TIMEFRAME = "1h"
HTF_TIMEFRAME = "4h"

INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0
LEVERAGE = 1.0
FEE_RATE = 0.0006
SLIPPAGE = 0.0002

# Fixed design target: 1R stop, 2R target.
TP_R = 2.0
MAX_RISK_PCT_OF_ENTRY = 0.025

# 4H regime
EMA_FAST_4H = 50
EMA_SLOW_4H = 200
ATR_LEN_4H = 14
ADX_LEN_4H = 14
ADX_MIN_4H = 18.0

# 1H setup
EMA_FAST_1H = 20
EMA_MID_1H = 50
ATR_LEN_1H = 14
RSI_LEN = 14
ADX_LEN_1H = 14
RVOL_LEN = 20

# Impulse / pullback geometry
DONCHIAN_LEN = 20
IMPULSE_ATR_MIN = 1.00
PULLBACK_MAX_ATR = 1.60
PULLBACK_MIN_BARS = 1
PULLBACK_MAX_BARS = 5
EMA_TOUCH_ATR = 0.35
REJECTION_MIN_BODY_ATR = 0.15
REJECTION_CLOSE_LOCATION = 0.65
STOP_BUFFER_ATR = 0.15

# Portfolio
MAX_OPEN_POSITIONS = 3

# If both SL and TP are touched inside one candle, assume SL first.
CONSERVATIVE_INTRABAR = True


# =========================
# INDICATORS — CAUSAL
# =========================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_down = down.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_w = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    plus = 100.0 * plus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_w
    minus = 100.0 * minus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_w

    dx = 100.0 * (plus - minus).abs() / (plus + minus).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def add_indicators(df: pd.DataFrame, htf: bool = False) -> pd.DataFrame:
    x = df.copy()

    if htf:
        x["ema_fast"] = ema(x["close"], EMA_FAST_4H)
        x["ema_slow"] = ema(x["close"], EMA_SLOW_4H)
        x["atr"] = atr(x, ATR_LEN_4H)
        x["adx"] = adx(x, ADX_LEN_4H)
        x["ema_fast_slope"] = x["ema_fast"] - x["ema_fast"].shift(3)
    else:
        x["ema20"] = ema(x["close"], EMA_FAST_1H)
        x["ema50"] = ema(x["close"], EMA_MID_1H)
        x["atr"] = atr(x, ATR_LEN_1H)
        x["rsi"] = rsi(x["close"], RSI_LEN)
        x["adx"] = adx(x, ADX_LEN_1H)
        x["vol_ma"] = x["volume"].rolling(RVOL_LEN, min_periods=RVOL_LEN).mean()
        x["rvol"] = x["volume"] / x["vol_ma"].replace(0, np.nan)

        # CRITICAL: current candle is excluded from Donchian.
        x["donchian_high"] = x["high"].shift(1).rolling(
            DONCHIAN_LEN, min_periods=DONCHIAN_LEN
        ).max()
        x["donchian_low"] = x["low"].shift(1).rolling(
            DONCHIAN_LEN, min_periods=DONCHIAN_LEN
        ).min()

    return x


# =========================
# LBank DATA
# =========================
def make_exchange():
    return ccxt.lbank({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })


def fetch_ohlcv(
    exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
) -> pd.DataFrame:
    rows = []
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    cursor = since_ms

    while cursor < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, cursor, limit=1000)
        if not batch:
            break

        rows.extend(batch)
        last_open = batch[-1][0]
        next_cursor = last_open + tf_ms

        if next_cursor <= cursor:
            break

        cursor = next_cursor

        if len(batch) < 1000:
            break

        time.sleep(exchange.rateLimit / 1000.0)

    if not rows:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        )

    df = pd.DataFrame(
        rows,
        columns=["ts", "open", "high", "low", "close", "volume"],
    )
    df = df.drop_duplicates("ts").sort_values("ts")
    df["Date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)

    df = df[
        (df["ts"] >= since_ms)
        & (df["ts"] < until_ms)
    ]

    df = df.set_index("Date")[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)

    valid = (
        (df["high"] >= df[["open", "close"]].max(axis=1))
        & (df["low"] <= df[["open", "close"]].min(axis=1))
        & (df["high"] >= df["low"])
        & (df["volume"] >= 0)
    )
    df = df[valid]

    # Remove the currently open candle.
    now_ms = exchange.milliseconds()
    last_complete_open_ms = (now_ms // tf_ms) * tf_ms - tf_ms
    last_complete = pd.to_datetime(
        last_complete_open_ms,
        unit="ms",
        utc=True,
    )
    df = df[df.index <= last_complete]

    return df


def prepare_symbol(
    exchange,
    symbol: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame:
    warm_start = test_start - pd.Timedelta(days=WARMUP_DAYS)

    since_ms = int(warm_start.timestamp() * 1000)
    until_ms = int(test_end.timestamp() * 1000)

    one = fetch_ohlcv(
        exchange, symbol, TIMEFRAME, since_ms, until_ms
    )
    four = fetch_ohlcv(
        exchange, symbol, HTF_TIMEFRAME, since_ms, until_ms
    )

    one = add_indicators(one, htf=False)
    four = add_indicators(four, htf=True)

    if one.empty or four.empty:
        return pd.DataFrame()

    # A 4H candle opened at t becomes usable only after t+4h.
    h = four.copy()
    h["regime_available_at"] = h.index + pd.Timedelta(hours=4)

    bull = (
        (h["ema_fast"] > h["ema_slow"])
        & (h["close"] > h["ema_fast"])
        & (h["ema_fast_slope"] > 0)
        & (h["adx"] >= ADX_MIN_4H)
    )
    bear = (
        (h["ema_fast"] < h["ema_slow"])
        & (h["close"] < h["ema_fast"])
        & (h["ema_fast_slope"] < 0)
        & (h["adx"] >= ADX_MIN_4H)
    )

    h["regime"] = np.where(
        bull,
        "BULL",
        np.where(bear, "BEAR", "NEUTRAL"),
    )

    h = h[[
        "regime_available_at",
        "regime",
    ]].sort_values("regime_available_at")

    one = one.reset_index()
    one = pd.merge_asof(
        one.sort_values("Date"),
        h,
        left_on="Date",
        right_on="regime_available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    one = one.set_index("Date").sort_index()

    return one


# =========================
# SETUP STATE
# =========================
@dataclass
class PendingEntry:
    symbol: str
    direction: str
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    stop_ref: float
    setup_type: str
    regime: str


@dataclass
class Position:
    symbol: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop: float
    target: float
    notional: float
    risk_price: float
    signal_time: pd.Timestamp
    setup_type: str
    regime: str


def candle_body(row) -> float:
    return abs(float(row["close"]) - float(row["open"]))


def close_location(row) -> float:
    rng = float(row["high"]) - float(row["low"])
    if rng <= 0:
        return 0.5
    return (float(row["close"]) - float(row["low"])) / rng


def finite(row, names) -> bool:
    return all(np.isfinite(row.get(k, np.nan)) for k in names)


# =========================
# SIGNALS
# =========================
def long_signal(
    df: pd.DataFrame,
    i: int,
) -> Optional[Tuple[float, str]]:
    """
    Signal is evaluated only on the fully closed candle i.
    Entry is scheduled for i+1 open.
    """
    if i < max(DONCHIAN_LEN + 5, 60):
        return None

    r = df.iloc[i]
    p = df.iloc[i - 1]

    if r["regime"] != "BULL":
        return None

    needed = [
        "close", "open", "high", "low", "atr",
        "ema20", "ema50", "rsi", "adx", "rvol",
        "donchian_high",
    ]
    if not finite(r, needed):
        return None

    # 1H trend alignment.
    if not (r["close"] > r["ema20"] > r["ema50"]):
        return None
    if r["adx"] < ADX_MIN_1H:
        return None

    # Find a recent completed impulse. No future bars are examined.
    found = None
    start = max(1, i - PULLBACK_MAX_BARS - 2)

    for j in range(start, i):
        x = df.iloc[j]
        if not finite(
            x,
            ["close", "open", "high", "low", "atr", "donchian_high"],
        ):
            continue

        impulse = x["close"] > x["donchian_high"]
        impulse_size = (
            (x["high"] - x["low"]) / x["atr"]
            if x["atr"] > 0 else 0
        )

        if (
            impulse
            and impulse_size >= IMPULSE_ATR_MIN
            and x["close"] > x["open"]
        ):
            found = j
            break

    if found is None:
        return None

    bars_since = i - found
    if bars_since < PULLBACK_MIN_BARS:
        return None
    if bars_since > PULLBACK_MAX_BARS:
        return None

    imp = df.iloc[found]
    pull = df.iloc[found + 1:i]

    if len(pull) < PULLBACK_MIN_BARS:
        return None

    pull_low = float(pull["low"].min())

    # Pullback may not erase the impulse structure.
    if pull_low < imp["low"] - 0.20 * imp["atr"]:
        return None

    # Pullback must interact with the fast trend area OR remain close to
    # the impulse high. This prevents chasing extended candles.
    ema_distance = abs(float(r["low"]) - float(r["ema20"])) / float(r["atr"])
    if ema_distance > EMA_TOUCH_ATR and r["low"] > imp["high"]:
        return None

    pullback_depth = (float(imp["high"]) - pull_low) / float(imp["atr"])
    if pullback_depth > PULLBACK_MAX_ATR:
        return None

    # Rejection / continuation candle.
    if r["close"] <= r["open"]:
        return None
    if candle_body(r) < REJECTION_MIN_BODY_ATR * r["atr"]:
        return None
    if close_location(r) < REJECTION_CLOSE_LOCATION:
        return None
    if r["rsi"] < 52:
        return None
    if r["rvol"] < 1.0:
        return None
    if r["close"] <= p["high"]:
        return None

    # Structural stop below the pullback.
    stop_ref = pull_low - STOP_BUFFER_ATR * r["atr"]

    if stop_ref >= r["close"]:
        return None

    if (r["close"] - stop_ref) / r["close"] > MAX_RISK_PCT_OF_ENTRY:
        return None

    return stop_ref, "BULL_IMPULSE_PULLBACK_RECLAIM"


def short_signal(
    df: pd.DataFrame,
    i: int,
) -> Optional[Tuple[float, str]]:
    if i < max(DONCHIAN_LEN + 5, 60):
        return None

    r = df.iloc[i]
    p = df.iloc[i - 1]

    if r["regime"] != "BEAR":
        return None

    needed = [
        "close", "open", "high", "low", "atr",
        "ema20", "ema50", "rsi", "adx", "rvol",
        "donchian_low",
    ]
    if not finite(r, needed):
        return None

    if not (r["close"] < r["ema20"] < r["ema50"]):
        return None
    if r["adx"] < ADX_MIN_1H:
        return None

    found = None
    start = max(1, i - PULLBACK_MAX_BARS - 2)

    for j in range(start, i):
        x = df.iloc[j]
        if not finite(
            x,
            ["close", "open", "high", "low", "atr", "donchian_low"],
        ):
            continue

        impulse = x["close"] < x["donchian_low"]
        impulse_size = (
            (x["high"] - x["low"]) / x["atr"]
            if x["atr"] > 0 else 0
        )

        if (
            impulse
            and impulse_size >= IMPULSE_ATR_MIN
            and x["close"] < x["open"]
        ):
            found = j
            break

    if found is None:
        return None

    bars_since = i - found
    if bars_since < PULLBACK_MIN_BARS:
        return None
    if bars_since > PULLBACK_MAX_BARS:
        return None

    imp = df.iloc[found]
    pull = df.iloc[found + 1:i]

    if len(pull) < PULLBACK_MIN_BARS:
        return None

    pull_high = float(pull["high"].max())

    if pull_high > imp["high"] + 0.20 * imp["atr"]:
        return None

    ema_distance = abs(float(r["high"]) - float(r["ema20"])) / float(r["atr"])
    if ema_distance > EMA_TOUCH_ATR and r["high"] < imp["low"]:
        return None

    pullback_depth = (pull_high - float(imp["low"])) / float(imp["atr"])
    if pullback_depth > PULLBACK_MAX_ATR:
        return None

    # Rejection / continuation candle.
    if r["close"] >= r["open"]:
        return None
    if candle_body(r) < REJECTION_MIN_BODY_ATR * r["atr"]:
        return None
    if close_location(r) > (1.0 - REJECTION_CLOSE_LOCATION):
        return None
    if r["rsi"] > 48:
        return None
    if r["rvol"] < 1.0:
        return None
    if r["close"] >= p["low"]:
        return None

    stop_ref = pull_high + STOP_BUFFER_ATR * r["atr"]

    if stop_ref <= r["close"]:
        return None

    if (stop_ref - r["close"]) / r["close"] > MAX_RISK_PCT_OF_ENTRY:
        return None

    return stop_ref, "BEAR_IMPULSE_PULLBACK_RECLAIM"


# =========================
# BACKTEST ENGINE
# =========================
def run_backtest(
    data: Dict[str, pd.DataFrame],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
):
    cash = INITIAL_CAPITAL
    positions: Dict[str, Position] = {}
    pending: Dict[str, PendingEntry] = {}
    trades: List[dict] = []
    last_exit_time: Dict[str, pd.Timestamp] = {}
    equity_curve: List[dict] = []

    timestamps = sorted(set(
        ts
        for df in data.values()
        for ts in df.index
        if test_start <= ts < test_end
    ))

    for ts in timestamps:

        # --------------------------------
        # 1. FILL SIGNALS FROM PREVIOUS BAR
        # --------------------------------
        for symbol in list(pending.keys()):
            pe = pending[symbol]

            if pe.entry_time != ts:
                continue

            if symbol in positions:
                del pending[symbol]
                continue

            if last_exit_time.get(symbol) == ts:
                del pending[symbol]
                continue

            if len(positions) >= MAX_OPEN_POSITIONS:
                # The signal is not shifted forward. It expires.
                del pending[symbol]
                continue

            row = data[symbol].loc[ts]
            raw_open = float(row["open"])

            if pe.direction == "LONG":
                entry = raw_open * (1.0 + SLIPPAGE)
                stop = pe.stop_ref
                risk = entry - stop
                if risk <= 0:
                    del pending[symbol]
                    continue
                target = entry + TP_R * risk
            else:
                entry = raw_open * (1.0 - SLIPPAGE)
                stop = pe.stop_ref
                risk = stop - entry
                if risk <= 0:
                    del pending[symbol]
                    continue
                target = entry - TP_R * risk

            if risk / entry > MAX_RISK_PCT_OF_ENTRY:
                del pending[symbol]
                continue

            positions[symbol] = Position(
                symbol=symbol,
                direction=pe.direction,
                entry_time=ts,
                entry_price=entry,
                stop=stop,
                target=target,
                notional=TRADE_MARGIN * LEVERAGE,
                risk_price=risk,
                signal_time=pe.signal_time,
                setup_type=pe.setup_type,
                regime=pe.regime,
            )

            del pending[symbol]

        # --------------------------------
        # 2. MANAGE OPEN POSITIONS
        # --------------------------------
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            row = data[symbol].loc[ts]

            if pos.direction == "LONG":
                stop_hit = row["low"] <= pos.stop
                target_hit = row["high"] >= pos.target
            else:
                stop_hit = row["high"] >= pos.stop
                target_hit = row["low"] <= pos.target

            exit_reason = None

            if stop_hit and target_hit:
                exit_reason = "SL" if CONSERVATIVE_INTRABAR else "TP"
            elif stop_hit:
                exit_reason = "SL"
            elif target_hit:
                exit_reason = "TP"

            if exit_reason is None:
                continue

            raw_exit = pos.stop if exit_reason == "SL" else pos.target

            if pos.direction == "LONG":
                exit_price = raw_exit * (1.0 - SLIPPAGE)
                gross = (
                    (exit_price - pos.entry_price)
                    * pos.notional
                    / pos.entry_price
                )
                r_mult = (
                    (exit_price - pos.entry_price)
                    / pos.risk_price
                )
            else:
                exit_price = raw_exit * (1.0 + SLIPPAGE)
                gross = (
                    (pos.entry_price - exit_price)
                    * pos.notional
                    / pos.entry_price
                )
                r_mult = (
                    (pos.entry_price - exit_price)
                    / pos.risk_price
                )

            entry_fee = pos.notional * FEE_RATE
            exit_notional = pos.notional * (exit_price / pos.entry_price)
            exit_fee = exit_notional * FEE_RATE
            fees = entry_fee + exit_fee
            net = gross - fees

            trades.append({
                "symbol": symbol,
                "direction": pos.direction,
                "signal_time": pos.signal_time,
                "entry_time": pos.entry_time,
                "exit_time": ts,
                "entry_price": pos.entry_price,
                "stop": pos.stop,
                "target": pos.target,
                "exit_price": exit_price,
                "risk_price": pos.risk_price,
                "notional": pos.notional,
                "gross_pnl": gross,
                "fees": fees,
                "net_pnl": net,
                "r_multiple": r_mult,
                "result": exit_reason,
                "setup_type": pos.setup_type,
                "regime": pos.regime,
                "holding_hours": (
                    ts - pos.entry_time
                ).total_seconds() / 3600.0,
            })

            cash += net
            last_exit_time[symbol] = ts
            del positions[symbol]

        # --------------------------------
        # 3. GENERATE SIGNALS
        # --------------------------------
        # Signals use the just-closed candle only.
        # Entry happens at the NEXT candle open.
        for symbol, df in data.items():

            if ts not in df.index:
                continue

            if symbol in positions or symbol in pending:
                continue

            # No same-candle re-entry after an exit.
            if last_exit_time.get(symbol) == ts:
                continue

            idx = df.index.get_loc(ts)

            if idx + 1 >= len(df.index):
                continue

            next_ts = df.index[idx + 1]

            if next_ts >= test_end:
                continue

            sig = long_signal(df, idx)

            if sig:
                stop_ref, setup = sig
                pending[symbol] = PendingEntry(
                    symbol=symbol,
                    direction="LONG",
                    signal_time=ts,
                    entry_time=next_ts,
                    stop_ref=stop_ref,
                    setup_type=setup,
                    regime=str(df.iloc[idx]["regime"]),
                )
                continue

            sig = short_signal(df, idx)

            if sig:
                stop_ref, setup = sig
                pending[symbol] = PendingEntry(
                    symbol=symbol,
                    direction="SHORT",
                    signal_time=ts,
                    entry_time=next_ts,
                    stop_ref=stop_ref,
                    setup_type=setup,
                    regime=str(df.iloc[idx]["regime"]),
                )

        equity_curve.append({
            "Date": ts,
            "equity_closed": cash,
        })

    return (
        pd.DataFrame(trades),
        pd.DataFrame(equity_curve),
        cash,
    )


# =========================
# REPORTING
# =========================
def consecutive_loss_streaks(trades: pd.DataFrame) -> List[int]:
    if trades.empty:
        return []

    x = trades.sort_values("exit_time")
    streaks = []
    n = 0

    for result in x["result"]:
        if result == "SL":
            n += 1
        else:
            if n:
                streaks.append(n)
                n = 0

    if n:
        streaks.append(n)

    return streaks


def print_report(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    final_cash: float,
):
    print("=" * 78)
    print("HUNTER V15 — PRECISION REGIME PULLBACK")
    print("=" * 78)

    n = len(trades)

    if n:
        wins = int((trades["result"] == "TP").sum())
        losses = int((trades["result"] == "SL").sum())
        wr = 100.0 * wins / n
        pnl = final_cash - INITIAL_CAPITAL

        gross_win = trades.loc[
            trades["net_pnl"] > 0,
            "net_pnl",
        ].sum()

        gross_loss = -trades.loc[
            trades["net_pnl"] < 0,
            "net_pnl",
        ].sum()

        pf = (
            gross_win / gross_loss
            if gross_loss > 0
            else float("inf")
        )

        avg_win_r = trades.loc[
            trades["result"] == "TP",
            "r_multiple",
        ].mean()

        avg_loss_r = trades.loc[
            trades["result"] == "SL",
            "r_multiple",
        ].mean()

        avg_hold = trades["holding_hours"].mean()

    else:
        wins = losses = 0
        wr = 0.0
        pnl = final_cash - INITIAL_CAPITAL
        pf = 0.0
        avg_win_r = avg_loss_r = avg_hold = 0.0

    streaks = consecutive_loss_streaks(trades)
    max_streak = max(streaks) if streaks else 0

    if not equity.empty:
        eq = equity["equity_closed"].astype(float)
        peak = eq.cummax()
        dd = eq - peak
        max_dd = float(dd.min())

        max_dd_pct = float(
            (dd / peak.replace(0, np.nan)).min()
            * 100.0
        )
    else:
        max_dd = max_dd_pct = 0.0

    print(f"Trades             = {n}")
    print(f"Wins               = {wins}")
    print(f"Losses             = {losses}")
    print(f"Win Rate           = {wr:.2f}%")
    print(f"Net PnL            = ${pnl:+.2f}")
    print(f"Final Balance      = ${final_cash:.2f}")
    print(f"Profit Factor      = {pf:.3f}")
    print(f"Max Drawdown       = ${max_dd:.2f} ({max_dd_pct:.2f}%)")
    print(f"Max Loss Streak    = {max_streak}")
    print(f"Loss Streaks       = {streaks}")
    print(f"Avg Win R          = {avg_win_r:+.3f}")
    print(f"Avg Loss R         = {avg_loss_r:+.3f}")
    print(f"Avg Holding Hours  = {avg_hold:.2f}")
    print("Designed RR        = 1:2 FIXED")
    print("Timeout            = DISABLED")
    print("Lookahead          = NONE BY DESIGN")

    if n:
        print("\nPER SYMBOL")
        rows = []

        for sym, z in trades.groupby("symbol"):
            rows.append({
                "symbol": sym,
                "trades": len(z),
                "wins": int((z["result"] == "TP").sum()),
                "losses": int((z["result"] == "SL").sum()),
                "win_rate": 100.0 * (z["result"] == "TP").mean(),
                "net_pnl": z["net_pnl"].sum(),
            })

        print(
            pd.DataFrame(rows)
            .sort_values("net_pnl")
            .to_string(index=False)
        )

        print("\nDIRECTION")
        d = trades.groupby("direction").agg(
            trades=("result", "size"),
            win_rate=(
                "result",
                lambda x: 100.0 * (x == "TP").mean(),
            ),
            net_pnl=("net_pnl", "sum"),
        )
        print(d.to_string())

        print("\nRESULT BY MONTH")
        m = trades.copy()
        m["month"] = m["exit_time"].dt.strftime("%Y-%m")

        print(
            m.groupby("month").agg(
                trades=("result", "size"),
                win_rate=(
                    "result",
                    lambda x: 100.0 * (x == "TP").mean(),
                ),
                net_pnl=("net_pnl", "sum"),
            ).to_string()
        )

    print("\nACCEPTANCE TARGETS")
    print("Target Win Rate    >= 60%")
    print("Target Max Streak  <= 4 (NOT guaranteed)")
    print("Target RR          = 1:2")
    print("=" * 78)


# =========================
# MAIN
# =========================
def main():
    exchange = make_exchange()

    # Test window = last 365 completed days.
    end = pd.Timestamp.now(tz="UTC").floor("h")
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)

    print("=" * 78)
    print("HUNTER V15 — CAUSAL LBank DATASET")
    print(f"Test window: {start} -> {end}")
    print("=" * 78)

    data: Dict[str, pd.DataFrame] = {}

    for symbol in SYMBOLS:
        try:
            print(f"Downloading {symbol}...")

            df = prepare_symbol(
                exchange,
                symbol,
                start,
                end,
            )

            if len(df) < 300:
                print(
                    f"  SKIP: insufficient 1H candles ({len(df)})"
                )
                continue

            data[symbol] = df
            print(f"  OK: {len(df)} x 1H candles")

        except Exception as exc:
            print(
                f"  ERROR {symbol}: "
                f"{type(exc).__name__}: {exc}"
            )

    if not data:
        raise RuntimeError(
            "No symbol data was downloaded from LBank."
        )

    trades, equity, final_cash = run_backtest(
        data,
        start,
        end,
    )

    os.makedirs("backtest_results", exist_ok=True)

    trades_path = (
        "backtest_results/hunter_v15_trades.csv"
    )
    equity_path = (
        "backtest_results/hunter_v15_equity.csv"
    )

    trades.to_csv(trades_path, index=False)
    equity.to_csv(equity_path, index=False)

    print_report(
        trades,
        equity,
        final_cash,
    )

    print(f"\nSaved: {trades_path}")
    print(f"Saved: {equity_path}")


if __name__ == "__main__":
    main()
