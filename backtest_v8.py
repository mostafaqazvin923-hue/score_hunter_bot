"""
HUNTER DTF V5 — Donchian Trend & Pullback Research Backtester
LBank USDT-M Futures / CCXT
Backtest only — no API key required.

Design goals:
- 1H execution + 4H regime
- Causal only: signals use closed candles; entry is next 1H candle open
- Donchian breakout + pullback/retest
- Fixed RR = 1:2
- No position timeout / max-bars exit
- No overlapping position per symbol
- Portfolio max open positions
- No new signal on the candle that closes a trade
- Conservative intrabar rule: if SL and TP are both touched, SL wins
- Real LBank OHLCV through CCXT
- Fees + slippage modeled
- Research variants + full-period / development / validation reporting
- CSV outputs; no matplotlib dependency (GitHub Actions friendly)

IMPORTANT:
This script is a research backtester. It does not place orders.
"""

import os
import sys
import time
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

try:
    import ccxt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
    import ccxt

try:
    import pandas as pd
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd

import numpy as np


# ============================================================
# CONFIG
# ============================================================

LOOKBACK_DAYS = 365

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "SUI/USDT",
    "ADA/USDT",
    "LINK/USDT",
    "AVAX/USDT",
    "DOT/USDT",
    "NEAR/USDT",
]

TIMEFRAME = "1h"
HTF_TIMEFRAME = "4h"

INITIAL_CAPITAL = 1000.0
TRADE_MARGIN = 100.0

# Keep this as notional unless you deliberately change the model.
# The backtester does NOT silently multiply $100 by leverage.
LEVERAGE = 1.0

FEE_RATE = 0.0006
SLIPPAGE = 0.0002

TP_R = 2.0
MAX_RISK_PCT_OF_ENTRY = 0.025

ATR_LEN = 14
ATR_STOP_BUFFER = 0.20

# Donchian settings are fixed before looking at the result.
DONCHIAN_FAST = 20
DONCHIAN_SLOW = 55

# Retest must occur shortly after a breakout.
# This is setup expiry, NOT a position timeout.
RETEST_WINDOW = 6

# Retest tolerance relative to ATR.
RETEST_ATR_TOL = 0.25

# Volume confirmation.
RVOL_LEN = 20
RVOL_MIN = 1.10

# 4H regime.
EMA_FAST = 50
EMA_SLOW = 200
ADX_LEN = 14
ADX_MIN = 18.0

# Portfolio controls.
MAX_OPEN_POSITIONS = 3

GROUPS = {
    "MAJOR": {"BTC/USDT", "ETH/USDT"},
    "L1_HIGH_BETA": {"SOL/USDT", "SUI/USDT", "AVAX/USDT"},
    "PAYMENTS": {"XRP/USDT", "ADA/USDT"},
    "INFRA": {"LINK/USDT", "NEAR/USDT"},
    "OTHER": {"DOT/USDT"},
}

CONSERVATIVE_INTRABAR = True


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Position:
    symbol: str
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    stop: float
    target: float
    risk_per_unit: float
    qty: float
    margin: float
    entry_fee: float
    entry_slippage: float
    group: str


# ============================================================
# HELPERS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def floor_ms(ms, timeframe_ms):
    return (ms // timeframe_ms) * timeframe_ms


def timeframe_to_ms(tf):
    unit = tf[-1]
    n = int(tf[:-1])
    if unit == "m":
        return n * 60_000
    if unit == "h":
        return n * 3_600_000
    if unit == "d":
        return n * 86_400_000
    raise ValueError(f"Unsupported timeframe: {tf}")


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def symbol_group(symbol):
    for g, members in GROUPS.items():
        if symbol in members:
            return g
    return "OTHER"


def previous_donchian(df, length):
    # IMPORTANT: shift first, then rolling.
    # The current candle is never part of its own breakout threshold.
    return df["high"].shift(1).rolling(length, min_periods=length).max()


def previous_donchian_low(df, length):
    return df["low"].shift(1).rolling(length, min_periods=length).min()


def true_range(df):
    prev_close = df["close"].shift(1)
    a = df["high"] - df["low"]
    b = (df["high"] - prev_close).abs()
    c = (df["low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr_wilder(df, length):
    tr = true_range(df)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def ema(series, length):
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def rvol(df, length):
    vol_ma = df["volume"].rolling(length, min_periods=length).mean()
    return df["volume"] / vol_ma.replace(0, np.nan)


def adx_wilder(df, length=14):
    up = df["high"].diff()
    down = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0),
        index=df.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0),
        index=df.index,
        dtype=float,
    )

    tr = true_range(df)

    atr = tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    plus = plus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    minus = minus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    plus_di = 100 * plus / atr.replace(0, np.nan)
    minus_di = 100 * minus / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def add_indicators(df):
    out = df.copy()

    out["atr"] = atr_wilder(out, ATR_LEN)
    out["rvol"] = rvol(out, RVOL_LEN)

    out["dc_fast_high"] = previous_donchian(out, DONCHIAN_FAST)
    out["dc_fast_low"] = previous_donchian_low(out, DONCHIAN_FAST)
    out["dc_slow_high"] = previous_donchian(out, DONCHIAN_SLOW)
    out["dc_slow_low"] = previous_donchian_low(out, DONCHIAN_SLOW)

    out["ema_fast"] = ema(out["close"], EMA_FAST)
    out["ema_slow"] = ema(out["close"], EMA_SLOW)
    out["adx"] = adx_wilder(out, ADX_LEN)

    return out


def prepare_htf(htf):
    x = htf.copy()
    x["ema_fast"] = ema(x["close"], EMA_FAST)
    x["ema_slow"] = ema(x["close"], EMA_SLOW)
    x["adx"] = adx_wilder(x, ADX_LEN)

    x["bull"] = (
        (x["ema_fast"] > x["ema_slow"])
        & (x["close"] > x["ema_fast"])
        & (x["ema_fast"] > x["ema_fast"].shift(1))
        & (x["adx"] >= ADX_MIN)
    )

    x["bear"] = (
        (x["ema_fast"] < x["ema_slow"])
        & (x["close"] < x["ema_fast"])
        & (x["ema_fast"] < x["ema_fast"].shift(1))
        & (x["adx"] >= ADX_MIN)
    )

    x["regime"] = np.where(x["bull"], "BULL", np.where(x["bear"], "BEAR", "NEUTRAL"))

    # A completed 4H candle becomes usable only after that 4H candle closes.
    x["available_at"] = x.index + pd.Timedelta(hours=4)
    return x


def merge_htf_regime(hourly, htf):
    h = hourly.reset_index().rename(columns={"index": "Date"})
    q = htf.reset_index()[["available_at", "regime", "adx"]].sort_values("available_at")

    merged = pd.merge_asof(
        h.sort_values("Date"),
        q,
        left_on="Date",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.set_index("Date")
    return merged


# ============================================================
# LBank DATA
# ============================================================

def make_exchange():
    return ccxt.lbank({
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
        },
    })


def fetch_ohlcv_paginated(exchange, symbol, timeframe, since_ms, until_ms):
    tf_ms = timeframe_to_ms(timeframe)
    rows = []
    cursor = since_ms
    limit = 1000

    while cursor < until_ms:
        try:
            batch = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=cursor,
                limit=limit,
            )
        except Exception as e:
            print(f"  ERROR fetching {symbol} {timeframe}: {e}")
            time.sleep(3)
            continue

        if not batch:
            break

        for row in batch:
            ts = int(row[0])
            if ts < since_ms:
                continue
            if ts >= until_ms:
                break
            rows.append(row)

        last_ts = int(batch[-1][0])
        next_cursor = last_ts + tf_ms

        if next_cursor <= cursor:
            break

        cursor = next_cursor

        if len(batch) < limit:
            # We may have reached the end.
            break

        time.sleep(exchange.rateLimit / 1000.0)

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    df["Date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("Date").sort_values("Date").set_index("Date")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    valid = (
        (df["high"] >= df[["open", "close"]].max(axis=1))
        & (df["low"] <= df[["open", "close"]].min(axis=1))
        & (df["high"] >= df["low"])
        & (df["volume"] >= 0)
    )
    df = df.loc[valid]

    # Remove the currently open candle.
    now_ms = exchange.milliseconds()
    last_complete_open_ms = floor_ms(now_ms, tf_ms) - tf_ms
    last_complete = pd.to_datetime(last_complete_open_ms, unit="ms", utc=True)
    df = df[df.index <= last_complete]

    return df


def fetch_all_data(exchange):
    end = utc_now()
    start = end - timedelta(days=LOOKBACK_DAYS)

    since_ms = int(start.timestamp() * 1000)
    until_ms = int(end.timestamp() * 1000)

    data_1h = {}
    data_4h = {}

    print("=" * 78)
    print("Downloading LBank Futures data")
    print("=" * 78)

    for symbol in SYMBOLS:
        print(f"[1H] {symbol}")
        d1 = fetch_ohlcv_paginated(exchange, symbol, TIMEFRAME, since_ms, until_ms)
        if len(d1) < 300:
            print(f"  WARNING: only {len(d1)} 1H candles")
        data_1h[symbol] = d1

        print(f"[4H] {symbol}")
        d4 = fetch_ohlcv_paginated(exchange, symbol, HTF_TIMEFRAME, since_ms, until_ms)
        if len(d4) < 100:
            print(f"  WARNING: only {len(d4)} 4H candles")
        data_4h[symbol] = d4

    return data_1h, data_4h


# ============================================================
# STRATEGY VARIANTS
# ============================================================

VARIANTS = {
    "A_BREAKOUT": {
        "require_retest": False,
        "require_rvol": False,
        "require_btc_regime": False,
    },
    "B_BREAKOUT_RETEST": {
        "require_retest": True,
        "require_rvol": False,
        "require_btc_regime": False,
    },
    "C_RETEST_RVOL": {
        "require_retest": True,
        "require_rvol": True,
        "require_btc_regime": False,
    },
    "D_RETEST_RVOL_BTC": {
        "require_retest": True,
        "require_rvol": True,
        "require_btc_regime": True,
    },
}


def get_btc_regime_at(btc_frame, ts):
    if btc_frame is None or btc_frame.empty:
        return "NEUTRAL"

    idx = btc_frame.index.searchsorted(ts, side="right") - 1
    if idx < 0:
        return "NEUTRAL"

    return str(btc_frame.iloc[idx]["regime"])


def signal_allowed(side, symbol_regime, btc_regime, require_btc):
    if side == "LONG" and symbol_regime != "BULL":
        return False
    if side == "SHORT" and symbol_regime != "BEAR":
        return False

    if require_btc:
        if side == "LONG" and btc_regime != "BULL":
            return False
        if side == "SHORT" and btc_regime != "BEAR":
            return False

    return True


# ============================================================
# BACKTEST ENGINE
# ============================================================

def slipped_entry(raw_open, side):
    # Adverse entry slippage.
    return raw_open * (1 + SLIPPAGE if side == "LONG" else 1 - SLIPPAGE)


def slipped_exit(raw_price, side):
    # Adverse exit slippage.
    return raw_price * (1 - SLIPPAGE if side == "LONG" else 1 + SLIPPAGE)


def fee(notional):
    return notional * FEE_RATE


def calculate_position(symbol, side, entry, stop, margin):
    if side == "LONG":
        risk = entry - stop
    else:
        risk = stop - entry

    if risk <= 0:
        return None

    risk_pct = risk / entry
    if risk_pct > MAX_RISK_PCT_OF_ENTRY:
        return None

    if side == "LONG":
        target = entry + TP_R * risk
    else:
        target = entry - TP_R * risk

    notional = margin * LEVERAGE
    qty = notional / entry

    return {
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk": risk,
        "risk_pct": risk_pct,
        "target": target,
        "qty": qty,
        "notional": notional,
    }


def setup_signal(row, prev_row, state, variant):
    """
    Returns:
      ("BREAKOUT_LONG", level) / ("BREAKOUT_SHORT", level) / None

    Breakout is always evaluated on a closed candle.
    Donchian thresholds exclude the current candle.
    """
    if pd.isna(row["atr"]):
        return None

    long_break = (
        pd.notna(row["dc_fast_high"])
        and row["close"] > row["dc_fast_high"]
        and prev_row is not None
        and row["close"] > prev_row["close"]
    )

    short_break = (
        pd.notna(row["dc_fast_low"])
        and row["close"] < row["dc_fast_low"]
        and prev_row is not None
        and row["close"] < prev_row["close"]
    )

    if long_break:
        return ("BREAKOUT_LONG", float(row["dc_fast_high"]))

    if short_break:
        return ("BREAKOUT_SHORT", float(row["dc_fast_low"]))

    return None


def retest_signal(row, setup):
    """
    Retest logic:
    Long:
      candle trades back to breakout level/tolerance and closes above it.
    Short:
      candle trades back to breakout level/tolerance and closes below it.

    The setup candle itself is not used as the retest candle.
    """
    if setup is None:
        return None

    side = setup["side"]
    level = setup["level"]

    atr = row["atr"]
    if pd.isna(atr):
        return None

    tol = float(atr) * RETEST_ATR_TOL

    if side == "LONG":
        touched = row["low"] <= level + tol
        held = row["close"] > level
        if touched and held:
            return "LONG"

    else:
        touched = row["high"] >= level - tol
        held = row["close"] < level
        if touched and held:
            return "SHORT"

    return None


def run_backtest(frames_1h, variant_name, variant_cfg, start=None, end=None):
    """
    Event-driven portfolio backtest.

    Each timestamp:
      1. Existing positions are checked for SL/TP.
      2. A position closed on this candle cannot be replaced on this same candle.
      3. New entries are generated only from closed candle information and
         are filled at the NEXT candle open.
    """

    prepared = {}
    all_times = set()

    for symbol, df in frames_1h.items():
        if df is None or df.empty:
            continue

        x = add_indicators(df)
        x = x.sort_index()

        if start is not None:
            x = x[x.index >= start]
        if end is not None:
            x = x[x.index < end]

        prepared[symbol] = x
        all_times.update(x.index.tolist())

    if not prepared:
        return empty_result(variant_name)

    all_times = sorted(all_times)

    cash = float(INITIAL_CAPITAL)
    positions = {}
    pending_entries = {}
    setup_states = {}

    trades = []
    equity_points = []
    last_exit_timestamp = {}

    # BTC regime source for optional global filter.
    btc_frame = prepared.get("BTC/USDT")

    for ts in all_times:
        closed_this_candle = set()

        # --------------------------------------------------------
        # 1) Fill entries scheduled from previous candle.
        # --------------------------------------------------------
        for symbol in list(pending_entries.keys()):
            if symbol not in prepared:
                del pending_entries[symbol]
                continue

            if ts not in prepared[symbol].index:
                continue

            if symbol in positions:
                del pending_entries[symbol]
                continue

            if symbol in last_exit_timestamp and last_exit_timestamp[symbol] == ts:
                del pending_entries[symbol]
                continue

            signal = pending_entries.pop(symbol)
            row = prepared[symbol].loc[ts]

            side = signal["side"]
            raw_open = float(row["open"])

            entry = slipped_entry(raw_open, side)

            pos_info = calculate_position(
                symbol,
                side,
                entry,
                signal["stop"],
                TRADE_MARGIN,
            )

            if pos_info is None:
                continue

            notional = pos_info["notional"]
            entry_fee = fee(notional)

            if cash < entry_fee:
                continue

            group = symbol_group(symbol)

            # Group lock + portfolio cap.
            active_groups = {
                positions[s].group
                for s in positions
            }

            if len(positions) >= MAX_OPEN_POSITIONS:
                continue

            if group in active_groups:
                continue

            cash -= entry_fee

            positions[symbol] = Position(
                symbol=symbol,
                side=side,
                entry_time=ts,
                entry_price=entry,
                stop=float(signal["stop"]),
                target=float(pos_info["target"]),
                risk_per_unit=float(pos_info["risk"]),
                qty=float(pos_info["qty"]),
                margin=TRADE_MARGIN,
                entry_fee=entry_fee,
                entry_slippage=abs(entry - raw_open) * pos_info["qty"],
                group=group,
            )

        # --------------------------------------------------------
        # 2) Manage open positions using current candle.
        # --------------------------------------------------------
        for symbol in list(positions.keys()):
            if symbol not in prepared or ts not in prepared[symbol].index:
                continue

            row = prepared[symbol].loc[ts]
            pos = positions[symbol]

            high = float(row["high"])
            low = float(row["low"])

            hit_stop = False
            hit_target = False

            if pos.side == "LONG":
                hit_stop = low <= pos.stop
                hit_target = high >= pos.target
            else:
                hit_stop = high >= pos.stop
                hit_target = low <= pos.target

            exit_reason = None
            raw_exit = None

            if hit_stop and hit_target:
                if CONSERVATIVE_INTRABAR:
                    exit_reason = "SL"
                    raw_exit = pos.stop
                else:
                    # Not used by default.
                    exit_reason = "SL"
                    raw_exit = pos.stop
            elif hit_stop:
                exit_reason = "SL"
                raw_exit = pos.stop
            elif hit_target:
                exit_reason = "TP"
                raw_exit = pos.target

            if exit_reason is None:
                continue

            exit_price = slipped_exit(raw_exit, pos.side)
            gross_pnl = (
                (exit_price - pos.entry_price) * pos.qty
                if pos.side == "LONG"
                else (pos.entry_price - exit_price) * pos.qty
            )

            exit_fee = fee(pos.qty * exit_price)
            net_pnl = gross_pnl - exit_fee - pos.entry_fee

            cash += net_pnl

            trade = {
                "symbol": symbol,
                "side": pos.side,
                "entry_time": pos.entry_time,
                "exit_time": ts,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "stop": pos.stop,
                "target": pos.target,
                "risk_per_unit": pos.risk_per_unit,
                "qty": pos.qty,
                "entry_fee": pos.entry_fee,
                "exit_fee": exit_fee,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "result": "WIN" if exit_reason == "TP" else "LOSS",
                "exit_reason": exit_reason,
                "bars_held": max(
                    1,
                    int((ts - pos.entry_time).total_seconds() // 3600),
                ),
            }
            trades.append(trade)

            del positions[symbol]
            closed_this_candle.add(symbol)
            last_exit_timestamp[symbol] = ts

            # Explicitly destroy any stale setup for this symbol.
            setup_states.pop(symbol, None)

        # --------------------------------------------------------
        # 3) Generate setups/signals from CLOSED current candle.
        #    Fill is scheduled for next candle, never current open.
        # --------------------------------------------------------
        for symbol, df in prepared.items():
            if ts not in df.index:
                continue

            if symbol in positions:
                continue

            if symbol in closed_this_candle:
                continue

            if symbol in pending_entries:
                continue

            row = df.loc[ts]

            # Find previous candle.
            loc = df.index.get_loc(ts)
            if loc == 0:
                continue

            prev_row = df.iloc[loc - 1]

            symbol_regime = row.get("regime", "NEUTRAL")
            btc_regime = get_btc_regime_at(btc_frame, ts)

            # ----------------------------------------------
            # Existing retest setup.
            # ----------------------------------------------
            setup = setup_states.get(symbol)

            if setup is not None:
                age = int(setup["age"])

                if age > RETEST_WINDOW:
                    setup_states.pop(symbol, None)
                    setup = None
                else:
                    side = setup["side"]

                    if not signal_allowed(
                        side,
                        symbol_regime,
                        btc_regime,
                        variant_cfg["require_btc_regime"],
                    ):
                        # Keep setup alive; regime may align later.
                        setup["age"] += 1
                    else:
                        if variant_cfg["require_rvol"]:
                            if pd.isna(row["rvol"]) or row["rvol"] < RVOL_MIN:
                                setup["age"] += 1
                            else:
                                retest_side = retest_signal(row, setup)
                                if retest_side is not None:
                                    if retest_side == "LONG":
                                        stop = float(row["low"] - ATR_STOP_BUFFER * row["atr"])
                                    else:
                                        stop = float(row["high"] + ATR_STOP_BUFFER * row["atr"])

                                    pending_entries[symbol] = {
                                        "side": retest_side,
                                        "stop": stop,
                                        "signal_time": ts,
                                    }
                                    setup_states.pop(symbol, None)
                                else:
                                    setup["age"] += 1
                        else:
                            retest_side = retest_signal(row, setup)
                            if retest_side is not None:
                                if retest_side == "LONG":
                                    stop = float(row["low"] - ATR_STOP_BUFFER * row["atr"])
                                else:
                                    stop = float(row["high"] + ATR_STOP_BUFFER * row["atr"])

                                pending_entries[symbol] = {
                                    "side": retest_side,
                                    "stop": stop,
                                    "signal_time": ts,
                                }
                                setup_states.pop(symbol, None)
                            else:
                                setup["age"] += 1

                    # Do not also create a new breakout setup on the same candle.
                    continue

            # ----------------------------------------------
            # New breakout.
            # ----------------------------------------------
            if not signal_allowed(
                "LONG",
                symbol_regime,
                btc_regime,
                variant_cfg["require_btc_regime"],
            ) and not signal_allowed(
                "SHORT",
                symbol_regime,
                btc_regime,
                variant_cfg["require_btc_regime"],
            ):
                continue

            breakout = setup_signal(row, prev_row, None, variant_cfg)
            if breakout is None:
                continue

            breakout_type, level = breakout

            if breakout_type == "BREAKOUT_LONG":
                if not signal_allowed(
                    "LONG",
                    symbol_regime,
                    btc_regime,
                    variant_cfg["require_btc_regime"],
                ):
                    continue

                if variant_cfg["require_rvol"]:
                    if pd.isna(row["rvol"]) or row["rvol"] < RVOL_MIN:
                        continue

                if variant_cfg["require_retest"]:
                    setup_states[symbol] = {
                        "side": "LONG",
                        "level": level,
                        "created_at": ts,
                        "age": 0,
                    }
                else:
                    stop = float(row["low"] - ATR_STOP_BUFFER * row["atr"])
                    pending_entries[symbol] = {
                        "side": "LONG",
                        "stop": stop,
                        "signal_time": ts,
                    }

            elif breakout_type == "BREAKOUT_SHORT":
                if not signal_allowed(
                    "SHORT",
                    symbol_regime,
                    btc_regime,
                    variant_cfg["require_btc_regime"],
                ):
                    continue

                if variant_cfg["require_rvol"]:
                    if pd.isna(row["rvol"]) or row["rvol"] < RVOL_MIN:
                        continue

                if variant_cfg["require_retest"]:
                    setup_states[symbol] = {
                        "side": "SHORT",
                        "level": level,
                        "created_at": ts,
                        "age": 0,
                    }
                else:
                    stop = float(row["high"] + ATR_STOP_BUFFER * row["atr"])
                    pending_entries[symbol] = {
                        "side": "SHORT",
                        "stop": stop,
                        "signal_time": ts,
                    }

        # --------------------------------------------------------
        # 4) Mark-to-market equity for reporting.
        # --------------------------------------------------------
        unrealized = 0.0

        for symbol, pos in positions.items():
            if symbol not in prepared or ts not in prepared[symbol].index:
                continue

            close = float(prepared[symbol].loc[ts]["close"])

            if pos.side == "LONG":
                unrealized += (close - pos.entry_price) * pos.qty
            else:
                unrealized += (pos.entry_price - close) * pos.qty

        equity_points.append({
            "timestamp": ts,
            "cash": cash,
            "unrealized": unrealized,
            "equity": cash + unrealized,
            "open_positions": len(positions),
        })

    # Open positions at the end are not force-closed.
    # The user explicitly requires no artificial timeout/forced end exit.
    return build_result(
        variant_name,
        trades,
        equity_points,
        prepared,
        cash,
        positions,
    )


# ============================================================
# RESULTS
# ============================================================

def empty_result(name):
    return {
        "variant": name,
        "trades": pd.DataFrame(),
        "equity": pd.DataFrame(),
        "summary": {},
        "symbol_stats": pd.DataFrame(),
        "streaks": [],
    }


def consecutive_loss_streaks(trades_df):
    if trades_df.empty:
        return []

    streaks = []
    current = 0

    for result in trades_df["result"]:
        if result == "LOSS":
            current += 1
        else:
            if current > 0:
                streaks.append(current)
                current = 0

    if current > 0:
        streaks.append(current)

    return streaks


def build_result(name, trades, equity_points, prepared, cash, positions):
    t = pd.DataFrame(trades)

    if not t.empty:
        t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True)
        t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
        t = t.sort_values("exit_time").reset_index(drop=True)

    e = pd.DataFrame(equity_points)

    if not e.empty:
        e["timestamp"] = pd.to_datetime(e["timestamp"], utc=True)
        e = e.sort_values("timestamp")
        e["peak"] = e["equity"].cummax()
        e["drawdown"] = e["equity"] - e["peak"]
        e["drawdown_pct"] = e["drawdown"] / e["peak"].replace(0, np.nan)
        max_dd = float(e["drawdown"].min())
        max_dd_pct = float(e["drawdown_pct"].min() * 100)
    else:
        max_dd = 0.0
        max_dd_pct = 0.0

    if t.empty:
        summary = {
            "variant": name,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "net_pnl": 0.0,
            "final_cash": cash,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_loss_streak": 0,
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd_pct,
            "open_positions_at_end": len(positions),
        }
    else:
        wins = t[t["net_pnl"] > 0]
        losses = t[t["net_pnl"] <= 0]

        gross_profit = float(wins["net_pnl"].sum())
        gross_loss = float(abs(losses["net_pnl"].sum()))

        pf = gross_profit / gross_loss if gross_loss > 0 else math.inf

        streaks = consecutive_loss_streaks(t)

        summary = {
            "variant": name,
            "trades": len(t),
            "wins": int((t["result"] == "WIN").sum()),
            "losses": int((t["result"] == "LOSS").sum()),
            "win_rate_pct": float((t["result"] == "WIN").mean() * 100),
            "net_pnl": float(t["net_pnl"].sum()),
            "final_cash": float(cash),
            "profit_factor": float(pf),
            "avg_win": float(wins["net_pnl"].mean()) if not wins.empty else 0.0,
            "avg_loss": float(losses["net_pnl"].mean()) if not losses.empty else 0.0,
            "max_loss_streak": max(streaks) if streaks else 0,
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd_pct,
            "open_positions_at_end": len(positions),
        }

    if not t.empty:
        rows = []
        for symbol, g in t.groupby("symbol"):
            rows.append({
                "symbol": symbol,
                "trades": len(g),
                "wins": int((g["result"] == "WIN").sum()),
                "losses": int((g["result"] == "LOSS").sum()),
                "win_rate_pct": float((g["result"] == "WIN").mean() * 100),
                "net_pnl": float(g["net_pnl"].sum()),
            })
        symbol_stats = pd.DataFrame(rows).sort_values("symbol")
    else:
        symbol_stats = pd.DataFrame(
            columns=["symbol", "trades", "wins", "losses", "win_rate_pct", "net_pnl"]
        )

    return {
        "variant": name,
        "trades": t,
        "equity": e,
        "summary": summary,
        "symbol_stats": symbol_stats,
        "streaks": consecutive_loss_streaks(t),
    }


def print_result(result, title=None):
    s = result["summary"]

    print()
    print("=" * 78)
    print(title or result["variant"])
    print("=" * 78)

    if not s:
        print("No result.")
        return

    print(f"Trades              : {s['trades']}")
    print(f"Wins / Losses       : {s['wins']} / {s['losses']}")
    print(f"Win Rate            : {s['win_rate_pct']:.2f}%")
    print(f"Net PnL             : ${s['net_pnl']:.2f}")
    print(f"Final Cash          : ${s['final_cash']:.2f}")
    print(f"Profit Factor       : {s['profit_factor']:.3f}")
    print(f"Average Win         : ${s['avg_win']:.4f}")
    print(f"Average Loss        : ${s['avg_loss']:.4f}")
    print(f"Max Drawdown        : ${s['max_drawdown']:.2f}")
    print(f"Max Drawdown %      : {s['max_drawdown_pct']:.2f}%")
    print(f"Max Loss Streak     : {s['max_loss_streak']}")
    print(f"Open at End         : {s['open_positions_at_end']}")

    print()
    print("Per Symbol")
    print("-" * 78)

    ss = result["symbol_stats"]
    if ss.empty:
        print("No closed trades.")
    else:
        print(
            ss.to_string(
                index=False,
                formatters={
                    "win_rate_pct": "{:.2f}".format,
                    "net_pnl": "{:.2f}".format,
                },
            )
        )

    print()
    print("All Consecutive Loss Streaks")
    print(result["streaks"] if result["streaks"] else "[]")


def write_result_files(result, outdir):
    os.makedirs(outdir, exist_ok=True)
    name = result["variant"]

    trades_path = os.path.join(outdir, f"{name}_trades.csv")
    equity_path = os.path.join(outdir, f"{name}_equity.csv")
    symbols_path = os.path.join(outdir, f"{name}_symbols.csv")

    result["trades"].to_csv(trades_path, index=False)
    result["equity"].to_csv(equity_path, index=False)
    result["symbol_stats"].to_csv(symbols_path, index=False)

    return trades_path, equity_path, symbols_path


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("HUNTER DTF V5 — Donchian Trend & Pullback Research Backtester")
    print("=" * 78)
    print("Causal backtest | LBank | 1H execution | 4H regime")
    print(f"Lookback             : {LOOKBACK_DAYS} days")
    print(f"Symbols              : {len(SYMBOLS)}")
    print(f"Initial capital      : ${INITIAL_CAPITAL:.2f}")
    print(f"Trade margin         : ${TRADE_MARGIN:.2f}")
    print(f"Modeled leverage     : {LEVERAGE:.2f}x")
    print(f"Fee                  : {FEE_RATE:.4%} each side")
    print(f"Slippage             : {SLIPPAGE:.4%} each side")
    print(f"Risk / Reward        : 1:{TP_R:.1f}")
    print(f"Donchian             : {DONCHIAN_FAST}/{DONCHIAN_SLOW}")
    print(f"Retest window        : {RETEST_WINDOW} candles")
    print(f"RVOL filter          : >= {RVOL_MIN:.2f}")
    print(f"Max open positions   : {MAX_OPEN_POSITIONS}")
    print()

    exchange = make_exchange()
    data_1h_raw, data_4h_raw = fetch_all_data(exchange)

    # Prepare 4H regimes and merge them causally into each 1H frame.
    frames = {}

    for symbol in SYMBOLS:
        d1 = data_1h_raw.get(symbol)
        d4 = data_4h_raw.get(symbol)

        if d1 is None or d1.empty:
            continue
        if d4 is None or d4.empty:
            continue

        htf = prepare_htf(d4)
        merged = merge_htf_regime(d1, htf)

        frames[symbol] = merged

    if not frames:
        print("ERROR: no usable data.")
        return 1

    # Split only for reporting.
    # Parameters are NOT optimized separately on these periods.
    common_start = min(df.index.min() for df in frames.values())
    common_end = max(df.index.max() for df in frames.values())

    total_days = (common_end - common_start).total_seconds() / 86400.0
    dev_end = common_start + pd.Timedelta(days=total_days * 0.67)

    print()
    print(f"Data start           : {common_start}")
    print(f"Data end             : {common_end}")
    print(f"Development cutoff  : {dev_end}")
    print()
    print("Running 4 predeclared research variants...")

    all_results = []

    for variant_name, cfg in VARIANTS.items():
        result = run_backtest(frames, variant_name, cfg)
        all_results.append(result)
        print_result(result)

    # Development / validation reporting for the most restrictive variant.
    # These are NOT used to optimize parameters.
    selected_name = "D_RETEST_RVOL_BTC"
    selected_cfg = VARIANTS[selected_name]

    validation_start = dev_end

    dev_result = run_backtest(
        frames,
        selected_name + "_DEV",
        selected_cfg,
        start=common_start,
        end=validation_start,
    )

    val_result = run_backtest(
        frames,
        selected_name + "_VALIDATION",
        selected_cfg,
        start=validation_start,
        end=common_end + pd.Timedelta(seconds=1),
    )

    print_result(dev_result, "D_RETEST_RVOL_BTC — DEVELOPMENT PERIOD")
    print_result(val_result, "D_RETEST_RVOL_BTC — VALIDATION PERIOD")

    # Write files.
    outdir = "backtest_results"

    for result in all_results:
        write_result_files(result, outdir)

    write_result_files(dev_result, outdir)
    write_result_files(val_result, outdir)

    # Compact comparison CSV.
    comparison = pd.DataFrame([r["summary"] for r in all_results])
    comparison.to_csv(
        os.path.join(outdir, "variant_comparison.csv"),
        index=False,
    )

    print()
    print("=" * 78)
    print("RESEARCH COMPARISON")
    print("=" * 78)

    if not comparison.empty:
        cols = [
            "variant",
            "trades",
            "win_rate_pct",
            "net_pnl",
            "profit_factor",
            "max_drawdown",
            "max_drawdown_pct",
            "max_loss_streak",
        ]
        print(
            comparison[cols].to_string(
                index=False,
                formatters={
                    "win_rate_pct": "{:.2f}".format,
                    "net_pnl": "{:.2f}".format,
                    "profit_factor": "{:.3f}".format,
                    "max_drawdown": "{:.2f}".format,
                    "max_drawdown_pct": "{:.2f}".format,
                },
            )
        )

    print()
    print("Files written to:", outdir)
    print("NOTE: This script deliberately does not select a 'winner' by")
    print("looking at the same period and then pretending it is out-of-sample.")
    print("Use the validation period as the first robustness check.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
